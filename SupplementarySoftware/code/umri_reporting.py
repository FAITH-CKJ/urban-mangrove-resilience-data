






from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from umri_core import LAMBDA,WEIGHTS,normalize
COMPONENTS=('d','r','g','h','T')
TERMS=[k+'_contribution' for k in COMPONENTS]
MEMBER_KEYS=['scheme','support','member_id','scenario','region_id','year']
GROUP_KEYS=['scheme','support','scenario','region_id','year','contrast']

def secant(s0,s1):
    
    a,b=np.broadcast_arrays(np.asarray(s0,dtype=float),np.asarray(s1,dtype=float))
    aa,bb=np.abs(a),np.abs(b)
    crossing=np.signbit(a)!=np.signbit(b)
    cross_term=np.divide(2*aa*bb,aa+bb,out=np.zeros_like(a),where=(aa+bb)>0)
    return (LAMBDA+np.where(crossing,cross_term,0))/((LAMBDA+aa)*(LAMBDA+bb))

def read_csv(p):return pd.read_csv(p,float_precision='round_trip')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_csv(d,p):d.to_csv(p,index=False,float_format='%.17g')

def historical_attribution(frame):
    out=frame.copy();factor=1/(LAMBDA+np.abs(out.S))
    for k in COMPONENTS:out[k+'_state_UMRI']=factor*out[k+'_contribution']
    base=out[out.year==2000][['region_id','S','UMRI']+TERMS].set_index('region_id')
    b=base.loc[out.region_id].reset_index(drop=True)
    coeff=secant(b.S.to_numpy(),out.S.to_numpy())
    out['baseline_year']=2000;out['baseline_UMRI']=b.UMRI.to_numpy()
    out['change_UMRI']=out.UMRI-out.baseline_UMRI
    for k in COMPONENTS:out[k+'_change_UMRI']=coeff*(out[k+'_contribution'].to_numpy()-b[k+'_contribution'].to_numpy())
    return out

def derive(source):
    source=Path(source).resolve()
    calibration=json.loads((source/'methods/classification_calibration.json').read_text('utf8'))
    assert calibration['lambda']['value']==LAMBDA
    input_files=[source/'reporting/primary_member_annual.parquet',source/'reporting/primary_paired_member_annual.parquet',source/'main_figures/Fig3_historical_contributions.csv',source/'main_figures/Fig3_gross_contribution_balance_2025.csv']
    input_hashes={p.relative_to(source).as_posix():sha(p) for p in input_files}
    branch_all=pd.read_parquet(input_files[0]);paired_all=pd.read_parquet(input_files[1])
    branch=branch_all[branch_all.support=='forecast'].copy()
    assert branch[TERMS].notna().all().all()
    np.testing.assert_allclose(branch[TERMS].sum(axis=1),branch.S,rtol=0,atol=1e-10)
    np.testing.assert_allclose(normalize(branch.S),branch.UMRI,rtol=0,atol=1e-12)
    parts=[];pair_checks=[]
    for contrast,(left,right) in {'urban_growth_minus_fixed':('fixed2025','growth'),'protection_minus_growth':('growth','protected')}.items():
        a=branch[branch.urban_branch==left][MEMBER_KEYS+['S','UMRI']+TERMS]
        b=branch[branch.urban_branch==right][MEMBER_KEYS+['S','UMRI']+TERMS]
        m=a.merge(b,on=MEMBER_KEYS,suffixes=('_0','_1'),validate='one_to_one')
        factor=secant(m.S_0,m.S_1);p=m[MEMBER_KEYS].copy();p['contrast']=contrast
        p['delta_UMRI']=m.UMRI_1-m.UMRI_0
        for k in COMPONENTS:p['delta_'+k+'_UMRI']=factor*(m[k+'_contribution_1']-m[k+'_contribution_0'])
        p['ecological_delta_UMRI']=p[['delta_'+k+'_UMRI' for k in COMPONENTS[1:]]].sum(axis=1)
        error=float(np.max(np.abs(p[['delta_'+k+'_UMRI' for k in COMPONENTS]].sum(axis=1)-p.delta_UMRI)))
        assert error<1e-12
        check=paired_all[(paired_all.support=='forecast')&(paired_all.contrast==contrast)][MEMBER_KEYS+['contrast','delta_UMRI']].merge(p,on=MEMBER_KEYS+['contrast'],validate='one_to_one',suffixes=('_stored',''))
        assert len(check)==len(p)
        np.testing.assert_allclose(check.delta_UMRI_stored,check.delta_UMRI,rtol=0,atol=1e-12)
        pair_checks.append({'contrast':contrast,'comparisons':len(p),'closure_max_UMRI':error,'equal_intermediate_sums':int((m.S_0==m.S_1).sum()),'cross_zero':int((m.S_0*m.S_1<0).sum())})
        parts.append(p)
    pairs=pd.concat(parts,ignore_index=True).sort_values(GROUP_KEYS+['member_id']).reset_index(drop=True)
    value_cols=['delta_UMRI']+['delta_'+k+'_UMRI' for k in COMPONENTS]+['ecological_delta_UMRI']
    grouped=pairs.groupby(GROUP_KEYS,sort=True,dropna=False)
    means=grouped[value_cols].mean().add_suffix('_mean').reset_index();means['members']=grouped.size().to_numpy()
    for q,label in [(.17,'q17'),(.5,'q50'),(.83,'q83')]:means=means.merge(grouped.delta_UMRI.quantile(q).rename('delta_UMRI_'+label).reset_index(),on=GROUP_KEYS,validate='one_to_one')
    means['ecological_share_percent']=np.divide(100*means.ecological_delta_UMRI_mean,means.delta_UMRI_mean,out=np.full(len(means),np.nan),where=means.delta_UMRI_mean.to_numpy()!=0)
    mean_error=float(np.max(np.abs(means[['delta_'+k+'_UMRI_mean' for k in COMPONENTS]].sum(axis=1)-means.delta_UMRI_mean)))
    assert mean_error<1e-12 and (means.members==15).all()
    for path in sorted(source.rglob('*.csv')):
        d=read_csv(path)
        if set(['year','region_id','S','UMRI']+TERMS)<=set(d.columns) and 2000 in set(d.year):write_csv(historical_attribution(d),path)
    for k in COMPONENTS:branch_all[k+'_state_UMRI']=branch_all[k+'_contribution']/(LAMBDA+np.abs(branch_all.S))
    branch_all.to_parquet(input_files[0],index=False,compression='zstd')
    pairs.to_parquet(source/'reporting/primary_paired_contributions_UMRI.parquet',index=False,compression='zstd')
    write_csv(means,source/'reporting/primary_paired_contributions_UMRI_annual.csv')
    write_csv(means[(means.year==2100)&(means.contrast=='urban_growth_minus_fixed')],source/'main_figures/Fig3_urban_UMRI_contributions_2100.csv')
    keys=['scheme','support','scenario','urban_branch','region_id','year'];state_cols=['UMRI']+[k+'_state_UMRI' for k in COMPONENTS]
    state_means=branch_all[branch_all.support=='forecast'].groupby(keys)[state_cols].mean().add_suffix('_mean').reset_index()
    write_csv(state_means,source/'reporting/primary_state_contributions_UMRI_annual.csv')
    gross_path=source/'main_figures/Fig3_gross_contribution_balance_2025.csv';gross=read_csv(gross_path);factor=1/(LAMBDA+np.abs(gross.S))
    for k in COMPONENTS:gross[k+'_state_UMRI']=factor*gross[k+'_contribution']
    gross['P_UMRI']=factor*gross.P;gross['N_UMRI']=factor*gross.N;gross['net_UMRI']=gross.P_UMRI-gross.N_UMRI
    gross['adverse_share_percent']=100*gross.N/(gross.P+gross.N);write_csv(gross,gross_path)
    hist=read_csv(source/'main_figures/Fig3_historical_contributions.csv')
    hist_state=float(np.max(np.abs(hist[[k+'_state_UMRI' for k in COMPONENTS]].sum(axis=1)-hist.UMRI)))
    hist_change=float(np.max(np.abs(hist[[k+'_change_UMRI' for k in COMPONENTS]].sum(axis=1)-hist.change_UMRI)))
    assert hist_state<1e-12 and hist_change<1e-12
    endpoint=hist[(hist.region_id==0)&(hist.year==2025)].iloc[0]
    report={'revision':'final-UMRI-score','status':'PASS','scientific_identity':'UMRI-v1.2.0','unit_definition':'Final dimensionless UMRI=N(S)=S/(lambda+abs(S)); range (-1,1)','cumulative_unit':'UMRI-years','lambda':LAMBDA,'weights':dict(zip(COMPONENTS,WEIGHTS.tolist())),'historical_states':len(hist),'historical_state_closure_max_UMRI':hist_state,'historical_change_closure_max_UMRI':hist_change,'paired_checks':pair_checks,'ensemble_mean_closure_max_UMRI':mean_error,
    'missing_component_semantics':'Matched-2025 branch rows retain missing component fields; component attribution uses forecast support.',
    'spatial_order':'Aggregate effective weighted intermediate terms over the defined reference support, then apply the fixed UMRI mapping.','ensemble_order':'Compute each paired member component attribution first, then average.',
    'historical_baseline_year':2000,'historical_overall_2000_to_2025':{c:float(endpoint[c]) for c in ['baseline_UMRI','UMRI','change_UMRI']+[k+'_change_UMRI' for k in COMPONENTS]},'overall_2100_effects':means[(means.year==2100)&(means.region_id==0)].to_dict(orient='records'),'gross_2025':gross[['P_UMRI','N_UMRI','net_UMRI','adverse_share_percent']].to_dict(orient='records'),'input_sha256':input_hashes,
    'formulas':{'UMRI':'UMRI=N(S)=S/(lambda+abs(S))','intermediate_sum':'S=sum(c_k)','state_component':'C_k=c_k/(lambda+abs(S))','change_component':'A_k=(c_k1-c_k0)*secant(S0,S1)','secant_same_sign':'lambda/((lambda+abs(S0))*(lambda+abs(S1)))','secant_opposite_sign':'(lambda+2*abs(S0)*abs(S1)/(abs(S0)+abs(S1)))/((lambda+abs(S0))*(lambda+abs(S1)))'},
    'field_semantics':{'S':'Computational intermediate: effective weighted sum','*_contribution':'Effective weighted component terms used as computational intermediates','UMRI':'Final dimensionless UMRI score','*_state_UMRI':'Component of the final UMRI state','*_change_UMRI':'Component of the final UMRI change','delta_*_UMRI':'Within-member component of the final UMRI contrast','*UMRI_years*':'Sum of annual final-UMRI differences','UMRI_from_mean_S':'UMRI evaluated at the ensemble-mean intermediate sum; mean member results are stored separately'}}
    (source/'methods/umri_reporting.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n','utf8')
    return report
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source-data',type=Path,default=Path(__file__).resolve().parents[2]/'SourceData');a=ap.parse_args();r=derive(a.source_data)
    print(json.dumps({k:r[k] for k in ['status','historical_states','historical_change_closure_max_UMRI','paired_checks','ensemble_mean_closure_max_UMRI','historical_overall_2000_to_2025']},indent=2))
