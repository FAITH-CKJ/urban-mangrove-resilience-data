
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from umri_core import compute,normalize,LAMBDA

PACKAGE=Path(__file__).resolve().parents[1]
DATA=PACKAGE.parent/'SourceData'
OUT=PACKAGE/'reproduced/qa';OUT.mkdir(parents=True,exist_ok=True)
def read(name):return pd.read_csv(DATA/name)
def primary(df):return df[(df.scheme=='pooled_centres')&(df.support=='forecast')].copy()


mf=json.loads((DATA/'MANIFEST.json').read_text(encoding='utf-8'))
for name,rec in mf.items():
 p=DATA/name
 assert p.stat().st_size==rec['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'],name


b=read('methods/historical_block_annual.csv')
records=[]
source_fields=['S','pressure_contribution','recovery_contribution','growth_contribution','health_contribution','trajectory_contribution']
target_fields=['S','d_contribution','r_contribution','g_contribution','h_contribution','T_contribution']
for year,y in b.groupby('year'):
 for rid in range(11):
  x=y if rid==0 else y[y.region_id==rid]
  n=x.valid_centres.to_numpy();row={'year':year,'region_id':rid}
  for a,c in zip(source_fields,target_fields):row[c]=np.sum(x[a].to_numpy()*n)/n.sum()
  row['UMRI']=float(normalize(row['S']));records.append(row)
rebuilt=pd.DataFrame(records).set_index(['year','region_id'])
expected=read('main_figures/Fig3_historical_contributions.csv').set_index(['year','region_id']).loc[rebuilt.index]
hist_error=float(np.max(np.abs(rebuilt[target_fields+['UMRI']].to_numpy()-expected[target_fields+['UMRI']].to_numpy())))
assert hist_error<1e-12

member=primary(pd.read_parquet(DATA/'reporting/primary_paired_member_annual.parquet'))
expected_pair=primary(read('supplementary/future_paired_annual.csv'))
keys=['scenario','region_id','year','contrast']
cols=['delta_UMRI_q17','delta_UMRI_q50','delta_UMRI_q83']
quant=member.groupby(keys).delta_UMRI.quantile([.17,.5,.83]).unstack();quant.columns=cols
exp=expected_pair.set_index(keys).loc[quant.index]
pair_error=float(np.max(np.abs(quant.to_numpy()-exp[cols].to_numpy())))
assert pair_error<1e-12
groups=['member_id','scenario','region_id','contrast']
assert (member.groupby(groups).year.nunique()==75).all()
summed=member.groupby(groups).delta_UMRI.sum().reset_index()
cum=summed.groupby(['scenario','region_id','contrast']).delta_UMRI.quantile([.17,.5,.83]).unstack();cum.columns=cols
ecum=primary(read('reporting/paired_cumulative_2026_2100.csv')).set_index(['scenario','region_id','contrast']).loc[cum.index]
cum_error=float(np.max(np.abs(cum.to_numpy()-ecum[cols].to_numpy())))
assert cum_error<1e-10

state=primary(pd.read_parquet(DATA/'reporting/primary_member_annual.parquet'))
state=state[state.year==2100]
cs=['d_contribution','r_contribution','g_contribution','h_contribution','T_contribution']
means=state.groupby(['scenario','region_id','urban_branch'])[cs].mean()
change=means.xs('growth',level='urban_branch')-means.xs('fixed2025',level='urban_branch')
expected_con=read('main_figures/Fig3_urban_contributions_2100.csv').set_index(['scenario','region_id']).loc[change.index]
con_error=float(np.max(np.abs(change.to_numpy()-expected_con[['delta_'+c for c in cs]].to_numpy())))
assert con_error<1e-12
assert np.max(np.abs(expected_con[['delta_'+c for c in cs]].sum(axis=1)-expected_con.delta_S))<1e-12

ledger=read('main_figures/Fig5_allocation_regional_annual.csv')
area_error=float(np.max(np.abs(ledger.in_cell_protected_growth_km2+ledger.relocated_growth_km2+ledger.additional_unmet_growth_km2-ledger.primary_realized_growth_km2)))
assert area_error<1e-8

z=np.load(DATA/'methods/frozen_definition_sample_500m.npz',allow_pickle=False)
result=compute(z)
assert np.nanmax(np.abs(result['contributions'].sum(-1)-result['S']))<1e-12
baseline=[]
for t in range(5):
 region_means=[]
 for rid in np.unique(z['region']):
  block_means=[]
  for bid in np.unique(z['block'][z['region']==rid]):
   x=result['A'][t,(z['region']==rid)&(z['block']==bid)]
   block_means.append(float(np.nanmean(x)))
  region_means.append(float(np.nanmean(block_means)))
 baseline.append(float(np.mean(region_means)))
lambda_error=abs(float(np.mean(baseline))-LAMBDA)
assert lambda_error<1e-15


hist=read('main_figures/Fig3_historical_contributions.csv')
base=hist[hist.year==2000].set_index('region_id')
s0=base.loc[hist.region_id,'S'].to_numpy();s1=hist.S.to_numpy()
factor=np.empty(len(hist));same=s0==s1
factor[same]=LAMBDA/(LAMBDA+abs(s0[same]))**2
factor[~same]=(normalize(s1[~same])-normalize(s0[~same]))/(s1[~same]-s0[~same])
hist_points=factor[:,None]*(hist[cs].to_numpy()-base.loc[hist.region_id,cs].to_numpy())
hist_point_error=float(np.max(abs(hist_points-hist[[k+'_change_UMRI' for k in ['d','r','g','h','T']]].to_numpy())))
assert hist_point_error<1e-10
branches=primary(pd.read_parquet(DATA/'reporting/primary_member_annual.parquet'))
allpairs=pd.read_parquet(DATA/'reporting/primary_paired_contributions_UMRI.parquet')
pair_keys=['scheme','support','scenario','member_id','region_id','year']
errors=[]
for contrast,left,right in [('urban_growth_minus_fixed','growth','fixed2025'),('protection_minus_growth','protected','growth')]:
 x=branches[branches.urban_branch==left].set_index(pair_keys)
 y=branches[branches.urban_branch==right].set_index(pair_keys).loc[x.index]
 q=allpairs[allpairs.contrast==contrast].set_index(pair_keys).loc[x.index]
 a0=y.S.to_numpy();b0=x.S.to_numpy();equal=a0==b0
 f=np.empty(len(x));f[equal]=LAMBDA/(LAMBDA+abs(a0[equal]))**2
 f[~equal]=(normalize(b0[~equal])-normalize(a0[~equal]))/(b0[~equal]-a0[~equal])
 point=f[:,None]*(x[cs].to_numpy()-y[cs].to_numpy())
 cols_point=['delta_'+k+'_UMRI' for k in ['d','r','g','h','T']]
 errors.append(float(np.max(abs(point-q[cols_point].to_numpy()))))
 assert np.max(abs(point.sum(axis=1)-q.delta_UMRI.to_numpy()))<1e-9
pair_point_error=max(errors);assert pair_point_error<1e-9
point_means=allpairs.groupby(keys)[cols_point+['delta_UMRI']].mean()
mean_point_closure=float(np.max(abs(point_means[cols_point].sum(axis=1)-point_means.delta_UMRI)))
assert mean_point_closure<1e-10

report={'status':'PASS','source_files_verified':len(mf),'historical_rows_recomputed':len(rebuilt),'historical_max_abs_error':hist_error,'annual_paired_groups':len(quant),'annual_paired_max_abs_error':pair_error,'cumulative_groups':len(cum),'cumulative_max_abs_error':cum_error,'endpoint_component_max_abs_error':con_error,'allocation_closure_max_abs_km2':area_error,'calibration_lambda_max_abs_error':lambda_error,'historical_point_attribution_max_error':hist_point_error,'paired_point_attribution_max_error':pair_point_error,'paired_point_rows':len(allpairs),'mean_point_closure_max_error':mean_point_closure,'score_units':'Final UMRI=N(S), range (-1,1)'}
(OUT/'reporting_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
