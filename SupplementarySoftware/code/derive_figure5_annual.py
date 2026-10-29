




from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
pa=argparse.ArgumentParser()
pa.add_argument('--source-data',type=Path,default=ROOT.parent/'SourceData')
pa.add_argument('--output',type=Path,default=ROOT/'reproduced/figure5_sources')
args=pa.parse_args()
S=args.source_data/'main_figures'
O=args.output
O.mkdir(parents=True,exist_ok=True)
inputs={}
def read(name):
    p=S/name
    inputs[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p,float_precision='round_trip')
def write(name,df):
    df.to_csv(O/name,index=False,float_format='%.17g')

a=read('Fig5_paired_native_annual.csv')
keys=['scheme','support','scenario','member_id','region_id','year']
z=a.pivot(index=keys,columns='contrast',values='delta_UMRI').reset_index()
z.columns.name=None
z['urban_loss_UMRI']=-z.urban_growth_minus_fixed
z['protection_benefit_UMRI']=z.protection_minus_growth
z['residual_loss_UMRI']=z.urban_loss_UMRI-z.protection_benefit_UMRI
z['avoided_loss_percent']=np.where(z.urban_loss_UMRI!=0,
    100*z.protection_benefit_UMRI/z.urban_loss_UMRI,np.nan)
z=z.sort_values(['scenario','member_id','year'])
assert len(z)==2250
write('Fig5_loss_members_annual.csv',z)
rows=[]
for (scenario,year),d in z.groupby(['scenario','year']):
    assert len(d)==15
    row={'scenario':scenario,'year':int(year),'members':len(d)}
    for col in ['urban_loss_UMRI','residual_loss_UMRI',
                'protection_benefit_UMRI','avoided_loss_percent']:
        vals=np.quantile(d[col],[.17,.5,.83],method='linear')
        row.update({f'{col}_q{q}':float(v) for q,v in zip([17,50,83],vals)})
    rows.append(row)
write('Fig5_loss_summary_annual.csv',pd.DataFrame(rows))

ledger=read('Fig5_allocation_overall_annual.csv')
phase=[]
for scenario in ['SSP2_protected','SSP5_protected']:
    d=ledger[ledger.scenario==scenario].sort_values('year').copy()
    assert len(d)==75 and (d.primary_realized_growth_km2>0).all()
    assert np.allclose(d.in_cell_protected_growth_km2+d.relocated_growth_km2+d.additional_unmet_growth_km2,
                       d.primary_realized_growth_km2,atol=1e-8,rtol=0)
    d['relocated_pct']=100*d.relocated_growth_km2/d.primary_realized_growth_km2
    d['unserved_pct']=100*d.additional_unmet_growth_km2/d.primary_realized_growth_km2
    d['within_pct']=100*d.in_cell_protected_growth_km2/d.primary_realized_growth_km2
    phase.extend(d.to_dict('records'))
write('Fig5_displayed_allocation_trajectories.csv',pd.DataFrame(phase))

pro=read('Fig5_protection_paired_annual.csv')
regional=read('Fig5_allocation_regional_annual.csv')
order=[5,2,6,9,3,10,8,7,4,1]
rows=[]
for j,(scenario,account) in enumerate([('ssp245','SSP2_protected'),('ssp585','SSP5_protected')]):
    benefits=pro[(pro.scenario==scenario)&(pro.year==2100)].set_index('region_id').loc[order]
    accounts=regional[(regional.scenario==account)&(regional.year==2100)].set_index('region_id').loc[order]
    for i,region in enumerate(order):
        row=benefits.loc[region]
        target=float(accounts.loc[region,'primary_realized_growth_km2'])
        pct=None if target==0 else float(100*accounts.loc[region,'additional_unmet_growth_km2']/target)
        rows.append({'region_id':region,'scenario':scenario,
            'benefit_q17_UMRI':float(row.delta_UMRI_q17),
            'benefit_q50_UMRI':float(row.delta_UMRI_q50),
            'benefit_q83_UMRI':float(row.delta_UMRI_q83),
            'unserved_pct':pct,'target_km2':target,'row':i+(-.22 if j==0 else .22)+.08})
write('Fig5_displayed_regions_2100.csv',pd.DataFrame(rows))
report={'status':'PASS','input_sha256':inputs,'member_years':2250,'annual_summaries':150,
        'annual_allocation_positions':150,'regional_benefits':20,'defined_regional_percentages':19,
        'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(O.glob('*.csv'))}}
(O/'derivation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k not in ['input_sha256','outputs']}))
