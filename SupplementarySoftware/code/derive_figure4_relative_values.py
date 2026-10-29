
from pathlib import Path
import argparse,hashlib,json
import numpy as np,pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--source-data',type=Path,default=Path(__file__).resolve().parents[2]/'SourceData');ap.add_argument('--block-member-data',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--receipt',type=Path,required=True);a=ap.parse_args()
out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
s=a.source_data.resolve(strict=True);block=a.block_member_data.resolve(strict=True)
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
cols=['member_id','scenario','urban_branch','year','region_id','valid_centres','m','C']
b=pd.read_parquet(block,columns=cols,filters=[('year','==',2100)]);b=b[b.urban_branch.isin(['growth','fixed2025'])].copy()
group=['member_id','scenario','urban_branch','year'];w=b[group+['region_id','valid_centres']].copy()
for v in ['m','C']:w[v]=b[v]*b.valid_centres
regional=w.groupby(group+['region_id'],as_index=False).sum(numeric_only=True)
overall=w.drop(columns='region_id').groupby(group,as_index=False).sum(numeric_only=True);overall['region_id']=0
states=pd.concat([regional,overall],ignore_index=True);states[['m','C']]=states[['m','C']].div(states.valid_centres,axis=0)
keys=['member_id','scenario','year','region_id'];f=states[states.urban_branch=='fixed2025'].set_index(keys).sort_index();g=states[states.urban_branch=='growth'].set_index(keys).loc[f.index]
assert len(f)==660 and (f.valid_centres==g.valid_centres).all()
native=pd.read_csv(s/'endpoint_2100/paired_habitat_2100_member.csv',float_precision='round_trip').query("contrast=='urban_growth_minus_fixed'").set_index(keys).loc[f.index]
d=f[['valid_centres']].copy();d['support']='forecast';errors={}
for v in ['m','C']:
 assert np.isfinite(f[v]).all() and (f[v]>0).all()
 pp=100*(g[v]-f[v]);errors[v]=float(np.max(np.abs(pp-native[f'delta_{v}_percentage_points'])));assert errors[v]<2e-12
 d[v+'_fixed2025']=f[v];d[v+'_growth']=g[v];d[f'delta_{v}_percentage_points']=pp;d[f'delta_{v}_relative_percent']=100*(g[v]-f[v])/f[v]
d.reset_index().to_csv(out/'Fig4_relative_habitat_2100.csv',index=False,float_format='%.17g')
cp=s/'reporting/primary_paired_contributions_UMRI.parquet';e=pd.read_parquet(cp).query("year==2100 and region_id==0 and contrast=='urban_growth_minus_fixed'").sort_values(['scenario','member_id']).copy()
assert len(e)==60 and e.groupby('scenario').size().eq(15).all() and (e.delta_UMRI<0).all()
terms=['d','T','r','g','h'];np.testing.assert_allclose(e[[f'delta_{k}_UMRI' for k in terms]].sum(axis=1),e.delta_UMRI,atol=1e-12)
for k in terms:e[k+'_loss_share_percent']=100*e[f'delta_{k}_UMRI']/e.delta_UMRI
np.testing.assert_allclose(e[[k+'_loss_share_percent' for k in terms]].sum(axis=1),100,atol=1e-12)
e.to_csv(out/'Fig4_relative_contributions_2100.csv',index=False,float_format='%.17g')
report={'status':'PASS','identity':'UMRI-v1.2.0','habitat_definition':'100*(growth-fixed2025)/fixed2025, spatial means before paired member ratios; ensemble summaries after ratios','contribution_definition':'100*component paired change / total paired change within each overall member; positive shares increase urban loss and negative shares offset loss; five shares sum to 100','habitat_members':660,'contribution_members':60,'contribution_values':300,'absolute_pp_reproduction_max_error':errors,'fixed_baseline_minimum':{v:float(f[v].min()) for v in ['m','C']},'total_UMRI_range':[float(e.delta_UMRI.min()),float(e.delta_UMRI.max())],'habitat_relative_ranges':{v:[float(d[f'delta_{v}_relative_percent'].min()),float(d[f'delta_{v}_relative_percent'].max())] for v in ['m','C']},'overall_relative_medians':d.reset_index().query('region_id==0').groupby('scenario')[[f'delta_{v}_relative_percent' for v in ['m','C']]].median().to_dict('index'),'input_sha256':{'block_member_data':sha(block),'endpoint_2100/paired_habitat_2100_member.csv':sha(s/'endpoint_2100/paired_habitat_2100_member.csv'),cp.relative_to(s).as_posix():sha(cp)},'output_sha256':{n:sha(out/n) for n in ['Fig4_relative_habitat_2100.csv','Fig4_relative_contributions_2100.csv']}}
a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(report,indent=2)+'\n','utf-8');print(json.dumps({k:report[k] for k in ['status','habitat_members','contribution_values','absolute_pp_reproduction_max_error','overall_relative_medians']}))
