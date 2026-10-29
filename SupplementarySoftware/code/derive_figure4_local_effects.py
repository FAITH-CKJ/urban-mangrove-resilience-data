





from pathlib import Path
import argparse,datetime as dt,hashlib,json
import numpy as np
import pandas as pd
ap=argparse.ArgumentParser()
ap.add_argument('--endpoint-directory',type=Path,required=True)
ap.add_argument('--centres',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--source-data',type=Path,default=Path(__file__).resolve().parents[2]/'SourceData')
ap.add_argument('--validation-output',type=Path)
args=ap.parse_args()
out=args.output
S=args.source_data.resolve(strict=True)
SC=['ssp126','ssp245','ssp370','ssp585'];SCI='UMRI-v1.2.0'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def dump(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
origin=args.endpoint_directory.resolve(strict=True)
protocol=load(origin/'protocol.json');terminal=load(origin/'terminal.json')
assert terminal['status']=='PASS' and terminal['identity']==SCI
centres_path=args.centres.resolve(strict=True)
centres=pd.read_parquet(centres_path)
assert len(centres)==934814 and not centres.duplicated(['block_id','centre_index']).any()
lookup={k:g.set_index('centre_index') for k,g in centres.groupby('block_id')}
pieces=[];sources={}
blocks=sorted((origin/'blocks').glob('*.npz'));assert len(blocks)==141
for bp in blocks:
    receipt=load(origin/'receipts'/(bp.stem+'.json'))
    digest=sha(bp);assert receipt['status']=='PASS' and receipt['payload_sha256']==digest
    with np.load(bp,allow_pickle=False) as z:
        c=lookup[bp.stem].loc[z['centre_index']].reset_index()
        np.testing.assert_array_equal(c.region_id,z['region_id'])
        np.testing.assert_array_equal(z['scenarios'],SC)
        assert len(c)==len(lookup[bp.stem])
        c=c[['centre_index','region_id','longitude','latitude']].copy()
        c['block_id']=bp.stem;c['valid']=z['valid']
        for j,s in enumerate(SC):
            
            c[s+'_delta']=z['urban_median_delta_UMRI_points'][j].astype('float64')/100
            c[s+'_votes']=z['urban_erosion_votes'][j]
            assert c.loc[c.valid,s+'_delta'].notna().all()
            assert c.loc[~c.valid,s+'_delta'].isna().all()
    pieces.append(c)
    sources[bp.name]={'sha256':digest,'receipt_sha256':sha(origin/'receipts'/(bp.stem+'.json'))}
full=pd.concat(pieces,ignore_index=True)
assert len(full)==934814 and int(full.valid.sum())==926220
assert not full.duplicated(['block_id','centre_index']).any()
full['longitude_bin']=np.floor(full.longitude).astype(int)
full['latitude_bin']=np.floor(full.latitude).astype(int)
rows=[]
for (rid,xb,yb),g in full.groupby(['region_id','longitude_bin','latitude_bin'],sort=True):
    good=g[g.valid]
    for s in SC:
        values=good[s+'_delta'].to_numpy();neg=values<0
        row={'scenario':s,'region_id':int(rid),'longitude_bin':int(xb),'latitude_bin':int(yb),
             'longitude':float(g.longitude.mean()),'latitude':float(g.latitude.mean()),
             'reference_n':len(g),'reference_area_km2':len(g)*.0009,
             'valid_n':len(good),'unknown_n':len(g)-len(good),
             'negative_n':int(neg.sum()),'zero_n':int((values==0).sum()),'positive_n':int((values>0).sum()),
             'all_members_negative_n':int((good[s+'_votes']==15).sum()),
             'mean_local_delta_UMRI':float(values.mean()) if len(values) else np.nan,
             'median_local_delta_UMRI':float(np.median(values)) if len(values) else np.nan,
             'local_delta_UMRI_q10':float(np.quantile(values,.1)) if len(values) else np.nan,
             'local_delta_UMRI_q90':float(np.quantile(values,.9)) if len(values) else np.nan,
             'min_local_delta_UMRI':float(values.min()) if len(values) else np.nan,
             'max_local_delta_UMRI':float(values.max()) if len(values) else np.nan}
        rows.append(row)
out.parent.mkdir(parents=True,exist_ok=True)
table=pd.DataFrame(rows);table.to_csv(out,index=False,float_format='%.17g')
fixed=pd.read_csv(S/'endpoint_2100/pixelwise_median_negative_area_2100.csv').query("region_id==0 and support=='forecast' and metric=='urban_erosion'").set_index('scenario')
for s in SC:
    z=table[table.scenario==s]
    assert z.reference_n.sum()==934814 and z.valid_n.sum()==926220
    assert z.negative_n.sum()==int(fixed.loc[s,'negative_n'])
    assert z.all_members_negative_n.sum()==int(fixed.loc[s,'all_members_negative_n'])
    assert (z.negative_n+z.zero_n+z.positive_n+z.unknown_n==z.reference_n).all()
published=pd.read_csv(S/'main_figures/Fig2_reference_distribution_1deg.csv').set_index(['region_id','longitude_bin','latitude_bin']).sort_index()
new=table[table.scenario==SC[0]].set_index(['region_id','longitude_bin','latitude_bin']).sort_index()
assert list(new.index)==list(published.index) and len(new)==73
np.testing.assert_array_equal(new.reference_n,published.reference_centres)
np.testing.assert_allclose(new[['longitude','latitude']],published[['longitude','latitude']],atol=1e-10)
dump(args.validation_output or out.with_suffix('.validation.json'),{'status':'PASS','scientific_identity':SCI,
    'definition':'Area-weighted mean of the within-centre 15-member median paired urban-growth minus fixed-2025 UMRI difference within each region-specific 1-degree display cell. Equal 30-m reference cells have equal weight.',
    'not_regional_estimator':'This local-map display average does not replace normalization of the regional spatial mean score before paired-member summary.',
    'reference_n':934814,'valid_n':926220,'unknown_n':8594,'display_cells':73,'scenarios':SC,
    'original_field_unit_conversion':'.astype(float64)/100 applied to archived *_UMRI_points arrays',
    'unknown_handling':'Unknown centres remain in reference counts and are excluded from local display means; no missing value is replaced with zero.',
    'coordinate_source':'explicit --centres input','coordinate_sha256':sha(centres_path),
    'source_blocks':sources,'output_sha256':sha(out),'global_and_region_support_reconciled':True,
    'same_1_degree_support_as_accepted_figure2':True,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
print('SPATIAL',len(table),table.query("scenario=='ssp585'").mean_local_delta_UMRI.agg(['min','max']).to_dict(),flush=True)
