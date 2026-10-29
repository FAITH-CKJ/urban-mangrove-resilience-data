




from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

import os
ROOT = Path(os.environ.get("UMRI_DATA_ROOT", Path(__file__).resolve().parents[1] / "full_data"))
from capacity_support import capacity_lower,align_capacity_support
LAMBDA = .05934161890467785
STATES = ['reference_retention','connected_capacity','confirmed_return_fraction','EVI','NDMI']
URBAN = ['urban_area_exposure_lower','urban_interface_exposure_lower','direct_urban_occupation_lower']
CLIMATE = ['tmean_c','tmax_p95_c','precip_mm_yr','dry_season_precip_mm','consecutive_dry_days','rsl_anomaly_m','rsl_rate_mm_yr']
TERRAIN = ['relative_elevation_m','slope_deg','landward_connected_space_km2','landward_urban_blocked_fraction','landward_class_coverage','baseline_direction_known_fraction']
MIGRATION = ['landward_connected_space_km2','landward_urban_blocked_fraction','landward_class_coverage']
STATIC = ['EVI_baseline','NDMI_baseline','capacity_baseline','longitude','latitude','region_id']

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()

def normalize(s):
    return s/(LAMBDA+np.abs(s))

def relative(a,b):
    a,b=np.broadcast_arrays(a,b)
    return np.where((a==0)&(b==0),0,np.divide(a-b,a+b,out=np.full_like(a,np.nan,dtype=float),where=(a+b)>0))

def score(state,urban,static,present=None):
    
    m,c,r,e,n=np.moveaxis(np.asarray(state,float),-1,0)
    absent=(m==0) if present is None else ~np.asarray(present,bool)
    d=-np.max(urban,axis=-1)
    g=np.where(absent,0,relative(e,static[...,0]))
    h=np.where(absent,-1,relative((n+1)/2,(static[...,1]+1)/2))
    t=relative(c,static[...,2])
    r=np.where(absent,0,r)
    gate=1-np.maximum.reduce([np.zeros_like(h),-h,-t])
    contributions=np.stack([.25*d,.15*r*gate,.10*(np.minimum(g,0)+gate*np.maximum(g,0)),.25*h,.25*t],-1)
    s=contributions.sum(-1)
    return {'S':s,'UMRI':normalize(s),'components':np.stack([d,r,g,h,t],-1),'contributions':contributions,'gate':gate}

def project_state(state,occupation,presence_probability=None,*,capacity_support):
    a=np.asarray(state).copy()
    a[...,0]=np.clip(a[...,0],0,1-np.clip(occupation,0,1))
    a[...,1]=np.clip(a[...,1],capacity_lower(a[...,0],capacity_support),a[...,0])
    a[...,2]=np.clip(a[...,2],0,a[...,0])
    a[...,3]=np.clip(a[...,3],0,1)
    a[...,4]=np.clip(a[...,4],-1,1)
    if presence_probability is not None:
        a[ np.asarray(presence_probability)<.5,:3]=0
    return a

def load_panel(table=None,drivers=None,formal=False):
    path=Path(table) if table else ROOT/'data/centre_year_primary.csv'
    df=pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path)
    if 'fixed_contact_exposure' in df and 'urban_interface_exposure_lower' not in df:
        df['urban_interface_exposure_lower']=df['fixed_contact_exposure']
    if 'longitude' not in df or 'latitude' not in df:
        index_path=ROOT/'forecast_2100/history/centres.parquet'
        index=pd.read_parquet(index_path,columns=['centre_id','longitude','latitude'])
        df=df.merge(index,on='centre_id',how='left',validate='many_to_one')
    if 'urban_interface_exposure_lower' not in df:
        if table is not None:
            raise ValueError('Production history requires the fixed-reference urban-interface exposure channel')
        z=np.load(ROOT/'data/primary_inputs_500m.npz')
        contact=np.where(z['fixed_landward_no_baseline_interface'],z['fixed_all_interface_urban_lower'],z['fixed_landward_urban_lower'])
        ids=[str(b)+':'+str(i) for b,i in zip(z['block'],z['centre_index'])]
        extra=pd.DataFrame({'centre_id':np.tile(ids,26),'year':np.repeat(z['years'],len(ids)),'urban_interface_exposure_lower':contact.ravel()})
        df=df.merge(extra,on=['centre_id','year'],how='left',validate='one_to_one')
    exo=URBAN.copy(); static_cols=STATIC.copy()
    driver_hash=None
    if drivers:
        p=Path(drivers)
        if p.is_dir():
            mapping_path=p/'centre_climate_mapping.parquet';cells_path=p/'climate_history_cells.csv'
            mapping=pd.read_parquet(mapping_path,columns=['centre_id','climate_cell_id'])
            mapping=mapping[mapping.centre_id.isin(df.centre_id.unique())]
            cells=pd.read_csv(cells_path)
            dr=mapping.merge(cells,on='climate_cell_id',validate='many_to_many')
            if dr.duplicated(['centre_id','year']).any():raise ValueError('Climate cell join produced duplicate centre/year')
            driver_hash=hashlib.sha256((sha256_file(mapping_path)+sha256_file(cells_path)).encode()).hexdigest()
        else:
            dr=pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p)
            driver_hash=sha256_file(p)
        keys=['centre_id','year'] if 'centre_id' in dr else ['block_id','year']
        dynamic=[c for c in CLIMATE+MIGRATION if c in dr]; terrain=[c for c in TERRAIN if c not in MIGRATION and c in dr]
        df=df.merge(dr[keys+dynamic+terrain],on=keys,how='left',validate='many_to_one')
        exo+=dynamic; static_cols+=terrain
    if formal:
        missing=set(CLIMATE+TERRAIN)-set(df.columns)
        if missing: raise ValueError('Formal model requires actual drivers: '+str(sorted(missing)))
    df=df.sort_values(['centre_id','year']).reset_index(drop=True)
    if df.duplicated(['centre_id','year']).any(): raise ValueError('Duplicate centre/year')
    counts=df.groupby('centre_id').year.agg(['count','min','max'])
    if not ((counts['count']==26)&(counts['min']==2000)&(counts['max']==2025)).all():
        raise ValueError('Expected all 26 historical rows per centre; preserve missing rows')
    n=len(counts); shape=(n,26)
    state=df[STATES].to_numpy(float).reshape(*shape,5)
    known=df.valid_primary.to_numpy(bool).reshape(shape)
    habitat=df.known_habitat_absence.to_numpy(bool).reshape(shape)
    mask=np.isfinite(state); mask[...,3:]=mask[...,3:] & ~habitat[...,None]
    exogenous=df[exo].to_numpy(float).reshape(*shape,len(exo))
    static=df[static_cols].to_numpy(float).reshape(*shape,len(static_cols))[:,0]
    if formal and (not np.isfinite(exogenous[...,3:]).all() or not np.isfinite(static[...,6:]).all()):
        raise ValueError('Formal driver coverage incomplete; inspect and resolve true source missingness')
    first=df.iloc[::26]
    
    blocks=first[['region_id','block_id']].drop_duplicates().sort_values(['region_id','block_id'])
    frozen_split=ROOT/'forecast_2100/models/spatial_holdout_blocks.json'
    if frozen_split.exists():
        held=json.loads(frozen_split.read_text(encoding='utf8'))['holdout_blocks']
    else:
        held=[]
        for _,g in blocks.groupby('region_id'):
            ordered=sorted(g.block_id,key=lambda x:hashlib.sha256(('UMRI2100:'+x).encode()).hexdigest())
            held.append(ordered[0])
    spatial=first.block_id.isin(held).to_numpy()
    panel=dict(state=state,mask=mask,valid=known,present=~habitat,exo=exogenous,static=static,
               S=df.S.to_numpy().reshape(shape),centre=first.centre_id.to_numpy(),block=first.block_id.to_numpy(),
               region=first.region_id.to_numpy(),holdout=spatial,
               evaluation_sample=(first['sample_role'].to_numpy()=='uniform') if 'sample_role' in first else np.ones(n,bool),
               sampling_weight=first['centre_sampling_weight'].to_numpy(float) if 'centre_sampling_weight' in first else np.ones(n),
               exo_names=exo,static_names=static_cols,
               source=str(path.resolve()),source_sha256=sha256_file(path),
               driver_source=str(Path(drivers).resolve()) if drivers else None,driver_sha256=driver_hash,
               status='FORMAL_HISTORICAL_DRIVERS' if formal else ('PILOT_PARTIAL_OBSERVED_DRIVERS_NOT_FINAL_SCENARIO' if drivers else 'PILOT_URBAN_ONLY_NOT_FINAL_SCENARIO'))
    panel['capacity_support']=align_capacity_support(panel['centre'])
    return panel,df

def export_contract(panel,df,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    years=df.year.to_numpy(); spatial=np.repeat(panel['holdout'],26)
    split=np.where(years<=2015,'train',np.where(years<=2020,'validation','test'))
    rows=df[['centre_id','block_id','region_id','year']].copy()
    rows['source_row_after_sort']=np.arange(len(df));rows['time_split']=split;rows['whole_block_holdout']=spatial
    rows['evaluation_uniform_sample']=np.repeat(panel['evaluation_sample'],26)
    if len(rows)<100000:rows.to_csv(out/'split_rows.csv',index=False)
    else:rows.to_parquet(out/'split_rows.parquet',index=False)
    if panel['driver_source']:
        snapshot_columns=['centre_id','year']+[c for c in panel['exo_names']+panel['static_names'] if c in CLIMATE+TERRAIN]
        df[snapshot_columns].to_parquet(out/'driver_feature_snapshot.parquet',index=False)
    obs=score(panel['state'],panel['exo'][...,:3],panel['static'][:,None,:],panel['present'])
    valid=panel['valid'] & np.isfinite(obs['S'])
    err=float(np.max(np.abs(obs['S'][valid]-panel['S'][valid])))
    if err>1e-12:raise AssertionError(f'Score reconstruction parity failed {err}')
    facts={'schema':'UMRI2100_STATE_V1','status':panel['status'],'state_order':STATES,'dynamic_order':panel['exo_names'],
           'static_order':panel['static_names'],'axes':'centre,year,feature','historical_years':[2000,2025],
           'source':panel['source'],'source_sha256':panel['source_sha256'],
           'driver_source':panel['driver_source'],'driver_sha256':panel['driver_sha256'],'centres':len(panel['centre']),
           'blocks':len(np.unique(panel['block'])),'rows':len(df),'valid_score_rows':int(valid.sum()),
           'score_parity_max_abs_error':err,'whole_block_holdout':sorted(np.unique(panel['block'][panel['holdout']]).tolist()),
           'known_absent_rows':int((~panel['present']).sum()),'missing_state_values':int((~panel['mask']).sum()),
           'training_absent_rows_2006_2015':int((~panel['present'][~panel['holdout'],6:16]).sum()),
           'uniform_evaluation_centres':int(panel['evaluation_sample'].sum()),
           'lambda':LAMBDA,'constraints':['max(m/sqrt(H),m/sqrt(K_B+3*H*(1-m)))<=C<=m','0<=r<=m','0<=m<=1-occupation'],
           'future_driver_required_keys':['centre_id','year','scenario','urban_branch','member_id'],
           'future_annual_years':[2026,2100],'test_access':'not evaluated in pilot model selection'}
    (out/'input_contract.json').write_text(json.dumps(facts,indent=2),encoding='utf8')
    return facts
