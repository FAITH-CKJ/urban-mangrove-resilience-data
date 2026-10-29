



from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

CONTEXT=['horizon','split','method']
STATES=['reference_retention','connected_capacity','confirmed_return_fraction','EVI','NDMI']

def read(folder,name):
    return pd.read_csv(folder/name,float_precision='round_trip',keep_default_na=True)

def verify(folder):
    cfg=json.loads((folder/'evaluation_definitions.json').read_text(encoding='utf-8'))
    lam=float(cfg['normalization_lambda']); direction_tolerance=float(cfg['direction_zero_tolerance_S'])
    atol=float(cfg['numerical_comparison_absolute_tolerance'])
    normalize=lambda v: v/(lam+np.abs(v))
    score=read(folder,'holdout_score_pairs_by_block_year.csv')
    state=read(folder,'holdout_state_error_components.csv')
    direction=read(folder,'holdout_direction_pairs_by_block.csv')
    absence=read(folder,'holdout_absence_counts_by_block_year.csv')
    expected=read(folder.parent,'forecast_holdout_metrics.csv').set_index(CONTEXT).sort_index()
    ncentre={tuple(c[k] for k in CONTEXT):int(c['centres']) for c in cfg['contexts']}
    assert len(ncentre)==len(expected)==8
    assert not score.duplicated(CONTEXT+['region_id','block_id','year']).any()
    assert not state.duplicated(CONTEXT+['state','region_id','block_id','year']).any()
    assert not direction.duplicated(CONTEXT+['region_id','block_id']).any()
    assert not absence.duplicated(CONTEXT+['region_id','block_id','year']).any()
    assert sorted(score.year.unique().tolist())==cfg['scoring_years']
    assert sorted(state.year.unique().tolist())==cfg['scoring_years']
    assert (score.valid_centres>=0).all() and (state.valid_centre_years>=0).all()
    assert (state.eligible_centres>=0).all()
    rows=[];direction_rows=[]
    for key,sc in score.groupby(CONTEXT,sort=True):
        mask=lambda df: np.logical_and.reduce([df[k].eq(v).to_numpy() for k,v in zip(CONTEXT,key)])
        region_year=sc.assign(pred=sc.predicted_S_sum/sc.valid_centres,obs=sc.observed_S_sum/sc.valid_centres).groupby(['region_id','year'])[['pred','obs']].mean()
        np.testing.assert_allclose(sc.predicted_S_sum/sc.valid_centres,sc.predicted_S_mean,atol=atol,rtol=0,equal_nan=True)
        np.testing.assert_allclose(sc.observed_S_sum/sc.valid_centres,sc.observed_S_mean,atol=atol,rtol=0,equal_nan=True)
        mae_S=np.abs(region_year.pred-region_year.obs).groupby('region_id').mean().mean()
        mae_UMRI=np.abs(normalize(region_year.pred)-normalize(region_year.obs)).groupby('region_id').mean().mean()
        row=dict(zip(CONTEXT,key));row.update(S_MAE=float(mae_S),UMRI_MAE=float(mae_UMRI),valid_rows=int(sc.valid_centres.sum()),centres=ncentre[key])
        st=state.loc[mask(state)].copy()
        gb=st.groupby(['state','region_id','block_id'],sort=True)
        assert (gb.eligible_centres.nunique()==1).all()
        assert (gb.year.nunique()==len(cfg['scoring_years'])).all()
        components=gb.agg(normalized_error_sum=('centre_year_normalized_absolute_error_sum','sum'),eligible_centres=('eligible_centres','first'))
        components['mean_absolute_error']=components.normalized_error_sum/components.eligible_centres
        state_means=components.mean_absolute_error.groupby(['state','region_id']).mean().groupby('state').mean()
        assert set(state_means.index)==set(STATES)
        row.update({v:float(state_means[v]) for v in STATES})
        di=direction.loc[mask(direction)].copy()
        columns=['observed_origin_S','predicted_origin_S','observed_terminal_S','predicted_terminal_S']
        for c in columns:
            di[c]=di[c+'_sum']/di.paired_centres
            np.testing.assert_allclose(di[c],di[c+'_mean'],atol=atol,rtol=0,equal_nan=True)
        regions=di.groupby('region_id')[columns].mean()
        dp=regions.predicted_terminal_S-regions.predicted_origin_S
        do=regions.observed_terminal_S-regions.observed_origin_S
        pred_dir=np.where(np.abs(dp)<=direction_tolerance,0,np.sign(dp)).astype(int)
        obs_dir=np.where(np.abs(do)<=direction_tolerance,0,np.sign(do)).astype(int)
        correct=pred_dir==obs_dir
        row['direction_accuracy']=float(correct.mean())
        for rid,pd_,od_,ok in zip(regions.index,pred_dir,obs_dir,correct):
            direction_rows.append({**dict(zip(CONTEXT,key)),'region_id':int(rid),'predicted_direction':int(pd_),'observed_direction':int(od_),'correct':bool(ok)})
        ab=absence.loc[mask(absence)]
        countcols=['true_absent_pred_absent','true_absent_pred_present','true_present_pred_absent','true_present_pred_present']
        assert (ab[countcols]>=0).all().all()
        row['false_absence']=int(ab.true_present_pred_absent.sum())
        rows.append(row)
    actual=pd.DataFrame(rows).set_index(CONTEXT).sort_index()
    assert actual.index.equals(expected.index)
    differences={}
    for c in expected:
        x=actual[c].to_numpy(float);y=expected[c].to_numpy(float)
        if c in ['valid_rows','centres','false_absence']:
            assert np.array_equal(x,y),(c,x,y)
        else:
            np.testing.assert_allclose(x,y,atol=atol,rtol=0,equal_nan=True,err_msg=c)
        differences[c]=float(np.max(np.abs(x-y)))
    return {'status':'PASS','contexts_verified':len(actual),'state_error_rows':len(state),'score_block_year_rows':len(score),'direction_block_rows':len(direction),'absence_block_year_rows':len(absence),'max_absolute_difference_by_metric':differences,'metrics':actual.reset_index().to_dict('records'),'direction_by_region':direction_rows,'aggregation':'Centre-specific valid-year normalization is retained before equal-block and equal-region averaging; final UMRI is applied after the regional S means.','model_fitting_or_prediction_performed':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-directory',type=Path,default=Path(__file__).resolve().parents[2]/'SourceData/supplementary/holdout_evaluation')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=verify(args.data_directory)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['metrics','direction_by_region']},indent=2))

if __name__=='__main__': main()
