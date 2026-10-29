
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from derive_retention_reconnection_limit import derive

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-data',type=Path,default=Path(__file__).resolve().parents[2]/'SourceData')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'reproduced/qa/neighbourhood_observations.json')
    args=parser.parse_args()
    source=args.source_data/'figure_story'
    observations=pd.read_parquet(source/'individual_neighbourhoods_2025.parquet')
    assert len(observations)==925801 and not observations.duplicated(['block_id','centre_index']).any()
    assert int(observations.complete_history.sum())==876841
    published=pd.read_csv(source/'historical_centre_distribution_2025.csv',float_precision='round_trip')
    max_error=0.
    for row in published.itertuples(index=False):
        group=observations if row.region_id==0 else observations[observations.region_id==row.region_id]
        values=group[row.variable].dropna().to_numpy()
        assert len(values)==row.n
        actual=np.r_[values.mean(),np.quantile(values,[.1,.25,.5,.75,.9]),values.min(),values.max()]
        expected=np.array([row.mean,row.q10,row.q25,row.q50,row.q75,row.q90,row.minimum,row.maximum])
        error=float(np.max(np.abs(actual-expected)))
        assert error<1e-12,(row.region_id,row.variable,error)
        max_error=max(max_error,error)
    keys=['support','region_id','group']
    actual=derive(source/'individual_neighbourhoods_2025.parquet').set_index(keys).sort_index()
    expected=pd.read_csv(source/'retention_reconnection_limit.csv',float_precision='round_trip').set_index(keys).sort_index()
    assert actual.index.equals(expected.index)
    np.testing.assert_allclose(actual.to_numpy(),expected[actual.columns].to_numpy(),rtol=0,atol=1e-12,equal_nan=True)
    report={'status':'PASS','individual_rows':len(observations),'distribution_rows':len(published),'distribution_max_abs_error':max_error,'retention_limit_rows':len(actual),'complete_history_rows':int(observations.complete_history.sum()),'missing_canopy_values':{c:int(observations[c].isna().sum()) for c in ['EVI_delta','NDMI_delta']}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n','utf8')
    print(json.dumps(report))

if __name__=='__main__':main()
