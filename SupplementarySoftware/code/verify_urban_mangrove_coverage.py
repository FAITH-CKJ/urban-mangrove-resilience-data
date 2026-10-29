
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd

def unique_keys(x,y):
    keys=np.rec.fromarrays([x,y],names=['x','y'])
    assert len(np.unique(keys))==len(keys),'Repeated reference/grid coordinates'
    return keys
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[2]/'SourceData/urban_coverage')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'reproduced/qa/global_coverage_verification.json')
    a=parser.parse_args();d=a.data;expected=json.loads((d/'coverage_summary.json').read_text('utf-8'))
    for name,h in {**expected['inputs'],**expected['outputs']}.items():
        assert hashlib.sha256((d/name).read_bytes()).hexdigest()==h,name
    with np.load(d/'reference_cells.npz') as z:
        rx=z['grid_x'];ry=z['grid_y'];region=z['region_id'];assert int(z['pixel_area_m2'])==900
    with np.load(d/'global_baseline_cells.npz') as z:
        gx=z['grid_x'];gy=z['grid_y'];assert int(z['pixel_area_m2'])==900
    reference=unique_keys(rx,ry);global_keys=unique_keys(gx,gy)
    inside=np.isin(reference,global_keys,assume_unique=True)
    assert inside.all(),'Some actual study-reference cells are outside the global denominator'
    blocks=pd.read_csv(d/'coverage_by_block.csv')
    assert not blocks.duplicated(['block_x','block_y']).any()
    gb=pd.DataFrame({'block_x':gx//1024,'block_y':gy//1024}).value_counts().to_dict()
    rb=pd.DataFrame({'block_x':rx//1024,'block_y':ry//1024}).value_counts().to_dict()
    assert set(gb).issubset(set(zip(blocks.block_x,blocks.block_y)))
    assert set(rb).issubset(set(zip(blocks.block_x,blocks.block_y)))
    for r in blocks.itertuples():
        pair=(r.block_x,r.block_y)
        assert r.global_baseline_centres==gb.get(pair,0)
        assert r.actual_reference_centres==r.reference_inside_global==rb.get(pair,0)
        assert r.reference_outside_global==0
    regional=pd.read_csv(d/'reference_population_by_region.csv')
    for r in regional.itertuples():assert int((region==r.region_id).sum())==r.reference_centres
    n=int(inside.sum());total=len(global_keys)
    assert n==len(reference)==expected['actual_reference_centres']==934814
    assert total==expected['global_baseline_centres']==1204702
    assert int(blocks.global_baseline_centres.sum())==total
    fraction=n/total;assert abs(fraction-expected['coverage_fraction'])<1e-15
    assert 4*n>3*total
    result={'status':'PASS','global_centres':total,'study_centres':n,'unmatched_study_centres':int((~inside).sum()),
      'candidate_blocks':len(blocks),'nonempty_blocks':len(gb),'reference_area_km2':n*.0009,
      'global_area_km2':total*.0009,'coverage_percent':100*fraction,'strictly_exceeds_three_quarters':True,
      'unique_coordinates':True,'all_block_counts_match_coordinates':True,'all_source_hashes_match':True}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n','utf-8');print(json.dumps(result))
if __name__=='__main__':main()
