





from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def derive(source):
    d=pd.read_parquet(source)
    assert len(d)==925801 and not d.duplicated(['block_id','centre_index']).any()
    for c in ['m_2025','C_2025','C_baseline','r_2025','g_2025','T_2025']:
        assert np.isfinite(d[c]).all(),c
    assert d.C_baseline.ge(0).all()
    assert (d.C_2025<=d.m_2025+1e-12).all()
    assert (d.r_2025<=d.m_2025+1e-12).all()
    denominator=(d.C_2025+d.C_baseline).to_numpy()
    trajectory=np.divide((d.C_2025-d.C_baseline).to_numpy(),denominator,out=np.zeros(len(d)),where=denominator>0)
    np.testing.assert_allclose(d.T_2025,trajectory,atol=1e-12,rtol=0)
    rows=[]
    for support in ['available','complete_history']:
        ss=d if support=='available' else d[d.complete_history]
        for rid in [0,*sorted(d.region_id.unique().tolist())]:
            z=ss if rid==0 else ss[ss.region_id.eq(rid)]
            canopy=z.m_2025.gt(0)&np.isfinite(z.EVI_2025)&np.isfinite(z.EVI_baseline)
            greener=canopy&z.EVI_2025.gt(z.EVI_baseline)
            assert np.array_equal(greener,canopy&z.g_2025.gt(0))
            depleted=z.C_2025.lt(z.C_baseline)
            assert np.array_equal(depleted,z.T_2025.lt(0))
            for group,mask in [('all_valid',np.ones(len(z),bool)),('greener',greener),('greener_depleted',greener&depleted),('recovery',z.r_2025.gt(0)),('recovery_depleted',z.r_2025.gt(0)&depleted)]:
                a=z.loc[mask]
                limited=a.m_2025.lt(a.C_baseline-1e-12)
                assert ((a.C_2025[limited]<a.C_baseline[limited])).all()
                n=len(a); nlow=int(limited.sum())
                deficit=(a.C_baseline-a.m_2025).clip(lower=0)
                rows.append({'support':support,'region_id':rid,'group':group,'n':n,'n_m_below_C_baseline':nlow,'fraction_m_below_C_baseline':nlow/n if n else np.nan,'mean_minimum_capacity_deficit_pp':100*deficit.mean() if n else np.nan})
    return pd.DataFrame(rows)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source-data',type=Path,default=Path(__file__).resolve().parent.parent.parent/'SourceData')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    out=derive(args.source_data/'figure_story/individual_neighbourhoods_2025.parquet')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.output,index=False,float_format='%.17g')
    print(out[out.region_id.eq(0)].to_string(index=False))
