
from pathlib import Path
import os,time,json
start=time.perf_counter()
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
import torch
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ecological_model/code'))
from model_predict import StateForecaster
model=StateForecaster(ROOT/'ecological_model/models/central.joblib',device='cpu',cpu_threads=2)
out=ROOT/'reproduced/forecast_demo';out.mkdir(parents=True,exist_ok=True)
report={'status':'PASS','n_centres':16,'years':[2026,2100],'cpu_threads':2,'branches':{}}
keys=['history_states','history_mask','history_drivers','future_drivers','static','capacity_support']
for branch in ['fixed2025','growth','protected']:
    with np.load(ROOT/'ecological_model/demo'/f'input_{branch}.npz') as z:
        inputs={k:z[k] for k in keys}
    actual=model.predict_with_quality(**inputs,batch_size=16)
    errors={}
    with np.load(ROOT/'ecological_model/demo'/f'expected_{branch}.npz') as z:
        for k in z.files:
            np.testing.assert_array_equal(np.isfinite(actual[k]),np.isfinite(z[k]))
            np.testing.assert_allclose(actual[k],z[k],rtol=0,atol=1e-7,equal_nan=True)
            errors[k]=float(np.nanmax(np.abs(actual[k].astype(float)-z[k].astype(float)),initial=0))
    np.savez_compressed(out/f'{branch}.npz',**{k:v for k,v in actual.items() if isinstance(v,np.ndarray)})
    report['branches'][branch]=errors
report['elapsed_seconds']=time.perf_counter()-start
(out/'checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
