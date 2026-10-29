






from pathlib import Path
import numpy as np

import os
REV=Path(os.environ.get("UMRI_CAPACITY_ROOT", Path(__file__).resolve().parents[1] / "full_data"))

def validate_capacity_support(support,n):
    x=np.asarray(support,dtype=np.float64)
    if x.shape!=(n,2):raise ValueError('Required centre-aligned H,K_B metadata must have shape (N,2)')
    if not np.isfinite(x).all() or not np.equal(x,np.floor(x)).all():raise ValueError('Fixed geometry must contain finite integer counts')
    if not ((x[:,0]>=1)&(x[:,1]>=1)&(x[:,1]<=x[:,0])).all():raise ValueError('Require 1 <= K_B <= H')
    return x

def capacity_lower(m,support):
    m=np.asarray(m)
    x=validate_capacity_support(support,len(m));H,K=x[:,0],x[:,1]
    while H.ndim<m.ndim:H=H[:,None];K=K[:,None]
    return np.maximum(m/np.sqrt(H),m/np.sqrt(K+3*H*(1-m)))

def align_capacity_support(centre_ids):
    ids=np.asarray(centre_ids).astype(str);result=np.empty((len(ids),2),np.float64)
    blocks=np.array([s.rsplit(':',1)[0] for s in ids])
    for bid in np.unique(blocks):
        take=np.flatnonzero(blocks==bid);ix=np.array([int(ids[i].rsplit(':',1)[1]) for i in take])
        with np.load(REV/'geometry/blocks'/(bid+'.npz')) as z:
            position=np.searchsorted(z['centre_index'],ix)
            if not np.array_equal(z['centre_index'][position],ix):raise ValueError('Geometry/centre identity mismatch')
            result[take]=np.stack([z['H'][position],z['K_B'][position]],axis=-1)
    return validate_capacity_support(result,len(ids))
