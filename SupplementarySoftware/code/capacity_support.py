






import numpy as np

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
