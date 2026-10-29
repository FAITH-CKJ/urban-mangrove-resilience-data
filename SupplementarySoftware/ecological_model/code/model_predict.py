
from pathlib import Path
import torch  
import joblib
import numpy as np
from model_data import project_state,score
from model_networks import GRUState,TFTState


def stamp_residual_calibration(checkpoint,alpha):
    
    path=Path(checkpoint);alpha=float(alpha)
    if not 0<alpha<=1:raise ValueError('Production residual alpha must be in (0,1]')
    bundle=joblib.load(path) if path.suffix=='.joblib' else torch.load(path,map_location='cpu',weights_only=False)
    calibration={'method':'annual_state_increment_scaling_before_physical_projection','alpha':alpha,
                 'selection_years':[2016,2020],'presence_gate_unchanged':True,'raw_training_alpha':1.,
                 'applied_at_every_recursive_annual_step':True}
    if bundle.get('residual_alpha')==alpha and bundle.get('residual_calibration')==calibration:return bundle
    bundle['residual_alpha']=alpha;bundle['residual_calibration']=calibration
    temporary=path.with_suffix('.calibration.tmp')
    if path.suffix=='.joblib':joblib.dump(bundle,temporary)
    else:torch.save(bundle,temporary)
    temporary.replace(path)
    return bundle


def prepare_observed_history(all_states,all_masks,static):
    




    x=np.asarray(all_states).copy();mask=np.asarray(all_masks,bool)
    if x.shape[-1]!=5 or x.shape[1]<5 or mask.shape!=x.shape:raise ValueError('Expected N,T>=5,5 observed states and masks')
    n=len(x);base=np.stack([np.ones(n),static[:,2],np.zeros(n),static[:,0],static[:,1]],-1)
    for t in range(x.shape[1]):x[:,t]=np.where(np.isfinite(x[:,t]),x[:,t],base if t==0 else x[:,t-1])
    return np.nan_to_num(x[:,-5:]).astype('float32'),mask[:,-5:].copy()


from capacity_support import validate_capacity_support


class StateForecaster:
    def __init__(self,checkpoint,allow_pilot=False,device=None,allow_unfinalized=False,cpu_threads=None):
        self.path=Path(checkpoint);self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if self.path.suffix=='.joblib':
            self.bundle=joblib.load(self.path);self.family='LightGBM_state';self.model=None
            if cpu_threads is not None:
                for estimator in self.bundle['state_models']+[self.bundle['presence_model']]:
                    if estimator is not None:estimator.set_params(n_jobs=int(cpu_threads))
        else:
            self.bundle=torch.load(self.path,map_location='cpu',weights_only=False)
            self.family=self.bundle['family'];cls=GRUState if self.family=='GRU' else TFTState
            self.model=cls(len(self.bundle['exo_names']),len(self.bundle['static_names']),self.bundle['width'],
                           presence_gate=self.bundle.get('architecture_version',1)>=2,residual_alpha=self.bundle.get('residual_alpha',1.))
            self.model.load_state_dict(self.bundle['state_dict']);self.model.to(self.device).eval()
        self.status=self.bundle['status']
        if self.status!='FORMAL_HISTORICAL_DRIVERS' and not allow_pilot:
            raise ValueError('Pilot checkpoint cannot generate formal scenarios. Explicit diagnostic allow_pilot required.')
        if self.status=='FORMAL_HISTORICAL_DRIVERS' and self.bundle.get('end_year')!=2025:
            if not allow_unfinalized:raise ValueError('Formal future release requires final fit through 2025; this checkpoint is a hindcast comparison fit')
            self.status='FORMAL_HISTORY_COMPARISON_CHECKPOINT_DIAGNOSTIC_ONLY'
        self.exo_names=self.bundle['exo_names'];self.static_names=self.bundle['static_names']
        self.scalers=self.bundle['scalers']

    def predict_with_quality(self,history_states,history_mask,history_drivers,future_drivers,static,batch_size=512,*,capacity_support):
        





        n=len(history_states);steps=future_drivers.shape[1];flags=np.zeros(n,np.uint16)
        capacity_support=validate_capacity_support(capacity_support,n)
        flags[~np.isfinite(static).all(1)]|=1
        flags[~np.isfinite(history_drivers).all((1,2))]|=2
        flags[~np.isfinite(future_drivers).all((1,2))]|=4
        flags[(~np.isfinite(history_states[:,-1,:3]).all(1))|(~np.asarray(history_mask[:,-1,:3],bool).all(1))]|=8
        ids=np.where(flags==0)[0]
        result={'states':np.full((n,steps,5),np.nan),'presence_probability':np.full((n,steps),np.nan),
                'S':np.full((n,steps),np.nan),'UMRI':np.full((n,steps),np.nan),'gate':np.full((n,steps),np.nan),
                'components':np.full((n,steps,5),np.nan),'contributions':np.full((n,steps,5),np.nan)}
        if len(ids):
            valid=self.predict(history_states[ids],history_mask[ids],history_drivers[ids],future_drivers[ids],static[ids],batch_size,capacity_support=capacity_support[ids])
            for key in result:result[key][ids]=valid[key]
        result['quality_flags']=np.repeat(flags[:,None],steps,axis=1)
        result['status']=self.status
        return result

    def predict(self,history_states,history_mask,history_drivers,future_drivers,static,batch_size=512,*,capacity_support):
        





        n=len(history_states);steps=future_drivers.shape[1]
        capacity_support=validate_capacity_support(capacity_support,n)
        if history_states.shape!=(n,5,5) or history_mask.shape!=(n,5,5):raise ValueError('Five observed encoder years required')
        if history_drivers.shape!=(n,5,len(self.exo_names)) or future_drivers.shape!=(n,steps,len(self.exo_names)):
            raise ValueError('Driver feature order/width mismatch')
        if static.shape!=(n,len(self.static_names)):raise ValueError('Static feature order/width mismatch')
        if steps<1 or n<1:raise ValueError('At least one centre and future year required')
        for name,a in [('history_drivers',history_drivers),('future_drivers',future_drivers),('static',static)]:
            if not np.isfinite(a).all():raise ValueError(name+' contains unresolved missing values')
        x=np.array(history_states,copy=True)
        base=np.stack([np.ones(n),static[:,2],np.zeros(n),static[:,0],static[:,1]],-1)
        for t in range(5):x[:,t]=np.where(np.isfinite(x[:,t]),x[:,t],base if t==0 else x[:,t-1])
        exmean=self.scalers['exmean'];exstd=self.scalers['exstd']
        he=(history_drivers-exmean)/exstd;fu=(future_drivers-exmean)/exstd
        st=((static-self.scalers['stmean'])/self.scalers['ststd']).astype('float32')
        output=[];presence=[]
        for lo in range(0,n,batch_size):
            hi=min(lo+batch_size,n);hist=np.concatenate([x[lo:hi],history_mask[lo:hi]],-1).astype('float32')
            eh=he[lo:hi].astype('float32');seq=[];prs=[]
            stride=1 if self.model is None else 5
            for start in range(0,steps,stride):
                count=min(stride,steps-start);future=fu[lo:hi,start:start+count].astype('float32')
                occupation=future_drivers[lo:hi,start:start+count,2]
                if self.model is None:
                    xx=np.concatenate([hist.reshape(hi-lo,-1),eh.reshape(hi-lo,-1),future[:,0],st[lo:hi]],-1)
                    pred=hist[:,-1,:5]+self.bundle.get('residual_alpha',1.)*np.stack([m.predict(xx) for m in self.bundle['state_models']],-1)
                    classifier=self.bundle['presence_model'];pr=classifier.predict_proba(xx)[:,1] if classifier else np.ones(hi-lo)
                    pred=project_state(pred,occupation[:,0],pr,capacity_support=capacity_support[lo:hi])[:,None];pr=pr[:,None]
                else:
                    with torch.no_grad():
                        args=[hist,eh,future,st[lo:hi],occupation,capacity_support[lo:hi]]
                        pred,logit=self.model(*[torch.as_tensor(a,dtype=torch.float32,device=self.device) for a in args])
                        pr=logit.sigmoid().cpu().numpy();pred=project_state(pred.cpu().numpy(),occupation,pr,capacity_support=capacity_support[lo:hi])
                seq.append(pred);prs.append(pr)
                mask=np.ones_like(pred)
                if self.bundle.get('architecture_version',1)>=2:mask[...,3:]=(pred[...,0:1]>0)
                hist=np.concatenate([hist,np.concatenate([pred,mask],-1)],1)[:,-5:]
                eh=np.concatenate([eh,future],1)[:,-5:]
            output.append(np.concatenate(seq,1));presence.append(np.concatenate(prs,1))
        states=np.concatenate(output);prob=np.concatenate(presence)
        values=score(states,future_drivers[...,:3],static[:,None,:],states[...,0]>0)
        return {'states':states,'presence_probability':prob,**values,'status':self.status}
