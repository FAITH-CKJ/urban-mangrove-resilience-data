







import argparse,copy,hashlib,json,os,time,shutil
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
import torch
from torch import nn
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor,LGBMClassifier
from model_data import load_panel,export_contract,score,normalize,project_state,STATES
from model_networks import GRUState,TFTState,frozen_score_torch

torch.set_num_threads(2)
SEED=2100


def write_json(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:float(x)),encoding='utf8');tmp.replace(path)


class Preparation:
    def __init__(self,p,end=2015,include_holdout=False,bootstrap_seed=None):
        self.p=p;centres=np.ones(len(p['centre']),bool) if include_holdout else ~p['holdout'];times=np.arange(26)<=end-2000
        ex=p['exo'][centres][:,times];st=p['static'][centres]
        self.exmean=np.nanmean(ex,axis=(0,1));self.exstd=np.nanstd(ex,axis=(0,1));self.exstd[self.exstd<1e-5]=1
        self.stmean=np.nanmean(st,axis=0);self.ststd=np.nanstd(st,axis=0);self.ststd[self.ststd<1e-5]=1
        self.ex=np.nan_to_num((p['exo']-self.exmean)/self.exstd).astype('float32')
        self.st=np.nan_to_num((p['static']-self.stmean)/self.ststd).astype('float32')
        x=p['state'].copy()
        
        base=np.stack([np.ones(len(x)),p['static'][:,2],np.zeros(len(x)),p['static'][:,0],p['static'][:,1]],-1)
        for t in range(26):
            x[:,t]=np.where(np.isfinite(x[:,t]),x[:,t],base if t==0 else x[:,t-1])
        self.x=np.nan_to_num(x).astype('float32')
        self.hist=np.concatenate([self.x,p['mask'].astype('float32')],-1)
        self.weight=np.zeros(len(x))
        for r in np.unique(p['region']):
            bs=np.unique(p['block'][(p['region']==r)&centres])
            for b in bs:
                take=(p['block']==b)&centres
                sw=p['sampling_weight'][take];self.weight[take]=sw/(len(bs)*sw.sum())
        self.weight=self.weight/(self.weight[centres].mean())
        self.bootstrap_seed=bootstrap_seed
        if bootstrap_seed is not None:
            rng=np.random.default_rng(bootstrap_seed)
            for r in np.unique(p['region']):
                bs=np.unique(p['block'][(p['region']==r)&centres])
                draws=rng.choice(bs,size=len(bs),replace=True)
                for b in bs:self.weight[(p['block']==b)&centres]*=np.sum(draws==b)

    def sequences(self,end=2015,horizon=5,include_holdout=False):
        ids=np.arange(len(self.x)) if include_holdout else np.where(~self.p['holdout'])[0]
        origins=range(5,end-2000-horizon+1)
        ci=[];ot=[]
        for o in origins:
            ok=np.isfinite(self.p['exo'][ids,o-4:o+1+horizon]).all((1,2))&np.isfinite(self.p['static'][ids]).all(1)
            ci.extend(ids[ok]);ot.extend([o]*int(ok.sum()))
        ci=np.asarray(ci);ot=np.asarray(ot)
        if len(ci)==0:raise ValueError('No finite training windows')
        h=np.stack([self.hist[c,o-4:o+1] for c,o in zip(ci,ot)])
        he=np.stack([self.ex[c,o-4:o+1] for c,o in zip(ci,ot)])
        fu=np.stack([self.ex[c,o+1:o+1+horizon] for c,o in zip(ci,ot)])
        occ=np.stack([self.p['exo'][c,o+1:o+1+horizon,2] for c,o in zip(ci,ot)])
        y=np.stack([self.x[c,o+1:o+1+horizon] for c,o in zip(ci,ot)])
        mask=np.stack([self.p['mask'][c,o+1:o+1+horizon] for c,o in zip(ci,ot)])
        pr=np.stack([self.p['present'][c,o+1:o+1+horizon] for c,o in zip(ci,ot)])
        return (h,he,fu,self.st[ci],occ,y,mask,pr,self.weight[ci]),ci,ot

    def scalers(self):
        return {k:getattr(self,k) for k in ['exmean','exstd','stmean','ststd']}


def metrics(p,pred,origin,ids=None,direct=None,predicted_origin_S=None,forecast_support=None):
    ids=np.arange(len(p['centre'])) if ids is None else np.asarray(ids)
    ids=ids[p['evaluation_sample'][ids]]
    horizon=pred.shape[1];sl=slice(origin+1,origin+1+horizon)
    sc=score(pred,p['exo'][:,sl,:3],p['static'][:,None,:],pred[...,0]>0)
    if direct is not None:sc={'S':direct,'UMRI':normalize(direct)}
    true=p['S'][:,sl];valid=p['valid'][:,sl]&np.isfinite(sc['S'])&np.isfinite(true)
    driver_origin=np.isfinite(p['exo'][:,origin-4:origin+1]).all((1,2))&np.isfinite(p['static']).all(1)
    driver_future=np.logical_and.accumulate(np.isfinite(p['exo'][:,sl]).all(-1),axis=1)
    if forecast_support is not None:driver_future=driver_future&np.asarray(forecast_support,bool)
    valid=valid&driver_origin[:,None]&driver_future
    frame=pd.DataFrame({'block':np.repeat(p['block'][ids],horizon),'region':np.repeat(p['region'][ids],horizon),
                        'year':np.tile(np.arange(horizon)+origin+2001,len(ids)),
                        'pred':np.where(valid[ids],sc['S'][ids],np.nan).ravel(),
                        'true':np.where(valid[ids],true[ids],np.nan).ravel()})
    frame['centre_abs_error']=np.abs(frame.pred-frame.true)
    blocks=frame.groupby(['region','block','year'])[['pred','true','centre_abs_error']].mean()
    regions=blocks.groupby(['region','year']).mean()
    rs=np.abs(regions.pred-regions.true).groupby('region').mean().mean()
    ru=np.abs(normalize(regions.pred)-normalize(regions.true)).groupby('region').mean().mean()
    
    
    paired=valid[ids,-1]&p['valid'][ids,origin]&np.isfinite(p['S'][ids,origin])
    predicted_origin=p['S'][:,origin] if predicted_origin_S is None else np.asarray(predicted_origin_S)
    paired=paired&np.isfinite(predicted_origin[ids])
    paired_ids=ids[paired]
    direction_frame=pd.DataFrame({'region':p['region'][paired_ids],'block':p['block'][paired_ids],
                 'origin':p['S'][paired_ids,origin],'predicted_origin':predicted_origin[paired_ids],
                 'pred':sc['S'][paired_ids,-1],'true':true[paired_ids,-1]})
    direction_regions=direction_frame.groupby(['region','block'])[['origin','predicted_origin','pred','true']].mean().groupby('region').mean()
    direction_regions['predicted_change_S']=direction_regions.pred-direction_regions.predicted_origin
    direction_regions['observed_change_S']=direction_regions.true-direction_regions.origin
    tolerance=1e-8
    for name in ['predicted','observed']:
        delta=direction_regions[name+'_change_S']
        direction_regions[name+'_direction']=np.where(np.abs(delta)<=tolerance,0,np.sign(delta)).astype(int)
    direction_regions['correct']=direction_regions.predicted_direction==direction_regions.observed_direction
    direction=float(direction_regions.correct.mean())
    errs={}
    for j,name in enumerate(STATES):
        m=p['mask'][ids,sl,j]&driver_origin[ids,None]&driver_future[ids]
        error=np.where(m,np.abs(pred[ids,:,j]-p['state'][ids,sl,j]),np.nan)
        count=np.isfinite(error).sum(1)
        centre_error=np.divide(np.nansum(error,axis=1),count,out=np.full(len(ids),np.nan),where=count>0)
        f=pd.DataFrame({'region':p['region'][ids],'block':p['block'][ids],'e':centre_error})
        errs[name]=float(f.groupby(['region','block']).e.mean().groupby('region').mean().mean())
    truth_abs=~p['present'][ids,sl];pred_abs=pred[ids,:,0]==0
    absence_valid=p['mask'][ids,sl,0]&driver_origin[ids,None]&driver_future[ids]
    confusion={'true_absent_pred_absent':int((truth_abs&pred_abs&absence_valid).sum()),
               'true_absent_pred_present':int((truth_abs&~pred_abs&absence_valid).sum()),
               'true_present_pred_absent':int((~truth_abs&pred_abs&absence_valid).sum()),
               'true_present_pred_present':int((~truth_abs&~pred_abs&absence_valid).sum())}
    return {'region_balanced_five_year_S_MAE':float(rs),'UMRI_MAE':float(ru),'regional_change_direction_accuracy':direction,
            'direction_zero_tolerance_S':tolerance,'direction_regions_scored':len(direction_regions),
            'regional_change_direction_details':direction_regions.reset_index().to_dict(orient='records'),
            'region_balanced_centre_S_MAE':float(regions.centre_abs_error.groupby('region').mean().mean()),
            'normalized_mean_state_MAE':float(np.mean([errs[name]/(2 if name=='NDMI' else 1) for name in STATES])),
            'state_target_MAE':errs,'absence_confusion_counts':confusion,'valid_rows':int(valid[ids].sum()),'centres':len(ids)},regions


def tree_features(pre,history,history_exo,future,ci):
    return np.concatenate([history.reshape(len(ci),-1),history_exo.reshape(len(ci),-1),future,pre.st[ci]],-1)


def fit_trees(pre,end,out,include_holdout=False):
    raise RuntimeError('Tree fitting is disabled in this distribution. Use the supplied fitted checkpoints with code/reproduce_forecast_demo.py; refitting requires a separately defined training protocol.')
    data,ci,ot=pre.sequences(end,1,include_holdout);h,he,fu,st,occ,y,mask,pr,w=data
    X=tree_features(pre,h,he,fu[:,0],ci)
    params=dict(n_estimators=180,num_leaves=15,max_depth=5,learning_rate=.035,min_child_samples=50,reg_lambda=3,
                n_jobs=2,verbosity=-1,random_state=SEED)
    models=[]
    for j in range(5):
        use=mask[:,0,j].astype(bool)
        model=LGBMRegressor(**params);model.fit(X[use],(y[:,0,j]-h[:,-1,j])[use],sample_weight=w[use]);models.append(model)
    p=pre.p
    use=p['valid'][ci,ot]&p['valid'][ci,ot+1]
    direct=LGBMRegressor(**params);direct.fit(X[use],(p['S'][ci,ot+1]-p['S'][ci,ot])[use],sample_weight=w[use])
    classifier=None
    if np.unique(pr).size>1:
        classifier=LGBMClassifier(**params);classifier.fit(X,pr[:,0].astype(int),sample_weight=w)
    bundle={'state_models':models,'presence_model':classifier,'direct_S':direct,'scalers':pre.scalers(),'end_year':end,'architecture_version':2,
            'fit_includes_spatial_holdout':include_holdout,'fit_role':'production' if end==2025 and include_holdout else 'hindcast',
            'exo_names':p['exo_names'],'static_names':p['static_names'],'status':p['status'],'bootstrap_seed':pre.bootstrap_seed,
            'source_sha256':p.get('source_sha256'),'driver_sha256':p.get('driver_sha256')}
    temporary=out/'lightgbm.tmp';joblib.dump(bundle,temporary);temporary.replace(out/'lightgbm.joblib');return bundle


def tree_rollout(pre,bundle,origin=15,horizon=5,return_presence=False):
    p=pre.p;n=len(p['centre']);ci=np.arange(n)
    hist=pre.hist[:,origin-4:origin+1].copy();he=pre.ex[:,origin-4:origin+1].copy()
    outputs=[];direct=[];probabilities=[];last_s=p['S'][:,origin].copy()
    for t in range(origin+1,origin+1+horizon):
        X=tree_features(pre,hist,he,pre.ex[:,t],ci)
        pred=hist[:,-1,:5]+bundle.get('residual_alpha',1.)*np.stack([m.predict(X) for m in bundle['state_models']],-1)
        pr=bundle['presence_model'].predict_proba(X)[:,1] if bundle['presence_model'] else np.ones(n)
        probabilities.append(pr)
        pred=project_state(pred,p['exo'][:,t,2],pr,capacity_support=p['capacity_support'])
        last_s=last_s+bundle['direct_S'].predict(X);direct.append(last_s.copy());outputs.append(pred)
        mask=np.ones_like(pred)
        if bundle.get('architecture_version',1)>=2:mask[...,3:]=(pred[...,0:1]>0)
        hist=np.concatenate([hist[:,1:],np.concatenate([pred,mask],-1)[:,None]],1)
        he=np.concatenate([he[:,1:],pre.ex[:,t:t+1]],1)
    result=(np.stack(outputs,1),np.stack(direct,1))
    return (*result,np.stack(probabilities,1)) if return_presence else result


def network_rollout(pre,model,origin=15,horizon=5,device='cpu',batch=256):
    model.eval();out=[];probs=[]
    with torch.no_grad():
        for lo in range(0,len(pre.x),batch):
            hi=min(lo+batch,len(pre.x));hist=pre.hist[lo:hi,origin-4:origin+1].copy();he=pre.ex[lo:hi,origin-4:origin+1].copy()
            seq=[];ps=[]
            for start in range(0,horizon,5):
                steps=min(5,horizon-start);t=origin+1+start
                args=[hist,he,pre.ex[lo:hi,t:t+steps],pre.st[lo:hi],pre.p['exo'][lo:hi,t:t+steps,2],pre.p['capacity_support'][lo:hi]]
                pred,logit=model(*[torch.as_tensor(a,dtype=torch.float32,device=device) for a in args])
                pr=logit.sigmoid().cpu().numpy();pred=project_state(pred.cpu().numpy(),args[-2],pr,capacity_support=pre.p['capacity_support'][lo:hi])
                seq.append(pred);ps.append(pr)
                mask=np.ones_like(pred)
                if getattr(model,'presence_gate',False):mask[...,3:]=(pred[...,0:1]>0)
                hist=np.concatenate([hist,np.concatenate([pred,mask],-1)],1)[:,-5:]
                he=np.concatenate([he,args[2]],1)[:,-5:]
            out.append(np.concatenate(seq,1));probs.append(np.concatenate(ps,1))
    return np.concatenate(out),np.concatenate(probs)


def fit_network(pre,family,end,out,epochs=40,width=32,include_holdout=False,select_validation=True,fixed_epochs=None,resume=False,score_loss_weight=2.):
    raise RuntimeError('Neural-network fitting is disabled in this distribution. Prediction reproduction uses the supplied fitted checkpoints; refitting requires a separately defined training protocol.')
    torch.manual_seed(SEED);np.random.seed(SEED)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    cls=GRUState if family=='GRU' else TFTState
    model=cls(len(pre.p['exo_names']),len(pre.p['static_names']),width).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003)
    best=float('inf');best_state=None;log=[];best_epoch=0
    phases=[(1,max(8,epochs//4)),(5,epochs)] if family=='GRU' else [(5,epochs)]
    if fixed_epochs is not None:phases=([(1,max(8,epochs//4))] if family=='GRU' else [])+[(5,fixed_epochs)]
    resume_phase=-1;resume_epoch=0
    identity={'source_sha256':pre.p.get('source_sha256'),'driver_sha256':pre.p.get('driver_sha256'),
              'end_year':end,'include_holdout':include_holdout,'width':width,'family':family,'phases':phases,'bootstrap_seed':pre.bootstrap_seed,
              'architecture_version':2,'score_loss_weight':score_loss_weight}
    checkpoint_path=out/(family.lower()+'_resume.pt')
    if resume and checkpoint_path.exists():
        saved=torch.load(checkpoint_path,map_location=device,weights_only=False)
        if saved.get('training_identity')!=identity:raise ValueError('Resume identity differs from current training inputs/settings')
        model.load_state_dict(saved['state_dict']);opt.load_state_dict(saved['optimizer'])
        resume_phase=saved['phase_index'];resume_epoch=saved['epoch']
        best=saved['best_validation'];best_state=saved['best_state'];best_epoch=saved['best_epoch'];log=saved['log']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device=='cuda':torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    start_time=time.time()
    for phase_index,(horizon,nepoch) in enumerate(phases):
        if phase_index<resume_phase:continue
        arrays,ci,ot=pre.sequences(end,horizon,include_holdout)
        tensors=[torch.as_tensor(a,dtype=torch.float32,device=device) for a in arrays]
        for epoch in range(nepoch):
            if phase_index==resume_phase and epoch<resume_epoch:continue
            model.train();order=torch.randperm(len(ci),device=device);total=torch.zeros((),device=device,dtype=torch.float64)
            for inds in order.split(256):
                h,he,fu,st,occ,y,mask,pr,w=[a[inds] for a in tensors]
                pred,logit=model(h,he,fu,st,occ)
                scales=torch.tensor([1,1,1,2,2],device=device)
                loss_state=((pred-y).square()*scales*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)
                loss_presence=nn.functional.binary_cross_entropy_with_logits(logit,pr,reduction='none').mean(1)
                baseline=st[:,:3]*torch.as_tensor(pre.ststd[:3],device=device,dtype=st.dtype)+torch.as_tensor(pre.stmean[:3],device=device,dtype=st.dtype)
                urban=fu[...,:3]*torch.as_tensor(pre.exstd[:3],device=device,dtype=fu.dtype)+torch.as_tensor(pre.exmean[:3],device=device,dtype=fu.dtype)
                predicted_s=frozen_score_torch(pred,urban,baseline)
                target_s=frozen_score_torch(y,urban,baseline,pr)
                score_valid=(mask[...,:3]>0).all(-1)&((pr==0)|(mask[...,3:]>0).all(-1))
                loss_score=((predicted_s-target_s).square()*score_valid).sum(1)/score_valid.sum(1).clamp_min(1)
                loss=((loss_state+score_loss_weight*loss_score+.01*loss_presence)*w).mean()
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=loss.detach().double()*len(inds)
            row={'family':family,'horizon':horizon,'epoch':epoch+1,'loss':float(total)/len(ci),'elapsed_seconds':time.time()-start_time}
            if select_validation and horizon==5 and ((epoch+1)%5==0 or epoch==nepoch-1):
                pred,_=network_rollout(pre,model,device=device)
                met,_=metrics(pre.p,pred,15,np.where(~pre.p['holdout'])[0]);value=met['region_balanced_five_year_S_MAE']
                row['validation_S_MAE']=value
                if value<best:
                    best=value;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
            log.append(row)
            if (epoch+1)%5==0:print(json.dumps(row),flush=True)
            checkpoint={'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'family':family,'width':width,'architecture_version':2,
                        'horizon':horizon,'epoch':epoch+1,'end_year':end,'status':pre.p['status'],'scalers':pre.scalers(),
                        'exo_names':pre.p['exo_names'],'static_names':pre.p['static_names'],'training_identity':identity,
                        'phase_index':phase_index,'best_validation':best,'best_state':best_state,'best_epoch':best_epoch,
                        'log':log,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if device=='cuda' else None}
            temp=checkpoint_path.with_suffix('.tmp');torch.save(checkpoint,temp);temp.replace(checkpoint_path)
            pd.DataFrame(log).to_csv(out/(family.lower()+'_training_log.csv'),index=False)
    if best_state is not None:model.load_state_dict(best_state)
    checkpoint={'state_dict':model.state_dict(),'family':family,'width':width,'architecture_version':2,'selected_epoch':best_epoch or nepoch,
                'fit_includes_spatial_holdout':include_holdout,'fit_role':'production' if end==2025 and include_holdout else 'hindcast',
                'objective':{'state_mse':1.,'frozen_S_mse':score_loss_weight,'presence_bce':.01},
                'end_year':end,'status':pre.p['status'],'scalers':pre.scalers(),
                'exo_names':pre.p['exo_names'],'static_names':pre.p['static_names'],'bootstrap_seed':pre.bootstrap_seed,
                'source_sha256':pre.p.get('source_sha256'),'driver_sha256':pre.p.get('driver_sha256')}
    temporary=out/(family.lower()+'.tmp');torch.save(checkpoint,temporary);temporary.replace(out/(family.lower()+'.pt'))
    return model,best_epoch or nepoch


def main():
    raise RuntimeError('Run python code/reproduce_forecast_demo.py from SupplementarySoftware to replay the supplied fitted models. Refitting requires a separately defined training protocol and complete historical inputs.')
    ap=argparse.ArgumentParser();ap.add_argument('--table');ap.add_argument('--drivers');ap.add_argument('--formal',action='store_true')
    ap.add_argument('--stage',choices=['pilot','compare'],default='pilot');ap.add_argument('--out',required=True)
    ap.add_argument('--epochs',type=int,default=40);ap.add_argument('--width',type=int,default=32);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    snapshot=out/'code_snapshot';snapshot.mkdir(exist_ok=True)
    for name in ['model_data.py','model_networks.py','model_predict.py','train_models.py','capacity_support.py']:
        source=Path(__file__).parent/name;target=snapshot/name
        if not target.exists():shutil.copy2(source,target)
    write_json(out/'run_protocol.json',{'stage':args.stage,'formal':args.formal,'epochs':args.epochs,'width':args.width,
                'state_target_order':STATES,'training_target_years':[2006,2015],'validation_target_years':[2016,2020],
                'reserved_test_target_years':[2021,2025],'TFT_encoder_years':5,'TFT_decoder_annual_steps':5,
                'neural_objective':{'state_mse':1.,'frozen_S_mse':2.,'presence_bce':.01},
                'TFT_method_reference':'https://arxiv.org/abs/1912.09363',
                'TFT_adaptation':'TFT variable selection, static contexts, recurrent encoder-decoder, gated static enrichment and interpretable causal attention; joint constrained ecological point-state and presence heads',
                'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')}})
    p,df=load_panel(args.table,args.drivers,args.formal);contract=export_contract(p,df,out)
    if args.stage!='pilot' and not args.formal:raise ValueError('Formal stages need complete external drivers')
    pre=Preparation(p);results={};predictions={};device='cuda' if torch.cuda.is_available() else 'cpu'
    for name in ['persistence_state','persistence_S']:
        pred=np.repeat(pre.x[:,15:16],5,axis=1)
        for j in range(5):pred[:,j]=project_state(pred[:,j],p['exo'][:,16+j,2])
        direct=np.repeat(p['S'][:,15:16],5,axis=1) if name=='persistence_S' else None
        results[name]={}
        for label,ids in [('temporal',np.where(~p['holdout'])[0]),('spatial',np.where(p['holdout'])[0])]:
            results[name][label],_=metrics(p,pred,15,ids,direct)
        predictions[name]=pred
    write_json(out/'validation_metrics.json',results)
    trees=fit_trees(pre,2015,out);tp,ds=tree_rollout(pre,trees)
    for name,direct in [('LightGBM_state',None),('LightGBM_direct_S',ds)]:
        results[name]={}
        for label,ids in [('temporal',np.where(~p['holdout'])[0]),('spatial',np.where(p['holdout'])[0])]:
            results[name][label],reg=metrics(p,tp,15,ids,direct);reg.to_csv(out/(name+'_'+label+'_regional_validation.csv'))
        predictions[name]=tp
    write_json(out/'validation_metrics.json',results)
    selected_epochs={}
    for family in ['GRU','TFT']:
        model,ep=fit_network(pre,family,2015,out,args.epochs,args.width,resume=args.resume)
        pred,prob=network_rollout(pre,model,device=device);results[family]={};selected_epochs[family]=ep
        np.savez_compressed(out/(family.lower()+'_validation_states.npz'),states=pred,presence_probability=prob,centre_id=p['centre'])
        for label,ids in [('temporal',np.where(~p['holdout'])[0]),('spatial',np.where(p['holdout'])[0])]:
            results[family][label],reg=metrics(p,pred,15,ids);reg.to_csv(out/(family.lower()+'_'+label+'_regional_validation.csv'))
        predictions[family]=pred;write_json(out/'validation_metrics.json',results)
        drift,_=network_rollout(pre,model,origin=10,horizon=10,device=device)
        
        dm,_=metrics(p,drift,10,np.where(~p['holdout'])[0]);write_json(out/(family.lower()+'_ten_year_diagnostic.json'),{'status':'TRAINING_OVERLAP_2011_2015_NOT_TEST','metrics':dm})
        del model;torch.cuda.empty_cache()
    
    eligible=['LightGBM_state','GRU','TFT']
    winner=min(eligible,key=lambda n:results[n]['temporal']['region_balanced_five_year_S_MAE'])
    gain=1-results[winner]['temporal']['region_balanced_five_year_S_MAE']/results['persistence_state']['temporal']['region_balanced_five_year_S_MAE']
    selection={'status':p['status'],'selected_family':winner,'selection_data':'2016-2020 validation only, non-held-out blocks',
               'selected_epochs':selected_epochs,'width':args.width,'training_epochs':args.epochs,'gain_over_state_persistence':gain,
               'test_years_2021_2025_evaluated':False,'formal_future_release_eligible':bool(args.formal),
               'input_contract':contract,'runtime':{'torch':torch.__version__,'device':device,'seed':SEED,'cpu_threads':2}}
    write_json(out/'selection.json',selection)
    print(json.dumps(selection,indent=2),flush=True)


if __name__=='__main__':main()
