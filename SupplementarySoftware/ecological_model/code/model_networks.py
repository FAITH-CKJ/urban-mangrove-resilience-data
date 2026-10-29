





import torch
from torch import nn


class GLU(nn.Module):
    def __init__(self,d):
        super().__init__();self.fc=nn.Linear(d,2*d)
    def forward(self,x):
        a,b=self.fc(x).chunk(2,-1);return a*torch.sigmoid(b)


class GRN(nn.Module):
    def __init__(self,inp,hidden,out=None,context=0):
        super().__init__();out=out or hidden
        self.fc1=nn.Linear(inp,hidden);self.fc2=nn.Linear(hidden,out)
        self.context=nn.Linear(context,hidden,bias=False) if context else None
        self.skip=nn.Identity() if inp==out else nn.Linear(inp,out)
        self.glu=GLU(out);self.norm=nn.LayerNorm(out);self.drop=nn.Dropout(.05)
    def forward(self,x,context=None):
        y=self.fc1(x)
        if self.context is not None:
            while context.ndim<y.ndim:context=context.unsqueeze(1)
            y=y+self.context(context)
        y=self.fc2(torch.nn.functional.elu(y))
        return self.norm(self.skip(x)+self.glu(self.drop(y)))


class VariableSelection(nn.Module):
    def __init__(self,n,d,context=0):
        super().__init__();self.embed=nn.ModuleList([nn.Linear(1,d) for _ in range(n)])
        self.transforms=nn.ModuleList([GRN(d,d) for _ in range(n)])
        self.weights=GRN(n*d,d,n,context)
    def forward(self,x,context=None):
        embeddings=[f(x[...,i:i+1]) for i,f in enumerate(self.embed)]
        w=torch.softmax(self.weights(torch.cat(embeddings,-1),context),-1)
        v=torch.stack([f(e) for f,e in zip(self.transforms,embeddings)],-2)
        return (w.unsqueeze(-1)*v).sum(-2),w


class InterpretableAttention(nn.Module):
    
    def __init__(self,d,heads=4):
        super().__init__();self.h=heads;self.dk=d//heads
        self.q=nn.Linear(d,d);self.k=nn.Linear(d,d);self.v=nn.Linear(d,self.dk);self.out=nn.Linear(self.dk,d)
    def forward(self,q,context,mask):
        b,l,_=q.shape;t=context.shape[1]
        qq=self.q(q).view(b,l,self.h,self.dk).transpose(1,2)
        kk=self.k(context).view(b,t,self.h,self.dk).transpose(1,2)
        att=(qq@kk.transpose(-1,-2))/(self.dk**.5)
        att=att.masked_fill(mask[None,None],-1e9).softmax(-1)
        values=self.v(context).unsqueeze(1)
        return self.out((att@values).mean(1)),att.mean(1)


def constrain_torch(x,occupation,capacity_support):
    m=torch.minimum(x[...,0].clamp(0,1),1-occupation.clamp(0,1))
    H,K=capacity_support[:,0],capacity_support[:,1]
    floor=torch.maximum(m/torch.sqrt(H),m/torch.sqrt(K+3*H*(1-m)))
    c=torch.minimum(torch.maximum(x[...,1],floor),m);r=torch.minimum(x[...,2].clamp(0,1),m)
    return torch.stack([m,c,r,x[...,3].clamp(0,1),x[...,4].clamp(-1,1)],-1)


def frozen_score_torch(state,urban,baseline,present=None):
    
    def relative(a,b):
        denominator=a+b
        ratio=(a-b)/torch.where(denominator>0,denominator,torch.ones_like(denominator))
        return torch.where((a==0)&(b==0),torch.zeros_like(ratio),ratio)
    m,c,r,e,n=state.unbind(-1)
    absent=(m==0) if present is None else ~present.bool()
    g=torch.where(absent,0.,relative(e,baseline[:,None,0]))
    h=torch.where(absent,-1.,relative((n+1)/2,(baseline[:,None,1]+1)/2))
    t=relative(c,baseline[:,None,2]);r=torch.where(absent,0.,r)
    gate=1-torch.maximum(torch.zeros_like(h),torch.maximum(-h,-t))
    return -.25*urban.max(-1).values+.25*h+.25*t+.10*torch.minimum(g,torch.zeros_like(g))+gate*(.15*r+.10*torch.maximum(g,torch.zeros_like(g)))


class ContinuousDriver(nn.Module):
    
    def __init__(self,n,ns):
        super().__init__();self.linear=nn.Linear(n,5,bias=False);self.smooth=nn.Linear(n,5,bias=False)
        self.static_gate=nn.Linear(ns,5)
        nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.smooth.weight)
    def forward(self,delta,static):
        while static.ndim<delta.ndim:static=static.unsqueeze(1)
        return (self.linear(delta)+self.smooth(torch.tanh(delta)))*(.5+torch.sigmoid(self.static_gate(static)))


class GRUState(nn.Module):
    family='GRU'
    def __init__(self,nexo,nstatic,width=32,presence_gate=True,residual_alpha=1.):
        super().__init__();self.gru=nn.GRU(10+nexo,width,batch_first=True)
        self.presence_gate=presence_gate
        self.residual_alpha=float(residual_alpha)
        self.initial=nn.Linear(nstatic,width)
        self.decode=GRN(width+nexo+nstatic,width)
        self.delta=nn.Linear(width,5);self.presence=nn.Linear(width,1)
        self.external=ContinuousDriver(nexo,nstatic)
        nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
        nn.init.constant_(self.presence.bias,4.)
    def forward(self,hist,hist_exo,future,static,occupation,capacity_support):
        _,h=self.gru(torch.cat([hist,hist_exo],-1),torch.tanh(self.initial(static)).unsqueeze(0))
        outputs=[];prob=[];last=hist[:,-1,:5];last_exo=hist_exo[:,-1]
        for j in range(future.shape[1]):
            z=self.decode(torch.cat([h[-1],future[:,j],static],-1))
            
            state=constrain_torch(last+self.residual_alpha*(.1*self.delta(z))+
                                  self.residual_alpha*self.external(future[:,j]-last_exo,static),occupation[:,j],capacity_support)
            p=self.presence(z).squeeze(-1)
            if self.presence_gate:
                state=torch.cat([state[...,:3]*(p>=0).to(state.dtype).unsqueeze(-1),state[...,3:]],-1)
            outputs.append(state);prob.append(p)
            mask=torch.ones_like(state)
            if self.presence_gate:mask[...,3:]=(state[...,0:1]>0).to(state.dtype)
            obs=torch.cat([state,mask],-1)
            _,h=self.gru(torch.cat([obs,future[:,j]],-1).unsqueeze(1),h)
            last=state;last_exo=future[:,j]
        return torch.stack(outputs,1),torch.stack(prob,1)


class TFTState(nn.Module):
    family='TFT'
    def __init__(self,nexo,nstatic,width=32,presence_gate=True,residual_alpha=1.):
        super().__init__();self.static_vsn=VariableSelection(nstatic,width)
        self.presence_gate=presence_gate
        self.residual_alpha=float(residual_alpha)
        self.context_selection=GRN(width,width);self.context_enrichment=GRN(width,width)
        self.context_h=GRN(width,width);self.context_c=GRN(width,width)
        self.past_vsn=VariableSelection(10+nexo,width,width)
        self.future_vsn=VariableSelection(nexo,width,width)
        self.encoder=nn.LSTM(width,width,batch_first=True);self.decoder=nn.LSTM(width,width,batch_first=True)
        self.local_glu=GLU(width);self.local_norm=nn.LayerNorm(width)
        self.enrichment=GRN(width,width,context=width)
        self.attention=InterpretableAttention(width)
        self.attn_glu=GLU(width);self.attn_norm=nn.LayerNorm(width)
        self.position=GRN(width,width);self.output_glu=GLU(width);self.output_norm=nn.LayerNorm(width)
        self.delta=nn.Linear(width,5);self.presence=nn.Linear(width,1)
        self.external=ContinuousDriver(nexo,nstatic)
        nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias);nn.init.constant_(self.presence.bias,4.)
    def forward(self,hist,hist_exo,future,static,occupation,capacity_support):
        s,_=self.static_vsn(static);ctx=self.context_selection(s)
        past,_=self.past_vsn(torch.cat([hist,hist_exo],-1),ctx)
        known,_=self.future_vsn(future,ctx)
        enc,hc=self.encoder(past,(self.context_h(s).unsqueeze(0),self.context_c(s).unsqueeze(0)))
        dec,_=self.decoder(known,hc)
        selection=torch.cat([past,known],1)
        local=self.local_norm(selection+self.local_glu(torch.cat([enc,dec],1)))
        enriched=self.enrichment(local,self.context_enrichment(s))
        t=past.shape[1];steps=future.shape[1]
        mask=torch.arange(t+steps,device=hist.device)[None,:]>(t+torch.arange(steps,device=hist.device)[:,None])
        att,_=self.attention(enriched[:,t:],enriched,mask)
        z=self.attn_norm(enriched[:,t:]+self.attn_glu(att));z=self.position(z)
        z=self.output_norm(local[:,t:]+self.output_glu(z))
        
        increments=self.residual_alpha*(.1*self.delta(z)+self.external(future-torch.cat([hist_exo[:,-1:],future[:,:-1]],1),static))
        last=hist[:,-1,:5];outputs=[];presence=self.presence(z).squeeze(-1)
        for j in range(steps):
            last=constrain_torch(last+increments[:,j],occupation[:,j],capacity_support)
            if self.presence_gate:last=torch.cat([last[...,:3]*(presence[:,j]>=0).to(last.dtype).unsqueeze(-1),last[...,3:]],-1)
            outputs.append(last)
        return torch.stack(outputs,1),presence
