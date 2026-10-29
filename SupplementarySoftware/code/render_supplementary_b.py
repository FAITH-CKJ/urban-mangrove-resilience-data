



from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

INK='#263B40'; MID='#82928E'; TEAL='#397E70'; PURPLE='#8C74A6'; GOLD='#B38B43'; BLUE='#5796B0'
REG={0:'Overall',1:'Australia & New Zealand',2:'Central America',3:'Eastern Africa',4:'Eastern Asia',5:'Northern America',6:'South America',7:'South-Eastern Asia',8:'Southern Asia',9:'Western Africa',10:'Western Asia'}
ORDER=[0,5,2,6,9,3,10,8,7,4,1]
SC=['ssp126','ssp245','ssp370','ssp585']
SCOL={'ssp126':'#649B8D','ssp245':'#7398B1','ssp370':'#B68A5B','ssp585':'#AC6466'}
SLAB={'ssp126':'SSP1–2.6','ssp245':'SSP2–4.5','ssp370':'SSP3–7.0','ssp585':'SSP5–8.5'}
STATE=LinearSegmentedColormap.from_list('UMRI_state',['#684766','#B18D9F','#F7F5F0','#8BBDB0','#216E66'])
INPUTS={}; AUDIT={}
def style():
 mpl.rcParams.update({'font.family':'Arial','font.size':7.2,'axes.labelsize':7.2,'axes.titlesize':7.5,'xtick.labelsize':6.8,'ytick.labelsize':7.0,'legend.fontsize':6.8,'axes.linewidth':.55,'axes.edgecolor':MID,'axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,'xtick.major.size':2.2,'ytick.major.size':2.2,'xtick.major.width':.5,'ytick.major.width':.5,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','path.simplify':False,'mathtext.fontset':'dejavusans'})
def read(root,rel):
 p=root/rel;INPUTS[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
 return pd.read_csv(p,float_precision='round_trip')
def readjson(root,rel):
 p=root/rel;INPUTS[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text('utf-8-sig'))
def canvas(height):return plt.figure(figsize=(171/25.4,height/25.4),facecolor='white')
def axis(f,x,y,w,h,**kw):return f.add_axes([x/171,y/(f.get_figheight()*25.4),w/171,h/(f.get_figheight()*25.4)],**kw)
def text(f,x,y,s,**kw):return f.text(x/171,y/(f.get_figheight()*25.4),s,**kw)
def title(f,x,y,letter,words):
 text(f,x,y,letter,fontweight='bold',fontsize=8.4,va='baseline');text(f,x+4.2,y,words,fontsize=7.5,va='baseline')
def clean(a,which='y'):
 a.set_axisbelow(True);a.grid(axis=which,color='#E5E9E6',lw=.45)
def save(f,out,stem,details):
 out.mkdir(parents=True,exist_ok=True);f.canvas.draw()
 renderer=f.canvas.get_renderer();bounds=[]
 for t in f.findobj(mpl.text.Text):
  if not t.get_visible() or not t.get_text():continue
  box=t.get_window_extent(renderer)
  
  if box.width and box.height and (box.x0 < -2 or box.y0 < -2 or box.x1 > f.bbox.width+2 or box.y1 > f.bbox.height+2):
   bounds.append(t.get_text())
 outputs={}
 for ext in ('pdf','svg','png','tiff'):
  p=out/(stem+'.'+ext);kw={}
  if ext=='png':kw['dpi']=300
  if ext=='tiff':kw.update(dpi=600,pil_kwargs={'compression':'tiff_lzw'})
  f.savefig(p,**kw);outputs[ext]=hashlib.sha256(p.read_bytes()).hexdigest()
 details.update(width_mm=171,height_mm=round(f.get_figheight()*25.4,3),out_of_canvas_text=bounds,outputs=outputs)
 AUDIT[stem]=details;plt.close(f)
 print(stem+': '+json.dumps({k:v for k,v in details.items() if k not in ('outputs',)}))

def region_rows(a,names=True):
 a.set(ylim=(10.5,-.5),yticks=range(11),yticklabels=[REG[r] for r in ORDER] if names else [])
 a.tick_params(axis='y',length=0,pad=3)
 for sp in ('left','right','top'):a.spines[sp].set_visible(False)
 a.axhline(.5,color='#CBD2CF',lw=.7)

def annual_matrix(a,values,years,names=True,cmap=STATE,vmin=-1,vmax=1):
 assert values.shape==(11,len(years)) and np.isfinite(values).all()
 im=a.pcolormesh(np.r_[years-.5,years[-1]+.5],np.arange(-.5,11.5),values,cmap=cmap,vmin=vmin,vmax=vmax,shading='flat',rasterized=False)
 a.set(xlim=(years[0]-.5,years[-1]+.5),xticks=[2000,2010,2025]);region_rows(a,names)
 for yy in np.arange(.5,10.6):a.axhline(yy,color='white',lw=.8)
 return im


def historical_native(root,out):
 d=read(root,'figure_story/historical_native_evidence.csv')
 j=read(root,'figure_story/historical_g_T_histogram.csv')
 ep=read(root,'figure_story/historical_endpoint_comparison.csv')
 f=canvas(90)
 configs=[(15,45,'a','Habitat',['m_mean','C_mean'],[TEAL,PURPLE],100,'Habitat (%)'),
          (76,35,'b','Recovery',['r_mean'],['#228D7C'],100,'Recovered habitat (%)'),
          (130,36,'c','Canopy',['delta_EVI_present_mean'],['#849637'],1,'EVI change')]
 for x,w,letter,head,keys,cols,mul,label in configs:
  title(f,x-11,85,letter,head);a=axis(f,x,57,w,22)
  for support,ls in [('available','-'),('complete_history',(0,(3,1.7)))]:
   q=d[(d.support==support)&(d.region_id==0)].sort_values('year');assert len(q)==26
   for key,co in zip(keys,cols):a.plot(q.year,mul*q[key],color=co,ls=ls,lw=1.0)
  a.set(xlim=(2000,2025),xticks=[2000,2010,2025],ylabel=label);clean(a)
  if letter=='a':
   a.set(ylim=(68,101),yticks=[70,85,100]);a.text(2025.5,80,'$m$',color=TEAL,va='center');a.text(2025.5,71,'$C$',color=PURPLE,va='center')
  elif letter=='b':
   a.set(ylim=(0,2.5),yticks=[0,1,2]);a.axvspan(2000,2006.5,color='#EEF0EC',zorder=0,lw=0)
  else:a.set(ylim=(-.02,.065),yticks=[0,.03,.06]);a.axhline(0,color=MID,lw=.5)
  for ln in a.lines:
   if len(ln.get_ydata())==26:
    assert np.nanmin(ln.get_ydata())>=a.get_ylim()[0] and np.nanmax(ln.get_ydata())<=a.get_ylim()[1]
 f.legend(handles=[Line2D([0],[0],color=INK,lw=1,label='Available observations'),Line2D([0],[0],color=INK,lw=1,ls=(0,(3,1.7)),label='Complete history')],loc='center',bbox_to_anchor=(.5,49/90),ncol=2,columnspacing=2.2)
 title(f,4,42,'d','Greenness and habitat structure')
 a=axis(f,15,9,45,28)
 cmap=LinearSegmentedColormap.from_list('density',['#EDF4DC','#A5CCAC','#439D83','#155D59','#0B383C'])
 q=j[j.support=='available'];m=q.pivot(index='T_low',columns='g_low',values='count').sort_index().sort_index(axis=1)
 xe=np.r_[m.columns.to_numpy(),q.g_high.max()];ye=np.r_[m.index.to_numpy(),q.T_high.max()]
 im=a.pcolormesh(xe,ye,np.ma.masked_equal(m.to_numpy(),0),cmap=cmap,norm=LogNorm(1,100000),rasterized=True,shading='flat')
 a.axhline(0,color=MID,lw=.5);a.axvline(0,color=MID,lw=.5)
 a.set(xlim=(-1,1),ylim=(-1,1),xticks=[-1,0,1],yticks=[-1,0,1],xlabel='Greenness response, $g$',ylabel='Structural state, $T$')
 cb=f.colorbar(im,cax=axis(f,63,12,1.8,22),ticks=[1,100,10000]);cb.outline.set_visible(False);cb.ax.tick_params(labelsize=6,length=1,pad=1);cb.ax.minorticks_off()
 text(f,63,36,'Centres / bin',fontsize=6.2,va='bottom')
 title(f,83,42,'e','Population distributions')
 totals={}
 for k,(dim,lab,col) in enumerate([('g','Greenness response, $g$','#849637'),('T','Structural state, $T$',PURPLE)]):
  a=axis(f,88,27 if k==0 else 8,78,10)
  for sup,ls in [('available','-'),('complete_history',(0,(3,1.7)))]:
   z=j[j.support==sup].groupby(dim+'_low',sort=True)['count'].sum()
   total=int(z.sum());totals[sup]=total
   edges=np.r_[z.index.to_numpy(),j[dim+'_high'].max()]
   vals=100*z.to_numpy()/total
   np.testing.assert_allclose(vals.sum(),100,atol=1e-10)
   a.stairs(vals,edges,color=col,lw=.95,ls=ls)
  a.set(xlim=(-1,1),xticks=[-1,0,1],xlabel=lab,ylabel='Centres (%)')
  a.xaxis.labelpad=1.5;a.yaxis.labelpad=2
  a.axvline(0,color=MID,lw=.5);clean(a)
 assert j['count'].max()<=100000
 save(f,out,'SupplementaryFig3_historical_native_states',{'annual_estimates':208,'joint_density_bins':6400,'marginal_bins':320,'density_support':totals,'density_population':'Available forest-present observations, 2025','marginals':'Both populations, each normalized by its own total; exact original bins, no smoothing','endpoint_comparisons':'Reported without duplication in Supplementary Table 10','same_scientific_data':True})

def swarm_y(values,centre,width=.12,minimum_x_spacing=.0035):
 
 values=np.asarray(values);order=np.argsort(values,kind='stable');placed=[];ys=np.empty(len(values))
 levels=np.r_[0,np.column_stack([np.arange(1,14),-np.arange(1,14)]).ravel()]*width/13
 for i in order:
  for level in levels:
   if all(abs(values[i]-x)>=minimum_x_spacing or abs(level-y)>width/14 for x,y in placed):break
  ys[i]=centre+level;placed.append((values[i],level))
 return ys


def historical_sensitivity(root,out):
 hist=read(root,'main_figures/Fig3_historical_contributions.csv')
 cohorts=read(root,'supplementary/historical_fixed_cohorts.csv')
 cover=read(root,'supplementary/historical_coverage.csv')
 loo=read(root,'supplementary/historical_leave_one_out.csv')
 d=read(root,'city_buffer_sensitivity/buffer_annual.csv')
 e=read(root,'city_buffer_sensitivity/buffer_comparison_table.csv')
 h=hist[(hist.region_id==0)&(hist.scheme=='pooled_centres')].sort_values('year')
 assert len(h)==len(cover)==26
 assert not loo[['excluded_unit_type','excluded_unit']].duplicated().any()
 assert len(d)==858 and not d.duplicated(['buffer_km','region_id','year']).any()
 comps=['pressure','recovery','growth','health','trajectory']
 np.testing.assert_allclose(d[[s+'_mean' for s in comps]].sum(axis=1),d.S,rtol=0,atol=1e-12)
 np.testing.assert_allclose(d.UMRI,d.S/(.05934161890467785+abs(d.S)),rtol=0,atol=1e-12)
 f=canvas(106);title(f,3,101,'a','Observation support');title(f,100,101,'b','Spatial exclusions')
 a=axis(f,15,69,78,26)
 a.plot(h.year,h.UMRI,lw=1.3,color=INK,label='All valid')
 for support,col,ls,lab in [('paired2000_2025',PURPLE,(0,(3,1.5)),'Both endpoints'),('complete26years',TEAL,'-','All 26 years')]:
  q=cohorts[(cohorts.region_id==0)&(cohorts.scheme=='pooled_centres')&(cohorts.support==support)].sort_values('year');assert len(q)==26
  a.plot(q.year,q.UMRI,lw=1.05,color=col,ls=ls,label=lab)
 a.set(xlim=(2000,2025),ylim=(-.53,-.22),xticks=[2000,2010,2025],yticks=[-.5,-.4,-.3],ylabel='UMRI')
 a.tick_params(axis='x',labelbottom=False);clean(a)
 a.legend(loc='lower left',fontsize=6.5,labelspacing=.2,handlelength=1.7,borderpad=.1)
 a=axis(f,15,60,78,7)
 a.plot(cover.year,cover.valid_percent,color=TEAL,lw=.95)
 a.set(xlim=(2000,2025),ylim=(90,101),xticks=[2000,2010,2025],yticks=[90,100],ylabel='Valid (%)');clean(a)
 a=axis(f,112,60,54,35)
 ex=[]
 for typ,n,col,mark,lab in [('block_id',141,TEAL,None,'One block'),('region_id',10,PURPLE,'o','One region')]:
  q=loo[loo.excluded_unit_type==typ];assert len(q)==n and q.delta_UMRI.notna().all()
  x=np.sort(q.delta_UMRI.to_numpy());y=100*np.arange(1,n+1)/n
  a.step(np.r_[x[0],x],np.r_[0,y],where='post',color=col,lw=1.0,label=lab)
  if mark:a.plot(x,y,ls='none',marker=mark,ms=2.4,color=col,mec='white',mew=.25)
  ex.extend({'excluded_unit_type':typ,'delta_UMRI':float(v),'cumulative_percent':float(w)} for v,w in zip(x,y))
 ref=float(h[h.year==2025].UMRI.iloc[0]-h[h.year==2000].UMRI.iloc[0])
 a.axvline(ref,color='#BE7857',lw=.8,ls=(0,(2,1.4)))
 a.set(xlim=(-.315,-.15),ylim=(0,103),xticks=[-.30,-.24,-.18],yticks=[0,50,100],xlabel='UMRI change, 2000–2025',ylabel='Estimates (%)');clean(a,'x')
 a.legend(loc='lower right',fontsize=6.5,labelspacing=.2,handlelength=1.5,borderpad=.15)
 title(f,3,49,'c','City-domain sensitivity')
 a=axis(f,37,11,129,33);region_rows(a);a.tick_params(axis='y',labelsize=6.6)
 a.axvline(0,color='#7F918B',lw=.65,zorder=1);clean(a,'x')
 pivot=d.pivot(index=['region_id','year'],columns='buffer_km',values='UMRI')
 colors={1:'#BE7857',5:'#3C8F9D'};display=[]
 for i,rid in enumerate(ORDER):
  for buf,off,mark in [(1,-.16,'o'),(5,.16,'^')]:
   q=pivot.loc[rid].sort_index();assert list(q.index)==list(range(2000,2026))
   delta=(q[buf]-q[2]).to_numpy();assert len(delta)==26 and np.isfinite(delta).all()
   a.scatter(delta,swarm_y(delta,i+off),s=5.8,marker=mark,color=colors[buf],alpha=.58,lw=0,zorder=3)
   display.extend({'buffer_km':buf,'region_id':rid,'year':int(year),'difference_from_2km':float(x)} for year,x in zip(q.index,delta))
 a.set(xlim=(-.22,.25),xticks=[-.2,-.1,0,.1,.2],xlabel='Annual UMRI difference from the 2-km domain')
 f.legend(handles=[Line2D([0],[0],marker=mark,color=colors[buf],lw=0,ms=3.2,label=f'{buf} km') for buf,mark in [(1,'o'),(5,'^')]],loc='center',bbox_to_anchor=(.62,1.8/106),ncol=2,columnspacing=2,handletextpad=.4)
 alt=e[(e.region_id>0)&(e.buffer_km!=2)];assert int(alt.endpoint_direction_same_as_2km.sum())==19
 assert len(display)==572
 pd.DataFrame(display).to_csv(out/'city_buffer_annual_display.csv',index=False,float_format='%.17g')
 pd.DataFrame(ex).to_csv(out/'spatial_exclusion_display.csv',index=False,float_format='%.17g')
 save(f,out,'SupplementaryFig1_observation_domain_sensitivity',{'historical_estimates':78,'coverage_years':26,'exclusion_estimates':151,'source_city_domain_estimates':858,'annual_domain_differences':572,'years_per_region_buffer':26,'annual_display':'Exact annual differences of regional estimates from their 2-km counterpart; deterministic ordinate packing within category, no smoothing','source_populations_retained':True,'endpoint_comparisons':'Full values retained in Supplementary Table 12, not repeated as a figure','excluded_estimates':0,'no_bar_charts':True})

def main():
 p=argparse.ArgumentParser();p.add_argument('--source-data',type=Path,default=Path(__file__).resolve().parents[2]/'SourceData');p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'reproduced');p.add_argument('--figures',default='S1,S3');a=p.parse_args();style();a.output.mkdir(parents=True,exist_ok=True)
 for k,fun in [('S1',historical_sensitivity),('S3',historical_native)]:
  if k in a.figures.split(','):fun(a.source_data,a.output)
 (a.output/'supplementary_b_checks.json').write_text(json.dumps({'input_sha256':INPUTS,'figures':AUDIT},indent=2,ensure_ascii=False)+'\n',encoding='utf8')
 if any(v['out_of_canvas_text'] for v in AUDIT.values()):raise RuntimeError('Figure text extends beyond canvas; inspect checks.')
if __name__=='__main__':main()
