






from __future__ import annotations
import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle,Polygon,Circle,Ellipse,Arc,FancyArrowPatch,PathPatch
from matplotlib.path import Path as MPath
from matplotlib.colors import LinearSegmentedColormap,BoundaryNorm,ListedColormap
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
parser=argparse.ArgumentParser()
parser.add_argument('--source-data',type=Path,default=ROOT.parent/'SourceData')
parser.add_argument('--cartography',type=Path,default=ROOT/'cartography')
parser.add_argument('--derived-data',type=Path)
parser.add_argument('--output',type=Path,default=ROOT/'reproduced')
parser.add_argument('--draft',action='store_true')
parser.add_argument('--figures',default='2,3,4,5,6')
ARGS=parser.parse_args()
SOURCE=ARGS.source_data/'main_figures';DERIVED=ARGS.derived_data or SOURCE
OUT=ARGS.output/'figures';QA=ARGS.output/'qa'
OUT.mkdir(parents=True,exist_ok=True);QA.mkdir(parents=True,exist_ok=True)
W=179;DARK='#263B40';MID='#657A7C';GRID='#E5EAE8';TEAL='#397B73';CLAY='#B96756'
SCENARIOS=['ssp126','ssp245','ssp370','ssp585']
LABELS=dict(zip(SCENARIOS,['SSP1–2.6','SSP2–4.5','SSP3–7.0','SSP5–8.5']))
COLORS=dict(zip(SCENARIOS,['#0072B2','#B8860B','#009E49','#D44732']))
STYLES=dict(zip(SCENARIOS,[(0,(5.0,2.4)),(0,(2.5,1.8)),(0,(0.7,1.8)),(0,(4.5,1.6,0.8,1.6))]))
COMP=['d','r','g','h','T'];CC=['#C47761','#358F84','#9CAF60','#639DB7','#9276A2']
REGIONS={1:'Australia / NZ',2:'Central America',3:'Eastern Africa',4:'Eastern Asia',5:'Northern America',6:'South America',7:'Southeastern Asia',8:'Southern Asia',9:'Western Africa',10:'Western Asia'}
ABBR={1:'ANZ',2:'CAM',3:'EAF',4:'EAS',5:'NAM',6:'SAM',7:'SEA',8:'SAS',9:'WAF',10:'WAS'}
ORDER=[5,2,6,9,3,10,8,7,4,1]
LETTERS={r:chr(99+i) for i,r in enumerate(ORDER)}
ALLOC_COL=['#BACAC3','#4D8C7E','#BF7D65']
INPUTS={};AUDIT={}
mpl.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.labelsize':7.5,'axes.titlesize':8.2,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7.3,'axes.linewidth':.55,'xtick.major.width':.5,'ytick.major.width':.5,'xtick.major.size':2.2,'ytick.major.size':2.2,'text.color':DARK,'axes.labelcolor':DARK,'xtick.color':DARK,'ytick.color':DARK,'axes.edgecolor':MID,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','path.simplify':False})

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(name,derived=False):
 p=(DERIVED if derived else SOURCE)/name;INPUTS[name]=sha(p);return pd.read_csv(p,float_precision='round_trip')
def axmm(fig,x,y,w,h,**kw):return fig.add_axes([x/W,y/(fig.get_figheight()*25.4),w/W,h/(fig.get_figheight()*25.4)],**kw)
def tx(fig,x,y,s,**kw):return fig.text(x/W,y/(fig.get_figheight()*25.4),s,**kw)
def panel(fig,x,y,l,title,size=8.2):
 tx(fig,x,y,l,fontweight='bold',fontsize=10,va='baseline');tx(fig,x+4.5,y,title,fontsize=size,va='baseline')
def clean(ax,grid='y'):
 ax.set_axisbelow(True)
 if grid:ax.grid(axis=grid,color=GRID,lw=.42)
def legend(fig,y,scenarios=SCENARIOS,observed=False,x=.52,styles=None,ncol=None):
 styles=STYLES if styles is None else styles
 hs=[Line2D([0],[0],color=COLORS[s],ls=styles[s],lw=1.4,label=LABELS[s],dash_capstyle='round') for s in scenarios]
 if observed:hs.insert(0,Line2D([0],[0],color=DARK,lw=1.3,label='Observed'))
 fig.legend(handles=hs,loc='center',bbox_to_anchor=(x,y/(fig.get_figheight()*25.4)),ncol=len(hs) if ncol is None else ncol,handlelength=2.1,handletextpad=.5,columnspacing=1.6,borderaxespad=0)
def ribbon(ax,frame,s,prefix='delta_UMRI',lw=1.2,linestyle=None):
 d=frame[frame.scenario==s].sort_values('year')
 ax.fill_between(d.year,d[prefix+'_q17'],d[prefix+'_q83'],color=COLORS[s],alpha=.11,lw=0)
 ax.plot(d.year,d[prefix+'_q50'],color=COLORS[s],ls=STYLES[s] if linestyle is None else linestyle,lw=lw,zorder=3,dash_capstyle='round')
def save(fig,stem):
 fig.canvas.draw();r=fig.canvas.get_renderer();overflow=[]
 for art in fig.findobj(mpl.text.Text):
  if art.get_visible() and art.get_text() and not art.get_clip_on():
   b=art.get_window_extent(r)
   if b.x0<-.8 or b.y0<-.8 or b.x1>fig.bbox.x1+.8 or b.y1>fig.bbox.y1+.8:overflow.append(art.get_text())
 exports=[]
 formats=[('png',320 if stem=='Fig2_regional_trajectories' else 240),('pdf',None),('svg',None)]
 if not ARGS.draft:formats+=[('600dpi.png',600),('tiff',600)]
 for suffix,dpi in formats:
  p=OUT/(stem+('_'+suffix if suffix=='600dpi.png' else '.'+suffix));kw={'dpi':dpi} if dpi else {}
  if suffix=='tiff':kw['pil_kwargs']={'compression':'tiff_lzw'}
  fig.savefig(p,**kw);exports.append({'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size})
 AUDIT[stem]={'width_mm':W,'height_mm':round(fig.get_figheight()*25.4,3),'text_outside_canvas':overflow,'exports':exports,'status':'RENDERED_FOR_VISUAL_REVIEW'}
 plt.close(fig);print('EXPORTED',stem,'overflow',overflow,flush=True)


def city(ax,x,y,s=1,c=DARK,new=False):
 for dx,h in [(0,.48),(.25,.78),(.55,.58)]:
  ax.add_patch(Rectangle((x+dx*s,y),.19*s,h*s,fc=c,ec='none',zorder=4))
  for dy in np.arange(.13,h-.04,.18):
   ax.plot([x+(dx+.055)*s,x+(dx+.13)*s],[y+dy*s]*2,color='white',lw=.55,zorder=5)
 if new:
  ax.plot([x+.9*s,x+.9*s],[y+.25*s,y+.65*s],color=c,lw=1.1);ax.plot([x+.7*s,x+1.1*s],[y+.45*s]*2,color=c,lw=1.1)
def mangrove(ax,x,y,s=1,c=TEAL):
 ax.plot([x,x],[y+.08*s,y+.56*s],color=c,lw=1.05,zorder=3)
 for dx in [-.26,-.13,.14,.27]:ax.plot([x,x+dx*s],[y+.25*s,y],color=c,lw=.75,zorder=3)
 for dx,dy,rx,ry in [(-.2,.59,.43,.30),(.16,.64,.46,.32),(0,.82,.43,.34)]:ax.add_patch(Ellipse((x+dx*s,y+dy*s),rx*s,ry*s,fc=c,ec='none',zorder=4))
def leaf(ax,x,y,s,c):
 verts=[(x,y),(x-.35*s,y+.35*s),(x-.2*s,y+.85*s),(x+.4*s,y+s),(x+.55*s,y+.4*s),(x+.25*s,y+.02*s),(x,y)]
 path=MPath(verts,[MPath.MOVETO]+[MPath.CURVE3]*6);ax.add_patch(PathPatch(path,fc=c,ec='none'));ax.plot([x,y*0+x+.3*s],[y+.13*s,y+.78*s],color='white',lw=.85)
def symbol(fig,x,y,i,size=6.2):
 a=axmm(fig,x,y,size,size);a.set(xlim=(0,1),ylim=(0,1));a.axis('off');c=CC[i]
 if i==0:city(a,.08,.06,.92,c)
 elif i==1:
  mangrove(a,.51,.13,.7,c);a.add_patch(Arc((.5,.48),.88,.84,theta1=38,theta2=310,lw=1.1,color=c));a.add_patch(Polygon([[.89,.22],[.89,.43],[.72,.31]],fc=c,ec='none'))
 elif i==2:
  leaf(a,.39,.06,.82,c);a.add_patch(FancyArrowPatch((.08,.12),(.08,.77),arrowstyle='-|>',mutation_scale=6,lw=.8,color=c))
 elif i==3:
  v=[(.5,.96),(.35,.73),(.14,.43),(.17,.21),(.33,.04),(.64,.04),(.84,.24),(.84,.44),(.5,.96)]
  a.add_patch(Polygon(v,fc=c,ec='none'));a.add_patch(Arc((.49,.38),.42,.41,theta1=200,theta2=290,lw=1,color='white'))
 else:
  a.plot([.18,.5,.82],[.24,.68,.24],color=c,lw=1.2)
  for xx,yy in [(.18,.24),(.5,.68),(.82,.24)]:a.add_patch(Polygon([[xx-.13,yy-.12],[xx+.10,yy-.15],[xx+.16,yy+.04],[xx+.04,yy+.16],[xx-.15,yy+.11]],fc=c,ec='none'))

def trajectory(ax,rid,small=False,overlay=False):
 h=HIST[HIST.region_id==rid].sort_values('year');f=FUT[FUT.region_id==rid]
 for s in SCENARIOS:ribbon(ax,f,s,prefix='UMRI',lw=.90 if small else 1.10,linestyle=STYLES[s])
 ax.plot(h.year,h.UMRI,color=DARK,lw=1.15,zorder=6)
 ax.axvline(2025,color=MID,ls=(0,(2,2)),lw=.55);ax.axhline(0,color=MID,lw=.5,zorder=1)
 ax.set(xlim=(2000,2100),ylim=(-.9,.3));ax.set_yticks([-.8,-.4,0]);ax.set_xticks([2000,2050,2100]);clean(ax,grid=None)
 ax.tick_params(pad=2,labelsize=6.5 if small else 7)

def figure2():
 height_mm=160;top_shift_mm=16;regional_height_mm=29.5
 fig=plt.figure(figsize=(W/25.4,height_mm/25.4))
 def topax(x,y,*args,**kw):return axmm(fig,x,y+top_shift_mm,*args,**kw)
 def toptx(x,y,*args,**kw):return tx(fig,x,y+top_shift_mm,*args,**kw)
 
 
 row_scales=[]
 for row in range(2):
  ids=ORDER[row*5:(row+1)*5]
  observed=HIST.loc[HIST.region_id.isin(ids),'UMRI'].to_numpy()
  projected=FUT.loc[FUT.region_id.isin(ids),['UMRI_q17','UMRI_q50','UMRI_q83']].to_numpy().ravel()
  values=np.concatenate([observed,projected]);assert np.isfinite(values).all()
  lower=float(values.min());upper=float(values.max());padding=.05*(upper-lower)
  limits=[lower-padding,upper+padding]
  ticks=np.arange(np.ceil(limits[0]/.2),np.floor(limits[1]/.2)+1)*.2
  ticks[np.isclose(ticks,0,atol=1e-12)]=0
  row_scales.append({'region_ids':ids,'data_minimum':lower,'data_maximum':upper,
   'padding_fraction':.05,'limits':limits,'ticks':ticks.tolist()})
 panel(fig,3,139+top_shift_mm,'a','Mangrove distribution and study coverage')
 
 region_colours={5:'#0076C8',2:'#EAAA00',6:'#75BF8A',9:'#AA2040',3:'#B58A9F',10:'#4A2A00',8:'#D46020',7:'#00BFEA',4:'#6A0B8A',1:'#606075'}
 proj=ccrs.Robinson(central_longitude=0);ax=topax(2,76,175,62,projection=proj)
 ax.set_global();ax.set_ylim(proj.transform_point(0,-45,ccrs.PlateCarree())[1],proj.transform_point(0,40,ccrs.PlateCarree())[1])
 ax.set_xlim(proj.transform_point(-120,0,ccrs.PlateCarree())[0],proj.transform_point(180,0,ccrs.PlateCarree())[0])
 ax.set_facecolor('#FAFCFC')
 land=ARGS.cartography/'ne_110m_land.shp';INPUTS['ne_110m_land.shp']=sha(land)
 ax.add_geometries(shpreader.Reader(str(land)).geometries(),ccrs.PlateCarree(),facecolor='#EBEEEA',edgecolor='#CAD3CB',lw=.23,zorder=1)
 ax.gridlines(xlocs=np.arange(-180,181,60),ylocs=[-30,0,30],color='white',lw=.45,alpha=.9,zorder=2)
 context=read('Fig2_global_mangrove_context_2025_005deg.csv',True)
 ax.scatter(context.longitude,context.latitude,s=.40,color='#9CAE9E',edgecolors='none',transform=ccrs.PlateCarree(),zorder=3,rasterized=False)
 reference=read('Fig2_reference_footprint_005deg.csv',True)
 
 
 assert len(reference)==645 and int(reference.reference_centres.sum())==934814
 reference=reference.assign(display_lon_bin=reference.lon_bin//20,display_lat_bin=reference.lat_bin//20)
 rows=[]
 for (rid,xb,yb),g in reference.groupby(['region_id','display_lon_bin','display_lat_bin'],sort=True):
  n=int(g.reference_centres.sum());area=n*.0009
  rows.append({'region_id':int(rid),'longitude_bin':int(xb),'latitude_bin':int(yb),
   'longitude':float(np.average(g.longitude,weights=g.reference_centres)),
   'latitude':float(np.average(g.latitude,weights=g.reference_centres)),
   'reference_centres':n,'reference_area_km2':area,'source_005deg_cells':len(g)})
 m=pd.DataFrame(rows)
 assert len(m)==73 and int(m.reference_centres.sum())==934814
 assert m.groupby('region_id').reference_centres.sum().to_dict()==reference.groupby('region_id').reference_centres.sum().to_dict()
 
 
 breaks=np.array([1.,10.,50.]);class_sizes=np.array([8.,16.,30.,50.])
 labels_area=['≤1','1–10','10–50','>50']
 m['area_class']=np.searchsorted(breaks,m.reference_area_km2,side='left')
 m['area_class_label']=[labels_area[i] for i in m.area_class]
 m['marker_area_pt2']=class_sizes[m.area_class]
 m['region_colour']=[region_colours[r] for r in m.region_id]
 m.to_csv(QA/'map_reference_distribution_1deg.csv',index=False,float_format='%.15g')
 m=m.sort_values(['marker_area_pt2','reference_area_km2'],ascending=False,kind='stable').reset_index(drop=True)
 sc=ax.scatter(m.longitude,m.latitude,s=m.marker_area_pt2,c=m.region_colour,edgecolors='white',linewidths=.36,
  transform=ccrs.PlateCarree(),zorder=5)
 label_positions={1:(159,-29),2:(-103,19),3:(37,-31),4:(126.5,27.5),5:(-93,31),6:(-55.5,-32),7:(108.5,8.8),8:(85,19),9:(-20,12),10:(57,29.5)}
 label_artists=[]
 for rid in ORDER:
  dest=proj.transform_point(*label_positions[rid],ccrs.PlateCarree())
  label=ax.text(*dest,LETTERS[rid],ha='center',va='center',fontsize=8.0,fontweight='bold',color=DARK,zorder=8)
  label_artists.append((rid,label))
 ax.spines['geo'].set_visible(False)
 toptx(105.0,101.5,'b',fontweight='bold',fontsize=8.7,va='baseline');toptx(109.5,101.5,'Overall trajectory',fontsize=6.5,va='baseline')
 inset=topax(109.0,85.0,26.0,14.5);trajectory(inset,0,True,overlay=True)
 inset.set_facecolor('none');inset.patch.set_visible(False);inset.grid(False)
 for spine in ['left','bottom']:
  inset.spines[spine].set_linewidth(.4);inset.spines[spine].set_color('#73898C')
 inset.set_ylabel('UMRI',fontsize=5.2,labelpad=.6)
 inset.set_ylim(-.72,-.18);inset.set_yticks([-.6,-.4,-.2]);inset.tick_params(labelsize=5.5,pad=1.0)
 inset.get_xticklabels()[0].set_ha('left');inset.get_xticklabels()[-1].set_ha('right')

 
 
 zoom_extent=[-83.2,-79.8,25.25,29.8]
 zoom=topax(3,80.5,17,23,projection=ccrs.Mercator(latitude_true_scale=27.5))
 zoom.set_extent(zoom_extent,crs=ccrs.PlateCarree());zoom.set_facecolor('#FAFCFC')
 from shapely.geometry import box
 land_detail=ARGS.cartography/'ne_50m_land.shp';INPUTS['ne_50m_land.shp']=sha(land_detail)
 selection=box(zoom_extent[0],zoom_extent[2],zoom_extent[1],zoom_extent[3])
 detail_land=[g for g in shpreader.Reader(str(land_detail)).geometries() if g.intersects(selection)]
 zoom.add_geometries(detail_land,ccrs.PlateCarree(),facecolor='#EBEEEA',edgecolor='#CAD3CB',lw=.32,zorder=1)
 zcontext=context[context.longitude.between(*zoom_extent[:2]) & context.latitude.between(*zoom_extent[2:])]
 zoom.scatter(zcontext.longitude,zcontext.latitude,s=.55,color='#9CAE9E',edgecolors='none',transform=ccrs.PlateCarree(),zorder=2)
 zm=m[m.region_id==5].copy()
 assert len(zm)==11 and int(zm.reference_centres.sum())==290329
 zoom.scatter(zm.longitude,zm.latitude,s=zm.marker_area_pt2,c=zm.region_colour,edgecolors='white',linewidths=.36,
  transform=ccrs.PlateCarree(),zorder=5)
 zoom.spines['geo'].set_edgecolor('#596A6D');zoom.spines['geo'].set_linewidth(.55)
 zoom_title=toptx(3.8,105.1,'c',fontsize=7.4,fontweight='bold',color=DARK,va='bottom');zoom_name=toptx(6.6,105.1,'Florida',fontsize=6.0,color=DARK,va='bottom')
 locator_extent=[-89.0,-75.0,22.0,33.0]
 x0,x1,y0,y1=locator_extent
 ax.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],color='#596A6D',lw=.6,transform=ccrs.PlateCarree(),zorder=7)
 trajectory_axes=[inset]
 
 area_title=toptx(51.0,131.0,'Reference habitat\narea (km²)',fontsize=5.5,ha='center',va='center',linespacing=1.1)
 area_key_axes=[];area_key_texts=[]
 for x,y,s,label in zip([43.5,55.5,43.5,55.5],[125.8,125.8,121.4,121.4],class_sizes,labels_area):
  a=topax(x-1.5,y-1.5,3,3);a.set_facecolor('none');a.patch.set_visible(False);a.scatter([.5],[.5],s=[s],facecolors='white',edgecolors=DARK,linewidths=.55)
  a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
  area_key_axes.append(a);area_key_texts.append(toptx(x+2.2,y,label,fontsize=5.3,va='center'))
 sw=topax(73.3,79.3,2.8,2.5);sw.set(xlim=(0,1),ylim=(0,1));sw.axis('off')
 
 for cx,cy,rad in [(.30,.64,.18),(.50,.73,.21),(.71,.63,.18)]:
  sw.add_patch(Circle((cx,cy),rad,facecolor='#9CAE9E',edgecolor='none'))
 sw.plot([.50,.50],[.54,.27],color='#9CAE9E',lw=.7,solid_capstyle='round')
 for rx in [.19,.50,.81]:sw.plot([.50,rx],[.29,.07],color='#9CAE9E',lw=.55,solid_capstyle='round')
 context_key=toptx(77.0,80.55,'Global mangroves (2025)',fontsize=5.8,color=DARK,va='center')
 legend(fig,73.9+top_shift_mm,observed=True,x=.5,styles=STYLES,ncol=5)
 for i,rid in enumerate(ORDER):
  row,col=divmod(i,5);x=13+col*33.3;y=51 if row==0 else 10
  a=axmm(fig,x,y,29,regional_height_mm);trajectory(a,rid,True);trajectory_axes.append(a)
  a.set_ylim(row_scales[row]['limits']);a.set_yticks(row_scales[row]['ticks'])
  tx(fig,x-1.8,y+33.3,LETTERS[rid],fontsize=8.7,fontweight='bold')
  tx(fig,x+2.0,y+33.3,REGIONS[rid],fontsize=6.85)
  fig.patches.append(Rectangle((x/W,(y+31.4)/height_mm),29/W,.55/height_mm,transform=fig.transFigure,fc=region_colours[rid],ec='none'))
  if col!=0:a.set_yticklabels([])
  else:a.set_ylabel('UMRI',fontsize=7,labelpad=2)
  if row==0:a.set_xticklabels([])
  else:
   a.get_xticklabels()[0].set_ha('left');a.get_xticklabels()[-1].set_ha('right')
 tx(fig,92,1.8,'Year',ha='center',fontsize=7.3)
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 from matplotlib.transforms import Bbox
 def centre_key(axes,texts,centre_mm=89.5):
  boxes=[a.get_window_extent(renderer) for a in axes]+[t.get_window_extent(renderer) for t in texts]
  bb=Bbox.union(boxes);dx=(centre_mm/25.4*fig.dpi-(bb.x0+bb.x1)/2)/fig.bbox.width
  for a in axes:
   pos=a.get_position();a.set_position([pos.x0+dx,pos.y0,pos.width,pos.height])
  for t in texts:
   x,y=t.get_position();t.set_position((x+dx,y))
 centre_key(area_key_axes,area_key_texts,centre_mm=51.0)
 centre_key([sw],[context_key])
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 def screen_points(frame,axis=ax):
  xy=axis.projection.transform_points(ccrs.PlateCarree(),frame.longitude.to_numpy(),frame.latitude.to_numpy())[:,:2]
  return axis.transData.transform(xy)
 def inset_overlap(frame,overlay=inset):
  xy=screen_points(frame);bb=overlay.get_window_extent()
  return int(((xy[:,0]>=bb.x0)&(xy[:,0]<=bb.x1)&(xy[:,1]>=bb.y0)&(xy[:,1]<=bb.y1)).sum())
 def overlap_pairs(frame,sizes,axis=ax):
  xy=screen_points(frame,axis);d=np.linalg.norm(xy[:,None,:]-xy[None,:,:],axis=2)
  radii=(np.sqrt(np.asarray(sizes))*.5+.18)*fig.dpi/72
  mask=np.triu(d<(radii[:,None]+radii[None,:]),1)
  ri=frame.region_id.to_numpy()
  return {'intersecting_symbol_pairs':int(mask.sum()),'cross_region_symbol_pairs':int((mask&(ri[:,None]!=ri[None,:])).sum())}
 label_conflicts=[];label_edge_conflicts=[];point_xy=screen_points(m)
 for rid,label in label_artists:
  bb=label.get_window_extent(renderer).padded(.6*fig.dpi/72)
  hits=np.where((point_xy[:,0]>=bb.x0)&(point_xy[:,0]<=bb.x1)&(point_xy[:,1]>=bb.y0)&(point_xy[:,1]<=bb.y1))[0]
  if len(hits):label_conflicts.append({'label_region':rid,'covered_point_indices':hits.tolist()})
  dx=np.maximum(np.maximum(bb.x0-point_xy[:,0],0),point_xy[:,0]-bb.x1);dy=np.maximum(np.maximum(bb.y0-point_xy[:,1],0),point_xy[:,1]-bb.y1)
  radii=(np.sqrt(m.marker_area_pt2.to_numpy())*.5+.18)*fig.dpi/72
  edge_hits=np.where(np.hypot(dx,dy)<radii)[0]
  if len(edge_hits):label_edge_conflicts.append({'label_region':rid,'overlapping_symbol_indices':edge_hits.tolist()})
 bounds=ax.get_window_extent(renderer)
 def in_main_view(frame):
  xy=screen_points(frame)
  return (xy[:,0]>=bounds.x0)&(xy[:,0]<=bounds.x1)&(xy[:,1]>=bounds.y0)&(xy[:,1]<=bounds.y1)
 visible_context=context[in_main_view(context)]
 future_lines=[];historical_lines=[]
 for ta in trajectory_axes:
  for line in ta.lines:
   xx=np.asarray(line.get_xdata())
   if len(xx)==75 and xx[0]==2026 and xx[-1]==2100:future_lines.append(line.is_dashed())
   if len(xx)==26 and xx[0]==2000 and xx[-1]==2025:historical_lines.append(not line.is_dashed())
 assert len(future_lines)==44 and all(future_lines) and len(historical_lines)==11 and all(historical_lines)
 def bbox_hits(frame,bb):
  xy=screen_points(frame)
  return int(((xy[:,0]>=bb.x0)&(xy[:,0]<=bb.x1)&(xy[:,1]>=bb.y0)&(xy[:,1]<=bb.y1)).sum())
 inset_tight=inset.get_tightbbox(renderer)
 context_key_bb=context_key.get_window_extent(renderer)
 b_text=[t for t in inset.findobj(mpl.text.Text) if t.get_visible() and t.get_text()]
 b_text += [t for t in fig.texts if t.get_text() in ['b','Overall trajectory']]
 b_text_conflicts=[]
 for t in b_text:
  bb=t.get_window_extent(renderer).padded(.3*fig.dpi/72)
  scount=bbox_hits(m,bb);ccount=bbox_hits(visible_context,bb)
  if scount or ccount:b_text_conflicts.append({'text':t.get_text(),'study':scount,'context':ccount})
 
 
 from shapely.geometry import Point
 zxy=screen_points(zm,zoom)
 zr=np.sqrt(zm.marker_area_pt2.to_numpy())*.5*fig.dpi/72
 disks=[Point(*xy).buffer(float(rad),quad_segs=64) for xy,rad in zip(zxy,zr)]
 zoom_visibility=[]
 for i,disk in enumerate(disks):
  remaining=disk
  for j in range(i+1,len(disks)):
   remaining=remaining.difference(disks[j].buffer(.18*fig.dpi/72))
  zoom_visibility.append(float(np.clip(remaining.area/disk.area,0,1)))
 
 all_text=[t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text()]
 text_boxes=[t.get_window_extent(renderer) for t in all_text]
 text_collisions=[]
 for i in range(len(all_text)):
  for j in range(i+1,len(all_text)):
   p,q=text_boxes[i],text_boxes[j]
   if min(p.x1,q.x1)-max(p.x0,q.x0)>.25 and min(p.y1,q.y1)-max(p.y0,q.y0)>.25:
    text_collisions.append([all_text[i].get_text(),all_text[j].get_text()])
 
 def symbol_clips(frame,axis):
  xy=screen_points(frame,axis);bb=axis.get_window_extent(renderer)
  radii=(np.sqrt(frame.marker_area_pt2.to_numpy())*.5+.18)*fig.dpi/72
  return int(((xy[:,0]-radii<bb.x0)|(xy[:,0]+radii>bb.x1)|(xy[:,1]-radii<bb.y0)|(xy[:,1]+radii>bb.y1)).sum())
 from shapely.geometry import Polygon as GeoPolygon
 locator_corners=pd.DataFrame({'longitude':[locator_extent[0],locator_extent[1],locator_extent[1],locator_extent[0]],'latitude':[locator_extent[2],locator_extent[2],locator_extent[3],locator_extent[3]]})
 locator_polygon=GeoPolygon(screen_points(locator_corners))
 located_centres=screen_points(zm)
 locator_clearance=[]
 for xy,sz in zip(located_centres,zm.marker_area_pt2):
  radius=(np.sqrt(sz)*.5+.18)*fig.dpi/72
  disc=Point(*xy).buffer(radius,quad_segs=64)
  locator_clearance.append(float(locator_polygon.boundary.distance(disc)/fig.dpi*25.4) if locator_polygon.contains(disc) else -1)
 regional_boxes=[a.get_position().bounds for a in trajectory_axes[1:]]
 regional_alignment={'maximum_width_difference_mm':float(np.ptp([a[2] for a in regional_boxes])*179),
  'maximum_height_difference_mm':float(np.ptp([a[3] for a in regional_boxes])*height_mm),
  'row_baseline_spread_mm':[float(np.ptp([a[1] for a in regional_boxes[k:k+5]])*height_mm) for k in [0,5]]}
 area_bb=Bbox.union([a.get_window_extent(renderer) for a in area_key_axes]+[t.get_window_extent(renderer) for t in area_key_texts])
 context_bb=Bbox.union([sw.get_window_extent(renderer),context_key.get_window_extent(renderer)])
 
 
 review={'status':'RENDERED_FOR_VISUAL_REVIEW','map_role':'Geographic study coverage by region and baseline habitat area',
  'colour_quantity':'Ten categorical macroregions','region_colours':region_colours,'symbol_quantity':'Historical reference habitat area, binned km2',
  'size_class_boundaries_km2':breaks.tolist(),'marker_areas_pt2':class_sizes.tolist(),
  'size_class_counts':m.area_class.value_counts().sort_index().to_dict(),
  'mapped_region_cells':len(m),'full_reference_centres':int(m.reference_centres.sum()),'full_reference_area_km2':float(m.reference_area_km2.sum()),
  'source_005deg_cells':len(reference),'display_grid_degrees':1,'coordinates':'Reference-centre-weighted means within region-specific display cells',
  'latitude_limits':[-45,40],'longitude_window_at_equator':[-120,180],'map_scale_factor_relative_to_full_longitude_view':1.2,'study_latitude_range':[float(reference.latitude.min()),float(reference.latitude.max())],
  'context_latitude_range':[float(context.latitude.min()),float(context.latitude.max())],
  'study_centres_clipped':int(reference.loc[~in_main_view(reference),'reference_centres'].sum()),
  'global_context_cells_clipped':int((~in_main_view(context)).sum()),
  'global_context_cells_in_view':int(in_main_view(context).sum()),
  'global_context_cells_in_source':len(context),
  'global_context_latitude_clipped':int((~context.latitude.between(-45,40)).sum()),
  'actual_map_width_mm':bounds.width/fig.dpi*25.4,'actual_map_height_mm':bounds.height/fig.dpi*25.4,
  'inset_plot_width_mm':26.0,'inset_plot_height_mm':14.5,'background_alpha':inset.patch.get_facecolor()[3],'overall_patch_visible':inset.patch.get_visible(),'overall_grid_visible':any(g.get_visible() for g in inset.get_xgridlines()+inset.get_ygridlines()),
  'study_cells_under_inset':inset_overlap(m),'global_context_cells_under_inset':inset_overlap(visible_context),
  'study_cells_under_zoom':inset_overlap(m,zoom),'global_context_cells_under_zoom':inset_overlap(visible_context,zoom),
  'overall_trajectory_location':'Indian Ocean within the main map','opaque_inset_backing':False,'historical_period_fill_in_inset':False,'historical_period_fill_in_regional_panels':False,'regional_horizontal_grid_visible':False,'zero_reference_line_retained':True,'year_2025_separator_retained':True,'label_point_centre_conflicts':label_conflicts,'label_symbol_edge_conflicts':label_edge_conflicts,
  'original_005deg_fixed_symbol_overlap_at_current_extent':overlap_pairs(reference,np.full(len(reference),9.8)),
  'current_symbol_overlap':overlap_pairs(m,m.marker_area_pt2.to_numpy()),'draw_order':'Large area classes first; smaller symbols on top',
  'regional_keys':'Direct black letters c-l beside their regions; matching panel header strips; no leaders or inset connectors',
  'trajectory_support_unchanged':925774,'trajectory_colours_unchanged':False,'scenario_colours':COLORS,'scenario_dash_patterns':STYLES,'uncertainty_band_alpha':.11,
  'future_line_style':'All44 forecast curves use four pathway-specific dash patterns,2026-2100','historical_line_style':'All 11 historical curves solid, 2000-2025',
  'forecast_curves_checked':len(future_lines),'historical_curves_checked':len(historical_lines),
  'layout_status':'REGIONAL_COVERAGE_MAP_WITH_SCENARIO_STYLES',
  'text_objects_checked':len(all_text),'text_bbox_intersections':text_collisions,
  'main_map_symbol_outlines_clipped':symbol_clips(m,ax),'zoom_symbol_outlines_clipped':symbol_clips(zm,zoom),
  'locator_frame_minimum_symbol_clearance_mm':min(locator_clearance),
  'regional_panel_alignment':regional_alignment,'regional_panel_height_mm':regional_height_mm,
  'regional_row_scales':row_scales,'regional_vertical_scale_rule':'Shared within each row; historical UMRI and all projected 17th-83rd percentile values plus 5 percent range padding',
  'context_legend_icon':'Editable mangrove canopy, trunk and prop-root silhouette in context colour',
  'study_points_under_area_key':bbox_hits(m,area_bb)+bbox_hits(m,area_title.get_window_extent(renderer)),
  'context_points_under_area_key':bbox_hits(visible_context,area_bb)+bbox_hits(visible_context,area_title.get_window_extent(renderer)),
  'area_key_background_transparent':all(not a.patch.get_visible() for a in area_key_axes),
  'area_key_clearance_margin_mm':0.5,
  'context_points_within_area_key_clearance':sum(bbox_hits(visible_context,Bbox.from_extents(bb.x0-fig.dpi/25.4*.5,bb.y0-fig.dpi/25.4*.5,bb.x1+fig.dpi/25.4*.5,bb.y1+fig.dpi/25.4*.5)) for bb in [area_bb,area_title.get_window_extent(renderer)]),
  'study_points_under_overall_with_axis_labels':bbox_hits(m,inset_tight),
  'context_points_in_overall_union_bbox_including_unpainted_space':bbox_hits(visible_context,inset_tight),
  'context_points_under_context_key':bbox_hits(visible_context,context_key_bb),
  'study_points_under_context_key':bbox_hits(m,context_key_bb),
  'context_points_under_zoom_title':bbox_hits(visible_context,zoom_title.get_window_extent(renderer)),
  'overall_text_context_conflicts':b_text_conflicts,
  'area_legend_title_center_mm':51.0,'area_legend_location':'North Atlantic, west of Africa','area_legend_group_center_mm':(area_bb.x0+area_bb.x1)/2/fig.dpi*25.4,
  'context_legend_group_center_mm':(context_bb.x0+context_bb.x1)/2/fig.dpi*25.4,'scenario_legend_columns':5,
  'map_leader_count':0,'inset_connector_count':0,'northern_america_zoom':{'projection':'Mercator, true scale at 27.5N','extent':zoom_extent,
   'display_units':len(zm),'reference_centres':int(zm.reference_centres.sum()),
   'same_symbols_and_sizes_as_main_map':True,'linkage':'Matching c labels and a thin charcoal regional locator frame; no connector','locator_frame_extent':locator_extent,'locator_frame_role':'Encloses the study region and its plotted symbols; the local zoom has separately recorded tighter bounds','basemap':'Natural Earth 1:50 million land',
   'overlap':overlap_pairs(zm,zm.marker_area_pt2.to_numpy(),zoom),
   'minimum_unoccluded_symbol_fraction':min(zoom_visibility),'fully_occluded_symbols':sum(x<.01 for x in zoom_visibility),
   'symbol_visible_fractions':zoom_visibility,
   'width_mm':zoom.get_window_extent().width/fig.dpi*25.4,'height_mm':zoom.get_window_extent().height/fig.dpi*25.4}}
 
 (QA/'map_display_checks.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf8')
 save(fig,'Fig2_regional_trajectories')


def story_figures(numbers):
 command=[sys.executable,'-X','utf8','-B',str(HERE/'reproduce_figure_story.py'),'--source-data',str(ARGS.source_data),'--output',str(ARGS.output),'--figures',','.join(numbers)]
 if ARGS.draft:command.append('--draft')
 subprocess.run(command,check=True)
 manifest=json.loads((QA/'figure_story_manifest.json').read_text(encoding='utf8'))
 AUDIT.update(manifest['figures']);INPUTS.update(manifest['inputs'])

def verify():
 assert len(HIST)==286 and set(HIST.region_id)==set(range(11))
 assert set(FUT.year)==set(range(2026,2101)) and (FUT.members==15).all()
 for frame in [URBAN,PROT,CUM]:
  assert (frame.members==15).all();assert (frame.delta_UMRI_q17<=frame.delta_UMRI_q50).all();assert (frame.delta_UMRI_q50<=frame.delta_UMRI_q83).all()
 h=float(abs(HISTFULL[[k+'_contribution' for k in COMP]].sum(axis=1)-HISTFULL.S).max());f=float(abs(CONTRIB[['delta_'+k+'_contribution' for k in COMP]].sum(axis=1)-CONTRIB.delta_S).max())
 a=float(abs(BUDGET[['in_cell_protected_growth_km2','relocated_growth_km2','additional_unmet_growth_km2']].sum(axis=1)-BUDGET.primary_realized_growth_km2).max())
 assert h<1e-10 and f<1e-10 and a<1e-8
 gross=read('Fig3_gross_contribution_balance_2025.csv',True).iloc[0]
 assert abs(float(gross.P-gross.N)-float(HISTFULL[(HISTFULL.region_id==0)&(HISTFULL.year==2025)].S.iloc[0]))<1e-10
 return {'historical_component_closure':h,'future_component_closure':f,'allocation_closure_km2':a,'ensemble_quantiles':'PASS','UMRI_definition':'UMRI=N(S), range (-1,1)','model_weights_and_calibration':'unchanged','temporal_smoothing':'none'}

HIST=read('Fig2_history_matched2025.csv');HISTFULL=read('Fig3_historical_contributions.csv');FUT=read('Fig2_future_growth_matched2025.csv');URBAN=read('Fig4_urban_paired_annual.csv');CUM=read('Fig4_urban_paired_cumulative.csv');CONTRIB=read('Fig3_urban_contributions_2100.csv');PROT=read('Fig5_protection_paired_annual.csv');BUDGET=read('Fig5_allocation_regional_annual.csv');GB=read('Fig5_allocation_overall_annual.csv')
CONTRIBSCORE=read('Fig3_urban_UMRI_contributions_2100.csv')
checks=verify()
checks['historical_change_closure_UMRI']=float(abs(HISTFULL[[k+'_change_UMRI' for k in COMP]].sum(axis=1)-HISTFULL.change_UMRI).max())
checks['paired_mean_closure_UMRI']=float(abs(CONTRIBSCORE[['delta_'+k+'_UMRI_mean' for k in COMP]].sum(axis=1)-CONTRIBSCORE.delta_UMRI_mean).max())
assert max(checks['historical_change_closure_UMRI'],checks['paired_mean_closure_UMRI'])<1e-10
wanted=ARGS.figures.split(',')
for number in wanted:
 if number == '2':figure2()
story=[n for n in wanted if n in ['3','4','5','6']]
if story:story_figures(story)
(QA/'refined_figure_data_checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf8')
(QA/'refined_figure_manifest.json').write_text(json.dumps({'scientific_identity':'UMRI-v1.2.0','layout_revision':'figure_story_reproduction_v1','inputs':INPUTS,'figures':AUDIT},indent=2,ensure_ascii=False)+'\n',encoding='utf8')

