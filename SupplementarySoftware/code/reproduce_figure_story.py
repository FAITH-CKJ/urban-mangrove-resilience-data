





from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,Rectangle,FancyArrowPatch
from matplotlib.colors import ListedColormap,BoundaryNorm,LogNorm,LinearSegmentedColormap
from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator

HERE=Path(__file__).resolve().parent
pa=argparse.ArgumentParser();pa.add_argument('--source-data',type=Path,default=HERE.parent.parent/'SourceData');pa.add_argument('--output',type=Path,default=HERE.parent/'reproduced');pa.add_argument('--figures',default='3,4,5,6');pa.add_argument('--draft',action='store_true');A=pa.parse_args()
S=A.source_data;OUT=A.output/'figures';Q=A.output/'qa';OUT.mkdir(parents=True,exist_ok=True);Q.mkdir(parents=True,exist_ok=True)
W=183.;INK='#24343B';MID='#67777C';GRID='#E5E9E8';TEAL='#297F70';CLAY='#BE6B53';GOLD='#A68B48';PURPLE='#80709D';BLUE='#4A88A8'
SC=['ssp126','ssp245','ssp370','ssp585'];LABELS=dict(zip(SC,['SSP1–2.6','SSP2–4.5','SSP3–7.0','SSP5–8.5']));COLORS=dict(zip(SC,['#3C7C9B','#B38B35','#9274A5','#B55755']));MARKERS=dict(zip(SC,['o','D','^','s']));STYLES=dict(zip(SC,['-',(0,(4,1.4)),(0,(1,1.1)),'-']))
REG={0:'Overall',1:'Australia / NZ',2:'Central America',3:'Eastern Africa',4:'Eastern Asia',5:'Northern America',6:'South America',7:'Southeastern Asia',8:'Southern Asia',9:'Western Africa',10:'Western Asia'}
ORDER=[5,2,6,9,3,10,8,7,4,1];ALL=[0]+ORDER
COMP=['d','r','g','h','T'];CC=['#BE6A52','#228D7C','#94A746','#5796B0','#8C74A6'];CNAME=['Urban pressure','Recovery','Growth','Health','Structure']
MAP_COL=['#ECEDED','#DCEAF1','#F0EBDF','#347C62','#CAB071','#ACCEB8','#929BA1','#C96C4F','#7865A5'];ALLOC=[MAP_COL[7],MAP_COL[8],'#D7DADB']
INPUTS={};AUDIT={}
mpl.rcParams.update({'font.family':'Arial','font.size':6.7,'axes.labelsize':6.7,'axes.titlesize':7,'xtick.labelsize':6.2,'ytick.labelsize':6.4,'legend.fontsize':6.3,'axes.linewidth':.55,'xtick.major.width':.5,'ytick.major.width':.5,'xtick.major.size':2.1,'ytick.major.size':2.1,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.edgecolor':MID,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','path.simplify':False,'hatch.linewidth':.35,'mathtext.fontset':'dejavusans'})
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(rel):
    p=S/rel;INPUTS[rel]=sha(p);return pd.read_csv(p,float_precision='round_trip')
def js(rel):
    p=S/rel;INPUTS[rel]=sha(p);return json.loads(p.read_text(encoding='utf8'))
def fig(h):return plt.figure(figsize=(W/25.4,h/25.4))
def ax(f,x,y,w,h,**kw):return f.add_axes([x/W,y/(f.get_figheight()*25.4),w/W,h/(f.get_figheight()*25.4)],**kw)
def txt(f,x,y,s,**kw):return f.text(x/W,y/(f.get_figheight()*25.4),s,**kw)
def panel(f,x,y,letter,title):
    txt(f,x,y,letter,fontsize=8,fontweight='bold',va='baseline');txt(f,x+4.1,y,title,fontsize=7,va='baseline')
def clean(a,axis='y'):
    a.set_axisbelow(True)
    if axis:a.grid(axis=axis,color=GRID,lw=.4)
def region_axis(a,order=ALL,labels=True):
    a.set_ylim(len(order)-.5,-.5);a.set_yticks(range(len(order)));a.set_yticklabels([REG[r] for r in order] if labels else []);a.tick_params(axis='y',length=0,pad=3);a.spines['left'].set_visible(False)
    for i in range(len(order)):
        if i==0 and order[0]==0:a.axhspan(-.5,.5,color='#EDF2F0',zorder=0)
        elif i%2==0:a.axhspan(i-.5,i+.5,color='#F7F8F7',zorder=0)
    if order[0]==0 and labels:a.get_yticklabels()[0].set_fontweight('bold')
def signed_asinh(a,scale,ticks,limits):
    a.set_xscale('function',functions=(lambda x:np.arcsinh(np.asarray(x)/scale),lambda x:np.sinh(np.asarray(x))*scale));a.set_xlim(*limits);a.xaxis.set_major_locator(FixedLocator(ticks));a.xaxis.set_minor_locator(NullLocator());a.xaxis.set_major_formatter(FuncFormatter(lambda x,p:('0' if x==0 else f'{x:g}').replace('-','−')))
def scenario_key(f,y,scenarios=SC,x=.55):
    handles=[Line2D([0],[0],color=COLORS[s],marker=MARKERS[s],ms=3,lw=.9,ls=STYLES[s],label=LABELS[s]) for s in scenarios]
    f.legend(handles=handles,loc='center',bbox_to_anchor=(x,y/(f.get_figheight()*25.4)),ncol=len(scenarios),handlelength=1.9,columnspacing=1.7,handletextpad=.55)
def interval(a,mid,lo,hi,y,s,ms=2.7):
    a.errorbar(mid,y,xerr=[np.maximum(0,np.asarray(mid)-lo),np.maximum(0,np.asarray(hi)-mid)],fmt=MARKERS[s],ms=ms,color=COLORS[s],mec='white',mew=.25,lw=.65,capsize=1.1,zorder=4)
def save(f,stem,extra=None):
    f.canvas.draw();renderer=f.canvas.get_renderer();bad=[];texts=[]
    for t in f.findobj(mpl.text.Text):
        if t.get_visible() and t.get_text():
            b=t.get_window_extent(renderer)
            if b.x0<-.7 or b.y0<-.7 or b.x1>f.bbox.x1+.7 or b.y1>f.bbox.y1+.7:bad.append(t.get_text())
            texts.append({'text':t.get_text(),'font_pt':t.get_fontsize(),'bbox_mm':[v/f.dpi*25.4 for v in [b.x0,b.y0,b.x1,b.y1]]})
    exports=[]
    for ext,dpi in [('pdf',None),('svg',None),('png',300)]+([] if A.draft else [('600dpi.png',600),('tiff',600)]):
        p=OUT/(stem+('_600dpi.png' if ext=='600dpi.png' else '.'+ext));kw={'dpi':600 if ext in ('pdf','svg') else dpi}
        if ext=='tiff':kw['pil_kwargs']={'compression':'tiff_lzw'}
        f.savefig(p,**kw)
        if ext=='tiff' and stem in ['Fig4_urban_effect','Fig5_protection_benefit','Fig6_habitat_compensation','SupplementaryFig5_regional_protection']:
            from PIL import Image
            with Image.open(p) as im:
                assert im.mode=='RGBA' and im.getchannel('A').getextrema()==(255,255)
                rgb=im.convert('RGB')
            rgb.save(p,compression='tiff_lzw',dpi=(600,600))
        exports.append({'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size})
    AUDIT[stem]={'width_mm':W,'height_mm':round(f.get_figheight()*25.4,4),'text_outside_canvas':bad,'status':'RENDERED_REQUIRES_VISUAL_REVIEW','exports':exports,'text':texts,'semantics':extra or {}}
    plt.close(f);print('EXPORTED',stem,'overflow',bad,flush=True)

def trim_bottom(f,mm):
    
    old_h=f.get_figheight()*25.4;new_h=old_h-mm
    apos=[(a,a.get_position().bounds) for a in f.axes]
    tpos=[(t,t.get_position()) for t in f.texts]
    lpos=[(l,l.get_bbox_to_anchor().transformed(f.transFigure.inverted()).bounds) for l in f.legends]
    ppos=[(p,p.get_xy(),p.get_height()) for p in f.patches if isinstance(p,Rectangle)]
    f.set_size_inches(W/25.4,new_h/25.4)
    for a,(x,y,w,h) in apos:a.set_position([x,(y*old_h-mm)/new_h,w,h*old_h/new_h])
    for t,(x,y) in tpos:t.set_position((x,(y*old_h-mm)/new_h))
    for l,(x,y,w,h) in lpos:l.set_bbox_to_anchor((x,(y*old_h-mm)/new_h,w,h*old_h/new_h),transform=f.transFigure)
    for p,(x,y),h in ppos:p.set_xy((x,(y*old_h-mm)/new_h));p.set_height(h*old_h/new_h)




def bio_icon(f,kind,x,y,w=4.4,h=4.4,color='#297F70'):
    
    from matplotlib.path import Path as MP
    from matplotlib.patches import PathPatch,Ellipse
    from matplotlib.transforms import Affine2D
    tr=Affine2D().scale(w/W,h/(f.get_figheight()*25.4)).translate((x-w/2)/W,(y-h/2)/(f.get_figheight()*25.4))+f.transFigure
    def path(v,c,fc='none',ec=None,lw=.55,z=8):
        p=PathPatch(MP(v,c),fc=fc,ec=ec,lw=lw,transform=tr,zorder=z,capstyle='round',joinstyle='round');f.add_artist(p)
    def line(v,co=color,lw=.55):
        f.add_artist(Line2D([p[0] for p in v],[p[1] for p in v],color=co,lw=lw,transform=tr,zorder=9,solid_capstyle='round'))
    def leaf(x0,y0,x1,y1,width=.25,co=color):
        dx=x1-x0;dy=y1-y0;le=(dx*dx+dy*dy)**.5;nx=-dy/le*width;ny=dx/le*width
        v=[(x0,y0),(x0+dx*.35+nx,y0+dy*.35+ny),(x1+nx*.7,y1+ny*.7),(x1,y1),(x1-nx*.7,y1-ny*.7),(x0+dx*.35-nx,y0+dy*.35-ny),(x0,y0)]
        path(v,[MP.MOVETO]+[MP.CURVE4]*6,fc=co,ec='none')
        line([(x0,y0),(x1,y1)],'#F2F4E8',.3)
    if kind=='recovery':
        line([(.14,.11),(.86,.11)],'#9A8A6A',.5)
        path([(.48,.12),(.47,.35),(.54,.50),(.51,.80)],[MP.MOVETO,MP.CURVE4,MP.CURVE4,MP.CURVE4],ec=color,lw=.8)
        leaf(.51,.48,.13,.71,.16);leaf(.52,.64,.88,.89,.16)
        line([(.48,.13),(.33,.04)],'#9A8A6A',.4);line([(.48,.13),(.64,.05)],'#9A8A6A',.4)
    elif kind=='leaf':
        leaf(.14,.12,.86,.9,.28)
        for a,b in [((.40,.40),(.30,.60)),((.55,.56),(.49,.80)),((.56,.57),(.76,.51))]:line([a,b],'#D4DEB8',.25)
    elif kind=='moisture':
        path([(.5,.96),(.39,.72),(.13,.42),(.18,.23),(.23,-.02),(.76,-.02),(.82,.23),(.87,.42),(.61,.72),(.5,.96)],[MP.MOVETO]+[MP.CURVE4]*9,fc=color,ec='none')
        path([(.31,.23),(.25,.37),(.35,.52),(.39,.58)],[MP.MOVETO]+[MP.CURVE4]*3,ec='#E5F0F1',lw=.55)
    elif kind=='resilience':
        path([(.03,.7),(.22,.7),(.26,.20),(.42,.22),(.62,.23),(.68,.77),(.97,.77)],[MP.MOVETO]+[MP.CURVE4]*6,ec=color,lw=.8)
        f.add_artist(Ellipse((.43,.23),.14,.14,fc=color,ec='none',transform=tr,zorder=9))
    elif kind in ['retention','connection']:
        if kind=='connection':line([(.27,.34),(.73,.34)],'#B29C5E',1.5)
        for cx in ([.50] if kind=='retention' else [.26,.75]):
            sc=.95 if kind=='retention' else .57
            line([(cx,.2),(cx,.63)],'#796C51',.65)
            for dx in [-.18,.18]:line([(cx,.4),(cx+dx*sc,.14)],'#796C51',.45)
            for dx,dy,r in [(-.15,.62,.32),(.08,.72,.37),(.22,.60,.32)]:
                f.add_artist(Ellipse((cx+dx*sc,dy),r*sc,r*.9,fc=color,ec='none',transform=tr,zorder=9))
        line([(.06,.1),(.94,.1)],'#B6B49B',.4)

def mangrove_scene(a,x,y,w,h,linked=False):
    



    from matplotlib.path import Path as MP
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D
    tr=Affine2D().scale(w,h).translate(x,y)+a.transAxes
    records=[]
    def bez(v,fc='none',ec='none',lw=.4,z=8,closed=False,role='retained'):
        v=list(v);codes=[MP.MOVETO]+[MP.CURVE4]*(len(v)-1)
        if closed:v=v+[v[0]];codes=codes+[MP.CLOSEPOLY]
        a.add_patch(PathPatch(MP(v,codes),transform=tr,fc=fc,ec=ec,lw=lw,zorder=z,capstyle='round',joinstyle='round',clip_on=False))
        if role!='link':records.append({'vertices':v,'face':fc,'edge':ec,'role':role})
    def blob(points,colour,z=8,role='retained'):
        v=[points[0]];n=len(points)
        for i in range(n):
            p0=np.array(points[(i-1)%n]);p1=np.array(points[i]);p2=np.array(points[(i+1)%n]);p3=np.array(points[(i+2)%n])
            v.extend([tuple(p1+(p2-p0)/6),tuple(p2-(p3-p1)/6),tuple(p2)])
        bez(v,fc=colour,z=z,closed=True,role=role)
    blob([(.01,.13),(.18,.17),(.48,.15),(.77,.18),(.99,.13),(.94,.05),(.68,.02),(.30,.03),(.04,.07)],'#EAF1EF',5,'substrate')
    blob([(.04,.20),(.18,.29),(.40,.26),(.53,.24),(.70,.28),(.96,.24),(.99,.16),(.80,.10),(.58,.11),(.40,.09),(.16,.11)],'#DFD4B9',6,'substrate')
    bez([(.01,.08),(.20,.03),(.33,.07),(.43,.04)],ec='#AFCCCA',lw=.35,z=7,role='substrate')
    bez([(.57,.06),(.74,.02),(.85,.06),(.99,.04)],ec='#AFCCCA',lw=.35,z=7,role='substrate')
    blob([(.04,.23),(.12,.31),(.26,.31),(.37,.25),(.35,.17),(.24,.14),(.09,.16)],'#A4AD86',7)
    blob([(.59,.25),(.69,.33),(.88,.34),(.99,.27),(.97,.17),(.80,.14),(.64,.17)],'#A4AD86',7)
    def tree(cx,base,tw,th,link=False,z=10):
        role='link' if link else 'retained';trunk='#8F7538' if link else '#68573E'
        dark,mid,light=('#9D7D32','#B49A46','#C4B26A') if link else ('#276551','#3D805C','#6E9D6C')
        def pts(seq):return [(cx+tw*px,base+th*py) for px,py in seq]
        bez(pts([(0,.12),(-.015,.31),(.03,.48),(0,.68)]),ec=trunk,lw=.72 if not link else .50,z=z,role=role)
        for seq in [[(0,.41),(-.10,.34),(-.23,.16),(-.30,.01)],[(.01,.37),(.14,.28),(.28,.11),(.34,.00)],[(0,.26),(-.02,.18),(-.08,.08),(-.13,.00)],[(.02,.47),(.16,.50),(.23,.61),(.27,.66)],[(-.01,.47),(-.18,.48),(-.24,.57),(-.28,.64)]]:
            bez(pts(seq),ec=trunk,lw=.54 if not link else .40,z=z,role=role)
        crown=[(-.49,.65),(-.40,.80),(-.24,.83),(-.13,.94),(.04,.92),(.17,1.0),(.36,.91),(.40,.80),(.51,.71),(.41,.59),(.19,.57),(.02,.60),(-.17,.55),(-.36,.59)]
        blob(pts(crown),dark,z+1,role)
        blob(pts([(-.43,.70),(-.32,.82),(-.14,.84),(-.05,.93),(.13,.91),(.22,.95),(.36,.87),(.37,.77),(.22,.71),(.05,.72),(-.09,.67),(-.28,.68)]),mid,z+2,role)
        blob(pts([(-.30,.79),(-.16,.85),(-.03,.87),(.02,.92),(.17,.91),(.21,.84),(.11,.78),(-.03,.79),(-.15,.76)]),light,z+3,role)
    if linked:
        blob([(.34,.23),(.42,.30),(.56,.30),(.66,.26),(.67,.18),(.54,.17),(.41,.18)],'#B9A65C',8,'link')
        for cx,base,tw,th in [(.39,.19,.105,.26),(.46,.19,.11,.31),(.53,.19,.11,.27),(.60,.20,.10,.29)]:tree(cx,base,tw,th,link=True,z=8)
    for par in [(.17,.22,.16,.54),(.71,.24,.20,.64),(.86,.24,.23,.64),(.11,.17,.20,.55),(.25,.18,.20,.60),(.77,.17,.28,.77)]:tree(*par,z=11)
    digest=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    return {'retained_and_substrate_geometry_sha256':digest,'physical_axes_scale':[w,h],'retained_paths':len(records),'new_vegetated_link':linked,'historical_m_unchanged':True,'tree_counts_are_not_data':True}




def f3_glyph(f, a, x, y, kind, color, size=4.6):
    
    xy=f.transFigure.inverted().transform(a.transData.transform((x,y)))
    xx,yy=xy[0]*W,xy[1]*f.get_figheight()*25.4
    if kind!='pressure':
        bio_icon(f,kind,xx,yy,w=size,h=size,color=color)
        return
    from matplotlib.transforms import Affine2D
    tr=Affine2D().scale(size/W,size/(f.get_figheight()*25.4)).translate((xx-size/2)/W,(yy-size/2)/(f.get_figheight()*25.4))+f.transFigure
    for bx,by,bw,bh in [(.04,.10,.24,.45),(.34,.10,.27,.82),(.67,.10,.27,.61)]:
        f.add_artist(Rectangle((bx,by),bw,bh,facecolor=color,edgecolor='none',transform=tr,zorder=10))
        for wx in [bx+bw*.27,bx+bw*.68]:
            for wy in np.arange(by+.15,by+bh-.06,.17):
                f.add_artist(Rectangle((wx,wy),.035,.065,facecolor='white',edgecolor='none',transform=tr,zorder=11))
    f.add_artist(Line2D([0,1],[.08,.08],color=color,lw=.5,transform=tr,zorder=10))


def f3_rows(a, names=None):
    a.set_ylim(10.65,-.65)
    a.set_yticks(np.arange(11));a.set_yticklabels(names if names is not None else [])
    a.tick_params(axis='y',length=0,pad=4,labelsize=6.0)
    a.tick_params(axis='x',labelsize=5.8,pad=2)
    a.spines['left'].set_visible(False)
    a.axhspan(-.6,.52,color='#F0F3F1',zorder=0)
    for y in [3.5,5.5,9.5]:a.axhline(y,color='#E8ECE9',lw=.4,zorder=0)
    a.xaxis.set_major_formatter(FuncFormatter(lambda v,p:('0' if v==0 else f'{v:g}').replace('-','−')))
    if names:a.get_yticklabels()[0].set_fontweight('bold')


def f3_swarm(values, width, limits, max_offset=.18, row_mm=4.65, min_sep=.45):
    
    values=np.asarray(values,float);out=np.zeros(len(values));placed=[]
    slots=sorted(np.linspace(-max_offset,max_offset,17),key=lambda v:(abs(v),v))
    for i in np.argsort(values,kind='stable'):
        scores=[]
        for dy in slots:
            distance=min((((values[i]-values[j])*width/(limits[1]-limits[0]))**2+
                          ((dy-y)*row_mm)**2 for j,y in placed),default=100.)
            scores.append(distance)
        good=[k for k,v in enumerate(scores) if v>min_sep**2]
        chosen=good[0] if good else int(np.argmax(scores))
        out[i]=slots[chosen];placed.append((i,out[i]))
    return out


def figure3():
    f=fig(140)
    panel(f,18,133.2,'a','')
    allh=read('main_figures/Fig3_historical_contributions.csv')
    h=allh[allh.region_id==0].sort_values('year');end=h.iloc[-1]
    region_end=allh[(allh.year==2025)&(allh.region_id>0)].set_index('region_id').loc[ORDER]
    a=ax(f,22,89,74,42)
    positive=np.zeros(len(h));negative=np.zeros(len(h))
    for k,col in zip(COMP,CC):
        v=h[k+'_change_UMRI'].to_numpy();vp=np.maximum(v,0);vn=np.minimum(v,0)
        a.fill_between(h.year,positive,positive+vp,color=col,lw=0,antialiased=True)
        a.fill_between(h.year,negative,negative+vn,color=col,lw=0,antialiased=True)
        positive+=vp;negative+=vn
    np.testing.assert_allclose(positive+negative,h.change_UMRI,atol=2e-12)
    a.plot(h.year,h.change_UMRI,color=INK,lw=1.05,zorder=5,
           solid_capstyle='round',solid_joinstyle='round')
    a.axhline(0,color=MID,lw=.45)
    a.set(xlim=(2000,2025),ylim=(-.31,.06),xticks=[2000,2010,2020,2025],
          yticks=[-.3,-.2,-.1,0],ylabel='ΔUMRI')
    clean(a)
    a.annotate(f'{end.change_UMRI:+.4f}'.replace('-','−'),
               (2025,end.change_UMRI),xytext=(-2,5),textcoords='offset points',
               ha='right',fontsize=7.0,fontweight='bold')
    symbols=['$d$','$r^*$','$g^*$','$h$','$T$']
    kinds=['pressure','recovery','leaf','moisture','connection']
    
    for x,co,symbol in zip([22,30,38,46,54],CC,symbols):
        f.add_artist(Line2D([x/W,(x+2.6)/W],[134/140]*2,color=co,lw=2.1,
                            solid_capstyle='round',transform=f.transFigure,zorder=12))
        txt(f,x+3.5,134,symbol,fontsize=6.1,va='center',color=co)
    f.add_artist(Line2D([66/W,69/W],[134/140]*2,color=INK,lw=1.05,
                        solid_capstyle='round',transform=f.transFigure,zorder=12))
    txt(f,70.5,134,'Net',fontsize=5.4,va='center',color=INK)
    
    mini=ax(f,27.6,95,29,17)
    mini.set_facecolor('white')
    for rid in ORDER:
        z=allh[allh.region_id==rid].sort_values('year')
        assert z.year.tolist()==list(range(2000,2026))
        mini.plot(z.year,z.change_UMRI,color='#87989A',lw=.48,alpha=.68,
                  solid_capstyle='round',solid_joinstyle='round')
    mini.plot(h.year,h.change_UMRI,color=INK,lw=.85,
              solid_capstyle='round',solid_joinstyle='round',zorder=4)
    mini.axhline(0,color='#C3CDCA',lw=.4,zorder=0)
    mini.set(xlim=(2000,2025),ylim=(-.55,.55),xticks=[2000,2025],yticks=[-.4,0,.4])
    mini.tick_params(labelsize=5.2,length=1.4,pad=1)
    for sp in mini.spines.values():sp.set_linewidth(.4);sp.set_color('#AAB6B1')
    txt(f,27.6,114.1,'10 regions',fontsize=5.2,color=MID)
    
    led=ax(f,106,89,72,42);limits=(-.36,.28)
    led.set(xlim=limits,ylim=(4.65,-.65),xticks=[-.3,-.15,0,.15],yticks=[])
    led.spines['left'].set_visible(False);led.axvline(0,color='#B3BCB7',lw=.55,zorder=0)
    led.tick_params(axis='x',labelsize=5.8,pad=2)
    led.set_xlabel('ΔUMRI',fontsize=6.1,labelpad=2)
    txt(f,106,134,'2025',fontsize=6.0,color=MID)
    bio_icon(f,'leaf',126,134.7,w=2.9,h=2.9,color=MID)
    txt(f,128.6,134,'Overall',fontsize=5.6,color=MID)
    bio_icon(f,'leaf',151,134.7,w=1.35,h=1.35,color=MID)
    txt(f,153,134,'10 regions',fontsize=5.6,color=MID)
    component_labels=['Urban\npressure','Recovery','Growth','Canopy\nmoisture','Habitat\nstructure']
    vals=[];glyph_positions=[]
    f.canvas.draw()
    for j,(k,col,symbol,kind) in enumerate(zip(COMP,CC,symbols,kinds)):
        v=float(end[k+'_change_UMRI']);vals.append(v)
        led.plot([0,v],[j-.22,j-.22],color=col,lw=1.05,solid_capstyle='round',zorder=2)
        f3_glyph(f,led,v,j-.22,kind,col,4.4)
        yy=89+42*(4.65-(j-.22))/5.3
        txt(f,101,yy,component_labels[j],fontsize=5.3,color=col,ha='center',va='center',linespacing=1.05)
        regional=region_end[k+'_change_UMRI'].to_numpy()
        ys=j+.31+f3_swarm(regional,72,limits,.12,42/5.3,.97)
        for rid,value,y in zip(ORDER,regional,ys):
            f3_glyph(f,led,float(value),float(y),kind,col,1.45)
            glyph_positions.append({'component':k,'region_id':rid,'x':float(value),'y':float(y)})
        assert regional.min()>limits[0] and regional.max()<limits[1]
    np.testing.assert_allclose(sum(vals),end.change_UMRI,atol=2e-12)
    
    panel(f,19.5,73.5,'b','')
    panel(f,103.5,73.5,'c','')
    
    b_key=f.legend(handles=[
        Line2D([],[],marker='o',ms=3.7,mfc='white',mec=MID,mew=.7,lw=0,label='2000'),
        Line2D([],[],marker='o',ms=3.1,mfc=MID,mec='white',mew=.3,lw=0,label='2025'),
        Line2D([],[],marker='o',ms=2.1,mfc='#B8C1BF',mec='white',mew=.2,lw=0,label='2025 blocks')],
        loc='center',bbox_to_anchor=(59/W,79.7/140),ncol=3,fontsize=5.6,
        labelcolor=MID,handlelength=.8,handletextpad=.45,columnspacing=1.3,
        borderpad=0,borderaxespad=0)
    c_key_artist_start=len(f.artists);c_key_text_start=len(f.texts)
    
    bio_icon(f,'leaf',109,79.7,w=2.3,h=2.3,color=MID)
    txt(f,111,79,'Mean',fontsize=5.4,color=MID)
    bio_icon(f,'connection',126,79.7,w=2.7,h=2.7,color=CC[4])
    txt(f,129,79.7,'Weaker habitat\nstructure',fontsize=5.4,va='center',color=INK,linespacing=1.05)
    f.add_artist(Line2D([150/W],[79.7/140],marker='s',ms=2.7,mfc=INK,mec='white',mew=.25,lw=0,transform=f.transFigure))
    txt(f,153,79.7,'Habitat recovery\nrequired',fontsize=5.4,va='center',color=INK,linespacing=1.05)
    
    from matplotlib.transforms import Affine2D,Bbox
    c_key_artists=f.artists[c_key_artist_start:];c_key_texts=f.texts[c_key_text_start:]
    f.canvas.draw();renderer=f.canvas.get_renderer()
    key_box=Bbox.union([item.get_window_extent(renderer) for item in [*c_key_artists,*c_key_texts]])
    key_dx=(142-(key_box.x0+key_box.x1)/2/f.dpi*25.4)/W
    for item in c_key_artists:
        base=(item.get_data_transform() if isinstance(item,mpl.patches.Patch) else item.get_transform())-f.transFigure
        item.set_transform(base+Affine2D().translate(key_dx,0)+f.transFigure)
    for item in c_key_texts:
        xx,yy=item.get_position();item.set_position((xx+key_dx,yy))
    d=read('figure_story/historical_native_endpoints.csv')
    d=d[d.support=='available'].set_index('region_id').loc[ALL]
    co=read('figure_story/historical_endpoint_comparison.csv')
    co=co[co.support=='available'].set_index('region_id').loc[ALL]
    block=read('figure_story/historical_native_block_sums.csv')
    block=block[block.support=='available']
    quant=read('figure_story/historical_centre_distribution_2025.csv').set_index(['region_id','variable'])
    limit=read('figure_story/retention_reconnection_limit.csv')
    limit=limit[(limit.support=='available')&(limit.group=='greener')].set_index('region_id').loc[ALL]
    np.testing.assert_array_equal(limit.n.to_numpy(),co.n_greening.to_numpy())
    names=['Overall','Northern America','Central America','South America','Western Africa','Eastern Africa',
           'Western Asia','Southern Asia','Southeastern Asia','Eastern Asia','Australia and\nNew Zealand']
    configs=[(22,15.5,'m','$m$ (%)',TEAL,[0,50,100],(-4,104)),
             (41.5,15.5,'C','$C$ (%)',CC[4],[0,50,100],(-4,104)),
             (61,15.5,'q','Δ$q$ (pp)',CC[4],[-10,0,10],(-16,16)),
             (80.5,15.5,'p','$p$ (%)',CC[0],[0,50,100],(-4,104))]
    for j,(x,w,key,title,col,ticks,xlim) in enumerate(configs):
        b=ax(f,x,11.5,w,58.5);f3_rows(b,names if j==0 else None);b.set_xlim(*xlim);b.set_xticks(ticks)
        b.tick_params(axis='x',labelsize=5.3)
        if j==0:b.tick_params(axis='y',labelsize=5.7,pad=3)
        if key=='q':
            b.axvline(0,color='#AAB6B1',lw=.55,zorder=1)
            for i,rid in enumerate(ALL):
                value=float(co.loc[rid,'delta_q_2000_pp'])
                assert xlim[0]<=value<=xlim[1]
                b.plot([0,value],[i,i],color=col,lw=.9,zorder=2,solid_capstyle='round')
                b.plot(value,i,'D',mfc=col,mec='white',mew=.3,ms=3.3,zorder=4)
            txt(f,x+w/2,73.5,title,fontsize=6.1,ha='center',linespacing=1.08)
            continue
        for i,rid in enumerate(ALL):
            if rid:
                bb=block[(block.region_id==rid)&(block.n_2000_2025>0)].sort_values('block_id')
                vv=100*bb[key+'_2025_sum']/bb.n_2000_2025
                offset=f3_swarm(vv,w,xlim,.26)
                b.scatter(vv,i+offset,s=4.2,color='#B8C1BF',edgecolor='white',linewidth=.14,alpha=.8,zorder=1)
            v0=float(d.loc[rid,key+'_2000_pct']);v1=float(d.loc[rid,key+'_2025_pct'])
            assert xlim[0]<=v0<=xlim[1] and xlim[0]<=v1<=xlim[1]
            b.plot([v0,v1],[i,i],color=col,lw=1.05,zorder=2,alpha=.9,solid_capstyle='round')
            b.plot(v0,i,'o',mfc='white',mec=col,mew=.8,ms=4.0,zorder=4,clip_on=False)
            b.plot(v1,i,'o',mfc=col,mec='white',mew=.3,ms=3.2,zorder=5,clip_on=False)
        txt(f,x+w/2,73.5,title,fontsize=6.1,ha='center',linespacing=1.18)
    cconfigs=[
        (106,15,'r_2025_pct','$r$ (%)',CC[1],(-.65,15.65),[0,5,10,15],'recovery'),
        (125,15,'EVI_delta','ΔEVI',CC[2],(-.115,.24),[-.1,0,.1,.2],'leaf'),
        (144,15,'NDMI_delta','ΔNDMI',CC[3],(-.18,.28),[-.1,0,.2],'moisture'),
        (163,15,'capacity_decline_among_greening_pct','Share (%)',CC[4],(-4,104),[0,50,100],'connection')]
    for j,(x,w,key,title,col,xlim,ticks,kind) in enumerate(cconfigs):
        c=ax(f,x,11.5,w,58.5);f3_rows(c);c.set_xlim(*xlim);c.set_xticks(ticks)
        c.axvline(0,color='#BBC3BF',lw=.5,zorder=1)
        c.tick_params(axis='x',labelsize=5.5)
        txt(f,x+w/2,73.5,title,fontsize=6.1,ha='center',linespacing=1.12)
        f.canvas.draw();values=co[key].to_numpy() if j==3 else d[key].to_numpy()
        for i,(rid,v) in enumerate(zip(ALL,values)):
            mean_y=i-.17
            c.plot([0,v],[mean_y,mean_y],color=col,lw=.6,alpha=.25,zorder=1)
            f3_glyph(f,c,float(v),mean_y,kind,col,(3.1 if i else 3.7) if j==3 else (2.6 if i else 3.0))
            if j==3:
                limited=100*float(limit.loc[rid,'fraction_m_below_C_baseline'])
                assert 0<=limited<=float(v)+1e-10
                c.plot([0,limited],[i+.29,i+.29],color=INK,lw=.65,alpha=.8,zorder=3)
                c.plot(limited,i+.29,'s',mfc=INK,mec='white',mew=.25,ms=2.5,zorder=4)
                if i==0:
                    c.annotate(f'{float(v):.1f}%',(float(v),mean_y),xytext=(-3,1),textcoords='offset points',ha='right',va='center',fontsize=5.2,color=col,bbox={'fc':'#EDF2F0','ec':'none','pad':.1})
                    c.annotate(f'{limited:.1f}%',(limited,i+.29),xytext=(-3,0),textcoords='offset points',ha='right',va='center',fontsize=5.2,color=INK,bbox={'fc':'#EDF2F0','ec':'none','pad':.1})
            if j<3:
                q=quant.loc[(rid,key)];yy=i+.29
                assert xlim[0]<=q.q10<=q.q25<=q.q50<=q.q75<=q.q90<=xlim[1]
                np.testing.assert_allclose(q['mean'],v,atol=2e-11,rtol=0)
                c.plot([q.q10,q.q90],[yy,yy],color=col,lw=.55,alpha=.85,zorder=3)
                c.plot([q.q25,q.q75],[yy,yy],color=col,lw=2.5,alpha=.65,
                       solid_capstyle='butt',zorder=4)
                c.plot([q.q50,q.q50],[yy-.09,yy+.09],color='white',lw=.6,zorder=5)
        assert values.min()>=xlim[0] and values.max()<=xlim[1]
    
    
    from matplotlib.transforms import Affine2D
    icon_transforms=[(artist,(artist.get_data_transform() if isinstance(artist,mpl.patches.Patch)
                              else artist.get_transform())-f.transFigure) for artist in f.artists]
    trim_bottom(f,4)
    
    horizontal_shift=-.4
    for aa in f.axes:
        bb=aa.get_position();aa.set_position([bb.x0+horizontal_shift/W,bb.y0,bb.width,bb.height])
    for tt in f.texts:
        xx,yy=tt.get_position();tt.set_position((xx+horizontal_shift/W,yy))
    for legend in f.legends:
        xx,yy,ww,hh=legend.get_bbox_to_anchor().transformed(f.transFigure.inverted()).bounds
        legend.set_bbox_to_anchor((xx+horizontal_shift/W,yy,ww,hh),transform=f.transFigure)
    shift=Affine2D().scale(1,140/136).translate(horizontal_shift/W,-4/136)
    for artist,base_transform in icon_transforms:
        artist.set_transform(base_transform+shift+f.transFigure)
    f.canvas.draw()
    renderer=f.canvas.get_renderer()
    b_key_box=b_key.get_window_extent(renderer)
    c_key_box=Bbox.union([item.get_window_extent(renderer) for item in [*c_key_artists,*c_key_texts]])
    measured_key_centres=[(bb.x0+bb.x1)/2/f.dpi*25.4 for bb in [b_key_box,c_key_box]]
    np.testing.assert_allclose(measured_key_centres,[58.6,141.6],atol=.03)
    actual=[]
    for aa in f.axes:
        bb=aa.get_position();actual.append([bb.x0*W,bb.y0*136,bb.width*W,bb.height*136])
    np.testing.assert_allclose([actual[0][0],sum(actual[0][::2])],[actual[3][0],actual[6][0]+actual[6][2]],atol=1e-10)
    np.testing.assert_allclose([actual[2][0],sum(actual[2][::2])],[actual[7][0],actual[10][0]+actual[10][2]],atol=1e-10)
    np.testing.assert_allclose(np.array(actual[3:])[:,[1,3]],np.tile([7.5,58.5],(8,1)),atol=1e-10)
    save(f,'Fig3_component_contributions',{
        'actual_axes_mm':actual,
        'measured_key_centres_mm':measured_key_centres,
        'presentation_revision':'figure3_component_layout_v1',
        'scope':'Figure 3 only',
        'retention_reconnection_limit':{'source':'figure_story/retention_reconnection_limit.csv','group':'greener','support':'available','markers':'Upper connection glyph: C<C0; lower square: m<C0; both use all greener neighbourhoods as denominator','overall_n':int(limit.loc[0,'n']),'overall_limited_n':int(limit.loc[0,'n_m_below_C_baseline']),'overall_limited_percent':100*float(limit.loc[0,'fraction_m_below_C_baseline'])},
        'historical_attribution_support':'Annually valid centres; overall and each region mapped separately',
        'regional_trajectory_count':10,'regional_trajectory_years':[2000,2025],
        'regional_component_points':glyph_positions,
        'native_structural_support':'925583 valid at both endpoints',
        'patch_concentration_change':{
            'source':'figure_story/historical_endpoint_comparison.csv',
            'field':'delta_q_2000_pp','unit':'percentage points',
            'support':'Available; positive retention and valid q at both 2000 and 2025 endpoints',
            'overall_n':int(co.loc[0,'n_persistent_2000_2025']),
            'regional_values':{str(rid):float(co.loc[rid,'delta_q_2000_pp']) for rid in ALL},
            'regional_n':{str(rid):int(co.loc[rid,'n_persistent_2000_2025']) for rid in ALL},
            'encoding':'Diamond at mean paired endpoint change, line to zero; descriptive, no interval or block distribution'},
        'recovery_support':'925801 valid 2025 observations, includes known absence as zero',
        'native_canopy_support':'922143 valid forest-present neighbourhood observations',
        'region_order':ALL,
        'panel_b_background':'m, C and p columns only: 141 spatial processing block means in 2025, paired support; fixed-size grey points with display-only vertical offsets. Region and overall means pool reference centres.',
        'native_intervals':'10th–90th percentile thin lines; 25th–75th thick lines; white median tick; descriptive centre distributions, not confidence intervals. Full ranges retained in Source Data.',
        'glyph_encoding':'Original vector glyph x coordinates encode means; large overall and small regional silhouettes in a; all glyph sizes fixed within their layer.',
        'coexistence':{
            'encoding':'Fourth column of c, aligned with the same eleven regions',
            'quantity':'100 * greening_capacity_decline_n / greening_n',
            'overall_denominator':713029,'overall_numerator':532032,
            'overall_percent':float(co.loc[0,'capacity_decline_among_greening_pct']),
            'regional_percent':{str(i):float(co.loc[i,'capacity_decline_among_greening_pct']) for i in ALL},
            'baseline':'Local 2000-2004','classification':'Descriptive; no significance test'},
        'layout':{'panel_a_axes_mm':[actual[0],actual[2]],
                  'regional_inset_mm':actual[1],'region_body_mm':[7.5,58.5],
                  'panel_b_column_x_mm':[aa[0] for aa in actual[3:7]],
                  'panel_b_column_width_mm':15.5,
                  'panel_c_column_x_mm':[aa[0] for aa in actual[7:]],
                  'panel_c_column_width_mm':15,'panel_c_column_gap_mm':4,
                  'panel_c_mean_offset_rows':-.17,'panel_c_interval_offset_rows':.29,
                  'keys_shared_midline_mm':75.7,'footer_legend_removed':True,'panel_letter_x_mm':[18+horizontal_shift,19.5+horizontal_shift,103.5+horizontal_shift],'lower_panel_letter_baseline_mm':69.5,'key_centres_mm':[59+horizontal_shift,142+horizontal_shift],'c_legend':['Mean','Weaker habitat structure','Habitat recovery required'],'c_legend_midline_mm':75.7,'panel_titles':False,'full_region_labels':names,'metric_headers':['m (%)','C (%)','Δq (pp)','p (%)','r (%)','ΔEVI','ΔNDMI','Share (%)'],'aligned_data_edges_mm':[[actual[0][0],actual[0][0]+actual[0][2]],[actual[2][0],actual[2][0]+actual[2][2]]],'horizontal_centering_shift_mm':horizontal_shift}})



F4_LAYOUT = 'asinh'

def f4_heading(f, a, letter, title):
    
    x,y,w,h=a.get_position().bounds
    H=f.get_figheight()*25.4
    left,top=x*W,(y+h)*H
    baseline=top+2.2
    label=txt(f,left,baseline,letter,fontsize=8,fontweight='bold',va='baseline')
    heading=txt(f,left+4.1,baseline,title,fontsize=7,va='baseline')
    return {'panel':letter,'title':title,'axes_bbox_mm':[left,y*H,w*W,h*H],
            'title_anchor_mm':[left+4.1,baseline],'letter_anchor_mm':[left,baseline],
            '_title':heading,'_label':label}

def f4_rows(a, labels=False):
    centres = np.r_[5.4, 15 + 4.6*np.arange(10)]
    names = ['Overall','N. America','C. America','S. America','W. Africa',
             'E. Africa','W. Asia','S. Asia','SE. Asia','E. Asia','Australia / NZ']
    a.set(ylim=(60,-1),yticks=centres,yticklabels=names if labels else [])
    a.spines['left'].set_visible(False)
    a.tick_params(axis='y',length=0,pad=4,labelsize=6.0)
    a.tick_params(axis='x',labelsize=5.7,pad=2,length=1.8)
    a.axhspan(-.5,11.2,color='#EDF2EF',zorder=0)
    a.axhline(11.2,color='#CBD5CF',lw=.5,zorder=1)
    for y in [26.5,35.7,54.1]:
        a.axhline(y,color='#EBEFEC',lw=.4,zorder=0)
    if labels:a.get_yticklabels()[0].set_fontweight('bold')
    a.xaxis.set_major_formatter(FuncFormatter(
        lambda v,p:('0' if v==0 else f'{v:g}').replace('-','−')))
    return centres

def f4_estimates(f,a,mid,lo,hi,j,s,kind,detail=False):
    yy = np.r_[1.5+2.6*j, 15+4.6*np.arange(10)+(j-1.5)*.83]
    use=np.ones(len(mid),dtype=bool)
    if detail:
        left,right=a.get_xlim()
        use=(lo>=left)&(hi<=right)
    a.errorbar(np.asarray(mid)[use],yy[use],
        xerr=[(np.asarray(mid)-lo)[use],(np.asarray(hi)-mid)[use]],
        fmt=MARKERS[s],ms=2.35,lw=.66,capsize=1.0,color=COLORS[s],
        mec='white',mew=.22,zorder=4)
    if not detail:
        
        
        
        import matplotlib.patheffects as pe
        f3_glyph(f,a,float(mid[0]),float(yy[0]),kind,COLORS[s],2.5)
        coordinates=np.c_[[lo[0],mid[0],hi[0]],[yy[0]]*3]
        pos=f.transFigure.inverted().transform(a.transData.transform(coordinates))
        caps=.33/(f.get_figheight()*25.4)
        halo=[pe.Stroke(linewidth=1.25,foreground='white'),pe.Normal()]
        f.add_artist(Line2D(pos[[0,2],0],pos[[0,2],1],transform=f.transFigure,
                           color=COLORS[s],lw=.62,zorder=21,path_effects=halo))
        for xx in pos[[0,2],0]:
            f.add_artist(Line2D([xx,xx],[pos[0,1]-caps,pos[0,1]+caps],
                               transform=f.transFigure,color=COLORS[s],lw=.62,
                               zorder=21,path_effects=halo))
        f.add_artist(Line2D([pos[1,0]],[pos[1,1]],transform=f.transFigure,
                           ls='none',marker=MARKERS[s],ms=1.75,mfc='white',
                           mec=COLORS[s],mew=.45,zorder=22))
    return int(use.sum())

def figure4():
    import sys
    from figure4_reference_rows import draw
    with mpl.rc_context():
        draw(sys.modules[__name__])


from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch,Ellipse
from matplotlib.offsetbox import DrawingArea,AnnotationBbox
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

F5_H=170.

F5_LAYOUT={
    'panel_bd_top':86.5,'panel_c_top':127.,
    'b_plot_top':92.5,'b_plot_h':26.,
    'c_plot_top':131.,'c_plot_h':28.7,
    'd_plot_top':96.4,
    'plot_left':17.,'left_group_right':87.,
    'metric_label_top':90.5,'metric_unit_top':94.1,
}
F5_BOTTOM=F5_H-F5_LAYOUT['c_plot_top']-F5_LAYOUT['c_plot_h']

def f5_compact_map_legend(f,map_col):
    
    rows=[[(3,'Mangroves in 2025'),(4,'Unbuilt historical habitat'),(5,'Connected landward space'),(6,'Existing construction')],
          [(7,'New construction'),(8,'Relocated construction'),(1,'Water / tidal'),(2,'Other known land'),(0,'Unknown')]]
    fp=FontProperties(family='Arial',size=5.6)
    legend_geometry=[]
    for j,row in enumerate(rows):
        widths=[TextPath((0,0),label,prop=fp,size=5.6).get_extents().width*25.4/72 for _,label in row]
        total=sum(widths)+len(row)*2.8+(len(row)-1)*4.0
        x=(W-total)/2; y=F5_H-75.0-j*3.2
        for (co,label),w in zip(row,widths):
            f.patches.append(Rectangle((x/W,(y-.65)/F5_H),1.8/W,1.3/F5_H,transform=f.transFigure,
                fc=map_col[co],ec='#B4BDB7' if co==0 else 'none',lw=.35))
            txt(f,x+2.8,y,label,va='center',fontsize=5.6)
            legend_geometry.append({'label':label,'x_mm':x,'y_mm':y,'text_width_mm':w})
            x+=2.8+w+4.0
    return legend_geometry

def f5_symbol_box(kind,color,h_mm=1.95):
    
    k=h_mm/25.4*72/6
    da=DrawingArea(6*k,6*k,0,0)
    def path(vertices,codes,fc='none',ec=None,lw=.45):
        v=[(x*k,y*k) for x,y in vertices]
        p=PathPatch(MPath(v,codes),fc=fc,ec=color if ec is None else ec,lw=lw,
                    joinstyle='round',capstyle='round')
        da.add_artist(p)
    def line(v,lw=.45,co=None):
        da.add_artist(Line2D([x*k for x,y in v],[y*k for x,y in v],color=co or color,lw=lw,
                            solid_capstyle='round',solid_joinstyle='round'))
    if kind=='mangrove':
        line([(3,0),(3,3.8)],.5)
        line([(3,1.4),(1.65,0)],.4);line([(3,1.4),(4.35,0)],.4)
        line([(3,2.4),(1.6,3.5)],.4);line([(3,2.8),(4.5,4)],.4)
        
        for v in [[(3,3.1),(1.5,2.7),(.35,3.8),(.65,4.55),(2.0,4.7),(2.8,4.0),(3,3.1)],
                  [(3,3.45),(3.65,4.85),(5.25,5.35),(5.55,4.6),(5.45,3.2),(3.8,2.8),(3,3.45)],
                  [(3.0,3.8),(2.2,4.85),(2.6,5.8),(3.45,6),(4.1,5.4),(3.8,4.35),(3,3.8)]]:
            path(v,[MPath.MOVETO]+[MPath.CURVE4]*6,fc=color,ec='none')
    else:
        
        line([(.5,0),(.5,3.6),(2.45,3.6),(2.45,5.5),(5.2,5.5),(5.2,0),(3.5,0),(3.5,1.25),(2.5,1.25),(2.5,0),(.5,0)],.48)
        for x,y in [(1.2,2.5),(3.2,4.4),(4.3,4.4),(3.2,3.15),(4.3,3.15)]:
            line([(x,y),(x+.38,y)],.38)
    return da

def f5_glyph(a,x,y,kind,color,h_mm=1.95,z=5):
    art=AnnotationBbox(f5_symbol_box(kind,color,h_mm),(x,y),frameon=False,
        box_alignment=(.5,0),pad=0,xycoords='data',annotation_clip=False,zorder=z)
    a.add_artist(art)
    return art

def f5_heading_glyph(f,x,y,kind,color,h_mm=3.1):
    art=AnnotationBbox(f5_symbol_box(kind,color,h_mm),(x/W,y/F5_H),frameon=False,
        box_alignment=(.5,0),pad=0,xycoords=f.transFigure,annotation_clip=False,zorder=8)
    f.add_artist(art)

def f5_metric_heading(f,centre,kind,label,unit,color):
    
    h=3.0
    bounds={'mangrove':(.35,5.55),'building':(.5,5.2)}[kind]
    ink_left=(bounds[0]/6-.5)*h
    ink_width=(bounds[1]-bounds[0])/6*h
    tb=TextPath((0,0),label,prop=FontProperties(family='Arial',size=6.2),size=6.2).get_extents()
    text_width=tb.width*25.4/72
    gap=1.4; left=centre-(ink_width+gap+text_width)/2
    label_y=F5_H-F5_LAYOUT['metric_label_top']
    unit_y=F5_H-F5_LAYOUT['metric_unit_top']
    icon_y=label_y-1.05
    f5_heading_glyph(f,left-ink_left,icon_y,kind,color,h)
    txt(f,left+ink_width+gap-tb.x0*25.4/72,label_y,label,fontsize=6.2)
    txt(f,centre,unit_y,unit,ha='center',fontsize=5.4,color=MID)
    return {'centre_mm':centre,'visible_width_mm':ink_width+gap+text_width,
            'visible_left_mm':left,'visible_right_mm':left+ink_width+gap+text_width,
            'icon_base_mm':icon_y,'label_baseline_mm':label_y,'unit_baseline_mm':unit_y}

def figure5():
    import sys
    from figure5_protection import draw
    draw(sys.modules[__name__])



def figure6():
    import sys
    from figure6_contours import draw
    global W
    previous_width = W
    try:
        W = 168.
        with mpl.rc_context():
            draw(sys.modules[__name__])
    finally:
        W = previous_width



def supplementary5():
    import sys
    from figure5_regional_supplement import draw
    draw(sys.modules[__name__])

if __name__=='__main__':
    wanted=A.figures.split(',')
    for number,fn in [('3',figure3),('4',figure4),('5',figure5),('6',figure6),('S5',supplementary5)]:
        if number in wanted:fn()
    (Q/('supplementary_figure5_manifest.json' if wanted==['S5'] else 'figure_story_manifest.json')).write_text(json.dumps({'revision':'figure_story_reproduction_v1','scientific_identity':'UMRI-v1.2.0','endpoint_revision':'endpoint-2100','inputs':INPUTS,'figures':AUDIT},ensure_ascii=False,indent=2)+'\n',encoding='utf8')


