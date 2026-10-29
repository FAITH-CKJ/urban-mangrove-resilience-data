




from pathlib import Path
import argparse,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--source-data',type=Path,default=ROOT.parent.parent/'SourceData')
ap.add_argument('--output-dir',type=Path,default=ROOT.parent/'reproduced/figures')
args=ap.parse_args()
DATA=args.source_data.resolve(strict=True); F=args.output_dir.resolve();F.mkdir(parents=True,exist_ok=True)
hist=pd.read_csv(DATA/'main_figures/Fig2_history_matched2025.csv')
regions=pd.read_csv(DATA/'supplementary/regional_population.csv')
names=dict(zip(regions.region_id,regions.region));names[0]='All regions'
ORDER=[0,5,2,6,9,3,10,8,7,4,1]
YEARS=list(range(2000,2026))
assert len(hist)==286
assert not hist.duplicated(['region_id','year']).any()
assert set(hist.region_id)==set(ORDER)
assert not hist[['year','region_id','UMRI']].isna().any().any()
assert hist.UMRI.between(-1,1).all()
W,H=164.,145.
DARK='#23383F';MID='#617077';AXIS='#829096';GRID='#E8EDEF'
BLUE='#215F75';ORANGE='#C17A39';PURPLE='#8B6AA0'
FILL='#E8F0F2'
STYLES={1:{'color':ORANGE,'ls':(0,(4.1,2.5)),'lw':1.1,'zorder':4},
        2:{'color':BLUE,'ls':'-','lw':1.35,'zorder':3},
        5:{'color':PURPLE,'ls':(0,(.9,1.65)),'lw':1.15,'zorder':5}}
plt.rcParams.update({'font.family':'Arial','font.size':7.8,'axes.labelsize':8,
 'xtick.labelsize':7.0,'ytick.labelsize':7.0,'axes.titlesize':8.0,
 'axes.linewidth':.45,'xtick.major.width':.45,'ytick.major.width':.45,
 'xtick.major.size':2.0,'ytick.major.size':2.0,'pdf.fonttype':42,
 'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
 'axes.spines.right':False,'legend.frameon':False,'savefig.facecolor':'white',
 'text.color':DARK,'axes.labelcolor':DARK,'xtick.color':MID,'ytick.color':MID,
 'lines.solid_capstyle':'round','lines.solid_joinstyle':'round',
 'lines.dash_capstyle':'butt'})

def figtext(fig,x,y,s,**kw):return fig.text(x/W,y/H,s,**kw)
def axes_mm(fig,x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])
def arrow(ax,x,y,direction,color,scale=1):
    
    dx=.063*scale;dy=.125*scale
    return ax.add_patch(FancyArrowPatch((x-dx/2,y-dy/2*direction),(x+dx/2,y+dy/2*direction),
       transform=ax.transAxes,arrowstyle='-|>',mutation_scale=6.7*scale,
       linewidth=.9*scale,color=color,zorder=7,clip_on=False))
def fmt(v):return f'{v:+.3f}'.replace('-','−')
def style(ax,row,col):
    
    ax.set_xlim(1999.4,2025.6);ax.set_ylim(-1,1)
    ax.set_xticks([2000,2010,2025]);ax.set_yticks([-1,0,1])
    ax.set_yticklabels(['−1','0','+1'] if col==0 else [])
    ax.tick_params(axis='both',pad=2)
    ax.spines['left'].set_color(AXIS);ax.spines['bottom'].set_color(AXIS)
    ax.spines['bottom'].set_bounds(2000,2025)
    for v in [-.5,.5]:ax.axhline(v,color=GRID,lw=.38,ls=(0,(1.8,2.5)),zorder=0)
    ax.axhline(0,color='#9CAAAE',lw=.6,zorder=1)
    
    ax.set_facecolor('white')

audits=[];series=[]
for mode in [1]:
    fig=plt.figure(figsize=(W/25.4,H/25.4))
    figtext(fig,2.5,73,'UMRI',fontsize=8.5,rotation=90,va='center',ha='center')
    figtext(fig,85,1.0,'Year',fontsize=8.0,ha='center',va='bottom')
    seen=[];axlist=[];changes={};text_boxes=[]
    for i,rid in enumerate(ORDER):
        row,col=divmod(i,3); x=11+col*51.3;y=111-row*34
        ax=axes_mm(fig,x,y,45.5,24.2);axlist.append(ax);style(ax,row,col)
        figtext(fig,x-3.5,y+26.5,chr(97+i),fontsize=9.0,weight='bold',ha='left',va='baseline')
        figtext(fig,x,y+26.5,names[rid],fontsize=7.9,weight='bold' if rid==0 else 'normal',ha='left',va='baseline')
        if mode==1:
            df=hist[hist.region_id==rid].sort_values('year');assert df.year.tolist()==YEARS
            xx=df.year.to_numpy(); yy=df.UMRI.to_numpy()
            ax.fill_between(xx,yy,0,color=FILL,linewidth=0,zorder=1)
            ln,=ax.plot(xx,yy,color=BLUE,lw=1.25,zorder=4)
            ax.scatter([xx[0]],[yy[0]],s=11,facecolor='white',edgecolor=BLUE,lw=.8,zorder=6,clip_on=False)
            ax.scatter([xx[-1]],[yy[-1]],s=12,facecolor=BLUE,edgecolor='white',lw=.35,zorder=6,clip_on=False)
            delta=float(yy[-1]-yy[0]);changes[rid]={2:delta}
            arrow(ax,.67,.82,np.sign(delta),BLUE,.8)
            ax.text(.98,.82,fmt(delta),transform=ax.transAxes,fontsize=7.4,color=MID,va='center',ha='right')
            assert np.array_equal(ln.get_xdata(),xx) and np.array_equal(ln.get_ydata(),yy)
            seen.extend(df.index.tolist())
            series.append({'figure':'Supplementary Figure 6','region_id':rid,'n':26,'year_start':2000,'year_end':2025,'UMRI_start':float(yy[0]),'UMRI_end':float(yy[-1]),'change_UMRI':delta,'source_rows':df.index.tolist()})
    key=axes_mm(fig,114,9,45,24.2);key.set(xlim=(0,1),ylim=(0,1));key.axis('off')
    if mode==1:
        key.text(0,1.12,'Reading the histories',fontsize=8.0,fontweight='bold',va='baseline')
        key.plot([.025,.105],[.82,.82],color=BLUE,lw=1.2)
        key.scatter([.025],[.82],s=11,fc='white',ec=BLUE,lw=.8)
        key.scatter([.105],[.82],s=12,fc=BLUE,ec='white',lw=.35)
        key.text(.18,.82,'2000 → 2025',fontsize=7.8,va='center')
        arrow(key,.06,.55,1,BLUE,.8);key.text(.18,.55,'Endpoint change (ΔUMRI)',fontsize=7.3,va='center')
        key.plot([.025,.105],[.28,.28],color='#9CAAAE',lw=.7)
        key.text(.18,.28,'Zero: balanced contributions',fontsize=7.2,va='center')
        key.text(.0,-.03,'26 annual estimates in every panel',fontsize=7.1,color=MID,va='center')
    assert len(seen)==len(set(seen))==286
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    overflow=[]
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():continue
        box=text.get_window_extent(renderer)
        if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.width+.5 or box.y1>fig.bbox.height+.5:overflow.append(text.get_text())
    assert not overflow,overflow
    stem=F/'SupplementaryFig6_full_histories'
    fig.savefig(stem.with_suffix('.pdf'),metadata={'Title':'Supplementary Figure 6','Author':'Ruili Li and co-authors','Subject':'Complete annual UMRI comparison','CreationDate':None})
    fig.savefig(stem.with_suffix('.svg'),metadata={'Date':None})
    fig.savefig(stem.with_suffix('.png'),dpi=600)
    fig.savefig(stem.with_suffix('.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    audits.append({'figure':'Supplementary Figure 6','records':len(seen),'panels':11,'years':[2000,2025],'vertical_limits':[-1,1],'smoothing':False,'excluded_records':0,'width_mm':W,'height_mm':H,'font_range_pt':[7.0,9.0],'visible_text_outside_canvas':overflow,'arrow_semantics':'Sign of the endpoint difference; no significance coding.','all_plotted_series_equal_source':True})

audit={'status':'PASS','backend':'Python matplotlib','figures':audits,
 'input_sha256':{str(p.relative_to(DATA)):hashlib.sha256(p.read_bytes()).hexdigest()
 for p in [DATA/'main_figures/Fig2_history_matched2025.csv',DATA/'supplementary/regional_population.csv']},
 'series':series}
(F/'SupplementaryFig6_verification.json').write_text(json.dumps(audit,indent=2)+'\n','utf8')
print(json.dumps({k:v for k,v in audit.items() if k!='series'},ensure_ascii=False))
