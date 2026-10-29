




from __future__ import annotations
import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import argparse,csv,hashlib,json,math,re
from fractions import Fraction as F
import numpy as np
from umri_reporting import secant
from umri_core import normalize, LAMBDA
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path as MP
from matplotlib.patches import PathPatch,Polygon
from matplotlib.transforms import Affine2D
from matplotlib.text import Text

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
p=argparse.ArgumentParser()
p.add_argument('--source-data',type=Path,default=ROOT.parent/'SourceData'/'main_figures')
p.add_argument('--output',type=Path,default=ROOT/'reproduced')
p.add_argument('--draft',action='store_true')
args=p.parse_args()
OUT=args.output/'figures';QA=args.output/'qa'
OUT.mkdir(parents=True,exist_ok=True);QA.mkdir(parents=True,exist_ok=True)
spec=json.loads((args.source_data/'Fig6_illustrative_parameters.json').read_text(encoding='utf-8-sig'))
W,H=171,73
C=dict(ink='#263B40',mid='#71817E',teal='#397B73',green='#568C70',leaf='#94AF83',
       clay='#B96756',urban='#C79B88',sand='#E7DFCD',mud='#B5A68C',water='#DCECEF',
       waterline='#A0C3CB',pale='#E5EFEB',purple='#9B86AA',blue='#719DB5',ochre='#AA8A54')
mpl.rcParams.update({'font.family':'Arial','font.size':7.2,'text.color':C['ink'],
 'axes.labelcolor':C['ink'],'axes.edgecolor':C['mid'],'xtick.color':C['ink'],
 'ytick.color':C['ink'],'axes.linewidth':.55,'xtick.major.width':.55,
 'ytick.major.width':.55,'xtick.major.size':2.2,'ytick.major.size':2.2,
 'hatch.linewidth':.22,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white',
 'path.simplify':False,'mathtext.fontset':'custom','mathtext.rm':'Arial',
 'mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold'})
fig=plt.figure(figsize=(W/25.4,H/25.4))
ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,W);ax.set_ylim(H,0);ax.set_axis_off()
ART=[];LAYER='figure_labels';ART_COUNTER=0
def tagged(a):
    global ART_COUNTER
    ART_COUNTER+=1
    a.set_gid(f"edit_{LAYER}__{ART_COUNTER:05d}")
    return a

def line(points,color=None,lw=.5,ls="-",z=4,alpha=1):
    q=np.asarray(points)
    return tagged(ax.plot(q[:,0],q[:,1],color=color or C["mid"],lw=lw,ls=ls,
                   solid_capstyle="round",zorder=z,alpha=alpha)[0])

def poly(points,fill,edge=None,lw=.35,z=2,alpha=1):
    q=Polygon(points,closed=True,facecolor=fill,edgecolor=edge or "none",lw=lw,zorder=z,alpha=alpha)
    ax.add_patch(q); return tagged(q)


def text(x,y,s,size=6.6,weight="normal",ha="left",va="center",color=None,rotation=0,z=30,**kw):
    t=ax.text(x,y,s,fontsize=size,fontweight=weight,ha=ha,va=va,
              color=color or C["ink"],rotation=rotation,zorder=z,**kw)
    ART.append(t);return tagged(t)


def group_editable_svg(target):
    import xml.etree.ElementTree as ET
    ns="http://www.w3.org/2000/svg"
    for prefix,uri in [("",ns),("xlink","http://www.w3.org/1999/xlink"),("dc","http://purl.org/dc/elements/1.1/"),("cc","http://creativecommons.org/ns#"),("rdf","http://www.w3.org/1999/02/22-rdf-syntax-ns#")]:ET.register_namespace(prefix,uri)
    tree=ET.parse(target);root=tree.getroot()
    
    
    for parent in list(root.iter()):
        for child in list(parent):
            if parent.get("id") and parent.get("id")==child.get("id"):
                at=list(parent).index(child)
                for k,item in enumerate(list(child)):parent.insert(at+k,item)
                parent.remove(child)
    for parent in list(root.iter()):
        children=list(parent);sets={}
        for child in children:
            gid=child.get("id","")
            if gid.startswith("edit_"):
                key=gid.split("__")[0][5:]
                if parent.get("id")==key:continue
                sets.setdefault(key,[]).append(child)
        for key,nodes in sets.items():
            where=list(parent).index(nodes[0]);group=ET.Element("{"+ns+"}g",{"id":key})
            parent.insert(where,group)
            for node in nodes:parent.remove(node);group.append(node)
    
    
    for node in root.iter():
        style={a.strip():b.strip() for a,b in (q.split(":",1) for q in node.get("style","").split(";") if ":" in q)}
        if "font" in style:
            value=style.pop("font")
            match=re.fullmatch(r"(.*?)\s*([0-9.]+)px\s+(.+)",value)
            if not match:raise ValueError("Unexpected SVG font specification: "+value)
            before,size,family=match.groups();family=family.strip().strip("\"'")
            props={"font-family":family,"font-size":size+"px"}
            for token in before.split():
                if token in ["italic","oblique"]:props["font-style"]=token
                elif token.isdigit() or token=="bold":props["font-weight"]=token
            style.update(props)
            for key,val in props.items():node.set(key,val)
            node.set("style","; ".join(f"{k}: {v}" for k,v in style.items()))
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag=="{"+ns+"}tspan" and child.text:
                xs=child.get("x","").split();ys=child.get("y","").split()
                if len(xs)>1:
                    if len(xs)!=len(child.text):raise ValueError("SVG character-position mismatch")
                    at=list(parent).index(child)
                    for index,char in enumerate(child.text):
                        attrs=dict(child.attrib);attrs["x"]=xs[index]
                        if ys:attrs["y"]=ys[index] if len(ys)>1 else ys[0]
                        span=ET.Element(child.tag,attrs);span.text=char
                        parent.insert(at+index,span)
                    parent.remove(child)
    
    
    
    
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    symbol_count=0
    for group in list(root.iter()):
        for text_node in list(group):
            if text_node.tag!="{"+ns+"}text":continue
            for span in list(text_node):
                family=span.get("font-family", "")
                if span.tag!="{"+ns+"}tspan" or family!="cmex10" or span.text!="q":continue
                size=float(span.get("font-size").removesuffix("px"))
                x=float(span.get("x","0"));y=float(span.get("y","0"))
                glyph=TextPath((0,0),span.text,size=size,
                               prop=FontProperties(family=family),usetex=False)
                glyph=glyph.transformed(Affine2D().scale(1,-1).translate(x,y))
                commands=[]
                letters={MP.MOVETO:"M",MP.LINETO:"L",MP.CURVE3:"Q",MP.CURVE4:"C",MP.CLOSEPOLY:"Z"}
                for vertices,kind in glyph.iter_segments(curves=True,simplify=False):
                    commands.append(letters[kind]+(" " .join(f"{v:.8f}" for v in vertices)
                                                  if kind!=MP.CLOSEPOLY else ""))
                symbol_count+=1
                outline=ET.Element("{"+ns+"}path",{
                    "id":f"math_symbol_sqrt_{symbol_count:03d}","d":" ".join(commands),
                    "data-math-symbol":"sqrt","aria-label":"square root",
                    "style":"stroke: none"})
                
                group.insert(list(group).index(text_node)+1,outline)
                text_node.remove(span)
    tree.write(target,encoding="utf-8",xml_declaration=True)


fw={k:F(str(v)) for k,v in spec['weights'].items()}
wd,wr,wg,wh,wT=[fw[k] for k in ('d','r','g','h','T')]
base=next(s for s in spec['settings'] if s['id']=='eroded')
joined=spec['reconnection']
m,ci,c0,h=[F(str(base[k])) for k in ('m','C','C0','h')]
pressure=F(str(joined['urban_pressure_p']))
assert spec['normalization_lambda']==LAMBDA
cii=F(str(joined['C']));rp=wr*F(str(joined['r']))+wg*F(str(joined['g_positive']))
def t_of(c):return (c-c0)/(c+c0)
def g_of(c):return min(F(1),1+h,1+t_of(c))
ti,tii,gi,gii=t_of(ci),t_of(cii),g_of(ci),g_of(cii)
rmax=wr*m+wg
pi=(wh*h+wT*ti+gi*rmax)/wd
pii=(wh*h+wT*tii+gii*rmax)/wd
cstar=c0*(1+h)/(1-h)
di=wT*(tii-ti);dg=rp*(gii-gi)
assert (pi,pii,cstar,di,dg)==(F(3,10),F(857,1500),F(36,55),F(8,195),F(17,3250))
assert di+dg==F(451,9750) and ci<=cii==m<c0
B={tuple(x) for x in joined['historical_cells']}
MI={tuple(x) for x in joined['current_before']}
LINK={tuple(x) for x in joined['new_link_cells']};MII=MI|LINK
def components(current):
    un=set(current);counts=[]
    while un:
        start=un.pop();todo=[start];seen={start}
        while todo:
            x,y=todo.pop()
            for q in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if q in un:un.remove(q);todo.append(q);seen.add(q)
        counts.append(len(seen&B))
    return sorted(counts)
assert len(B)==10 and len(MI&B)==len(MII&B)==7 and not B&LINK
assert components(MI)==[3,4] and components(MII)==[7]


P=dict(ink='#263B3B',I='#A57460',II='#285B54',shared='#ECE1D7',added='#E6EEE9',adverse='#F7F5F0',
       direct='#668493',direct_fill='#E7EEF0',gate='#DDC798',pale='#F4F6F1',axis='#82948F')
TITLE_Y,AX_TOP=3.8,16.0
for k in ['text.color','axes.labelcolor','xtick.color','ytick.color']:mpl.rcParams[k]=P['ink']
AX_HEIGHT=42.0; AX_WIDTH=68.0
B_YMAX=.18
layout_y=lambda y:y*B_YMAX/.0555
GUIDE_STYLE=(0,(.7,3.0))
SYMBOL_BASE=AX_TOP+AX_HEIGHT+8.2


def vpath(commands,fill='none',edge=None,lw=.4,z=3):
    vertices=[];codes=[]
    for command in commands:
        kind,*v=command
        if kind=='M':vertices.append(v);codes.append(MP.MOVETO)
        elif kind=='L':vertices.append(v);codes.append(MP.LINETO)
        elif kind=='C':vertices.extend([v[:2],v[2:4],v[4:]]);codes.extend([MP.CURVE4]*3)
        elif kind=='Q':vertices.extend([v[:2],v[2:]]);codes.extend([MP.CURVE3]*2)
        elif kind=='Z':vertices.append(vertices[0]);codes.append(MP.CLOSEPOLY)
    artist=PathPatch(MP(vertices,codes),facecolor=fill,edgecolor=edge or 'none',lw=lw,
                     capstyle='round',joinstyle='round',zorder=z)
    ax.add_patch(artist);return tagged(artist)






LAYER='habitat_configuration_I'
line([(14.5,9.5),(19.3,9.5)],P['I'],1.10,z=12)
text(20.8,9.5,'I  Separate patches',size=6.6,color=P['ink'])
LAYER='habitat_configuration_II'
line([(47.0,9.5),(51.8,9.5)],P['II'],1.10,z=12)
text(53.3,9.5,'II  Reconnected habitat',size=6.6,color=P['ink'])
LAYER='shared_ecological_header'
fixed=text(98.0,9.5,r'Fixed $p=0.20$, $r=0.20$, $g=0.10$',size=6.8,color='#6C8077')
fixed.set_gid('edit_b_fixed_response__00001')
LAYER='panel_titles'
text(3.8,TITLE_Y,'a',size=8.4,weight='bold')
text(8.0,TITLE_Y,'Potential compensation',size=7.5)
text(87.3,TITLE_Y,'b',size=8.4,weight='bold')
text(91.5,TITLE_Y,'Reconnection gain',size=7.5)

def coordinate_axes(x,top,w,height,group):
    a=fig.add_axes([x/W,(H-top-height)/H,w/W,height/H]);a.set_gid(group)
    for side in ('top','right'):a.spines[side].set_visible(False)
    for side in ('left','bottom'):a.spines[side].set_color(P['axis']);a.spines[side].set_linewidth(.55)
    a.tick_params(axis='both',labelsize=7.2,pad=2.2,direction='out',length=2.4,width=.55)
    return a
la=coordinate_axes(14.5,AX_TOP,AX_WIDTH,AX_HEIGHT,'panel_a_coordinate_system')
pa=np.sort(np.r_[np.linspace(0,1,1001),float(pi),float(pii)])
ri=(float(wd)*pa-float(wh*h+wT*ti))/float(gi)
rii=(float(wd)*pa-float(wh*h+wT*tii))/float(gii)
def agid(artist,name):artist.set_gid('edit_'+name+'__00001');return artist
agid(la.fill_between(pa,0,np.minimum(rii,float(rmax)),color=P['adverse'],linewidth=0,zorder=0),'a_adverse_domain')
agid(la.fill_between(pa,rii,float(rmax),where=pa<=float(pii),color=P['added'],linewidth=0,zorder=1),'a_compensation_reconnected')
agid(la.fill_between(pa,ri,float(rmax),where=pa<=float(pi),color=P['shared'],linewidth=0,zorder=2),'a_compensation_shared')
for setting,co,intercept,end in [('I',P['I'],float(ri[0]),float(pi)),('II',P['II'],float(rii[0]),float(pii))]:
    agid(la.plot([0,end],[intercept,float(rmax)],color=co,lw=1.25,solid_capstyle='round',zorder=5)[0],'a_balance_'+setting)
    agid(la.plot([end,end],[0,float(rmax)],color='#8D9C95',lw=.50,ls=GUIDE_STYLE,dash_capstyle='round',zorder=3)[0],'a_pressure_projection_'+setting)
    agid(la.plot(end,float(rmax),'o',ms=4.0,mec='white',mew=.65,color=co,zorder=7)[0],'a_upper_intersection_'+setting)
agid(la.plot([0,1],[float(rmax)]*2,color='#71817E',lw=.75,solid_capstyle='butt',zorder=4)[0],'a_response_cap_reference')
def at(a,x,y,s,group,**kw):
    t=a.text(x,y,s,**kw);t.set_gid('edit_'+group+'__00001');ART.append(t);return t
at(la,.78,.215,r'$R_{\max}$','a_upper_label',fontsize=7.0,ha='center',va='bottom')
agid(la.plot([float(pi),float(pi),float(pii),float(pii)],[.219,.223,.223,.219],color='#8B947F',lw=.48,zorder=8)[0],'a_pressure_gain_bracket')
at(la,float((pi+pii)/2),.234,r'$\Delta p^*$','a_pressure_gain_symbol',fontsize=6.5,ha='center',va='bottom',color=P['II'])
at(la,.105,.175,'I + II','a_shared_label',fontsize=7.0,ha='center',va='center')
at(la,.225,.139,'II only','a_added_label',fontsize=7.1,color=P['II'],ha='center',va='center')
at(la,.77,.075,r'$\mathrm{UMRI}<0$','a_adverse_formula',fontsize=6.8,ha='center',va='center')
balance_angle=math.degrees(math.atan(float(wd/gii)*AX_HEIGHT/(.238*AX_WIDTH)))
at(la,.36,.164,r'$\mathrm{UMRI}=0$','a_balance_label',fontsize=7.2,color=P['II'],rotation=balance_angle)
la.set(xlim=(0,1),ylim=(0,.238),xticks=[0,.3,float(pii),1],yticks=[0,.1,.2])
la.set_xticklabels(['0','0.30','0.57','1'])
for tick,label in zip(la.get_xticklabels(),['0','I','II','1']):
    if label in ('I','II'):tick.set_color(P[label])
la.set_yticklabels(['0','0.10','0.20'])
la.set_xlabel(r'Urban pressure, $p=-d$',fontsize=7.2,labelpad=7)
la.set_ylabel(r'Positive response, $R_+$',fontsize=7.2,labelpad=5)

ra=coordinate_axes(98.0,AX_TOP,AX_WIDTH,AX_HEIGHT,'panel_b_coordinate_system')
cc=np.sort(np.r_[np.linspace(float(ci),float(cii),801),float(cstar)])
tt=(cc-float(c0))/(cc+float(c0));gg=np.minimum(1+float(h),1+tt)
direct_intermediate=float(wT)*(tt-float(ti));gate_intermediate=float(rp)*(gg-float(gi))
s_initial=float(-wd*pressure+wh*h+wT*ti+rp*gi)
s_curve=-float(wd*pressure)+float(wh*h)+float(wT)*tt+float(rp)*gg
mapping=secant(s_initial,s_curve)
direct=mapping*direct_intermediate;gate=mapping*gate_intermediate
total=(normalize(s_curve)-normalize(s_initial))
point_closure=float(np.max(np.abs(direct+gate-total)))
assert point_closure<1e-10
assert float(total[-1])>0 and np.all(np.diff(total)>=-1e-12)
agid(ra.fill_between(cc,0,direct,color=P['direct_fill'],linewidth=0,zorder=1),'b_direct_area')
agid(ra.fill_between(cc,direct,total,color=P['gate'],linewidth=0,zorder=2),'b_added_compensation')
agid(ra.plot(cc,direct,color=P['direct'],lw=.90,ls=(0,(4,2.4)),dash_capstyle='butt',zorder=4)[0],'b_direct_curve')
agid(ra.plot(cc,total,color=P['II'],lw=1.30,solid_capstyle='round',zorder=5)[0],'b_total_curve')
agid(ra.plot([float(cstar)]*2,[0,float(np.interp(float(cstar),cc,total))],color='#8D9C95',lw=.50,ls=GUIDE_STYLE,dash_capstyle='round',zorder=3)[0],'b_limiting_switch')
for x,y,co,label in [(float(ci),0,P['I'],'I'),(float(cii),float(total[-1]),P['II'],'II')]:
    agid(ra.plot(x,y,'o',ms=4,mec='white',mew=.65,color=co,zorder=7,clip_on=False)[0],'b_state_'+label)
    at(ra,x+(.003 if label=='I' else -.002),y+layout_y(.0032),label,'b_state_label_'+label,fontsize=7.5,
       color=co,ha='left' if label=='I' else 'right',weight='bold')
at(ra,.601,layout_y(.0323),r'$\Delta\mathrm{UMRI}$','b_total_formula',fontsize=6.6,color=P['II'],ha='center')



def icon_xy(xc,yc,xy):
    q=np.asarray(xy,dtype=float)
    return np.c_[xc+q[:,0]*.21/AX_WIDTH,yc+q[:,1]*B_YMAX/AX_HEIGHT]

def small_patch(xc,yc,points,name):
    q=icon_xy(xc,yc,np.asarray(points)*STRUCTURE_SCALE)
    pp=Polygon(q,closed=True,facecolor='#E0E9DF',edgecolor='#6F8F7D',lw=.38,zorder=7,joinstyle='miter')
    ra.add_patch(pp);agid(pp,name)
    return pp

STRUCTURE_SCALE=1.73
sx=.5708;sy=layout_y(.01175)-.008-1.3*STRUCTURE_SCALE*B_YMAX/AX_HEIGHT
for j,ends in enumerate([[(2.0,.725),(3.25,.725)]]):
    qq=icon_xy(sx,sy,np.asarray(ends)*STRUCTURE_SCALE)
    agid(ra.plot(qq[:,0],qq[:,1],color='#B3975A',lw=.55,solid_capstyle='round',zorder=6)[0],f'b_structure_glyph_link_{j}')
sp1=small_patch(sx,sy,[(0,.15),(2,.15),(2,1.3),(1,1.3),(1,2.45),(0,2.45)],'b_structure_glyph_patch_1')
sp2=small_patch(sx,sy,[(3.25,.15),(5.25,.15),(5.25,2.45),(3.25,2.45)],'b_structure_glyph_patch_2')
for j,(cx,cy) in enumerate([(.50,.65),(.45,1.85),(3.80,.80),(4.60,1.75)]):
    for k,(scale,ox,oy,fill) in enumerate([(1,0,0,'#355D52'),(.82,-.053,.07,'#789C83')]):
        aa=np.linspace(0,2*np.pi,14,endpoint=False);rr=.44*scale*(1+.10*np.sin(5*aa+j*.5))
        q=icon_xy(sx,sy,np.c_[cx+ox+rr*np.cos(aa),cy+oy+.9*rr*np.sin(aa)]*STRUCTURE_SCALE)
        vv=[q[0]];codes=[MP.MOVETO]
        for u in range(len(q)):
            q0,q1,q2,q3=q[(u-1)%len(q)],q[u],q[(u+1)%len(q)],q[(u+2)%len(q)]
            vv.extend([q1+(q2-q0)/6,q2-(q3-q1)/6,q2]);codes.extend([MP.CURVE4]*3)
        vv.append(q[0]);codes.append(MP.CLOSEPOLY)
        pp=PathPatch(MP(vv,codes),facecolor=fill,edgecolor='#355D52',lw=.12,zorder=9+k*.1)
        pp.set_clip_path(sp1 if j<2 else sp2)
        ra.add_patch(pp);agid(pp,f'b_structure_glyph_canopy_{j}_{k}')
at(ra,sx+10.4*.21/AX_WIDTH,layout_y(.0095)-.008,r'$A_T$','b_direct_formula',fontsize=6.7,color=P['direct'],ha='left',va='center')

gx=.504;gy=layout_y(.04155)-1.3*B_YMAX/AX_HEIGHT
qq=icon_xy(gx,gy,[(3.1,.40),(5.6,1.45),(5.6,2.20),(3.1,.62)])
band=Polygon(qq,closed=True,facecolor=P['gate'],edgecolor='none',zorder=7)
ra.add_patch(band);agid(band,'b_gate_glyph_band')
for j,ends in enumerate([[(3.1,.40),(5.6,1.45)],[(3.1,.62),(5.6,2.20)]]):
    qq=icon_xy(gx,gy,ends)
    agid(ra.plot(qq[:,0],qq[:,1],color='#A38B54',lw=.48,solid_capstyle='round',zorder=8)[0],f'b_gate_glyph_boundary_{j}')
def response_leaf(xc,yc,name):
    
    for j,(points,fill) in enumerate([([(1.5,.1),(1.3,1.3),(1.8,2.8),(2.0,3.6)],None),
        ([(1.45,1.4),(.1,1.5),(.2,2.7),(1.65,2.5),(1.45,1.4)],'#ADC9B1'),
        ([(1.7,2.0),(2.95,2.2),(3.1,3.4),(1.8,3.15),(1.7,2.0)],'#7AA187')]):
        q=icon_xy(xc,yc,points)
        codes=[MP.MOVETO]+[MP.CURVE4]*3+([MP.CLOSEPOLY] if fill else [])
        pp=PathPatch(MP(q,codes),facecolor=fill or 'none',edgecolor=P['II'],lw=.40,zorder=9)
        ra.add_patch(pp);agid(pp,name+str(j))
response_leaf(gx,gy-.50*B_YMAX/AX_HEIGHT,'b_fixed_response_leaf_')
at(ra,gx+7.2*.21/AX_WIDTH,layout_y(.0391),r'$A_r + A_g$','b_gate_formula',fontsize=6.7,color='#947D47',ha='left',va='center')
cy=float(np.interp(float(cstar),cc,total))
agid(ra.plot(float(cstar),cy,marker='D',ms=3.8,mec=P['II'],mew=.65,mfc='white',zorder=8)[0],'b_exact_switch_node')


dx=float(cstar);rx=(2.1/2)/AX_WIDTH*.21;dh=3.0/AX_HEIGHT*B_YMAX
yt=layout_y(.0498);yb=yt-dh
vv=[(dx,yt),
    (dx-.12*rx,yt-.22*dh),(dx-rx,yt-.53*dh),(dx-rx,yb+.33*dh),
    (dx-rx,yb-.11*dh),(dx+rx,yb-.11*dh),(dx+rx,yb+.33*dh),
    (dx+rx,yt-.53*dh),(dx+.12*rx,yt-.22*dh),(dx,yt),(dx,yt)]
pc=PathPatch(MP(vv,[MP.MOVETO]+[MP.CURVE4]*9+[MP.CLOSEPOLY]),facecolor='white',edgecolor=P['direct'],lw=.65,zorder=9)
ra.add_patch(pc);agid(pc,'b_moisture_limitation_symbol')
agid(ra.plot([dx,dx],[yb-layout_y(.001),cy+layout_y(.0011)],color=P['direct'],lw=.34,zorder=7)[0],'b_moisture_symbol_leader')
at(ra,dx+.0045,(yt+yb)/2,r'$h$','b_moisture_symbol_h',fontsize=6.6,ha='left',va='center',color=P['direct'])
ra.set(xlim=(.495,.705),ylim=(0,B_YMAX),xticks=[.5,.6,float(cstar),.7],yticks=[0,.05,.1,.15])
ra.set_xticklabels(['0.5','0.6','0.655','0.7'])
ra.set_yticklabels(['0','.05','.10','.15'])
ra.get_xticklabels()[2].set_color('#687D72');ra.get_xticklabels()[3].set_color(P['II'])
ra.set_xlabel(r'Connected capacity, $C$',fontsize=7.2,labelpad=15)
ra.set_ylabel('Gain in UMRI',fontsize=7.2,labelpad=5)
la.xaxis.set_label_coords(.5,-.255)
ra.xaxis.set_label_coords(.5,-.255)
LAYER='aligned_axis_symbols'
for xpos,label,color in [(14.5+AX_WIDTH*float(pi),r'$p^*_{\rm I}$',P['I']),
 (14.5+AX_WIDTH*float(pii),r'$p^*_{\rm II}$',P['II']),
 (98+AX_WIDTH*(float(cstar)-.495)/.21,r'$C^*$','#687D72'),
 (98+AX_WIDTH*(.7-.495)/.21,r'$C=m$',P['II'])]:
    text(xpos,SYMBOL_BASE,label,size=7.2,color=color,ha='center',va='baseline')


LAYER='named_axis_pictograms'
ux,uy=29.7,H-5.3
for j,(x0,ytop,ww) in enumerate([(0,1.2,1.35),(1.3,0,1.6),(2.85,1.6,1.4)]):
    poly([(ux+x0,uy+3.8),(ux+x0,uy+ytop),(ux+x0+ww,uy+ytop),(ux+x0+ww,uy+3.8)],'#DFE6E6',edge='#6D8187',lw=.35,z=12)
    for wy in np.arange(ytop+.55,3.45,.85):
        line([(ux+x0+.3,uy+wy),(ux+x0+ww-.3,uy+wy)],'#80949A',.22,z=13)
line([(ux-.25,uy+3.8),(ux+4.55,uy+3.8)],'#6D8187',.4,z=13)
px,py=4.2,AX_TOP+2.5
vpath([('M',px,py+1.8),('C',px-.3,py+.5,px+.5,py-.5,px+.6,py-1.6)],edge=P['II'],lw=.45,z=12)
vpath([('M',px,py+.5),('C',px-2.0,py+.3,px-1.9,py-1.3,px+.2,py-.7),('C',px+.1,py-.2,px+.1,py+.2,px,py+.5),('Z',)],'#ADC9B1',P['II'],.35,z=12)
vpath([('M',px+.35,py-.2),('C',px+2.2,py-.4,px+1.8,py-1.9,px+.5,py-1.3),('C',px+.4,py-.9,px+.4,py-.5,px+.35,py-.2),('Z',)],'#7AA187',P['II'],.35,z=12)

fig.canvas.draw();renderer=fig.canvas.get_renderer();overflow=[];boxes=[]
for t in fig.findobj(match=Text):
    if not t.get_visible() or not t.get_text():continue
    bb=t.get_window_extent(renderer)
    if bb.width==0 or bb.height==0:continue
    boxes.append({'text':t.get_text(),'bbox_px':[bb.x0,bb.y0,bb.x1,bb.y1],'size_pt':t.get_fontsize()})
    if bb.x0<-.8 or bb.y0<-.8 or bb.x1>fig.bbox.x1+.8 or bb.y1>fig.bbox.y1+.8:overflow.append(t.get_text())
assert not overflow,overflow
NAME='SupplementaryFig4_habitat_compensation';exports=[]
for ext,dpi in [('pdf',None),('svg',None),('png',260)]+([] if args.draft else [('600dpi.png',600),('tiff',600)]):
    pathout=OUT/(NAME+('_'+ext if ext=='600dpi.png' else '.'+ext))
    options={'dpi':dpi} if dpi else {}
    if ext=='tiff':options['pil_kwargs']={'compression':'tiff_lzw'}
    fig.savefig(pathout,**options)
    if ext=='svg':group_editable_svg(pathout)
    exports.append({'file':pathout.name,'bytes':pathout.stat().st_size,'sha256':hashlib.sha256(pathout.read_bytes()).hexdigest()})

with (QA/'Fig6_analytical_boundary.csv').open('w',newline='',encoding='utf8') as fh:
    w=csv.writer(fh);w.writerow(['setting','urban_pressure_p','required_positive_response','positive_response_upper_bound','upper_bound_allows_compensation'])
    for label,vals,limit in [('eroded',ri,float(pi)),('reconnected',rii,float(pii))]:
        for xx,yy in zip(pa,vals):w.writerow([label,f'{xx:.17g}',f'{yy:.17g}',f'{float(rmax):.17g}',xx<=limit])
with (QA/'Fig6_reconnection_response.csv').open('w',newline='',encoding='utf8') as fh:
    w=csv.writer(fh);w.writerow(['C','T','G','direct_gain','compensation_gain','total_gain','urban_pressure_p','S_intermediate','UMRI','direct_UMRI','compensation_UMRI','total_UMRI'])
    for row in zip(cc,tt,gg,direct_intermediate,gate_intermediate,direct_intermediate+gate_intermediate,np.full(len(cc),float(pressure)),s_curve,normalize(s_curve),direct,gate,total):w.writerow([f'{v:.17g}' for v in row])
exact=lambda x:{'fraction':str(x),'value':float(x)}
manifest={'figure_revision':'final-UMRI-score','art_revision':'supplementary_figure4_layout_v1','scientific_identity':'UMRI-v1.2.0',
 'purpose':'Analytical illustration of habitat retention and condition-gated compensation','dimensions_mm':[W,H],
 'layout':{'axes_width_mm':AX_WIDTH,'axes_height_mm':AX_HEIGHT,'axes_top_mm':AX_TOP,'left_axis_x_mm':14.5,'right_axis_x_mm':98.0,
           'frame_gap_mm':15.5,'shared_symbol_baseline_mm':SYMBOL_BASE,'balance_label_angle_deg':balance_angle,
           'component_icon_strip':False,'adverse_domain_is_explicitly_shaded':True,
           'spatial_header_removed':True,'state_key_baseline_mm':9.5,'panel_title_baseline_mm':TITLE_Y,'right_axis_y_limits':[0,B_YMAX],'right_axis_x_limits':[.495,.705],'right_zero_aligned_with_axis':True},
 'settings':{'I':{'m':float(m),'C':float(ci),'h':float(h),'C0':float(c0),'T':exact(ti),'G':exact(gi),'p_star':exact(pi)},
             'II':{'m':float(m),'C':float(cii),'h':float(h),'C0':float(c0),'T':exact(tii),'G':exact(gii),'p_star':exact(pii)}},
 'response_upper_bound':exact(rmax),'fixed_response_in_b':exact(rp),'C_switch':exact(cstar),
  'intermediate_gains':{'direct':exact(di),'compensation':exact(dg),'total':exact(di+dg)},
 'fixed_urban_pressure_p':float(pressure),'lambda':LAMBDA,'unit':'UMRI',
 'UMRI_I':float(normalize(s_initial)),'UMRI_II':float(normalize(s_curve[-1])),
 'gains_UMRI':{'direct':float(direct[-1]),'compensation':float(gate[-1]),'total':float(total[-1])},
 'component_closure_max_UMRI':point_closure,
 'habitat_geometry':{'B':sorted(B),'current_I':sorted(MI),'current_II':sorted(MII),'external_link':sorted(LINK),
                     'historical_counts_I':components(MI),'historical_counts_II':components(MII)},
 'semantics':{'same_historical_retention':True,'current_mangrove_area_increases':True,'pressure_is_conditionally_fixed':True,
              'continuous_parameter_curve_not_a_bridge_time_series':True,'domains_are_outer_bounds':True,
              'right_states_absolute_scores_both_negative_for_p_nonnegative':True,'all_curves_end_at_C_equals_m':True,
              'condition_gate_fixed_after_switch':True,'compensation_point_contribution_varies_with_mapping_after_switch':True,'direct_and_total_gains_continue_after_switch':True},
 'line_semantics':{'balance_boundaries':'coloured solid','common_response_upper_bound':'uniform neutral solid','total_gain':'teal solid','structural_gain':'blue-grey dashed (4,2.4)','coordinate_projections':'neutral dotted (0.7,3.0)','annotation_leaders':'thin solid','domain_fill_edges':'none'},
 'palette':P,'visual_variant':'analytical_without_spatial_header','right_label_style':'named_contributions_with_formulas','extra_symbols':{'pressure_gain_bracket':True,'pressure_gain':exact(pii-pi),'switch_node_and_moisture_symbol':True,'moisture_symbol_dimensions_mm':[2.1,3.0],'moisture_symbol_semantics':'Moisture becomes limiting at C*, h itself remains fixed.','component_glyph_dimensions_mm':{'structure':[9.0825,3.979],'compensation_container':[5.6,3.5]},'component_glyph_semantics':'Two vegetated habitat patches and a visible gold link identify the structural contribution. A static shoot identifies unchanged positive response, and a narrow gold band identifies the compensation increment.','axis_pictogram_semantics':'The shoot categorises recovery and greening. The skyline categorises urban pressure and does not encode habitat loss or a quantitative pressure value.'},'spatial_header_removed':True,'state_key_labels':['I Separate patches','II Reconnected habitat'],'spatial_key_labels':[],'spatial_artwork':[],'text_outside_canvas':overflow,'text_geometry':boxes,'exports':exports}
(QA/'supplementary_figure8_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
plt.close(fig)
print(json.dumps({'status':'RENDERED_FOR_REVIEW','dimensions_mm':[W,H],'files':[x['file'] for x in exports],
                  'text_overflow':overflow,'total_gain_UMRI':float(total[-1])}))
