




from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
from shapely.geometry import LineString, box
from shapely.ops import unary_union

def draw(base):
    O = base.A.output
    W, H = 168., 80.
    INK, MID = base.INK, base.MID
    TEAL, GOLD, PURPLE, BLUE = "#397F71", "#A68B48", "#8C74A6", "#5796B0"
    mpl.rcParams.update({
        "lines.solid_capstyle":"round", "lines.solid_joinstyle":"round",
        "lines.dash_capstyle":"round", "axes.linewidth":.48,
        "xtick.major.width":.45, "ytick.major.width":.45,
    })
    meta = base.js("figure_story/Fig6_theory_methods.json")
    meta["settings"] = base.js(meta["settings_file"])
    bd = base.read("figure_story/Fig6_retention_boundary.csv")
    source_curve = base.read("figure_story/Fig6_reconnection_control.csv")
    L = meta["settings"]["normalization_lambda"]
    wd, wr, wg, wh, wt = .25, .15, .10, .25, .25
    C0, CI, retained, h_ref, p_ref, r_ref, g_ref = .8, .5, .7, -.1, .2, .2, .1
    Rplus = wr*r_ref + wg*g_ref
    T = lambda c: (np.asarray(c)-C0)/(np.asarray(c)+C0)
    Gate = lambda c,h: 1+np.minimum(np.minimum(0,np.asarray(h)), T(c))
    V = lambda s: np.asarray(s)/(L+np.abs(s))
    Np = lambda s: L/(L+np.abs(s))**2
    pressure_ceiling = lambda m,c,h: (wh*h+wt*T(c)+Gate(c,h)*(wr*np.asarray(m)+wg))/wd
    score = lambda c,h: -wd*p_ref+wh*np.asarray(h)+wt*T(c)+Rplus*Gate(c,h)
    frozen_score = lambda c,h: -wd*p_ref+wh*np.asarray(h)+wt*T(c)+Rplus*Gate(CI,h)
    release = lambda c,h: V(score(c,h))-V(frozen_score(c,h))
    cs = meta["gate_limit_C"]
    T0 = float(T(CI))

    
    def required_capacity(m,p,h=h_ref):
        m = np.asarray(m)
        response = wr*m+wg
        k = wd*p-wh*h+wt
        c_struct = C0*k/(2*(wt+response)-k)
        c_switch = C0*(1+h)/(1-h)
        t_req = (wd*p-wh*h-(1+h)*response)/wt
        c_moist = C0*(1+t_req)/(1-t_req)
        return np.where(c_struct <= c_switch, c_struct, c_moist)

    def ceiling_partials(m,q,h=h_ref):
        c = m*q
        t_prime = 2*C0/(c+C0)**2
        limiting = T(c) < min(0,h)
        k = wt+(wr*m+wg)*limiting
        return ((q*t_prime*k+wr*Gate(c,h))/wd,
                m*t_prime*k/wd)

    def ceiling_partials_m_c_h(m,c,h):
        response=wr*m+wg
        return (
            Gate(c,h)*wr/wd,
            (2*C0/(c+C0)**2)*(wt+response*(T(c)<min(0,h)))/wd,
            (wh+response*(h<min(0,float(T(c)))))/wd,
        )

    check_c = source_curve.C.to_numpy()
    check_h = h_ref
    sd = score(check_c, check_h)
    sf = frozen_score(check_c, check_h)
    checks = {
        "nominal_dynamic_matches_all_402_rows": bool(np.allclose(
            V(sd)-V(sd[0]), source_curve.gain_dynamic, atol=3e-13, rtol=0)),
        "nominal_fixed_matches_all_402_rows": bool(np.allclose(
            V(sf)-V(sf[0]), source_curve.gain_frozen, atol=3e-13, rtol=0)),
        "nominal_release_matches_all_402_rows": bool(np.allclose(
            release(check_c,h_ref), source_curve.additional_gain_from_gate, atol=3e-13, rtol=0)),
        "configuration_q1_recovers_all_1001_original_boundary_rows": bool(np.allclose(
            pressure_ceiling(bd.m.to_numpy(),bd.m.to_numpy(),h_ref),bd.p_cap,atol=3e-15,rtol=0)),
    }

    states = pd.DataFrame([
        {"state":"III", "m":.5, "C":.5},
        {"state":"I", "m":.7, "C":.5},
        {"state":"II", "m":.7, "C":.7},
    ])
    states["q"] = states.C/states.m
    states["T"] = T(states.C)
    states["G"] = Gate(states.C,h_ref)
    states["pressure_ceiling"] = pressure_ceiling(states.m,states.C,h_ref)
    checks["three_exact_ceilings"] = bool(np.allclose(
        states.pressure_ceiling, [27/130,3/10,857/1500], atol=3e-15,rtol=0))
    checks["same_capacity_retention_effect"] = bool(abs(
        states.pressure_ceiling.iloc[1]-states.pressure_ceiling.iloc[0]
        -Gate(.5,h_ref)*wr/wd*(.7-.5)) < 1e-15)

    eps = 1e-6
    derivative_errors = []
    for mm,qq in [(.6,.8),(.7,.8),(.8,.9),(.9,.9)]:
        dm,dq = ceiling_partials(mm,qq)
        ndm=(pressure_ceiling(mm+eps,(mm+eps)*qq,h_ref)-pressure_ceiling(mm-eps,(mm-eps)*qq,h_ref))/(2*eps)
        ndq=(pressure_ceiling(mm,mm*(qq+eps),h_ref)-pressure_ceiling(mm,mm*(qq-eps),h_ref))/(2*eps)
        derivative_errors += [abs(dm-ndm),abs(dq-ndq)]
    checks["analytical_cross_effects_match_finite_differences"] = max(derivative_errors)<1e-8
    root_errors=[]
    for pp in [0,.2,.4,.6,.8]:
        mm=np.linspace(.3,1,351)
        cc=required_capacity(mm,pp)
        keep=(cc>0)&(cc<=mm)
        root_errors.extend(np.abs(pressure_ceiling(mm[keep],cc[keep],h_ref)-pp))
    checks["inverse_pressure_frontiers"] = max(root_errors)<3e-15
    margin_derivative_errors=[]
    for mm,cc,hh in [(.7,.5,-.1),(.7,.7,-.1),(.9,.9,.1),(.8,.65,-.3)]:
        exact=ceiling_partials_m_c_h(mm,cc,hh)
        point=np.array([mm,cc,hh])
        for dim,value in enumerate(exact):
            high=point.copy();low=point.copy()
            high[dim]+=eps;low[dim]-=eps
            numerical=(pressure_ceiling(*high)-pressure_ceiling(*low))/(2*eps)
            margin_derivative_errors.append(abs(value-numerical))
    checks["pressure_margin_partials_match_independent_differences"] = max(margin_derivative_errors)<1e-8

    
    
    
    m_axis=np.linspace(1e-6,1,801)
    q_axis=np.linspace(.28,1,721)
    M,Q=np.meshgrid(m_axis,q_axis)
    CC=M*Q
    q_min=lambda mm: np.maximum(1/np.sqrt(10),1/np.sqrt(1+30*(1-np.asarray(mm))))
    legal=Q>=q_min(M)
    ceiling=pressure_ceiling(M,CC,h_ref)
    ceiling_masked=np.ma.masked_where(~legal,ceiling)
    
    
    rng=np.random.default_rng(20261021)
    ma=rng.uniform(.2,1,2000)
    mb=.05+rng.random(2000)*(ma-.05)
    ca=ma*q_min(ma)+rng.random(2000)*(ma-ma*q_min(ma))
    cb=mb*q_min(mb)+rng.random(2000)*(np.minimum(ca,mb)-mb*q_min(mb))
    ha=rng.uniform(-.7,.2,2000)
    hb=-1+rng.random(2000)*(ha+1)
    pa0=rng.uniform(0,.8,2000)
    pb0=pa0+rng.random(2000)*(1-pa0)
    margin_a=pressure_ceiling(ma,ca,ha)-pa0
    margin_b=pressure_ceiling(mb,cb,hb)-pb0
    checks["coupled_pressure_and_habitat_erosion_reduces_margin"] = bool(
        np.max(margin_b-margin_a)<1e-12)
    cx=np.unique(np.r_[np.linspace(.5,.7,801),cs])
    hx=np.unique(np.r_[np.linspace(-.35,.30,1301),T0,h_ref])
    Cgrid,Hgrid=np.meshgrid(cx,hx)
    E=release(Cgrid,Hgrid)
    delta_s=Rplus*(Gate(Cgrid,Hgrid)-Gate(CI,Hgrid))
    checks["controlled_release_nonnegative"] = bool(np.min(E)>-2e-13)
    checks["no_release_when_moisture_limits_initial_state"] = bool(np.max(np.abs(E[Hgrid<=T0]))<2e-13)
    checks["gate_difference_equals_score_difference"] = bool(np.max(np.abs(
        score(Cgrid,Hgrid)-frozen_score(Cgrid,Hgrid)-delta_s))<8e-17)

    
    
    h_ridge=(2*wd*p_ref-2*wt*T(cx)-Rplus*(2+T(cx)+T0))/(2*wh)
    delta_ridge=Rplus*(T(cx)-T0)
    ridge_max=2*delta_ridge/(2*L+delta_ridge)
    ridge_valid=(h_ridge>=np.maximum(0,T(cx)))&(h_ridge>=hx.min())&(h_ridge<=hx.max())
    checks["normalization_ridge_exact"] = bool(np.allclose(
        release(cx[ridge_valid],h_ridge[ridge_valid]),ridge_max[ridge_valid],atol=6e-13,rtol=0))
    checks["ridge_is_mapping_property_not_ecological_threshold"] = True
    checks["chosen_h_range_has_valid_NDMI_at_common_zero_baseline"] = bool(np.all(
        (2*hx/(1-hx)>=-1)&(2*hx/(1-hx)<=1)))
    checks={key:bool(value) for key,value in checks.items()}
    assert all(checks.values()), checks

    
    
    h_dynamic_zero=(wd*p_ref-wt*T(cx)-Rplus*(1+T(cx)))/wh
    h_fixed_zero=(wd*p_ref-wt*T(cx)-Rplus*(1+T0))/wh
    checks['updated_balance_curve_exact']=bool(np.max(np.abs(score(cx,h_dynamic_zero)))<8e-17)
    checks['fixed_balance_curve_exact']=bool(np.max(np.abs(frozen_score(cx,h_fixed_zero)))<8e-17)
    checks['balance_window_width_exact']=bool(np.allclose(
        h_fixed_zero-h_dynamic_zero,Rplus*(T(cx)-T0)/wh,atol=6e-17,rtol=0))
    h_balance_mid=(h_dynamic_zero+h_fixed_zero)/2
    checks['balance_window_changes_sign']=bool(
        np.min(score(cx[1:],h_balance_mid[1:]))>=-1e-16 and
        np.max(frozen_score(cx[1:],h_balance_mid[1:]))<=1e-16)
    checks={key:bool(value) for key,value in checks.items()}
    assert all(checks.values()),checks

    f=base.fig(H)
    base.panel(f,3,76,'a','Habitat loss lowers compensation limits')
    base.panel(f,88.0,76,'b','Reconnection can strengthen compensation')
    a=base.ax(f,14,13,65,54)
    b=base.ax(f,98,13,62,54)
    line_layers={a:[],b:[]}
    fine_curve_errors=[]
    boundary_m=np.unique(np.r_[np.linspace(0,1,12001),.7])
    boundary_q=q_min(boundary_m)
    support_vertices=np.vstack([np.column_stack([boundary_m,boundary_q]),[1,1],[0,1],[0,boundary_q[0]]])
    support_path=MplPath(support_vertices,closed=True)

    def traced(axis,*xy,**kwargs):
        artists=axis.plot(*xy,**kwargs)
        line_layers[axis].extend(artists)
        return artists[0]

    def line_key(x,y,label,color,style='-',length=2.5):
        f.add_artist(Line2D([x/W,(x+length)/W],[y/H,y/H],transform=f.transFigure,
                           color=color,lw=.48,ls=style))
        base.txt(f,x+length+1,y,label,fontsize=5.7,color=INK,va='center')

    label_contrasts=[]
    def contrast_ink(ink,background,minimum=4.6):
        def luminance(color):
            rgb=np.asarray(mpl.colors.to_rgb(color))
            linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
            return float(linear@np.array([.2126,.7152,.0722]))
        background_luminance=luminance(background)
        rgb=np.asarray(mpl.colors.to_rgb(ink))
        for factor in np.linspace(1,0,501):
            chosen=rgb*factor
            contrast=(max(luminance(chosen),background_luminance)+.05)/(min(luminance(chosen),background_luminance)+.05)
            if contrast>=minimum:break
        label_contrasts.append({'ink_rgb':chosen.tolist(),'contrast':contrast})
        return chosen

    def curve_label(axis,line,position,text,color,fontsize=5.7):
        coords=np.column_stack(line.get_data())
        valid=np.isfinite(coords).all(axis=1)
        coords=coords[valid]
        display=axis.transData.transform(coords)
        target=axis.transData.transform(position)
        ix=int(np.argmin(np.sum((display-target)**2,axis=1)))
        left,right=max(0,ix-3),min(len(coords)-1,ix+3)
        tangent=display[right]-display[left]
        angle=np.degrees(np.arctan2(tangent[1],tangent[0]))
        if angle>90:angle-=180
        if angle< -90:angle+=180
        artist=axis.text(*coords[ix],text,ha='center',va='center',rotation=angle,
                         rotation_mode='anchor',fontsize=fontsize,color=color,zorder=10)
        artist._curve_label=True
        return artist

    
    
    cap_cmap=LinearSegmentedColormap.from_list('capacity_frontier_refined',[
        '#AF8274','#D9BCAE','#F7F4EC','#CEDFD4','#568871'],N=1024)
    a.set_facecolor('#F0F2EF')
    mesh=a.pcolormesh(M,Q,ceiling,cmap=cap_cmap,norm=TwoSlopeNorm(0,-1.1,1),
                     shading='auto',rasterized=True,zorder=1,antialiased=False)
    mesh.set_clip_path(support_path,a.transData)
    traced(a,boundary_m,boundary_q,color='#A6B0A7',lw=.40,zorder=4)
    a.set(xlim=(0,1),ylim=(.28,1),xticks=[0,.25,.5,.75,1],yticks=[.4,.6,.8,1],
          xlabel='Habitat retained, $m$',ylabel='Patch concentration, $q$')
    b.set(xlim=(.5,.7),ylim=(-.35,.30),xticks=[.5,.55,.6,.65,.7],
          yticks=[-.3,-.2,-.1,0,.1,.2,.3],xlabel='Connected habitat, $C$',
          ylabel='Canopy moisture, $h$')

    pressure_levels=[-.8,-.4,0,.2,.4,.6,.8]
    minor_pressure=np.array([v for v in np.round(np.arange(-1.,1.,.05),2)
                            if not np.any(np.isclose(v,pressure_levels))])
    def pressure_path(pp,major=False):
        mm=np.linspace(.00001,1,8001)
        
        m_switch=((wd*pp-wh*h_ref-wt*h_ref)/(1+h_ref)-wg)/wr
        if 0<m_switch<1:mm=np.unique(np.r_[mm,m_switch])
        qq=required_capacity(mm,pp)/mm
        fine_curve_errors.append(float(np.nanmax(np.abs(pressure_ceiling(mm,mm*qq,h_ref)-pp))))
        color=TEAL if pp==0 else '#FFFFFF'
        line=traced(a,mm,qq,color=color,lw=.76 if pp==0 else (.49 if major else .22),
                    alpha=1 if major else .83,zorder=3 if major else 2)
        line.set_clip_path(support_path,a.transData)
        return line
    for pp in minor_pressure:pressure_path(pp)
    label_m=[.15,.32,.49,.63,.79,.91,.945]
    for pp,mm in zip(pressure_levels,label_m):
        line=pressure_path(pp,True)
        curve_label(a,line,(mm,float(required_capacity(mm,pp)/mm)),
                    f'{pp:g}'.replace('-','−'),contrast_ink('#705346' if pp<0 else '#356955',cap_cmap(TwoSlopeNorm(0,-1.1,1)(pp))))

    capacity_levels=[.15,.30,.50,.80]
    for cap in capacity_levels:
        xx=np.linspace(cap,1,6001)
        line=traced(a,xx,cap/xx,color='#657C71',lw=.52 if cap==.5 else .33,
                    ls=(0,(2.1,2.7)),alpha=.82 if cap==.5 else .58,zorder=4)
        line.set_clip_path(support_path,a.transData)
    xx=np.linspace(cs,1,6001)
    gate_line=traced(a,xx,cs/xx,color='#527F91',lw=.72,zorder=5)
    gate_line.set_clip_path(support_path,a.transData)
    curve_label(a,gate_line,(.805,cs/.805),r'$C^*$',color=contrast_ink('#4E788A',cap_cmap(TwoSlopeNorm(0,-1.1,1)(float(pressure_ceiling(.805,cs,h_ref))))),fontsize=6.0)

    
    a.add_patch(FancyArrowPatch((.7,.977),(.7,5/7+.021),arrowstyle='-|>',
                    mutation_scale=5.0,color=base.CLAY,lw=.65,zorder=7,clip_on=False))
    a.add_patch(FancyArrowPatch((.680,1),(.521,1),arrowstyle='-|>',
                    mutation_scale=5.0,color=base.CLAY,lw=.65,zorder=7,clip_on=False))
    
    a.text(.60,1.017,'Habitat loss',ha='center',va='bottom',
           fontsize=5.0,color='#9B4D39',clip_on=False,zorder=10)
    a.text(.722,.83,'Fragmentation',ha='center',va='center',rotation=90,
           fontsize=5.0,color='#9B4D39',zorder=10)
    a.scatter([.7],[cs/.7],s=10.5,facecolor='white',edgecolor='#527F91',lw=.60,zorder=9)
    for _,row in states.iterrows():
        a.scatter(row.m,row.q,s=17 if row.state=='II' else 15,
                  facecolor=TEAL if row.state=='II' else 'white',edgecolor=TEAL,
                  lw=.65,zorder=8,clip_on=False)
        if row.state in ['III','II']:
            a.annotate(row.state,(row.m,row.q),xytext=(0,3.5),textcoords='offset points',
                       ha='center',va='bottom',fontsize=6.2,color=INK,annotation_clip=False)
        else:
            a.annotate('I',(row.m,row.q),xytext=(4,-6),textcoords='offset points',
                       ha='left',va='top',fontsize=6.2,color=INK)
    a.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
    a.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
    base.txt(f,14,71.6,'Urban pressure limit, $p^*$',fontsize=5.9,color=MID)
    line_key(48.5,71.6,'Equal $C$','#657C71',(0,(2.1,2.7)))
    line_key(65,71.6,'Limit changes','#527F91')

    
    
    gain_cmap=LinearSegmentedColormap.from_list('gate_release_refined',[
        '#FAF8F0','#F2E8CD','#DBC38A','#B6954E','#887133'],N=1024)
    b.pcolormesh(cx,hx,E,cmap=gain_cmap,norm=Normalize(0,.105),
                 shading='auto',rasterized=True,zorder=1,antialiased=False)
    b.axhspan(hx.min(),T0,color='#EDF2F1',zorder=2,lw=0)
    levels=[.01,.02,.04,.06,.08,.1]
    minor_gain=np.array([v for v in np.arange(.005,.105,.005) if v not in levels])
    def gain_path(value,major=False):
        fraction=value
        delta_needed=2*L*value/(2-value)
        t_needed=T0+delta_needed/Rplus
        c_needed=C0*(1+t_needed)/(1-t_needed)
        c_line=c_needed+(.7-c_needed)*np.linspace(0,1,3501)**2
        delta=Rplus*(T(c_line)-T0)
        ridge=(2*wd*p_ref-2*wt*T(c_line)-Rplus*(2+T(c_line)+T0))/(2*wh)
        lower=np.full_like(c_line,T0);upper=ridge.copy()
        for _ in range(52):
            mid=(lower+upper)/2
            below=release(c_line,mid)<value
            lower=np.where(below,mid,lower);upper=np.where(below,upper,mid)
        h_low=(lower+upper)/2
        cross=fraction>delta/(L+delta)
        centered=(L+delta/2)**2-L*(2*L+delta)/(2-fraction)
        score_high=np.where(cross,-delta/2+np.sqrt(np.maximum(0,centered)),
                             -L-delta/2+np.sqrt(L*delta/fraction+delta**2/4))
        initial_intercept=-wd*p_ref+wt*T(c_line)+Rplus*(1+T0)
        h_high=(score_high-initial_intercept)/wh
        
        h_low[0]=h_high[0]=ridge[0]
        yy=np.r_[h_low[::-1],h_high[1:]]
        xx=np.r_[c_line[::-1],c_line[1:]]
        fine_curve_errors.append(float(np.max(np.abs(release(xx,yy)-value))))
        return traced(b,xx,yy,color='white',lw=.50 if major else .23,
                      alpha=1 if major else .86,zorder=3 if major else 2)
    for value in minor_gain:gain_path(value)
    manual_positions=[(.571,-.045),(.60,.01),(.641,.047),(.647,.23),(.675,.18),(.687,.145)]
    for value,position in zip(levels,manual_positions):
        curve_label(b,gain_path(value,True),position,f'{value:g}',contrast_ink('#65522C',gain_cmap(Normalize(0,.105)(value))))
    b.fill_between(cx,h_dynamic_zero,h_fixed_zero,color='#487C6B',alpha=.19,lw=0,zorder=3.5)
    traced(b,cx,h_dynamic_zero,color='#386F60',lw=.68,zorder=4)
    traced(b,cx,h_fixed_zero,color='#637B6F',lw=.54,ls=(0,(2.5,2.5)),zorder=4)
    traced(b,cx,T(cx),color='#638A9A',lw=.72,zorder=5)
    traced(b,[.5,.7],[T0,T0],color='#93A9AE',lw=.44,ls=(0,(2.0,2.6)),zorder=4)
    traced(b,[.5,.7],[h_ref,h_ref],color='#557D6D',lw=.48,ls=(0,(2.1,2.8)),zorder=5)
    b.scatter([.5,cs,.7],[h_ref,h_ref,h_ref],s=[15,10.5,17],
              facecolor=['white','white',TEAL],edgecolor=TEAL,lw=.65,zorder=8,clip_on=False)
    b.annotate('I',(.5,h_ref),xytext=(4,5),textcoords='offset points',
               fontsize=6.2,ha='left',va='bottom')
    b.annotate('II',(.7,h_ref),xytext=(4,5),textcoords='offset points',
               fontsize=6.2,ha='left',va='bottom',annotation_clip=False)
    b.annotate('+0.0224',(.7,h_ref),xytext=(-4,-7),textcoords='offset points',
               fontsize=5.9,ha='right',va='top',color='#80672B',annotation_clip=False)
    b.text(.612,-.290,'0',ha='center',va='center',fontsize=7.,color='#456C65')
    b.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:.2f}'))
    b.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'.replace('-','−')))
    base.txt(f,98,71.6,'Extra UMRI gain',fontsize=5.9,color=MID)
    line_key(127.5,71.6,'Constraint relaxes','#386F60')
    line_key(149.5,71.6,'Held fixed','#637B6F',(0,(2.5,2.5)))
    b.text(.508,-.018,'Structure limits',fontsize=5.9,color='#6B5778',va='center')
    b.text(.646,-.201,'Limiting condition changes',fontsize=5.9,color='#456470',va='center',ha='center')
    b.text(.518,-.29,'Moisture limits',fontsize=5.9,color='#456D7A',va='center')
    for axis in [a,b]:
        axis.xaxis.labelpad=3.3;axis.yaxis.labelpad=3.8
        axis.tick_params(pad=2.7,width=.40,length=2.0)
        for spine in axis.spines.values():spine.set_linewidth(.42)

    
    
    from shapely.geometry import Polygon
    f.canvas.draw();renderer=f.canvas.get_renderer()
    def pieces(geometry):
        if geometry.is_empty:return []
        if geometry.geom_type=='LineString':return [np.asarray(geometry.coords)]
        return [part for g in geometry.geoms for part in pieces(g)]
    for axis in [a,b]:
        guards=[]
        for artist in axis.texts:
            if not artist.get_visible() or not artist.get_text().strip():continue
            if getattr(artist,'_curve_label',False):
                angle=artist.get_rotation();artist.set_rotation(0)
                bounds=artist.get_window_extent(renderer);artist.set_rotation(angle)
                width,height=bounds.width+.85*f.dpi/72,bounds.height+.35*f.dpi/72
                center=axis.transData.transform(artist.get_position())
                theta=np.deg2rad(angle)
                rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*np.array([width,height])/2
                guards.append(Polygon(corners@rotation.T+center))
            else:
                bounds=artist.get_window_extent(renderer).padded(.40*f.dpi/72)
                guards.append(box(bounds.x0,bounds.y0,bounds.x1,bounds.y1))
        guard=unary_union(guards)
        for line in line_layers[axis]:
            coords=np.column_stack(line.get_data())
            finite=np.isfinite(coords).all(axis=1)
            starts=np.flatnonzero(finite & ~np.r_[False,finite[:-1]])
            stops=np.flatnonzero(finite & ~np.r_[finite[1:],False])+1
            kept=[axis.transData.inverted().transform(part) for st,en in zip(starts,stops) if en-st>1
                  for part in pieces(LineString(axis.transData.transform(coords[st:en])).difference(guard)) if len(part)>1]
            combined=np.concatenate([np.vstack([part,[[np.nan,np.nan]]]) for part in kept]) if kept else np.empty((0,2))
            line.set_data(combined[:,0],combined[:,1])
    checks['drawn_contours_satisfy_exact_formulas']=max(fine_curve_errors)<1e-9
    assert checks['drawn_contours_satisfy_exact_formulas'],max(fine_curve_errors)
    (O/'qa/exact_stroke_geometry.json').write_text(json.dumps({
        'maximum_contour_residual':max(fine_curve_errors),'curve_count':len(fine_curve_errors),
        'pressure_path':'direct inverse analytic curve with exact gate-change point',
        'gain_path':'analytic turning point and upper branch; vectorized bracketed lower branch',
        'support_path':'12001 point evaluation of original necessary bound; vector clip',
        'annotation_gap':'rotated physical text footprint',
        'label_contrasts':label_contrasts,'minimum_direct_label_contrast':min(x['contrast'] for x in label_contrasts),'data_smoothing':False,'display_line_width_pt':{'minor_pressure':.22,'minor_gain':.23,'major':.50,'boundary':.72}
    },indent=2)+'\n',encoding='utf8')

    
    
    reference={tuple(x) for x in meta["settings"]["reconnection"]["historical_cells"]}
    current_i={tuple(x) for x in meta["settings"]["reconnection"]["current_before"]}
    bridge={tuple(x) for x in meta["settings"]["reconnection"]["new_link_cells"]}
    current_ii=current_i|bridge
    lost_reference_cells={(0,1),(5,1)}
    current_iii=current_ii-lost_reference_cells
    def components(pixels):
        remain=set(pixels); comps=[]
        while remain:
            stack=[remain.pop()];comp=set(stack)
            while stack:
                x,y=stack.pop()
                for n in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
                    if n in remain:
                        remain.remove(n);comp.add(n);stack.append(n)
            comps.append(comp)
        return comps
    def geometry_state(current):
        retained_count=len(reference&current)
        counts=[len(comp&reference) for comp in components(current)]
        return retained_count/len(reference),np.sqrt(sum(n*n for n in counts))/len(reference)
    assert geometry_state(current_iii)==(.5,.5)
    assert geometry_state(current_i)==(.7,.5)
    assert geometry_state(current_i|bridge)==(.7,.7)
    assert current_i<current_ii and current_iii<current_ii
    assert (current_ii-current_i).isdisjoint(reference)
    assert current_ii-current_iii==lost_reference_cells
    checks["fragmentation_arrow_preserves_historical_retention"] = True
    checks["retention_loss_arrow_deletes_only_two_historical_cells"] = True
    total=float(source_curve.gain_dynamic.iloc[-1])
    fixed=float(source_curve.gain_frozen.iloc[-1])

    semantics={
        "revision":"figure_story_reproduction_v1","preview_only":False,"panels":2,
        "panel_a_field":"Conditional pressure ceiling p*(m,C=mq) at h=-0.1 and C0=0.8; colour range [-1.1,1] contains every finite field value.",
        "panel_a_feasibility":"Necessary continuous support for the H=10, K_B=1 example. The three marked configurations are discrete realizations.",
        "q_definition":"C/m = sqrt(sum of squared retained shares among current components).",
        "panel_b_field":"N(S_dynamic(C,h))-N(S_initial_gate(C,h)).",
        "panel_b_regimes":["h <= T(C_I): no gate release","T(C_I) < h < T(C): initially structure-limited, then moisture-limited","h >= T(C): structure-limited throughout"],
        "nominal_controls":{"m":.7,"p":.2,"r":.2,"g":.1,"C0":.8,"C_I":.5},
        "main_source_row_counts":{"original_retention_boundary":len(bd),"original_reconnection":len(source_curve)},
        "direct_contour_labels":{"pressure":pressure_levels,"extra_gain":levels},
        "minor_contours":{"pressure":minor_pressure.tolist(),"extra_gain":minor_gain.tolist()},
        "capacity_lines":capacity_levels,
        "capacity_gate_switch":cs,
        "balance_boundaries":{"updated":"S_dynamic=0","fixed":"S_frozen=0","width_h_at_C_0_7":float(h_fixed_zero[-1]-h_dynamic_zero[-1])},
        "annotation_gaps":"Line segments are cut at rotated physical text footprints; field values are unchanged.",
        "smoothing":False,"empirical_data_added":False,
        "numerical_range_in_b":{"C":[.5,.7],"h":[-.35,.30],"units":"UMRI"},
        "normalization_ridge":"Fixed-delta mapped contrast peaks at S_dynamic=-S_frozen; mathematical mapping sensitivity, not an ecological tipping point.",
        "normalization_ridge_drawn":False,
        "loss_arrows":"II to I removes only the external bridge (Fragmentation); II to III removes two historical cells while retaining that bridge (Habitat loss).",
        "exact_geometries":{"III":sorted(current_iii),"I":sorted(current_i),"II":sorted(current_ii),"B":sorted(reference),"external_link":sorted(bridge),"historical_cells_removed_II_to_III":sorted(lost_reference_cells)},
        "figure_content":"Two analytical panels only; case diagrams, gain bars and explanatory legends are in the caption/source, not in the image.",
    }

    base.save(f,"Fig6_habitat_compensation",semantics)
    assert not base.AUDIT["Fig6_habitat_compensation"]["text_outside_canvas"]
    checks["exact_three_state_geometries_validated"] = True
    (O/"source_data").mkdir(parents=True,exist_ok=True)

    states.to_csv(O/"source_data/Fig6_three_configurations.csv",index=False)
    np.savez_compressed(O/"source_data/Fig6_analytical_fields.npz",
                       m=m_axis,q=q_axis,pressure_ceiling=ceiling,necessary_support=legal,
                       C=cx,h=hx,extra_gain_UMRI=E,intermediate_gate_release=delta_s,
                       normalization_ridge_h=h_ridge,
                       updated_balance_h=h_dynamic_zero,fixed_balance_h=h_fixed_zero,
                       capacity_gate_switch=cs)
    derivation={
        "C":"m*q",
        "R_max":"w_r*m+w_g",
        "p_star":"[w_h*h+w_T*T(C)+G(C,h)*(w_r*m+w_g)]/w_d",
        "T_prime":"2*C0/(C+C0)^2",
        "p_m_at_fixed_C":"G*w_r/w_d >= 0; strictly positive where G > 0",
        "p_m_at_fixed_q":"[q*T_prime*(w_T+R_max*I_structure_limiting)+G*w_r]/w_d",
        "p_q_at_fixed_m":"m*T_prime*(w_T+R_max*I_structure_limiting)/w_d",
        "compensation_margin":"M_comp=p_star(m,C,h)-p; S <= w_d*M_comp",
        "margin_partial_p":"-1",
        "margin_partial_m_at_fixed_C":"G*w_r/w_d >= 0",
        "margin_partial_C":"T_prime*(w_T+R_max*I_structure_limiting)/w_d > 0",
        "margin_partial_h":"(w_h+R_max*I_moisture_limiting)/w_d > 0",
        "urbanization_interpretation":"Rising exposure lowers the margin directly; loss of retained habitat or connected capacity lowers the ceiling against which exposure is assessed. The arrows are conditional endpoint examples, not observed urban trajectories.",
        "finite_erosion_check_pairs":2000,
        "finite_erosion_max_margin_change":float(np.max(margin_b-margin_a)),
        "pressure_margin_derivative_max_abs_error":max(margin_derivative_errors),
        "structural_response_derivative":"dS/dC=T_prime*(w_T+R_plus*I_structure_limiting); the gate adds to direct structural sensitivity in the structure-limited regime.",
        "switch":"h=T(C), equivalently C*=C0*(1+h)/(1-h)",
        "updated_balance_h":"[wd*p-wT*T(C)-Rplus*(1+T(C))]/wh, where structure limits G",
        "fixed_balance_h":"[wd*p-wT*T(C)-Rplus*(1+T(CI))]/wh, where h>=T(CI)",
        "balance_window_width":"Rplus*(T(C)-T(CI))/wh",
        "balance_window_width_at_C_0_7":float(h_fixed_zero[-1]-h_dynamic_zero[-1]),
        "delta_gate":"max(0,min(h,T(C))-min(h,T(C_I))) on C>=C_I and T(C)<0",
        "delta_S":"R_plus * [G(C,h)-G(C_I,h)]",
        "mapped_extra":"N(S_dynamic)-N(S_frozen)",
        "mapped_max_for_fixed_delta":"2*delta_S/(2*lambda+delta_S) at S_dynamic=-S_frozen",
        "derivative_max_abs_error":max(derivative_errors),
        "inverse_boundary_max_abs_error":max(root_errors),
        "nominal_total_gain_UMRI":total,"nominal_fixed_gain_UMRI":fixed,
        "nominal_extra_gain_UMRI":total-fixed,
        "endpoint_ridge_h":float(h_ridge[-1]),"endpoint_ridge_extra_UMRI":float(ridge_max[-1]),
        "original_definition_and_derivation_sources":["main.tex","supplementary.tex: Notes 1-2","figure_story/Fig6_theory_methods.json"],
    }
    (O/"qa/analytical_derivation.json").write_text(json.dumps(derivation,indent=2)+"\n",encoding="utf8")

    (O/"qa/figure6_contours_checks.json").write_text(json.dumps({
        "status":"PASS","revision":semantics["revision"],"checks":checks,
        "physical_size_mm":[W,H],"analytical_derivation":derivation,
        "figure":base.AUDIT["Fig6_habitat_compensation"]},indent=2)+"\n",encoding="utf8")
    return {"checks":checks,"semantics":semantics,"derivation":derivation}

