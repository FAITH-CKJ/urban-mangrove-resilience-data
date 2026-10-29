
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Rectangle,Polygon
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

W,H=144.,114.
SC=['ssp245','ssp585']
COL={'ssp245':'#0072B2','ssp585':'#D55E00'}
MAP=['#ECEDED','#DCEAF1','#F0EBDF','#347C62','#CAB071','#ACCEB8','#929BA1','#C96C4F','#7865A5']
HEIGHT=np.sqrt(3)/2

def draw(base):
    with mpl.rc_context({'font.family':'Arial','font.size':6.3,'axes.labelsize':6.1,
        'axes.linewidth':.45,'xtick.major.width':.4,'ytick.major.width':.4,
        'xtick.major.size':1.5,'ytick.major.size':1.5,'lines.solid_capstyle':'round',
        'lines.dash_capstyle':'round','path.simplify':False}):
        previous_width=base.W
        try:
            base.W=W
            _draw(base)
        finally:
            base.W=previous_width

def _draw(base):
    f=base.fig(H);ink=base.INK;mid='#627277'
    text=lambda x,y,t,**kw:base.txt(f,x,y,t,**kw)
    axis=lambda x,y,w,h:base.ax(f,x,y,w,h)
    def panel(x,y,label):
        text(x,y,label,fontsize=8,fontweight='bold')
    def clean(a):
        a.grid(False);a.tick_params(labelsize=5.8,pad=1.8)
    def read_parquet(name):
        p=base.S/name;base.INPUTS[name]=base.sha(p);return pd.read_parquet(p)
    def shares(d):
        z=d.copy();target=z.primary_realized_growth_km2
        z['within_pct']=100*z.in_cell_protected_growth_km2/target
        z['relocated_pct']=100*z.relocated_growth_km2/target
        z['unserved_pct']=100*z.additional_unmet_growth_km2/target
        defined=target>0
        assert np.allclose(z.loc[defined,['within_pct','relocated_pct','unserved_pct']].sum(axis=1),100,atol=1e-10,rtol=0)
        assert (z.loc[defined,['within_pct','relocated_pct','unserved_pct']].to_numpy()>=0).all()
        return z
    def ternary(w,r,u):
        w=np.asarray(w);r=np.asarray(r);u=np.asarray(u)
        assert np.allclose(w+r+u,100,atol=1e-9,rtol=0)
        return (r+.5*u)/100,HEIGHT*u/100

    
    panel(3,110.7,'a')
    meta=base.js('figure_story/Fig5_spatial_example.json')
    p=base.S/'figure_story/Fig5_spatial_example.npz';base.INPUTS['figure_story/Fig5_spatial_example.npz']=base.sha(p)
    with np.load(p) as z:
        full={k:z[k].copy() for k in ['growth','protected']};original_extent=z['extent'].copy()
    rows,cols=(26,108),(45,147)
    extent=np.array([original_extent[0]+cols[0]*30,original_extent[0]+cols[1]*30,
                     original_extent[3]-rows[1]*30,original_extent[3]-rows[0]*30])
    maps={k:v[rows[0]:rows[1],cols[0]:cols[1]].copy() for k,v in full.items()}
    map_w=66.;map_h=map_w*(extent[3]-extent[2])/(extent[1]-extent[0]);map_geometry={}
    for x,key,title in [(3,'growth','Unprotected'),(75,'protected','Protected')]:
        a=axis(x,55.3,map_w,map_h)
        a.imshow(maps[key],extent=extent,cmap=ListedColormap(MAP),vmin=-.5,vmax=8.5,origin='upper',interpolation='nearest')
        a.set_aspect('equal');a.set_xticks([]);a.set_yticks([])
        for spine in a.spines.values():spine.set_visible(True);spine.set_linewidth(.3);spine.set_edgecolor('#A6B2AD')
        text(x+map_w/2,110.7,title,ha='center',fontsize=6.5)
        xx=extent[0]+150;yy=extent[2]+140
        a.plot([xx,xx+500],[yy,yy],color='white',lw=2,zorder=7)
        a.plot([xx,xx+500],[yy,yy],color=ink,lw=.7,zorder=8)
        a.text(xx+250,yy+75,'500 m',ha='center',fontsize=5.8,bbox={'fc':'white','ec':'none','alpha':.96,'pad':.2},zorder=8)
        map_geometry[key]={'nominal_mm':[x,55.3,map_w,map_h],'extent':extent.tolist(),
            'display_array_sha256':hashlib.sha256(maps[key].tobytes()).hexdigest(),
            'original_array_sha256':hashlib.sha256(full[key].tobytes()).hexdigest(),
            'display_class_counts':{str(i):int((maps[key]==i).sum()) for i in range(9)}}
    
    
    key_rows=[[(3,'Mangrove'),(4,'Unbuilt habitat'),(5,'Landward'),(6,'Existing'),(7,'New'),(8,'Relocated')]]
    key_boxes=[];fp=FontProperties(family='Arial',size=6)
    for row_i,row in enumerate(key_rows):
        widths=[TextPath((0,0),label,prop=fp).get_extents().width*25.4/72 for _,label in row]
        item_widths=[3.1+w for w in widths];gap=2.6
        x=W/2-(sum(item_widths)+gap*(len(row)-1))/2;y=52.0
        for (i,label),tw,iw in zip(row,widths,item_widths):
            f.patches.append(Rectangle((x/W,(y-.55)/H),2.1/W,1.1/H,transform=f.transFigure,
                fc=MAP[i],ec='#A1AAA9' if i==0 else 'none',lw=.3))
            text(x+3.1,y,label,va='center',fontsize=6)
            key_boxes.append({'label':label,'x_mm':x,'y_mm':y,'estimated_width_mm':iw});x+=iw+gap

    panel(3,46.7,'b')
    panel(50.5,46.7,'c')
    panel(98,46.7,'d')
    summary=base.read('main_figures/Fig5_loss_summary_annual.csv')
    loss=base.read('main_figures/Fig5_loss_members_annual.csv')
    overall=shares(base.read('main_figures/Fig5_allocation_overall_annual.csv'))
    b=axis(10.5,10.5,35.5,33)
    b.set(xlim=(82,101.2),ylim=(45,97),xticks=[85,90,95,100],yticks=[50,70,90],
          xlabel='Construction accommodated (%)',ylabel='Urban loss avoided (%)')
    clean(b);b.xaxis.labelpad=2.5;b.yaxis.labelpad=2.5
    b_records=[];b_summary=[]
    for j,sc in enumerate(SC):
        l=overall[overall.scenario==['SSP2_protected','SSP5_protected'][j]].sort_values('year').copy()
        assert l.year.tolist()==list(range(2026,2101))
        l['accommodated_percent']=l.within_pct+l.relocated_pct
        z=loss[loss.scenario==sc].merge(l[['year','accommodated_percent']],on='year',validate='many_to_one')
        assert len(z)==1125 and z.member_id.nunique()==15
        for _,g in z.groupby('member_id',sort=True):
            g=g.sort_values('year');assert g.year.tolist()==list(range(2026,2101))
            b.plot(g.accommodated_percent,g.avoided_loss_percent,color=COL[sc],lw=.35,alpha=.22,zorder=2)
        s=summary[summary.scenario==sc].merge(l[['year','accommodated_percent']],on='year',validate='one_to_one').sort_values('year')
        b.plot(s.accommodated_percent,s.avoided_loss_percent_q50,color=COL[sc],lw=1.0,ls='-',zorder=4)
        
        for year,marker,size in [(2026,'o',4.0),(2050,'^',4.6),(2100,'s',3.85)]:
            r=s[s.year==year].iloc[0]
            shape=MarkerStyle(marker,capstyle='round',joinstyle='round')
            b.plot(r.accommodated_percent,r.avoided_loss_percent_q50,marker=shape,ms=size,ls='none',
                   mfc=COL[sc] if year==2100 else 'white',
                   mec='white' if year==2100 else COL[sc],mew=.32 if year==2100 else .65,zorder=6)
        b_records+=z.to_dict('records');b_summary+=s.to_dict('records')
    year_handles=[Line2D([0],[0],marker=MarkerStyle(marker,capstyle='round',joinstyle='round'),
        ms=size,ls='none',mfc=mid if year==2100 else 'white',
        mec='white' if year==2100 else mid,mew=.28 if year==2100 else .58,label=str(year))
        for year,marker,size in [(2026,'o',3.3),(2050,'^',3.9),(2100,'s',3.2)]]
    year_key=f.legend(handles=year_handles,loc='center',bbox_to_anchor=(28.25/W,46.7/H),ncol=3,
        fontsize=5.6,labelcolor=mid,handlelength=1.0,handletextpad=.35,columnspacing=1.1,
        borderaxespad=0,borderpad=0)

    
    
    c=axis(58.46,10.5,35.5,33)
    c.set(xlim=(2024,2102),ylim=(0,1550),xticks=[2026,2050,2100],yticks=[0,500,1000,1500],
          xlabel='Year',ylabel='Construction (km²)')
    clean(c);c.xaxis.labelpad=2;c.yaxis.labelpad=2
    c_records=[]
    for j,sc in enumerate(SC):
        l=overall[overall.scenario==['SSP2_protected','SSP5_protected'][j]].sort_values('year').copy()
        assert l.year.tolist()==list(range(2026,2101))
        l['accommodated_km2']=l.in_cell_protected_growth_km2+l.relocated_growth_km2
        assert np.allclose(l.accommodated_km2+l.additional_unmet_growth_km2,l.primary_realized_growth_km2,atol=1e-9,rtol=0)
        c.fill_between(l.year,l.accommodated_km2,l.primary_realized_growth_km2,color=COL[sc],alpha=.105,lw=0,zorder=1)
        for field,style,lw in [('primary_realized_growth_km2',(0,(3.5,1.8)),.65),
                               ('accommodated_km2','-',.95),('relocated_growth_km2',(0,(1,1.8)),.72)]:
            c.plot(l.year,l[field],color=COL[sc],ls=style,lw=lw,zorder=3)
        c_records+=l.to_dict('records')
    role_handles=[Line2D([0],[0],color=mid,lw=lw,ls=style,label=label)
        for label,style,lw in [('Target',(0,(3.5,1.8)),.65),('Built','-',.95),('Moved',(0,(1,1.8)),.72)]]
    role_key=f.legend(handles=role_handles,loc='center',bbox_to_anchor=(76.21/W,46.7/H),ncol=3,
        fontsize=5.6,labelcolor=mid,handlelength=1.75,handletextpad=.4,columnspacing=.8,
        borderaxespad=0,borderpad=0)

    
    
    habitat=base.read('endpoint_2100/paired_habitat_2100_member.csv')
    habitat=habitat[(habitat.year==2100)&(habitat.contrast=='protection_minus_growth')]
    native=base.read('figure_story/Fig4_relative_habitat_2100.csv')
    joined=habitat.merge(native[['member_id','scenario','region_id','year','m_growth','C_growth','valid_centres']],
        on=['member_id','scenario','region_id','year'],validate='one_to_one',suffixes=('','_baseline'))
    assert len(joined)==len(habitat)==330
    assert (joined.valid_centres==joined.valid_centres_baseline).all()
    assert (joined.C_growth>0).all() and (joined.m_growth>0).all()
    joined['relative_capacity_gain_percent']=joined.delta_C_percentage_points/joined.C_growth
    joined['relative_habitat_gain_percent']=joined.delta_m_percentage_points/joined.m_growth
    d=axis(105.5,10.5,35.5,33)
    d.set(xlim=(-1,28),ylim=(-1,28),xticks=[0,10,20],yticks=[0,10,20],
          xlabel='Relative Δm (%)',ylabel='Relative ΔC (%)')
    clean(d);d.xaxis.labelpad=2.5;d.yaxis.labelpad=2.5
    reference_end=float(min(joined.relative_habitat_gain_percent.max(),joined.relative_capacity_gain_percent.max()))
    d.plot([0,reference_end],[0,reference_end],color='#9AA6A4',lw=.36,ls=(0,(2.4,2.4)),zorder=1)
    z=joined[joined.region_id>0];assert len(z)==300
    d.scatter(z.relative_habitat_gain_percent,z.relative_capacity_gain_percent,s=3.4,
        facecolor='#8E9FA1',edgecolors='white',linewidths=.12,alpha=.56,zorder=2)
    for sc in SC:
        z=joined[(joined.region_id==0)&(joined.scenario==sc)];assert len(z)==15
        d.scatter(z.relative_habitat_gain_percent,z.relative_capacity_gain_percent,s=12.0,
            marker=MarkerStyle(base.MARKERS[sc],capstyle='round',joinstyle='round'),
            facecolors=COL[sc],edgecolors='white',linewidths=.32,alpha=.96,zorder=4)
    region_handle=Line2D([0],[0],marker='o',ls='none',ms=2.6,mfc='#8E9FA1',mec='white',mew=.15)
    overall_handles=tuple(Line2D([0],[0],marker=MarkerStyle(base.MARKERS[sc],capstyle='round',joinstyle='round'),
        ls='none',ms=3.15,mfc=COL[sc],mec='white',mew=.28) for sc in SC)
    population_key=d.legend(handles=[region_handle,overall_handles],labels=['Regions','Overall'],
        handler_map={tuple:HandlerTuple(ndivide=None,pad=.35)},loc='upper left',
        borderaxespad=.2,borderpad=.1,fontsize=5.5,handlelength=1.7,handletextpad=.4,labelspacing=.4)
    assert joined.relative_habitat_gain_percent.between(-1,28).all()
    assert joined.relative_capacity_gain_percent.between(-1,28).all()
    d_records=joined.to_dict('records')
    handles=[Line2D([0],[0],color=COL[sc],lw=1.6,ls='-',label=base.LABELS[sc]) for sc in SC]
    f.legend(handles=handles,loc='center',bbox_to_anchor=(W/2/W,2.8/H),ncol=2,fontsize=6.4,
             handlelength=2.4,columnspacing=1.8,handletextpad=.6)
    pd.DataFrame(b_records).to_csv(base.Q/'Fig5_b_member_trajectories.csv',index=False,float_format='%.17g')
    pd.DataFrame(c_records).to_csv(base.Q/'Fig5_c_construction_annual.csv',index=False,float_format='%.17g')
    pd.DataFrame(d_records).to_csv(base.Q/'Fig5_d_relative_habitat.csv',index=False,float_format='%.17g')
    f.canvas.draw()
    for a,key in zip(f.axes[:2],['growth','protected']):
        bnd=a.get_position();map_geometry[key]['actual_mm']=[bnd.x0*W,bnd.y0*H,bnd.width*W,bnd.height*H]
    renderer=f.canvas.get_renderer()
    actual_panel_bounds={}
    for label,a in zip(['b','c','d'],f.axes[2:5]):
        bb=a.get_tightbbox(renderer).transformed(f.transFigure.inverted())
        actual_panel_bounds[label]=[bb.x0*W,bb.y0*H,bb.x1*W,bb.y1*H]
    polished_key_bounds={}
    for name,key in [('years',year_key),('construction_roles',role_key),('populations',population_key)]:
        bb=key.get_window_extent(renderer).transformed(f.transFigure.inverted())
        polished_key_bounds[name]=[bb.x0*W,bb.y0*H,bb.x1*W,bb.y1*H]
    base.save(f,'Fig5_protection_benefit',{'revision':'figure5_protection_layout_v1','preview_only':False,
        'scientific_identity':'UMRI-v1.2.0','panels':['a','b','c','d'],
        'figure_role':'Concentrated spatial placement, the full joint protection trajectory, annual absolute construction accounts, and complete relative habitat responses.',
        'spatial_example':meta,'map_crop':{'row_slice':list(rows),'column_slice':list(cols),'original_extent':original_extent.tolist(),
            'display_extent':extent.tolist(),'criterion':'Use the tighter common spatial window retaining the selected source and both receiving-cell centres; preserve UTM aspect and native pixels.',
            'display_resolution_m':30,'two_maps_same_extent':True,'no_spatial_resampling':True},
        'map_geometry_mm':map_geometry,'map_palette':MAP,'map_legend_position':'One centred row below both maps; background classes defined in caption',
        'map_legend_items':key_boxes,'map_legend_class_ids':[3,4,5,6,7,8],
        'map_background_classes_in_caption':{'0':'Unknown or uncovered: light grey','1':'Water/tidal: pale blue','2':'Other known land: cream'},
        'descriptive_panel_titles':False,
        
        'polish_scope':['b year-symbol contours and optical sizes','c line hierarchy and key spacing','d point edges and faithful population key'],
        'polished_key_bounds_mm':polished_key_bounds,'map_flow_arrows':False,'example_construction_bar':False,
        'scenario_colours':COL,'b_member_year_rows':len(b_records),'b_summary_records':b_summary,
        'b_definition':'Annual median of within-member avoided-loss percentages versus the cumulative construction accommodated share; thin lines retain all 15 members.',
        'b_year_markers':{'2026':'open circle','2050':'open triangle','2100':'filled square'},
        'c_annual_accounts':c_records,'c_annual_positions':len(c_records),
        'c_definition':'Cumulative area since 2025: dashed target, solid protected construction accommodated, dotted relocation. The target–built shaded gap is the unserved amount.',
        'regional_construction_destination':'The single combined supplementary figure retains all regional accounts and names.',
        'd_member_records':d_records,'d_regional_members':300,'d_overall_members':30,
        'd_denominators':'The paired unprotected spatial means of retained habitat and connected capacity in the same region, SSP and member.',
        'd_marker_identity':'Grey points show all regional paired members; larger coloured symbols show all overall paired members. The dashed diagonal marks equal relative changes.',
        'panel_units':{'b':'percent versus percent','c':'year versus km²','d':'relative percent versus relative percent'},
        'data_axes_mm':{'b':[10.5,10.5,35.5,33],'c':[58.46,10.5,35.5,33],'d':[105.5,10.5,35.5,33]},
        'panel_content_bounds_mm':actual_panel_bounds,
        'zero_crosshair_lines_in_d':False,'equal_change_reference_extent':[0,reference_end],'linear_axes_b_c_d':True,
        'formal_publication':True,'supplement_number':'Supplementary Fig. 5'})
