




from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,Polygon
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter,FixedLocator,NullLocator
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from scipy.spatial import ConvexHull

SC=['ssp126','ssp245','ssp370','ssp585']
COL=dict(zip(SC,['#0072B2','#B8860B','#009E49','#D44732']))
MARK=dict(zip(SC,['o','D','^','s']))
STYLE=dict(zip(SC,[(0,(5,2.4)),(0,(2.5,1.8)),(0,(.7,1.8)),(0,(4.5,1.6,.8,1.6))]))
ORDER=[5,2,6,9,3,10,8,7,4,1]
INK='#263B40';MID='#64797D';GRID='#E7ECEB'

def draw(F):
    H=216.;f=F.fig(H)
    def A(x,y,w,h,**kw):return F.ax(f,x,y,w,h,**kw)
    def T(x,y,s,**kw):return F.txt(f,x,y,s,**kw)
    def head(x,y,l,title):
        T(x,y,l,fontsize=8,fontweight='bold',va='baseline')
        
    fmt=FuncFormatter(lambda v,p:('0' if v==0 else f'{v:g}').replace('-','−'))
    annual=F.read('main_figures/Fig4_urban_paired_annual.csv')
    cumulative=F.read('main_figures/Fig4_urban_paired_cumulative.csv')
    native=F.read('endpoint_2100/paired_habitat_2100_summary.csv')
    native=native[native.contrast=='urban_growth_minus_fixed']
    intensity=F.read('figure_story/Fig4_loss_intensity_exceedance.csv')
    summary=F.read('figure_story/Fig4_loss_intensity_summary.csv').set_index('scenario').loc[SC]
    spatial=F.read('figure_story/Fig4_local_urban_effect_1deg.csv')
    comp=F.read('main_figures/Fig3_urban_UMRI_contributions_2100.csv')
    comp_all=comp.copy()
    comp=comp[comp.region_id==0].set_index('scenario').loc[SC]
    area=F.read('endpoint_2100/pixelwise_median_negative_area_2100.csv')
    area=area[(area.region_id==0)&(area.support=='forecast')&(area.metric=='urban_erosion')].set_index('scenario').loc[SC]
    head(4.7,212.1,'a','Local effect of urban expansion in 2100')

    
    pc=ccrs.PlateCarree();cart=Path(__file__).resolve().parent.parent/'cartography'
    country=cart/'ne_110m_admin_0_countries.shp'
    for ext in ['shp','shx','dbf','prj','cpg']:
        p=country.with_suffix('.'+ext);F.INPUTS['../SupplementarySoftware/cartography/'+p.name]=F.sha(p)
    p=cart/'ne_110m_admin_0_countries_provenance.json';F.INPUTS['../SupplementarySoftware/cartography/'+p.name]=F.sha(p)
    geoms=list(Reader(country).geometries())
    ys=[167.,118.,69.,20.]
    cmap=LinearSegmentedColormap.from_list('paired_urban_loss',['#EDBB00','#46B887','#00A5BD','#0D73C9','#193785'])
    scale=.01;upper=.8
    map_rows=[];zero_display_rows=[]
    for i,(s,y) in enumerate(zip(SC,ys)):
        ma=A(11,y,166,166*76/292,projection=pc)
        ma.set_extent([-112,180,-42,34],crs=pc)
        ma.set_facecolor('white')
        ma.add_geometries(geoms,pc,facecolor='white',edgecolor='#9BA6AA',linewidth=.28,zorder=0)
        ma.spines['geo'].set_visible(False)
        ma.set_yticks([-20,0,20],crs=pc)
        ma.set_yticklabels(['20° S','0°','20° N'],fontsize=5.8)
        ma.set_xticks([-100,-50,0,150],crs=pc)
        ma.set_xticklabels(['100° W','50° W','0°','150° E'] if i==3 else [],fontsize=5.8)
        ma.tick_params(length=1.4,width=.4,pad=2)
        if i<3:ma.set_xticks([],crs=pc)
        ma.set_ylabel('')
        if i==3:ma.set_xlabel('Longitude',fontsize=6.3,labelpad=2)
        
        header_y=y+45.9
        f.add_artist(Line2D([13./183,20.5/183],[header_y/H]*2,
            transform=f.transFigure,color=COL[s],lw=1.05,ls=STYLE[s],clip_on=False))
        f.add_artist(Line2D([16.75/183],[header_y/H],transform=f.transFigure,
            color=COL[s],lw=0,marker=MARK[s],ms=2.8,clip_on=False))
        T(23.,y+45.1,F.LABELS[s],fontsize=6.6,color=INK)
        m=spatial[spatial.scenario==s].copy().sort_values('mean_local_delta_UMRI',key=lambda x:x.abs())
        assert len(m)==73 and m.reference_n.sum()==934814 and m.valid_n.sum()==926220
        val=m.mean_local_delta_UMRI.to_numpy(float)
        loss=val<0;zero=val==0;positive=val>0;unknown=~np.isfinite(val)
        assert np.nanmin(val)>=-.8 and np.nanmax(val)<.001
        mag=np.arcsinh(-val[loss]/scale)/np.arcsinh(upper/scale)
        ma.scatter(m.loc[loss,'longitude'],m.loc[loss,'latitude'],s=11.5,c=mag,cmap=cmap,vmin=0,vmax=1,
                   ec='white',lw=.27,transform=pc,zorder=4)
        
        
        assert (m.loc[zero,'valid_n']>0).all()
        assert (m.loc[zero,'zero_n']==m.loc[zero,'valid_n']).all()
        f.canvas.draw()
        zero_points=ma.transData.transform(m.loc[zero,['longitude','latitude']].to_numpy(float))
        nonzero_points=ma.transData.transform(m.loc[loss|positive,['longitude','latitude']].to_numpy(float))
        clear_pt=(np.sqrt(11.5)+np.sqrt(9.))/2+.45
        distance=np.linalg.norm(zero_points[:,None,:]-nonzero_points[None,:,:],axis=2).min(axis=1)*72/f.dpi
        visible_zero=distance>clear_pt
        zd=m.loc[zero].copy();zd['nearest_nonzero_distance_pt']=distance;zd['drawn']=visible_zero
        zd['reason']=np.where(visible_zero,'valid exact zero; no nonzero symbol overlap','nonzero symbol takes precedence')
        zero_display_rows.extend(zd[['scenario','region_id','longitude_bin','latitude_bin','longitude','latitude','reference_n','valid_n','unknown_n','zero_n','nearest_nonzero_distance_pt','drawn','reason']].to_dict('records'))
        ma.scatter(zd.loc[visible_zero,'longitude'],zd.loc[visible_zero,'latitude'],s=9.,
                   facecolors='none',edgecolors='#74868D',lw=.5,transform=pc,zorder=3)
        ma.scatter(m.loc[positive,'longitude'],m.loc[positive,'latitude'],s=10.5,marker='+',c='#3E5461',lw=.55,transform=pc,zorder=6)
        np.testing.assert_allclose(m.negative_n.sum(),area.loc[s,'negative_n'],atol=0)
        map_rows.append({'scenario':s,'display_cells':len(m),'reference_n':int(m.reference_n.sum()),
            'valid_n':int(m.valid_n.sum()),'unknown_n':int(m.unknown_n.sum()),'negative_centres':int(m.negative_n.sum()),
            'zero_centres':int(m.zero_n.sum()),'positive_centres':int(m.positive_n.sum()),
            'negative_display_cells':int(loss.sum()),'exact_zero_cells':int(zero.sum()),
            'exact_zero_cells_drawn':int(visible_zero.sum()),'exact_zero_cells_not_drawn':int((~visible_zero).sum()),
            'unknown_cells_not_drawn':int(unknown.sum()),'positive_display_cells':int(positive.sum()),
            'displayed_locations':m.loc[loss|positive,['longitude','latitude']].to_dict('records')+zd.loc[visible_zero,['longitude','latitude']].to_dict('records'),
            'minimum_cell_mean':float(np.nanmin(val)),'maximum_cell_mean':float(np.nanmax(val)),'axes':ma})

    T(3.2,118,'Latitude',fontsize=6.5,rotation=90,va='center',ha='center')

    
    cb_label=T(78,9.2,'Local ΔUMRI',fontsize=5.8,color=INK,ha='center',va='bottom')
    cb=A(52,6.4,52,1.6)
    cb.imshow(np.linspace(1,0,500)[None,:],aspect='auto',cmap=cmap,extent=(0,1,0,1))
    positions=1-np.arcsinh(np.array([.8,.1,.01,0])/scale)/np.arcsinh(upper/scale)
    cb.set(yticks=[],xticks=positions,xticklabels=['−0.8','−0.1','−0.01','0'])
    cb.tick_params(labelsize=5.4,length=1,pad=1)
    for sp in cb.spines.values():sp.set_visible(False)
    symbol_legend=f.legend(handles=[
        Line2D([0],[0],color='#3E5461',lw=0,marker='+',mew=.55,ms=3.3,label='Positive'),
        Line2D([0],[0],color='#74868D',lw=0,marker='o',mfc='none',
               mec='#74868D',mew=.5,ms=3,label='Exact zero')],loc='center left',
        bbox_to_anchor=(110./183,7.4/H),fontsize=5.6,handlelength=1,handletextpad=.55,
        labelspacing=.55,borderpad=0,borderaxespad=0)

    
    right_x=108.8;right_w=29.0;inset_h=16.0
    head(right_x-5.6,ys[0]+19.7,'b','Urban loss through time')
    b=A(right_x,ys[0]+1.7,right_w,inset_h)
    overall=annual[annual.region_id==0]
    for s in SC:
        z=overall[overall.scenario==s].sort_values('year');assert z.year.tolist()==list(range(2026,2101))
        b.fill_between(z.year,z.delta_UMRI_q17,z.delta_UMRI_q83,color=COL[s],alpha=.13,lw=0)
        b.plot(z.year,z.delta_UMRI_q50,color=COL[s],ls=STYLE[s],lw=.95,dash_capstyle='round')
    b.set(xlim=(2026,2100),ylim=(-.24,.007),xticks=[2030,2060,2100],yticks=[-.2,-.1,0],ylabel='ΔUMRI')
    b.yaxis.set_major_formatter(fmt);b.tick_params(labelsize=5.5,pad=1.6,length=1.6)
    b.set_ylabel('')
    b.axhline(0,color=MID,lw=.4)
    ci=A(59.,ys[0]+5.8,22.,10.5)
    cum=cumulative[cumulative.region_id==0].set_index('scenario').loc[SC]
    for j,s in enumerate(SC):
        z=cum.loc[s]
        ci.errorbar(z.delta_UMRI_q50,j,xerr=[[z.delta_UMRI_q50-z.delta_UMRI_q17],[z.delta_UMRI_q83-z.delta_UMRI_q50]],
                    fmt=MARK[s],color=COL[s],ms=2.7,lw=.65,capsize=1.,mec='white',mew=.2)
    ci.set(xlim=(-7.6,0),ylim=(3.55,-.55),xticks=[-7.5,0],yticks=[])
    ci.xaxis.set_major_formatter(fmt);ci.tick_params(labelsize=5.5,pad=1.5,length=1.5)
    ci.spines['left'].set_visible(False);ci.spines['bottom'].set_linewidth(.35)
    ci.set_xlabel('UMRI-years',fontsize=5.3,labelpad=2,color=MID)

    
    head(right_x-5.6,ys[1]+19.7,'c','Loss extent and intensity')
    c=A(right_x,ys[1]+1.7,right_w,inset_h)
    F.signed_asinh(c,.01,[0,.01,.1,1],(0,1.6))
    c.set(ylim=(0,55),yticks=[0,20,40],ylabel='Reference area (%)')
    c.tick_params(labelsize=5.5,pad=1.6,length=1.6)
    c.set_xlabel('Local UMRI loss',fontsize=6.1,labelpad=2)
    c.set_ylabel('');c.set_title('Reference area (%)',loc='left',fontsize=5.7,pad=1.8)
    f.canvas.draw()
    for s in SC:
        z=intensity[intensity.scenario==s].sort_values('loss_threshold_UMRI')
        np.testing.assert_allclose(z.iloc[0].exceedance_pct_reference,area.loc[s,'negative_pct_reference'],atol=1e-12)
        assert (np.diff(z.exceedance_n)<=0).all() and z.iloc[-1].exceedance_n==0
        c.plot(z.loss_threshold_UMRI,z.exceedance_pct_reference,color=COL[s],ls=STYLE[s],lw=.95,dash_capstyle='round')
        med=summary.loc[s,'conditional_median_loss_UMRI'];yy=summary.loc[s,'exceedance_at_conditional_median_pct_reference']
        
        c.plot(med,yy,marker=MARK[s],ms=3.1,mfc=COL[s],mec='white',mew=.25,zorder=6)

    
    head(right_x-5.6,ys[2]+19.7,'d','')
    da=A(right_x,ys[2]+1.7,right_w,inset_h)
    da.set(xlim=(-22,2.2),ylim=(-22,2.2),xticks=[-20,-10,0],yticks=[-20,-10,0])
    da.xaxis.set_major_formatter(fmt);da.yaxis.set_major_formatter(fmt)
    da.tick_params(labelsize=5.3,pad=1.8,length=1.6)
    da.set_xlabel('Δm (%)',fontsize=5.8,labelpad=2);da.set_ylabel('')
    d_measure_label=da.text(.035,.965,'ΔC (%)',transform=da.transAxes,
        fontsize=5.4,va='top',ha='left',color=INK,zorder=8)
    for spine in da.spines.values():spine.set_visible(True);spine.set_linewidth(.4)
    diag=np.linspace(-22,2.2,300);da.plot(diag,diag,color='#A9B9BC',lw=.55,ls=(0,(2,2)),zorder=0)
    habitat_member=F.read('figure_story/Fig4_relative_habitat_2100.csv')
    assert len(habitat_member)==660 and habitat_member.groupby(['scenario','region_id']).size().eq(15).all()
    for k in ['m','C']:
        assert (habitat_member[k+'_fixed2025']>0).all()
        np.testing.assert_allclose(habitat_member[f'delta_{k}_relative_percent'],100*(habitat_member[k+'_growth']-habitat_member[k+'_fixed2025'])/habitat_member[k+'_fixed2025'],atol=1e-12)
    regional=habitat_member[habitat_member.region_id>0]
    da.scatter(regional.delta_m_relative_percent,regional.delta_C_relative_percent,s=2.7,color='#899B9F',alpha=.55,ec='none',zorder=1)
    habitat=[];habitat_regions=[];habitat_member_summary=[]
    for scenario in SC:
        z=habitat_member[(habitat_member.region_id==0)&(habitat_member.scenario==scenario)].sort_values('member_id')
        xy=z[['delta_m_relative_percent','delta_C_relative_percent']].to_numpy(float)
        assert len(xy)==15 and ((xy>-22)&(xy<.8)).all()
        hull=ConvexHull(xy)
        da.add_patch(Polygon(xy[hull.vertices],closed=True,facecolor=COL[scenario],edgecolor=COL[scenario],alpha=.11,lw=.6,zorder=2))
        da.scatter(xy[:,0],xy[:,1],s=4.6,marker=MARK[scenario],c=COL[scenario],alpha=.92,ec='white',lw=.12,zorder=3)
        for key,col in [('m',0),('C',1)]:
            lo,mid,hi=np.quantile(xy[:,col],[.17,.5,.83]);habitat.append({'variable':key,'scenario':scenario,'median':float(mid),'q17':float(lo),'q83':float(hi),'unit':'percent relative to fixed2025'})
        mid=np.median(xy,axis=0);da.plot(*mid,marker=MARK[scenario],ms=4.2,mfc=COL[scenario],mec='white',mew=.5,zorder=5)
        habitat_member_summary.append({'scenario':scenario,'members':15,'convex_envelope_vertices':xy[hull.vertices].tolist(),'minimum':xy.min(axis=0).tolist(),'maximum':xy.max(axis=0).tolist()})
    for (scenario,rid),rr in regional.groupby(['scenario','region_id']):
        habitat_regions.append({'scenario':scenario,'region_id':int(rid),'delta_m_median_relative_percent':float(rr.delta_m_relative_percent.median()),'delta_C_median_relative_percent':float(rr.delta_C_relative_percent.median())})

    
    
    head(right_x-5.6,ys[3]+19.7,'e','')
    ea=A(right_x,ys[3]+1.7,right_w,inset_h)
    ea.set_yscale('function',functions=(lambda x:np.arcsinh(np.asarray(x)/5),lambda x:np.sinh(np.asarray(x))*5))
    keys=['d','T','r','g','h']
    ea.set(xlim=(-.55,4.55),ylim=(-2,100),xticks=range(5),
           xticklabels=['Pressure','Structure','Recovery','Growth','Moisture'])
    ea.yaxis.set_major_locator(FixedLocator([0,5,25,100]));ea.yaxis.set_minor_locator(NullLocator())
    ea.yaxis.set_major_formatter(fmt);ea.tick_params(labelsize=5.3,pad=1.6,length=1.6)
    ea.tick_params(axis='x',length=0,pad=1.4)
    for tick in ea.get_xticklabels():tick.set_rotation(43);tick.set_ha('right');tick.set_rotation_mode('anchor')
    ea.axhline(0,color='#8FA5AB',lw=.5,ls=(0,(3,2)),zorder=1)
    ea.set_ylabel('');ea.set_title('Loss share (%)',loc='left',fontsize=5.8,pad=1.8)
    path=F.S/'reporting/primary_paired_contributions_UMRI.parquet'
    F.INPUTS['reporting/primary_paired_contributions_UMRI.parquet']=F.sha(path)
    contributions=F.pd.read_parquet(path)
    members=contributions[(contributions.year==2100)&(contributions.region_id==0)&
                          (contributions.contrast=='urban_growth_minus_fixed')].copy().sort_values(['scenario','member_id'])
    assert len(members)==60 and members.groupby('scenario').size().eq(15).all()
    np.testing.assert_allclose(members[[f'delta_{k}_UMRI' for k in ['d','r','g','h','T']]].sum(axis=1),members.delta_UMRI,atol=1e-12)
    relative_contrib=F.read('figure_story/Fig4_relative_contributions_2100.csv').set_index(['scenario','member_id']).loc[members.set_index(['scenario','member_id']).index].reset_index()
    assert (members.delta_UMRI<0).all()
    np.testing.assert_allclose(relative_contrib[[k+'_loss_share_percent' for k in keys]].sum(axis=1),100,atol=1e-12)
    f.canvas.draw();packing=[];member_contribution_ranges={}
    
    def offsets(vals,base,diameter_pt=1.0):
        yy=ea.transData.transform(np.column_stack([np.full(len(vals),base),vals]))[:,1]
        step=diameter_pt*f.dpi/72*1.06
        placed=[];xx=np.zeros(len(vals))
        for k in np.argsort(yy,kind='stable'):
            candidates=[0.]
            for px,py in placed:
                dy=yy[k]-py
                if abs(dy)<step:
                    dx=np.sqrt(step**2-dy**2)+1e-7
                    candidates.extend([px+dx,px-dx])
            possible=[dx for dx in sorted(candidates,key=lambda x:(abs(x),x))
                      if all((dx-px)**2+(yy[k]-py)**2>=step**2-1e-8 for px,py in placed)]
            assert possible,'Beeswarm packing has no free categorical offset'
            dx=possible[0];xx[k]=dx;placed.append((dx,yy[k]))
        px=ea.transData.transform([[base,0]])[0,0]
        result=ea.transData.inverted().transform(np.column_stack([px+xx,yy]))[:,0]
        
        
        spread=float(np.max(np.abs(result-base)))
        if spread>.43:result=base+(result-base)*(.43/spread)
        assert np.max(np.abs(result-base))<.47
        return result
    for j,key in enumerate(keys):
        values=relative_contrib[key+'_loss_share_percent'].to_numpy(float)
        np.testing.assert_allclose(values,100*members[f'delta_{key}_UMRI'].to_numpy()/members.delta_UMRI.to_numpy(),atol=1e-12)
        assert (values>-2).all() and (values<100).all()
        xx=offsets(values,j)
        for scenario in SC:
            keep=(members.scenario==scenario).to_numpy()
            ea.scatter(xx[keep],values[keep],s=2.4,marker=MARK[scenario],color=COL[scenario],alpha=1.0,
                       ec='white',lw=.06,zorder=4)
        packing.append({'component':key,'points':60,'maximum_categorical_offset':float(np.max(np.abs(xx-j)))})
        member_contribution_ranges[key]={'minimum':float(values.min()),'maximum':float(values.max())}
    budgets=[];regional_profiles=[]
    for scenario in SC:
        z=comp.loc[scenario];sub=members[members.scenario==scenario]
        for key in ['d','T','r','g','h']:np.testing.assert_allclose(sub[f'delta_{key}_UMRI'].mean(),z[f'delta_{key}_UMRI_mean'],atol=1e-12)
        direct=float(z.delta_d_UMRI_mean);structure=float(z.delta_T_UMRI_mean)
        other=float(z.delta_r_UMRI_mean+z.delta_g_UMRI_mean+z.delta_h_UMRI_mean)
        budgets.append({'scenario':scenario,'mean_delta_UMRI':float(z.delta_UMRI_mean),'urban_pressure':direct,
            'habitat_structure':structure,'other_ecology':other,'ecological_share_percent':float(z.ecological_share_percent),
            'component_means':{k:float(z[f'delta_{k}_UMRI_mean']) for k in ['d','T','r','g','h']}})
    footer_legend=da.legend(handles=[Line2D([0],[0],color=INK,lw=0,marker='o',ms=3.1,label='Overall'),
                      Line2D([0],[0],color='#9DACAE',lw=0,marker='o',ms=1.7,label='Regional')],
             loc='lower right',bbox_to_anchor=(.96,.08),ncol=1,fontsize=5.3,
             handlelength=.8,handletextpad=.5,labelspacing=.55,borderpad=0,borderaxespad=.2)

    
    f.canvas.draw();renderer=f.canvas.get_renderer()
    def footer_bounds():
        boxes=[cb.get_tightbbox(renderer),cb_label.get_window_extent(renderer),symbol_legend.get_window_extent(renderer)]
        return min(b.x0 for b in boxes)/f.bbox.width*183,max(b.x1 for b in boxes)/f.bbox.width*183
    bounds=footer_bounds();shift=(94.-(bounds[0]+bounds[1])/2)/183
    pos=cb.get_position();cb.set_position([pos.x0+shift,pos.y0,pos.width,pos.height])
    for text in [cb_label]:
        x,y=text.get_position();text.set_position((x+shift,y))
    symbol_legend.set_bbox_to_anchor((110./183+shift,7.4/H),transform=f.transFigure)
    f.canvas.draw();footer_ink_bounds=footer_bounds()
    assert abs(sum(footer_ink_bounds)/2-94.)<.02
    key_box=footer_legend.get_window_extent(renderer)
    right_key_centre=(key_box.x0+key_box.x1)/2/f.bbox.width*183
    
    habitat_xy=habitat_member[['delta_m_relative_percent','delta_C_relative_percent']].to_numpy(float)
    hp=da.transData.transform(habitat_xy)
    key_margin=f.dpi/72*2.3
    d_key_hits=((hp[:,0]>key_box.x0-key_margin)&(hp[:,0]<key_box.x1+key_margin)&
                (hp[:,1]>key_box.y0-key_margin)&(hp[:,1]<key_box.y1+key_margin))
    assert not d_key_hits.any(), 'Panel d key overlaps a member point'
    d_label_box=d_measure_label.get_window_extent(renderer)
    d_label_hits=((hp[:,0]>d_label_box.x0-key_margin)&(hp[:,0]<d_label_box.x1+key_margin)&
                  (hp[:,1]>d_label_box.y0-key_margin)&(hp[:,1]<d_label_box.y1+key_margin))
    assert not d_label_hits.any(), 'Panel d variable label overlaps a member point'
    d_point_margin=np.column_stack([hp[:,0]-da.bbox.x0,da.bbox.x1-hp[:,0],
                                   hp[:,1]-da.bbox.y0,da.bbox.y1-hp[:,1]])*72/f.dpi
    assert d_point_margin.min()>2.3, 'A panel d marker reaches the plot frame'
    assert da.bbox.contains(key_box.x0,key_box.y0) and da.bbox.contains(key_box.x1,key_box.y1)
    assert da.bbox.contains(d_label_box.x0,d_label_box.y0) and da.bbox.contains(d_label_box.x1,d_label_box.y1)

    
    from shapely.geometry import box
    inset_checks=[]
    for row,inset,panel_name in [(map_rows[0],b,'b annual'),(map_rows[0],ci,'b cumulative'),
                               (map_rows[1],c,'c'),(map_rows[2],da,'d'),(map_rows[3],ea,'e')]:
        ma=row['axes'];scenario=row['scenario']
        bbox=inset.get_tightbbox(renderer)
        candidates=F.pd.DataFrame(row['displayed_locations'])
        pts=ma.transData.transform(candidates[['longitude','latitude']].to_numpy(float))
        radius=(11.5**.5)/2*f.dpi/72
        
        
        boxes=[inset.bbox]
        texts=[inset.title,inset._left_title,inset.xaxis.label,inset.yaxis.label,*inset.get_xticklabels(),*inset.get_yticklabels()]
        texts += [t for t in f.texts if t.get_text()==panel_name[0]]
        boxes += [t.get_window_extent(renderer) for t in texts if t.get_visible() and t.get_text()]
        hits=np.zeros(len(pts),dtype=bool)
        for bb in boxes:
            hits |= ((pts[:,0]>bb.x0-radius)&(pts[:,0]<bb.x1+radius)&(pts[:,1]>bb.y0-radius)&(pts[:,1]<bb.y1+radius))
        geo=ma.transData.inverted().transform([[inset.bbox.x0,inset.bbox.y0],[inset.bbox.x1,inset.bbox.y1]])
        footprint=box(geo[0,0],geo[0,1],geo[1,0],geo[1,1])
        land=sum(g.intersection(footprint).area for g in geoms if g.intersects(footprint))
        inset_checks.append({'scenario':scenario,'panel':panel_name,'points_overlapped':int(hits.sum()),'axes_geographic_bounds':geo.ravel().tolist(),'land_fraction_under_axes':float(land/footprint.area),'tight_bbox_mm':[z/f.dpi*25.4 for z in bbox.bounds],'overlapped_point_locations':candidates.loc[hits,['longitude','latitude']].to_dict('records')})
        assert not hits.any(),inset_checks[-1]

    
    endpoints=[]
    for rid in [0]+ORDER:
        for s in SC:
            z=annual[(annual.year==2100)&(annual.region_id==rid)&(annual.scenario==s)].iloc[0]
            endpoints.append({'variable':'UMRI','scenario':s,'region_id':rid,
                'median':float(z.delta_UMRI_q50),'q17':float(z.delta_UMRI_q17),'q83':float(z.delta_UMRI_q83)})
            z=native[(native.region_id==rid)&(native.scenario==s)].iloc[0]
            for key in ['m','C']:
                p=f'delta_{key}_percentage_points'
                endpoints.append({'variable':key,'scenario':s,'region_id':rid,'median':float(z[p+'_q50']),
                                  'q17':float(z[p+'_q17']),'q83':float(z[p+'_q83'])})
    f.canvas.draw()
    for row in map_rows:
        pos=row.pop('axes').get_position()
        row['actual_axes_mm']=[pos.x0*183,pos.y0*H,pos.width*183,pos.height*H]
    assert len(endpoints)==132
    F.save(f,'Fig4_urban_effect',{
        'scope':'Spatial paired urban effects, native habitat responses and exact ecological contribution profiles.',
        'layout':'Four enlarged same-extent maps; b-e compare all pathways in Indian Ocean whitespace, with cumulative b separated into South Atlantic whitespace.',
        'template_reference':'Nature 10.1038/s41586-024-07702-8 Fig.1: information in ocean whitespace; inherited four-row comparison from Nature 10.1038/s41586-025-08663-2 Fig.2. No reference data or filtering is imported.',
        'scenario_colours':COL,'scenario_styles_match_figure2':True,
        'UMRI_display_unit':'Final dimensionless UMRI=N(S)','UMRI_cumulative_unit':'UMRI-years',
        'annual_years':[2026,2100],'annual_curves':4,'cumulative_member_first':True,
        'interval':'17th–83rd percentiles of 15 paired ensemble members, not confidence intervals',
        'all_endpoint_intervals':endpoints,'habitat_overall_relative_percent':habitat,'component_budget':budgets,
        'component_statistic':'Shares of total additional urban loss within each member: 100 times each signed component change divided by the negative total change. Positive shares increase loss; negative shares offset loss; each five-component budget sums to 100 percent. All 300 values retain exact categorical-only offsets.',
        'asinh_parameters':{'c_UMRI':.01,'map_colour_UMRI':.01,'e_percent':5.},'linear_axes':['b','b cumulative inset','d'],
        'habitat_regional_points':habitat_regions,'habitat_member_summary':habitat_member_summary,
        'habitat_member_counts':{'regional':600,'overall':60},'contribution_member_points':300,
        'contribution_member_ranges':member_contribution_ranges,'beeswarm_packing':packing,
        'panel_d_reference':'Nature 10.1038/s41586-024-08430-9 Fig.2b: complete joint point clouds; no fitted association is imported.',
        'panel_e_reference':'Nature 10.1038/s41586-024-08308-w Fig.4c: all individual points in compact grouped distributions; no test or boxplot definition is imported.',
        'panel_titles_present':False,'inline_numerical_annotations':False,
        'footer_ink_bounds_mm':footer_ink_bounds,'footer_centre_mm':sum(footer_ink_bounds)/2,
        'right_key_centre_mm':right_key_centre,'footer_baseline_mm':7.4,
        'legend_layout':'Scenario line/marker keys integrated into the four row headings; one centred map scale below all maps; Overall/Regional key inside the empty lower-right of d.',
        'panel_d_legend_point_overlaps':int(d_key_hits.sum()),
        'panel_d_label_point_overlaps':int(d_label_hits.sum()),
        'panel_d_minimum_point_frame_clearance_pt':float(d_point_margin.min()),
        'panel_d_limits':[-22,2.2],
        'panel_d_label_and_key_inside_frame':True,
        'panel_e_category_dividers':False,'panel_e_zero_line_style':'dashed',
        'b_descriptive_titles_in_caption_only':True,
        'right_axes_mm':[[ax.get_position().x0*183,ax.get_position().y0*H,ax.get_position().width*183,ax.get_position().height*H] for ax in [b,c,da,ea]],
        'inset_placement_checks':inset_checks,'all_insets_compare_all_scenarios':True,'map_rows':map_rows,
        'zero_display_audit':zero_display_rows,'zero_nonzero_clearance_pt':clear_pt,
        'map_projection':'Plate Carree','map_extent':[-112,180,-42,34],
        'map_statistic':'Mean of within-centre 15-member median paired UMRI effects over valid equal-area reference centres in each region-specific 1-degree display cell.',
        'map_symbol_size':'Equal-size symbols at contributing centres mean coordinates; no count or area encoding.',
        'habitat_relative_reference':'fixed2025 branch at 2100, within each member after spatial pooling','contribution_share_sum':100,'habitat_intervals_unclipped':True,'map_drawing_order':'Ascending effect magnitude; stronger negative effects drawn last.',
        'map_colour_range':[-.8,0],'map_positive_preserved':True,
        'map_zero_rule':'Hollow grey circles for exact-zero valid-centre effects where no nonzero map symbol overlaps. Zero rows hidden by nonzero symbols remain in Source Data. Unknown centres are never recoded.',
        'map_unknown_not_imputed_or_drawn':True,
        'map_background':'White land and ocean; Natural Earth outlines #9BA6AA at 0.28 pt.',
        'native_habitat_scope':'Panel d shows 600 regional and 60 Overall memberwise 100*(growth-fixed2025)/fixed2025 habitat changes. Each branch uses the same spatial support before ratios; all denominators are positive. Coloured polygons enclose the 15 Overall member points in each pathway; larger markers combine the two marginal medians. Envelopes are not confidence regions.',
        'map_and_primary_regional_estimator_differ_explicitly':True,
        'area_statistic':'Exact full-support exceedance of negative within-centre paired-member median effects.',
        'no_new_prediction_model_or_inferential_test':True})
