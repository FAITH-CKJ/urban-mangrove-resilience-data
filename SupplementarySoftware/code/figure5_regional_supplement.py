
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle,FancyArrowPatch
from matplotlib.colors import ListedColormap,FuncNorm
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator

W,H=183.,130.
SC=['ssp245','ssp585']
COL={'ssp245':'#0072B2','ssp585':'#D55E00'}
MAP=['#ECEDED','#DCEAF1','#F0EBDF','#347C62','#CAB071','#ACCEB8','#929BA1','#C96C4F','#7865A5']
UNSERVED='#D6DBDC'

def draw(base):
    with mpl.rc_context({'font.family':'Arial','font.size':6.0,'axes.labelsize':6.0,
        'axes.linewidth':.45,'xtick.major.width':.4,'ytick.major.width':.4,
        'xtick.major.size':1.5,'ytick.major.size':1.5,'lines.solid_capstyle':'round',
        'lines.dash_capstyle':'round','path.simplify':False}):
        _draw(base)

def _draw(base):
    f=base.fig(H);ink=base.INK;mid='#627277'
    text=lambda x,y,t,**kw:base.txt(f,x,y+53,t,**kw)
    axis=lambda x,y,w,h:base.ax(f,x,y+53,w,h)
    def panel(x,y,label,title):
        text(x,y,label,fontsize=8,fontweight='bold')
    def clean(a):
        a.grid(False);a.tick_params(labelsize=5.6,pad=2)
    def asinh(a,scale,limits,ticks):
        a.set_xscale('function',functions=(lambda x:np.arcsinh(np.asarray(x)/scale),lambda x:np.sinh(np.asarray(x))*scale))
        a.set_xlim(*limits);a.xaxis.set_major_locator(FixedLocator(ticks));a.xaxis.set_minor_locator(NullLocator())
        a.xaxis.set_major_formatter(FuncFormatter(lambda x,p:('0' if x==0 else f'{x:g}').replace('-','−')))

    
    
    
    panel(3,73.5,'a','Regional construction, resilience and habitat · 2100')
    handles=[Line2D([0],[0],color=COL[sc],marker=MarkerStyle(base.MARKERS[sc],joinstyle='round',capstyle='round'),ms=3.0,mec='white',mew=.25,lw=.85,ls='-',label=base.LABELS[sc]) for sc in SC]
    f.legend(handles=handles,loc='center',bbox_to_anchor=(146/W,126.5/H),ncol=2,fontsize=5.8,handlelength=2,columnspacing=1.4,handletextpad=.5)
    regled=base.read('main_figures/Fig5_allocation_regional_annual.csv')
    published=base.read('main_figures/Fig5_protection_paired_annual.csv')
    habitat=base.read('endpoint_2100/paired_habitat_2100_member.csv')
    def parquet(name):
        p=base.S/name;base.INPUTS[name]=base.sha(p);return pd.read_parquet(p)
    members=parquet('reporting/primary_paired_member_annual.parquet')
    components=parquet('reporting/primary_paired_contributions_UMRI.parquet')
    def subset(d):
        z=d[(d.year==2100)&(d.contrast=='protection_minus_growth')&(d.region_id>0)].copy()
        if 'scheme' in z:z=z[z.scheme=='pooled_centres']
        if 'support' in z:z=z[z.support=='forecast']
        return z
    members,components,habitat=map(subset,[members,components,habitat])
    assert len(members)==len(components)==len(habitat)==300
    y0,hh=10.8,52
    allocation=axis(33, y0,29,hh);benefit=axis(72,y0,29,hh)
    compax=axis(108,y0,21,hh);mh=axis(139,y0,18,hh);ch=axis(165,y0,17,hh)
    allocation.set(xlim=(0,100),xticks=[0,50,100]);asinh(benefit,.001,(-.0014,1.05),[0,.01,.1,1])
    for a in [mh,ch]:asinh(a,.05,(-.25,21),[0,.1,10])
    for a in [allocation,benefit,mh,ch]:
        a.set(ylim=(9.5,-.5),yticks=[]);a.spines['left'].set_visible(False);clean(a)
        if a is not allocation:a.axvline(0,color='#C5CCCD',lw=.4,zorder=1)
    compax.set(xlim=(0,5),ylim=(9.5,-.5));compax.set_axis_off()
    for i,rid in enumerate(base.ORDER):text(29.5,y0+hh*(9.5-i)/10,base.REG[rid],ha='right',va='center',fontsize=5.7)
    for x,label,unit in [(47.5,'Construction','% of target'),(86.5,'Benefit','UMRI'),
                         (118.5,'Contributions',''),(148,'Habitat retained','Δm (pp)'),(173.5,'Connected capacity','ΔC (pp)')]:
        text(x,69,label,ha='center',fontsize=5.9)
        if unit:text(x,65.6,unit,ha='center',fontsize=5.6,color=mid)
    comps=['d','r','g','h','T']
    for k,label in enumerate(['d','r*','g*','h','T']):text(108+(k+.5)*21/5,65.6,label,ha='center',fontsize=5.6)
    norm=FuncNorm((lambda x:np.arcsinh(x/.002),lambda x:np.sinh(x)*.002),vmin=-.6,vmax=.6)
    cmap=mpl.colors.LinearSegmentedColormap.from_list('component_gain',['#B35B16','#F8FAFC','#2166AC'])
    member_records=[];region_records=[];equal=[];no_target=[];component_records=[]
    offsets=np.linspace(-.067,.067,15)
    for i,rid in enumerate(base.ORDER):
        for j,sc in enumerate(SC):
            y=i+(-.18 if j==0 else .18)
            g=members[(members.region_id==rid)&(members.scenario==sc)].sort_values('member_id')
            h=habitat[(habitat.region_id==rid)&(habitat.scenario==sc)].sort_values('member_id')
            co=components[(components.region_id==rid)&(components.scenario==sc)].sort_values('member_id')
            assert g.member_id.tolist()==h.member_id.tolist()==co.member_id.tolist() and len(g)==15
            assert np.allclose(co[[f'delta_{k}_UMRI' for k in comps]].sum(axis=1),g.delta_UMRI,atol=3e-14,rtol=0)
            l=regled[(regled.region_id==rid)&(regled.scenario==['SSP2_protected','SSP5_protected'][j])&(regled.year==2100)].iloc[0]
            target=float(l.primary_realized_growth_km2);is_equal=bool(target>0 and l.additional_unmet_growth_km2==0)
            if target>0:
                amounts=[l.in_cell_protected_growth_km2,l.relocated_growth_km2,l.additional_unmet_growth_km2]
                assert abs(sum(amounts)-target)<1e-8
                left=0
                for value,colour in zip(amounts,[MAP[7],MAP[8],UNSERVED]):
                    share=100*value/target
                    allocation.barh(y,share,left=left,height=.27,color=colour,edgecolor='white',linewidth=.2,zorder=3);left+=share
                if is_equal:
                    allocation.text(103,y,'=',ha='left',va='center',fontsize=6.1,color=ink,clip_on=False)
                    equal.append({'region_id':rid,'scenario':sc,'target_km2':target})
            else:
                allocation.text(50,y,'No target',ha='center',va='center',fontsize=5.6,color=mid)
                no_target.append({'region_id':rid,'scenario':sc,'target_km2':0})
            row_record={'region_id':rid,'scenario':sc,'target_km2':target,'equal_construction':is_equal}
            for a,field,values in [(benefit,'benefit_UMRI',g.delta_UMRI),(mh,'delta_m_pp',h.delta_m_percentage_points),(ch,'delta_C_pp',h.delta_C_percentage_points)]:
                values=values.to_numpy();lo,med,hi=np.quantile(values,[.17,.5,.83])
                assert values.min()>=a.get_xlim()[0] and values.max()<=a.get_xlim()[1]
                a.scatter(values,y+offsets,s=2.0,color=COL[sc],alpha=.4,edgecolors='white',linewidths=.1,zorder=3)
                a.errorbar(med,y,xerr=[[med-lo],[hi-med]],fmt='',marker=MarkerStyle(base.MARKERS[sc],joinstyle='round',capstyle='round'),linestyle='none',ms=2.75,color=COL[sc],
                           mec='white',mew=.28,lw=.7,capsize=.9,zorder=5)
                row_record[field]={'q17':float(lo),'q50':float(med),'q83':float(hi),'min':float(values.min()),'max':float(values.max())}
            q=published[(published.region_id==rid)&(published.scenario==sc)&(published.year==2100)].iloc[0]
            assert np.allclose([row_record['benefit_UMRI'][f'q{k}'] for k in [17,50,83]],
                               [q[f'delta_UMRI_q{k}'] for k in [17,50,83]],atol=1e-13,rtol=0)
            means=co[[f'delta_{k}_UMRI' for k in comps]].mean()
            for k,field in enumerate(comps):
                value=float(means[f'delta_{field}_UMRI']);assert abs(value)<.6
                compax.add_patch(Rectangle((k+.02,y-.145),.96,.29,fc=cmap(norm(value)),ec='#E2E7E5',lw=.2))
                component_records.append({'region_id':rid,'scenario':sc,'component':field,'mean_UMRI':value})
            for z,hhrow,cc in zip(g.itertuples(),h.itertuples(),co.itertuples()):
                member_records.append({'region_id':rid,'scenario':sc,'member_id':z.member_id,'benefit_UMRI':z.delta_UMRI,
                    'delta_m_pp':hhrow.delta_m_percentage_points,'delta_C_pp':hhrow.delta_C_percentage_points,
                    **{f'delta_{k}_UMRI':getattr(cc,f'delta_{k}_UMRI') for k in comps}})
            region_records.append(row_record)
    assert len(equal)==7 and len(no_target)==1
    text(3,8.3,'= Same total built',fontsize=5.6,color=mid)
    for x,colour,label in [(3,MAP[7],'Within cell'),(23.5,MAP[8],'Relocated'),(44,UNSERVED,'Unserved')]:
        f.patches.append(Rectangle((x/W,55.8/H),1.8/W,1.1/H,transform=f.transFigure,fc=colour,ec='none'))
        text(x+2.7,3.35,label,va='center',fontsize=5.6)
    
    
    cbax=axis(108,3.1,21,1.2)
    cb=f.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cbax,orientation='horizontal',ticks=[-.5,0,.5])
    cb.ax.tick_params(labelsize=5.6,pad=1,length=1.2,width=.35);cb.outline.set_linewidth(.3)
    text(118.5,7.0,'Mean ΔUMRI',ha='center',fontsize=5.5,color=mid)
    pd.DataFrame(member_records).to_csv(base.Q/'Fig5_regional_members_2100.csv',index=False,float_format='%.17g')
    pd.DataFrame(component_records).to_csv(base.Q/'Fig5_regional_component_means.csv',index=False,float_format='%.17g')

    
    
    
    
    lower_shift=lambda x:9.2 if x<90 else -9.2
    lowtext=lambda x,y,t,**kw:base.txt(f,x+lower_shift(x),y,t,**kw)
    lowaxis=lambda x,y,w,h:base.ax(f,x+lower_shift(x),y,w,h)
    lowtext(3,48.5,'b',fontsize=8,fontweight='bold')
    lowtext(95,48.5,'c',fontsize=8,fontweight='bold')
    native=base.read('figure_story/Fig4_relative_habitat_2100.csv')
    joined=habitat.merge(native[['member_id','scenario','region_id','year','m_growth','C_growth']],
        on=['member_id','scenario','region_id','year'],validate='one_to_one')
    joined=joined.merge(components[['member_id','scenario','region_id','year','delta_UMRI','ecological_delta_UMRI']],
        on=['member_id','scenario','region_id','year'],validate='one_to_one')
    assert len(joined)==300 and (joined.C_growth>0).all()
    joined['relative_capacity_gain_percent']=joined.delta_C_percentage_points/joined.C_growth
    local=lowaxis(18,12,59,31.5)
    local.set(xlim=(-.25,6.1),ylim=(0,30),xticks=[0,2,4,6],yticks=[0,10,20,30],
        xlabel='Connected-capacity gain (%)',ylabel='Ecological contribution (% of benefit)')
    clean(local);local.xaxis.labelpad=2;local.yaxis.labelpad=2
    component_plot=lowaxis(118,12,61,31.5)
    component_plot.set(ylim=(4.5,-.5),yticks=list(range(5)),
        yticklabels=['Urban pressure','Recovery','Growth','Moisture','Structure'],xlabel='Contribution (% of member benefit)')
    clean(component_plot);component_plot.tick_params(axis='y',length=0,pad=3)
    component_plot.xaxis.labelpad=2;component_plot.spines['left'].set_visible(False)
    local_records=[];share_records=[];local_cases=[];share_values=[]
    for j,(rid,sc,label) in enumerate([(1,'ssp245','Australia / NZ'),(4,'ssp585','Eastern Asia')]):
        z=joined[(joined.region_id==rid)&(joined.scenario==sc)].sort_values('member_id').copy()
        co=components[(components.region_id==rid)&(components.scenario==sc)].sort_values('member_id').copy()
        assert len(z)==len(co)==15 and z.member_id.tolist()==co.member_id.tolist() and (z.delta_UMRI>0).all()
        z['ecological_share_percent']=100*z.ecological_delta_UMRI/z.delta_UMRI
        for row in z.itertuples():
            local.plot(row.relative_capacity_gain_percent,row.ecological_share_percent,marker=MarkerStyle(base.MARKERS[sc],joinstyle='round',capstyle='round'),ms=3.5,
                mfc=COL[sc] if row.delta_m_percentage_points>=0 else 'white',mec=COL[sc],mew=.65,alpha=.9)
        if sc=='ssp245':local.text(.05,14.3,label,fontsize=6,color=COL[sc],ha='left',va='center')
        else:local.text(4.1,28.4,label,fontsize=6,color=COL[sc],ha='center',va='center')
        sums=[]
        for k,field in enumerate(['d','r','g','h','T']):
            values=100*co[f'delta_{field}_UMRI'].to_numpy()/co.delta_UMRI.to_numpy()
            sums.append(values);share_values.extend(values.tolist())
            yy=k+(-.15 if j==0 else .15)
            component_plot.scatter(values,yy+np.linspace(-.06,.06,15),s=3.6,marker=MarkerStyle(base.MARKERS[sc],joinstyle='round',capstyle='round'),
                color=COL[sc],alpha=.46,edgecolors='white',linewidths=.1,zorder=3)
            lo,med,hi=np.quantile(values,[.17,.5,.83])
            component_plot.errorbar(med,yy,xerr=[[med-lo],[hi-med]],fmt='',marker=MarkerStyle(base.MARKERS[sc],joinstyle='round',capstyle='round'),linestyle='none',ms=2.9,
                color=COL[sc],mec='white',mew=.28,lw=.7,capsize=.9,zorder=4)
            for member,value in zip(co.member_id,values):
                share_records.append({'region_id':rid,'scenario':sc,'member_id':member,'component':field,'share_percent':float(value)})
        assert np.allclose(np.stack(sums).sum(axis=0),100,atol=1e-10,rtol=0)
        local_records+=z.to_dict('records')
        local_cases.append({'region_id':rid,'scenario':sc,'mean_benefit_ecological_share_percent':float(100*z.ecological_delta_UMRI.mean()/z.delta_UMRI.mean())})
    low_limit=min(-.2,min(share_values)*1.1)
    ticks=[0,.1,1,10,100]
    if low_limit<=-1:ticks=[-1]+ticks
    elif low_limit<=-.1:ticks=[-.1]+ticks
    asinh(component_plot,.2,(low_limit,110),ticks)
    component_plot.axvline(0,color='#BAC5C3',lw=.4,zorder=0)
    for x,label,fc in [(50,'m↓','white'),(68,'m↑',mid)]:
        f.add_artist(Line2D([(x+lower_shift(x))/W],[45.5/H],marker='o',ms=3,mfc=fc,mec=mid,mew=.6,ls='none',transform=f.transFigure))
        lowtext(x+2,45.5,label,fontsize=5.6,color=mid,va='center')
    pd.DataFrame(local_records).to_csv(base.Q/'equal_construction_relative_members.csv',index=False,float_format='%.17g')
    pd.DataFrame(share_records).to_csv(base.Q/'equal_construction_member_shares.csv',index=False,float_format='%.17g')

    base.save(f,'SupplementaryFig5_regional_protection',{
        'revision':'figure5_protection_layout_v1','preview_only':False,
        'scientific_identity':'UMRI-v1.2.0',
        'regional_order':base.ORDER,'regional_statistics':region_records,
        'regional_member_records':member_records,'regional_component_means':component_records,
        'equal_construction_cases':equal,'undefined_construction_cases':no_target,
        'regional_member_count':300,'habitat_member_count':300,'displayed_member_values':900,
        'component_mean_values':100,'scenario_colours':COL,
        'quantile_definition':'17th, 50th and 83rd percentiles of 15 paired members; linear interpolation',
        'component_definition':'Arithmetic means of exact memberwise native-UMRI component contributions; components sum to mean total effect.',
        'allocation_denominator':'Each region/scenario cumulative unprotected realized construction target; zero target remains undefined.',
        'display_transforms':{'benefit_asinh_scale_UMRI':.001,'habitat_asinh_scale_percentage_points':.05,
            'component_colour_asinh_scale_UMRI':.002,'component_colour_limits_UMRI':[-.6,.6]},
        'local_relative_members':local_records,'local_component_shares':share_records,'local_cases':local_cases,
        'local_component_share_transform':'asinh(percent / 0.2), native percentage ticks',
        'lower_panel_axes_mm':{'b':[27.2,12,59,31.5],'c':[108.8,12,61,31.5]},
        'lower_panel_translation_mm':{'b':9.2,'c':-9.2},
        'panels':['a','b','c'],'descriptive_panel_titles':False,'all_member_extremes_displayed':True,'formal_publication':True,
        'supplement_number':'Supplementary Fig. 5'})
