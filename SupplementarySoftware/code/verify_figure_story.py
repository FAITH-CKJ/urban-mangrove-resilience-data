




from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
pa = argparse.ArgumentParser()
pa.add_argument('--source-data', type=Path, default=ROOT.parent/'SourceData')
pa.add_argument('--output', type=Path, default=ROOT/'reproduced/qa/figure_story_verification.json')
args = pa.parse_args()
S = args.source_data/'figure_story'
checks = []
inputs = {}

def read(name):
    p = S/name
    inputs[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p, float_precision='round_trip') if p.suffix == '.csv' else json.loads(p.read_text('utf-8'))

def close(name, actual, expected, atol=1e-10):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    error = float(np.max(np.abs(a-b)))
    assert np.isfinite(error) and error <= atol, (name, error)
    checks.append({'name': name, 'max_abs_error': error, 'status': 'PASS'})

e = read('historical_native_endpoints.csv')
b = read('historical_native_block_sums.csv')
assert len(e) == 22 and len(b) == 282 and set(e.region_id) == set(range(11))
assert not e.duplicated(['support','region_id']).any()
assert not b.duplicated(['support','block_id']).any()
sumcols = [c for c in b if c.startswith('n_') or c.endswith('_sum')]
max_aggregation_error = 0.
for row in e.itertuples(index=False):
    sub = b[b.support == row.support]
    if row.region_id: sub = sub[sub.region_id == row.region_id]
    for c in sumcols:
        max_aggregation_error = max(max_aggregation_error, abs(float(sub[c].sum()) - getattr(row,c)))
assert max_aggregation_error < 1e-8
checks.append({'name':'22 endpoint rows from 282 block sums', 'status':'PASS', 'max_abs_error':max_aggregation_error})
for v in ['p','m','C']:
    for y in [2000,2025]:
        close(f'{v} {y} native percentage', e[f'{v}_{y}_pct'], 100*e[f'{v}_{y}_sum']/e.n_2000_2025)
    close(f'{v} native endpoint difference',e[f'delta_{v}_pp'],e[f'{v}_2025_pct']-e[f'{v}_2000_pct'])
close('Recovery stock, not rate', e.r_2025_pct, 100*e.r_2025_sum/e.n_2025)
for v in ['EVI','NDMI']:
    close(f'{v} conditional native canopy change',e[f'{v}_delta'],e[f'{v}_delta_sum']/e[f'n_canopy_{v}'])
assert (e.C_2000_pct <= e.m_2000_pct+1e-10).all()
assert (e.C_2025_pct <= e.m_2025_pct+1e-10).all()
assert (e.n_canopy_EVI <= e.n_2025-e.n_absent_2025).all()
assert (e.n_canopy_NDMI <= e.n_2025-e.n_absent_2025).all()
checks.append({'name':'Habitat and conditional-canopy support', 'status':'PASS'})



dist = read('historical_centre_distribution_2025.csv')
assert len(dist) == 33 and not dist.duplicated(['region_id','variable']).any()
assert set(dist.region_id) == set(range(11))
assert set(dist.variable) == {'r_2025_pct','EVI_delta','NDMI_delta'}
available = e[e.support == 'available'].set_index('region_id')
for variable,count_col in [('r_2025_pct','n_2025'),('EVI_delta','n_canopy_EVI'),('NDMI_delta','n_canopy_NDMI')]:
    part = dist[dist.variable == variable].set_index('region_id').loc[available.index]
    close('Centre distribution mean '+variable,part['mean'],available[variable])
    close('Centre distribution support '+variable,part.n,available[count_col],atol=0)
    ordered = part[['minimum','q10','q25','q50','q75','q90','maximum']].to_numpy(float)
    assert np.isfinite(ordered).all() and (np.diff(ordered,axis=1) >= 0).all()
    assert ((part['mean'] >= part.minimum) & (part['mean'] <= part.maximum)).all()
    checks.append({'name':'Centre percentile order and bounds '+variable,'status':'PASS'})
assert (dist[dist.variable == 'r_2025_pct'].minimum >= 0).all()
assert (dist[dist.variable == 'r_2025_pct'].maximum <= 100).all()
comp = read('historical_endpoint_comparison.csv')
ca = comp[comp.support == 'available'].set_index('region_id').loc[available.index]
raw_counts = ca.greener_capacity_lower_pct*ca.n_canopy_2025/100
joint_counts = np.rint(raw_counts)
close('Same-location joint fractions recover integer counts',raw_counts,joint_counts,atol=1e-7)
assert ((joint_counts >= 0) & (joint_counts <= ca.n_greening)).all()
assert int(joint_counts.loc[0]) == 532032 and int(ca.loc[0,'n_greening']) == 713029
close('Conditional capacity decline among greener locations',ca.capacity_decline_among_greening_pct,100*joint_counts/ca.n_greening)
hist_path = S.parent/'main_figures/Fig3_historical_contributions.csv'
inputs['../main_figures/Fig3_historical_contributions.csv'] = hashlib.sha256(hist_path.read_bytes()).hexdigest()
hist = pd.read_csv(hist_path,float_precision='round_trip')
assert len(hist) == 286 and not hist.duplicated(['region_id','year']).any()
assert set(hist.region_id) == set(range(11)) and set(hist.year) == set(range(2000,2026))
close('286 annual regional component budgets',hist[[k+'_change_UMRI' for k in ['d','r','g','h','T']]].sum(axis=1),hist.change_UMRI)


f4_curve = read('Fig4_loss_intensity_exceedance.csv')
f4_summary = read('Fig4_loss_intensity_summary.csv')
f4_method = read('../methods/figure4_parameters.json')
f4_scenarios = ['ssp126','ssp245','ssp370','ssp585']
assert set(f4_curve.scenario) == set(f4_summary.scenario) == set(f4_scenarios)
assert len(f4_summary) == 4 and not f4_curve.duplicated(['scenario','loss_threshold_UMRI']).any()
assert (f4_curve.reference_n == 934814).all()
assert (f4_curve.valid_n == 926220).all() and (f4_curve.unknown_n == 8594).all()
assert (f4_curve.valid_n + f4_curve.unknown_n == f4_curve.reference_n).all()
assert (f4_curve.exceedance_n >= 0).all() and (f4_curve.exceedance_n <= f4_curve.valid_n).all()
assert (f4_curve.unanimous_negative_exceedance_n >= 0).all()
assert (f4_curve.unanimous_negative_exceedance_n <= f4_curve.exceedance_n).all()
for col in ['exceedance_n','unanimous_negative_exceedance_n']:
    close('Integer empirical counts '+col, f4_curve[col], np.rint(f4_curve[col]), atol=0)
assert f4_curve.loss_threshold_UMRI.between(0,2).all()
close('Fixed-reference exceedance percentages', f4_curve.exceedance_pct_reference,
      100*f4_curve.exceedance_n/f4_curve.reference_n)
close('Distinct-cell exceedance areas', f4_curve.exceedance_area_km2,
      .0009*f4_curve.exceedance_n)
close('Unanimous-negative subset percentages', f4_curve.unanimous_negative_exceedance_pct_reference,
      100*f4_curve.unanimous_negative_exceedance_n/f4_curve.reference_n)
for col in ['conditional_median_loss','conditional_loss_q90','conditional_loss_q99','maximum_loss']:
    assert f4_summary[col+'_UMRI'].between(0,2).all()
f4_area = read('../endpoint_2100/pixelwise_median_negative_area_2100.csv')
f4_area = f4_area[(f4_area.region_id==0)&(f4_area.support=='forecast')&
                  (f4_area.metric=='urban_erosion')].set_index('scenario').loc[f4_scenarios]
f4_summary_index = f4_summary.set_index('scenario').loc[f4_scenarios]
for scenario in f4_scenarios:
    z = f4_curve[f4_curve.scenario==scenario].sort_values('loss_threshold_UMRI')
    r = f4_area.loc[scenario]
    t = f4_summary_index.loc[scenario]
    assert z.iloc[0].loss_threshold_UMRI == 0 and z.iloc[-1].loss_threshold_UMRI == 1.6
    assert (np.diff(z.loss_threshold_UMRI)>0).all()
    assert (np.diff(z.exceedance_n)<=0).all() and z.iloc[-1].exceedance_n == 0
    assert (np.diff(z.unanimous_negative_exceedance_n)<=0).all()
    close('Zero threshold affected counts '+scenario,
          [z.iloc[0].exceedance_n,t.affected_n],[r.negative_n,r.negative_n],atol=0)
    close('Zero threshold affected area '+scenario,
          [z.iloc[0].exceedance_pct_reference,z.iloc[0].exceedance_area_km2],
          [r.negative_pct_reference,r.negative_area_km2])
    close('Summary affected proportion '+scenario,t.affected_pct_reference,r.negative_pct_reference)
    close('Zero threshold unanimous-negative counts '+scenario,
          [z.iloc[0].unanimous_negative_exceedance_n,t.unanimous_negative_n],
          [r.all_members_negative_n,r.all_members_negative_n],atol=0)
    med = z[z.loss_threshold_UMRI == t.conditional_median_loss_UMRI]
    assert len(med)==1
    close('Median marker lies on empirical curve '+scenario,med.exceedance_pct_reference,
          [t.exceedance_at_conditional_median_pct_reference],atol=0)
    assert 0<t.conditional_median_loss_UMRI<=t.conditional_loss_q90_UMRI
    assert t.conditional_loss_q90_UMRI<=t.conditional_loss_q99_UMRI<=t.maximum_loss_UMRI<2
    close('Fixed reference classes '+scenario,
          r.negative_area_km2+r.zero_area_km2+r.positive_area_km2+r.unknown_area_km2,
          r.reference_area_km2)
checks.append({'name':'Four monotone empirical exceedance profiles and unknown-support semantics','status':'PASS'})
for rel, expected_rows in [('Fig4_urban_paired_annual.csv',3300),('Fig4_urban_paired_cumulative.csv',44)]:
    f4_paired = read('../main_figures/'+rel)
    assert len(f4_paired)==expected_rows
    if not rel.endswith('annual.csv'):
        for q in ['q17','q50','q83']:
            close('Cumulative UMRI-years '+q,f4_paired['delta_UMRI_'+q],f4_paired['delta_UMRI_years_'+q],atol=1e-12)
    assert (f4_paired.delta_UMRI_q17<=f4_paired.delta_UMRI_q50).all()
    assert (f4_paired.delta_UMRI_q50<=f4_paired.delta_UMRI_q83).all()
    if rel.endswith('annual.csv'):
        assert ((f4_paired.delta_UMRI_q17>-2)&(f4_paired.delta_UMRI_q83<2)).all()
assert f4_method['asinh_parameters']=={'c_UMRI':0.01,'map_colour_UMRI':0.01,'e_percent':5.0}
assert len(f4_curve)==f4_method['exceedance_rows']
for rel,digest in f4_method['source_sha256'].items():
    assert hashlib.sha256((args.source_data/rel).read_bytes()).hexdigest()==digest
checks.append({'name':'Figure 4 source files match method metadata SHA-256','status':'PASS'})


f4_map = read('Fig4_local_urban_effect_1deg.csv')
keys = ['region_id','longitude_bin','latitude_bin']
assert len(f4_map)==292 and not f4_map.duplicated(['scenario']+keys).any()
assert set(f4_map.scenario)==set(f4_scenarios)
f4_ref = read('../main_figures/Fig2_reference_distribution_1deg.csv').set_index(keys).sort_index()
assert len(f4_ref)==73
for scenario in f4_scenarios:
    z=f4_map[f4_map.scenario==scenario].set_index(keys).sort_index()
    assert list(z.index)==list(f4_ref.index)
    close('Map reference support '+scenario,z.reference_n,f4_ref.reference_centres,atol=0)
    close('Map coordinates '+scenario,z[['longitude','latitude']],f4_ref[['longitude','latitude']])
    close('Map class closure '+scenario,z.negative_n+z.zero_n+z.positive_n+z.unknown_n,z.reference_n,atol=0)
    close('Map validity closure '+scenario,z.valid_n+z.unknown_n,z.reference_n,atol=0)
    r=f4_area.loc[scenario]
    for col in ['reference_n','valid_n','unknown_n','negative_n','zero_n','positive_n','all_members_negative_n']:
        close('Map full-support '+col+' '+scenario,z[col].sum(),r[col],atol=0)
    close('Map fixed-reference area '+scenario,z.reference_area_km2,.0009*z.reference_n)
    assert z.longitude.between(-112,180).all() and z.latitude.between(-42,34).all()
    valid=z.valid_n>0
    assert np.isfinite(z.loc[valid,'mean_local_delta_UMRI']).all()
    assert z.loc[~valid,'mean_local_delta_UMRI'].isna().all()
    quantiles=z.loc[valid,['min_local_delta_UMRI','local_delta_UMRI_q10','median_local_delta_UMRI','local_delta_UMRI_q90','max_local_delta_UMRI']].to_numpy(float)
    assert (np.diff(quantiles,axis=1)>=0).all()
    assert (z.loc[valid,'mean_local_delta_UMRI']>=z.loc[valid,'min_local_delta_UMRI']).all()
    assert (z.loc[valid,'mean_local_delta_UMRI']<=z.loc[valid,'max_local_delta_UMRI']).all()
f4_components=read('../main_figures/Fig3_urban_UMRI_contributions_2100.csv')
f4_components=f4_components[f4_components.region_id==0].set_index('scenario').loc[f4_scenarios]
close('Four mean component budgets',f4_components[[f'delta_{k}_UMRI_mean' for k in ['d','r','g','h','T']]].sum(axis=1),f4_components.delta_UMRI_mean)
for row in f4_method['component_budget']:
    z=f4_components.loc[row['scenario']]
    for key in ['d','T','r','g','h']:
        close('Component mean '+key+' '+row['scenario'],row['component_means'][key],z[f'delta_{key}_UMRI_mean'])
    close('Ecological percentage '+row['scenario'],row['ecological_share_percent'],z.ecological_share_percent)
f4_members=read('../endpoint_2100/paired_habitat_2100_member.csv')
f4_members=f4_members[(f4_members.contrast=='urban_growth_minus_fixed')&(f4_members.year==2100)]
assert len(f4_members)==660 and f4_members.groupby(['scenario','region_id']).size().eq(15).all()
assert not f4_members.duplicated(['scenario','region_id','member_id']).any()
f4_native=read('../endpoint_2100/paired_habitat_2100_summary.csv')
f4_native=f4_native[f4_native.contrast=='urban_growth_minus_fixed'].set_index(['scenario','region_id'])
for (scenario,rid),rows in f4_members.groupby(['scenario','region_id']):
    for key in ['m','C']:
        col=f'delta_{key}_percentage_points'
        close('Native member quantiles '+str(rid)+' '+scenario+' '+key,np.quantile(rows[col],[.17,.5,.83]),f4_native.loc[(scenario,rid),[col+'_q17',col+'_q50',col+'_q83']])
f4_path=S.parent/'reporting/primary_paired_contributions_UMRI.parquet'
inputs['../reporting/primary_paired_contributions_UMRI.parquet']=hashlib.sha256(f4_path.read_bytes()).hexdigest()
f4_cm=pd.read_parquet(f4_path)
f4_cm=f4_cm[(f4_cm.year==2100)&(f4_cm.region_id==0)&(f4_cm.contrast=='urban_growth_minus_fixed')]
assert len(f4_cm)==60 and f4_cm.groupby('scenario').size().eq(15).all()
close('All 60 signed member contribution budgets',f4_cm[[f'delta_{k}_UMRI' for k in ['d','T','r','g','h']]].sum(axis=1),f4_cm.delta_UMRI)
for scenario,rows in f4_cm.groupby('scenario'):
    for key in ['d','T','r','g','h']:
        close('Member contribution mean '+key+' '+scenario,rows[f'delta_{key}_UMRI'].mean(),f4_components.loc[scenario,f'delta_{key}_UMRI_mean'])
assert f4_method['habitat_member_counts']=={'regional':600,'overall':60} and f4_method['contribution_member_points']==300

f4_relative=read('Fig4_relative_habitat_2100.csv').set_index(['scenario','region_id','member_id']).sort_index()
native_members=f4_members.set_index(['scenario','region_id','member_id']).sort_index()
assert len(f4_relative)==660 and f4_relative.index.equals(native_members.index)
assert f4_relative.index.is_unique
for key in ['m','C']:
    assert np.isfinite(f4_relative[key+'_fixed2025']).all() and (f4_relative[key+'_fixed2025']>0).all()
    close('Relative habitat preserves absolute '+key,f4_relative['delta_'+key+'_percentage_points'],native_members['delta_'+key+'_percentage_points'])
    close('Relative habitat native difference '+key,100*(f4_relative[key+'_growth']-f4_relative[key+'_fixed2025']),native_members['delta_'+key+'_percentage_points'])
    close('Relative habitat within-member denominator '+key,f4_relative['delta_'+key+'_relative_percent'],100*(f4_relative[key+'_growth']-f4_relative[key+'_fixed2025'])/f4_relative[key+'_fixed2025'])
for row in f4_method['habitat_overall_relative_percent']:
    values=f4_relative.loc[(row['scenario'],0),'delta_'+row['variable']+'_relative_percent']
    close('Relative habitat quantiles '+row['scenario']+' '+row['variable'],np.quantile(values,[.17,.5,.83]),[row['q17'],row['median'],row['q83']])
f4_share=read('Fig4_relative_contributions_2100.csv').set_index(['scenario','member_id']).sort_index()
absolute_components=f4_cm.set_index(['scenario','member_id']).sort_index()
assert f4_share.index.is_unique and f4_share.index.equals(absolute_components.index)
assert len(f4_share)==60 and (f4_share.delta_UMRI<0).all()
close('Relative contributions preserve member totals',f4_share.delta_UMRI,absolute_components.delta_UMRI)
for key in ['d','T','r','g','h']:
    close('Relative contribution preserves absolute '+key,f4_share['delta_'+key+'_UMRI'],absolute_components['delta_'+key+'_UMRI'])
    close('Relative contribution common denominator '+key,f4_share[key+'_loss_share_percent'],100*absolute_components['delta_'+key+'_UMRI']/absolute_components.delta_UMRI)
close('Relative contribution five shares sum to 100',f4_share[[k+'_loss_share_percent' for k in ['d','T','r','g','h']]].sum(axis=1),np.full(60,100.))
assert f4_method['linear_axes']==['b','b cumulative inset','d']
assert f4_method['habitat_unit'].startswith('Relative percent, 100*(growth-fixed2025)/fixed2025')
assert f4_method['absolute_habitat_source_unit'].startswith('Percentage points, 100*(growth-fixed2025)')
checks.append({'name':'Relative habitat and contribution denominators are defined and retain all 960 values','status':'PASS'})
checks.append({'name':'All 292 spatial rows, 660 habitat member pairs and 300 signed contribution values preserve accepted support and values','status':'PASS'})


f5_native = read('../main_figures/Fig5_paired_native_annual.csv')
f5_members = read('../main_figures/Fig5_loss_members_annual.csv')
f5_summary = read('../main_figures/Fig5_loss_summary_annual.csv')
f5_phase = read('../main_figures/Fig5_displayed_allocation_trajectories.csv')
f5_regions = read('../main_figures/Fig5_displayed_regions_2100.csv')
f5_method = read('../methods/figure5_parameters.json')
assert len(f5_native)==4500 and len(f5_members)==2250 and len(f5_summary)==150
assert set(f5_native.scheme)=={'pooled_centres'} and set(f5_native.support)=={'forecast'}
assert set(f5_native.region_id)=={0}
f5_keys = ['scenario','member_id','year']
assert not f5_native.duplicated(f5_keys+['contrast']).any()
assert not f5_members.duplicated(f5_keys).any()
assert not f5_summary.duplicated(['scenario','year']).any()
assert set(f5_native.scenario)=={'ssp245','ssp585'}
assert set(f5_native.year)==set(range(2026,2101))
assert f5_members.groupby(['scenario','year']).size().eq(15).all()
f5_pivot = f5_native.pivot(index=f5_keys,columns='contrast',values='delta_UMRI')
f5_aligned = f5_members.set_index(f5_keys).loc[f5_pivot.index]
f5_loss = -f5_pivot.urban_growth_minus_fixed
f5_benefit = f5_pivot.protection_minus_growth
assert (f5_loss>0).all()
close('Figure 5 annual urban loss from native paired contrasts',f5_aligned.urban_loss_UMRI,f5_loss)
close('Figure 5 annual protection benefit from native paired contrasts',f5_aligned.protection_benefit_UMRI,f5_benefit)
close('Figure 5 annual protected residual within member',f5_aligned.residual_loss_UMRI,f5_loss-f5_benefit)
close('Figure 5 annual avoided percentage within member',f5_aligned.avoided_loss_percent,100*f5_benefit/f5_loss)
f5_quantile_error = 0.
for (scenario,year),group in f5_members.groupby(['scenario','year']):
    row=f5_summary[(f5_summary.scenario==scenario)&(f5_summary.year==year)].iloc[0]
    assert row.members==15
    for variable in ['urban_loss_UMRI','protection_benefit_UMRI',
                     'residual_loss_UMRI','avoided_loss_percent']:
        observed=np.array([row[f'{variable}_q{q}'] for q in [17,50,83]])
        expected=np.quantile(group[variable],[.17,.5,.83],method='linear')
        f5_quantile_error=max(f5_quantile_error,float(np.max(np.abs(observed-expected))))
assert f5_quantile_error<1e-10
checks.append({'name':'Figure 5 all annual quantiles after member-wise arithmetic','status':'PASS','max_abs_error':f5_quantile_error})
f5_endpoint=read('../endpoint_2100/protection_avoided_loss_2100_member.csv')
f5_endpoint=f5_endpoint.merge(f5_members[f5_members.year==2100],on=['scenario','member_id'],
                             suffixes=('_published','_annual'),validate='one_to_one')
assert len(f5_endpoint)==30
for variable in ['urban_loss_UMRI','protection_benefit_UMRI',
                 'residual_loss_UMRI','avoided_loss_percent']:
    close('Figure 5 existing 2100 member identity '+variable,
          f5_endpoint[variable+'_published'],f5_endpoint[variable+'_annual'])
f5_protection=read('../main_figures/Fig5_protection_paired_annual.csv')
f5_overall=f5_protection[f5_protection.region_id==0].set_index(['scenario','year'])
f5_annual=f5_summary.set_index(['scenario','year']).loc[f5_overall.index]
for q in [17,50,83]:
    close('Figure 5 annual absolute benefit identity q'+str(q),
          f5_annual[f'protection_benefit_UMRI_q{q}'],f5_overall[f'delta_UMRI_q{q}'])
f5_ledger=read('../main_figures/Fig5_allocation_overall_annual.csv')
assert len(f5_phase)==150 and not f5_phase.duplicated(['scenario','year']).any()
f5_accounts=f5_ledger.set_index(['scenario','year']).loc[f5_phase.set_index(['scenario','year']).index]
f5_shares=f5_phase.set_index(['scenario','year'])
for column in ['primary_realized_growth_km2','in_cell_protected_growth_km2','relocated_growth_km2',
               'protected_served_growth_km2','additional_unmet_growth_km2']:
    close('Figure 5 cumulative account identity '+column,f5_shares[column],f5_accounts[column],atol=0)
assert (f5_accounts.primary_realized_growth_km2>0).all()
close('Figure 5 cumulative target closure',
      f5_accounts.in_cell_protected_growth_km2+f5_accounts.relocated_growth_km2+f5_accounts.additional_unmet_growth_km2,
      f5_accounts.primary_realized_growth_km2,atol=1e-8)
for share,column in [('within_pct','in_cell_protected_growth_km2'),('relocated_pct','relocated_growth_km2'),
                     ('unserved_pct','additional_unmet_growth_km2')]:
    close('Figure 5 annual target-weighted share '+share,f5_shares[share],
          100*f5_accounts[column]/f5_accounts.primary_realized_growth_km2)
close('Figure 5 allocation shares close to 100',f5_shares.within_pct+f5_shares.relocated_pct+f5_shares.unserved_pct,
      np.full(150,100.))
f5_regledger=read('../main_figures/Fig5_allocation_regional_annual.csv')
assert len(f5_regions)==20 and f5_regions.unserved_pct.notna().sum()==19
assert not f5_regions.duplicated(['region_id','scenario']).any()
assert set(f5_regions.region_id)==set(range(1,11))
f5_region_error=0.
for row in f5_regions.itertuples(index=False):
    pro=f5_protection[(f5_protection.region_id==row.region_id)&(f5_protection.scenario==row.scenario)&
                      (f5_protection.year==2100)].iloc[0]
    scenario={'ssp245':'SSP2_protected','ssp585':'SSP5_protected'}[row.scenario]
    ledger_row=f5_regledger[(f5_regledger.region_id==row.region_id)&(f5_regledger.scenario==scenario)&
                            (f5_regledger.year==2100)].iloc[0]
    assert row.target_km2==ledger_row.primary_realized_growth_km2
    for q in [17,50,83]:
        f5_region_error=max(f5_region_error,abs(getattr(row,f'benefit_q{q}_UMRI')-pro[f'delta_UMRI_q{q}']))
    if row.target_km2==0:
        assert row.region_id==2 and row.scenario=='ssp585' and row.benefit_q50_UMRI==0
        assert np.isnan(row.unserved_pct)
    else:
        f5_region_error=max(f5_region_error,abs(row.unserved_pct-100*ledger_row.additional_unmet_growth_km2/row.target_km2))
assert f5_region_error<1e-10
checks.append({'name':'Figure 5 twenty regional benefits and nineteen defined percentages','status':'PASS',
               'max_abs_error':f5_region_error})
f5_full_target_counts={}
for scenario in ['SSP2_protected','SSP5_protected']:
    z=f5_regledger[(f5_regledger.scenario==scenario)&(f5_regledger.year==2100)]
    z=z[z.primary_realized_growth_km2>0]
    f5_full_target_counts[scenario]={'positive_target_regions':len(z),
        'fully_accommodated_regions':int((z.additional_unmet_growth_km2==0).sum())}
assert f5_full_target_counts=={
    'SSP2_protected':{'positive_target_regions':10,'fully_accommodated_regions':4},
    'SSP5_protected':{'positive_target_regions':9,'fully_accommodated_regions':3}}
checks.append({'name':'Figure 5 full-accommodation counts exclude zero construction targets','status':'PASS',
               'counts':f5_full_target_counts})
for rel,digest in f5_method['source_sha256'].items():
    assert hashlib.sha256((args.source_data/rel).read_bytes()).hexdigest()==digest,rel
checks.append({'name':'Figure 5 source hashes and adopted display semantics','status':'PASS'})

meta = read('Fig5_spatial_example.json')
npz = S/'Fig5_spatial_example.npz'
inputs[npz.name] = hashlib.sha256(npz.read_bytes()).hexdigest()
assert inputs[npz.name] == meta['npz_sha256']
with np.load(npz, allow_pickle=False) as z:
    assert all(z[k].shape == z['base'].shape for k in ['growth','protected','coverage','classes','reference_B','mangrove','corridor','growth_year','protected_in_cell_year','protected_final_year'])
    assert set(np.unique(z['base'])) <= set(range(7))
    assert set(np.unique(z['growth'])) <= set(range(9))
    assert set(np.unique(z['protected'])) <= set(range(9))
    assert int(z['crs']) == 32756 and int(z['source_cell_id']) == meta['source_cell_id']
    close('Map extent', z['extent'], meta['extent'])
    assert z['transform'][0] == 30 and z['transform'][4] == -30
    assert float(z['coverage'].mean()) == meta['coverage_fraction'] == 1.
    for c,n in meta['category_counts'].items(): assert int((z['base']==int(c)).sum()) == n
ledger = meta['selected_ledger']
close('Illustrative source-cell construction balance',ledger['primary_realized_growth_km2'], ledger['in_cell_protected_growth_km2']+ledger['relocated_growth_km2']+ledger['additional_unmet_growth_km2'])
checks.append({'name':'Spatial example: exact payload, native 30-m local grid and recorded selection', 'status':'PASS'})

method = read('Fig6_theory_methods.json')
method['settings'] = json.loads((args.source_data/method['settings_file']).read_text('utf8'))
setting = method['settings']; w = setting['weights']
boundary = read('Fig6_retention_boundary.csv')
assert (boundary.m > 0).all() and boundary.m.max() <= 1
tmax = (boundary.m-boundary.C0)/(boundary.m+boundary.C0)
gmax = 1+np.minimum(0,np.minimum(boundary.h,tmax))
close('Retention boundary structural ceiling',boundary.T_max,tmax)
close('Retention boundary gate ceiling',boundary.G_max,gmax)
close('Retention-pressure necessary outer boundary',boundary.p_cap,(w['h']*boundary.h+w['T']*tmax+gmax*(w['r']*boundary.m+w['g']))/w['d'])
c = read('Fig6_reconnection_control.csv'); r = setting['reconnection']
t = (c.C-r['C0'])/(c.C+r['C0']); gate = 1+np.minimum(0,np.minimum(r['h'],t))
base = -w['d']*r['urban_pressure_p'] + w['h']*r['h'] + w['T']*t
response = w['r']*r['r'] + w['g']*r['g'] if 'g' in r else w['r']*r['r'] + w['g']*r['g_positive']
dynamic = base+gate*response; frozen = base+gate.iloc[0]*response
lam = setting['normalization_lambda']
ud = dynamic/(lam+np.abs(dynamic)); uf = frozen/(lam+np.abs(frozen))
for name,expected in [('T',t),('G',gate),('S_dynamic',dynamic),('S_frozen',frozen),('UMRI_dynamic',ud),('UMRI_frozen',uf),('gain_dynamic',ud-ud.iloc[0]),('gain_frozen',uf-uf.iloc[0]),('additional_gain_from_gate',ud-uf)]:
    close('Controlled response '+name,c[name],expected)
assert np.isclose(c.C.min(),0.5) and np.isclose(c.C.max(),0.7)
assert np.isclose(c.C.to_numpy(),36/55,rtol=0,atol=1e-14).sum() == 1
close('Controlled total gain',c.gain_dynamic.iloc[-1],method['reconnection_total_gain_UMRI'])
close('Dynamic versus frozen difference',c.additional_gain_from_gate.iloc[-1],method['additional_gain_dynamic_minus_frozen_UMRI'])

B = {tuple(x) for x in r['historical_cells']}
initial = {tuple(x) for x in r['current_before']}
bridge = {tuple(x) for x in r['new_link_cells']}
assert len(B)==10 and len(bridge)==2 and not B.intersection(bridge)
def capacity(habitat):
    unvisited = set(habitat); sizes=[]
    while unvisited:
        start=unvisited.pop(); stack=[start]; component={start}
        while stack:
            x,y=stack.pop()
            for n in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                if n in unvisited: unvisited.remove(n); component.add(n); stack.append(n)
        sizes.append(len(component.intersection(B)))
    return sorted(sizes), np.sqrt(sum(n*n for n in sizes))/len(B)
before,cb = capacity(initial); after,ca = capacity(initial|bridge)

assert before == [3,4] and after == [7]
close('Independent four-neighbour geometry capacity',[cb,ca],[.5,.7])
assert len(B&initial)==len(B&(initial|bridge))==7
checks.append({'name':'Historical retention preserved by outside-reference bridge', 'status':'PASS'})
from verify_figure6_fields import verify as verify_figure6_fields
figure6_fields=verify_figure6_fields(args.source_data)
checks.extend(figure6_fields['checks'])
inputs.update(figure6_fields['input_sha256'])

from figure5_equal_construction import derive,verify_published,SOURCE_FILES,PUBLISHED
table7=derive(S.parent);published_table7=verify_published(S.parent,table7)
method7=read('../methods/figure5_equal_construction.json')
assert method7['evidence_location']=='Supplementary Fig. 5b,c and Supplementary Table 7' and method7['panels']==['a','b','c']
for name in SOURCE_FILES+[PUBLISHED]:
    digest=hashlib.sha256((S.parent/name).read_bytes()).hexdigest()
    assert digest==method7['source_sha256'][name]
    inputs['../'+name]=digest
for row in table7.to_dict('records'):
    close('Table 7 equal construction '+row['scenario'],row['construction_unprotected_km2'],row['construction_protected_km2'])
    close('Table 7 mean benefit budget '+row['scenario'],row['urban_pressure_mean_UMRI']+row['ecological_mean_UMRI'],row['mean_UMRI_benefit'])
    close('Table 7 shares total '+row['scenario'],row['urban_pressure_share_percent']+row['ecological_share_percent'],100.)
checks.append({'name':'Supplementary Table 7 preserves both complete equal-construction cases and Figure 5 a-d','status':'PASS'})
report={'status':'PASS','scientific_identity':'UMRI-v1.2.0','figure_story_revision':'figure_story_reproduction_v1','checks':checks,'input_sha256':inputs,'figure5_member_years':len(f5_members),'figure5_annual_summaries':len(f5_summary),'figure5_allocation_positions':len(f5_phase),'figure5_regional_benefits':len(f5_regions),'figure5_defined_regional_percentages':int(f5_regions.unserved_pct.notna().sum()),'figure5_full_target_counts':f5_full_target_counts,'figure4_exceedance_rows':len(f4_curve),'figure4_summary_rows':len(f4_summary),'centre_distribution_rows':len(dist),'historical_endpoint_rows':len(e),'historical_blocks':141,'boundary_rows':len(boundary),'controlled_response_rows':len(c),'figure6_grid_counts':figure6_fields['grid_counts'],'scope':'Published numerical sources, including observation and model-evaluation tables.'}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['checks','input_sha256']}|{'checks_passed':len(checks)}))
