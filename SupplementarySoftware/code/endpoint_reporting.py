
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

SCENARIOS = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
LABELS = ['SSP1--2.6', 'SSP2--4.5', 'SSP3--7.0', 'SSP5--8.5']

def read(base, name):
    return pd.read_csv(Path(base) / 'endpoint_2100' / name, float_precision='round_trip')

def verify(base):
    base = Path(base)
    area = read(base, 'pixelwise_median_negative_area_2100.csv')
    member = read(base, 'member_negative_area_2100.csv')
    quant = read(base, 'member_area_quantiles_2100.csv')
    endpoints = read(base, 'regional_endpoint_2100.csv')
    assert len(area) == len(quant) == 176 and len(member) == 2640 and len(endpoints) == 80
    assert (area.negative_n + area.zero_n + area.positive_n == area.valid_n).all()
    assert (area.valid_n + area.unknown_n == area.reference_n).all()
    assert (member.negative_n + member.zero_n + member.positive_n == member.valid_n).all()
    assert (member.valid_n + member.unknown_n == member.reference_n).all()
    for frame in [area, member]:
        for name in ['reference', 'valid', 'unknown', 'negative', 'zero', 'positive']:
            np.testing.assert_allclose(frame[name + '_area_km2'], frame[name + '_n'] * 0.0009, rtol=0, atol=3e-12)
        np.testing.assert_allclose(frame.negative_pct_reference, 100 * frame.negative_n / frame.reference_n, rtol=0, atol=3e-12)
        np.testing.assert_allclose(frame.negative_pct_valid, 100 * frame.negative_n / frame.valid_n, rtol=0, atol=3e-12)
        g = frame.query('region_id==0 and support=="forecast"')
        assert g.reference_n.eq(934814).all() and g.valid_n.eq(926220).all() and g.unknown_n.eq(8594).all()
    keys = ['scenario', 'support', 'metric', 'region_id']
    q = quant.set_index(keys)
    for key, g in member.groupby(keys):
        assert len(g) == 15
        for field in ['negative_area_km2', 'negative_pct_reference', 'negative_pct_valid']:
            np.testing.assert_allclose(g[field].quantile([.17,.5,.83]), [q.loc[key, field + '_' + v] for v in ['q17','q50','q83']], rtol=0, atol=3e-12)
    counts = endpoints.assign(negative=endpoints.UMRI_q50 < 0).groupby(['support','scenario']).negative.sum()
    assert counts.eq(9).all()
    assert endpoints.loc[endpoints.UMRI_q50 >= 0, 'region_id'].eq(8).all()
    native = read(base, 'paired_habitat_2100_member.csv')
    summary = read(base, 'paired_habitat_2100_summary.csv').set_index(['scenario','contrast','region_id'])
    for key, g in native.groupby(['scenario','contrast','region_id']):
        assert len(g) == 15
        for field in ['delta_m_percentage_points','delta_C_percentage_points']:
            np.testing.assert_allclose(g[field].quantile([.17,.5,.83]), [summary.loc[key, field+'_'+v] for v in ['q17','q50','q83']], rtol=0, atol=3e-12)
    global_urban = native.query('region_id==0 and contrast=="urban_growth_minus_fixed"')
    assert len(global_urban) == 60 and (global_urban[['delta_m_percentage_points','delta_C_percentage_points']] < 0).all().all()
    avoided = read(base, 'protection_avoided_loss_2100_member.csv')
    paired = pd.read_parquet(base / 'reporting/primary_paired_member_annual.parquet').query('support=="forecast" and region_id==0 and year==2100')
    urban = paired.query('contrast=="urban_growth_minus_fixed"').set_index(['scenario','member_id'])
    protect = paired.query('contrast=="protection_minus_growth"').set_index(['scenario','member_id'])
    for _, row in avoided.iterrows():
        key = (row.scenario,row.member_id)
        loss = -urban.loc[key, 'delta_UMRI']
        benefit = protect.loc[key, 'delta_UMRI']
        np.testing.assert_allclose([row.urban_loss_UMRI, row.protection_benefit_UMRI, row.avoided_loss_percent, row.residual_loss_UMRI], [loss,benefit,100*benefit/loss,loss-benefit], rtol=0, atol=3e-12)
    aq = read(base, 'protection_avoided_loss_2100_summary.csv').set_index('scenario')
    for scenario, g in avoided.groupby('scenario'):
        assert len(g) == 15
        for field in ['urban_loss_UMRI','protection_benefit_UMRI','avoided_loss_percent','residual_loss_UMRI']:
            np.testing.assert_allclose(g[field].quantile([.17,.5,.83]), [aq.loc[scenario,field+'_'+v] for v in ['q17','q50','q83']], rtol=0, atol=3e-12)
    annual = read(base, 'regional_negative_counts_2026_2100.csv')
    assert len(annual) == 600 and annual.negative_regions.ge(8).all()
    assert annual.query('year==2100').negative_regions.eq(9).all()
    for support, end in [('forecast',2084),('matched2025',2086)]:
        low = annual.query('support==@support and negative_regions<9')
        assert set(low.scenario) == {'ssp370'} and low.year.tolist() == list(range(2075,end+1))
    files = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((base/'endpoint_2100').glob('*.csv'))}
    return {'status':'PASS','identity':'UMRI-v1.2.0','endpoint_year':2100,'area_rows':len(area),'member_area_rows':len(member),'native_member_rows':len(native),'avoided_loss_member_rows':len(avoided),'negative_regions_each_pathway':9,'source_csv_sha256':files,'area_estimand':'Area classified by pixelwise ensemble median','companion_estimand':'Quantiles of areas computed within each member','global_area':area.query('region_id==0 and support=="forecast"').to_dict('records'),'avoided_loss':aq.reset_index().to_dict('records')}

def supplementary_table(base, table):
    verify(base)
    area = read(base, 'pixelwise_median_negative_area_2100.csv').query('support=="forecast" and region_id==0').set_index(['scenario','metric'])
    from derive_endpoint_overlap import summarize
    overlap = summarize(base)
    supplied = read(base, 'state_and_urban_loss_overlap_2100.csv')
    pd.testing.assert_frame_equal(overlap, supplied, check_exact=True)
    overlap = overlap.query('support=="forecast" and region_id==0').set_index('scenario')
    rows = []
    for s, label in zip(SCENARIOS,LABELS):
        a = area.loc[(s,'state_negative')]; b = area.loc[(s,'urban_erosion')]
        rows.append([label,f'{a.negative_area_km2:.4f}',f'{a.negative_pct_reference:.2f}',f'{b.negative_area_km2:.4f}',f'{b.negative_pct_reference:.2f}',f'{overlap.loc[s, "joint_area_km2"]:.4f}',f'{overlap.loc[s, "joint_pct_additional_urban_loss"]:.2f}'])
    text = table('Spatial extent of resilience deficits and additional urban losses in 2100.','endpoint_area','lrrrrrr',['Pathway',r'\shortstack{Resilience deficit\\(km$^2$)}',r'\shortstack{Reference area\\(\%)}',r'\shortstack{Additional urban loss\\(km$^2$)}',r'\shortstack{Reference area\\(\%)}',r'\shortstack{Overlap\\(km$^2$)}',r'\shortstack{Overlap / urban\\loss (\%)}'],rows,r'Resilience deficit denotes negative pixelwise ensemble-median UMRI in the urban-growth branch. Additional urban loss denotes negative pixelwise ensemble medians of within-member growth-minus-fixed-2025 differences. Overlap is the area satisfying both conditions; its percentage uses the additional urban-loss area as denominator. Reference-area percentages use the fixed historical reference area of 841.3326 km$^2$ (934,814 distinct 30-m equal-area cells). Forecasts are valid for 833.5980 km$^2$; the remaining 7.7346 km$^2$ is unknown and retained in the denominator. Signs use archived score precision; unknown cells are separate from zero. Supplementary Method S4 specifies the estimands.')
    q = read(base,'member_area_quantiles_2100.csv').query('support=="forecast" and region_id==0').set_index(['scenario','metric'])
    rows = []
    for s,label in zip(SCENARIOS,LABELS):
        fields=[]
        for metric in ['state_negative','urban_erosion']:
            r=q.loc[(s,metric)]
            fields.append(f'{r.negative_pct_reference_q50:.2f} [{r.negative_pct_reference_q17:.2f}, {r.negative_pct_reference_q83:.2f}]')
        rows.append([label]+fields)
    text += '\n'+r'\begin{center}\small\begin{tabular}{lrr}\toprule Pathway & Resilience deficit (\%) & Additional urban loss (\%)\\\midrule'+'\n'+'\n'.join(' & '.join(r)+r'\\' for r in rows)+r'\bottomrule\end{tabular}\end{center}'+'\n'
    text += r'Companion estimates summarize the area calculated separately in each of the 15 members: median [17th, 83rd percentile], as percentages of the fixed reference area. These quantify ensemble variation in member-level affected area; they are a different estimand from the area of the pixelwise-median map above. Regional values, positive and zero areas, unknown coverage and both forecast supports are supplied in Source Data.'+'\n'
    return text

if __name__=='__main__':
    package=Path(__file__).resolve().parents[1]
    report=verify(package.parent/'SourceData')
    out=package/'reproduced/qa/endpoint_2100_verification.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('ENDPOINT_2100_REPORTING_PASS',report['area_rows'],report['member_area_rows'])
