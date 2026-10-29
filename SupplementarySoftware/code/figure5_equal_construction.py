
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

SOURCE_FILES = ['reporting/primary_paired_contributions_UMRI_annual.csv',
                'main_figures/Fig5_allocation_regional_annual.csv',
                'main_figures/Fig5_protection_paired_annual.csv']
PUBLISHED = 'supplementary/equal_construction_2100.csv'

def derive(source_data):
    root = Path(source_data)
    contribution, allocation, protection = [pd.read_csv(root/n, float_precision='round_trip') for n in SOURCE_FILES]
    rows = []
    for scenario, rid, region, ledger_scenario in [
            ('ssp245', 1, 'Australia and New Zealand', 'SSP2_protected'),
            ('ssp585', 4, 'Eastern Asia', 'SSP5_protected')]:
        q = contribution[(contribution.scheme == 'pooled_centres') &
            (contribution.support == 'forecast') & (contribution.scenario == scenario) &
            (contribution.region_id == rid) & (contribution.year == 2100) &
            (contribution.contrast == 'protection_minus_growth')]
        ledger = allocation[(allocation.scenario == ledger_scenario) &
            (allocation.region_id == rid) & (allocation.year == 2100)]
        pp = protection[(protection.scenario == scenario) &
            (protection.region_id == rid) & (protection.year == 2100)]
        assert len(q) == len(ledger) == len(pp) == 1
        q, ledger, pp = q.iloc[0], ledger.iloc[0], pp.iloc[0]
        target, realized, unserved = [float(ledger[k]) for k in
            ['primary_realized_growth_km2', 'protected_realized_growth_km2', 'additional_unmet_growth_km2']]
        np.testing.assert_allclose([realized, ledger.protected_served_growth_km2], target, atol=1e-9, rtol=0)
        np.testing.assert_allclose(unserved, 0, atol=1e-9, rtol=0)
        pressure = float(q.delta_d_UMRI_mean)
        ecology = float(sum(q[f'delta_{k}_UMRI_mean'] for k in ['r','g','h','T']))
        total = float(q.delta_UMRI_mean)
        np.testing.assert_allclose([pressure+ecology, pp.delta_UMRI_mean], total, atol=1e-10, rtol=0)
        np.testing.assert_allclose(ecology, q.ecological_delta_UMRI_mean, atol=1e-10, rtol=0)
        assert int(q.members) == int(pp.members) == 15 and total > 0
        row = dict(scenario=scenario, region_id=rid, region=region, year=2100,
            construction_unprotected_km2=target, construction_protected_km2=realized,
            unserved_km2=unserved, mean_UMRI_benefit=total,
            urban_pressure_mean_UMRI=pressure, ecological_mean_UMRI=ecology,
            urban_pressure_share_percent=100*pressure/total,
            ecological_share_percent=100*ecology/total,
            median_UMRI_benefit=float(pp.delta_UMRI_q50),
            q17_UMRI_benefit=float(pp.delta_UMRI_q17),
            q83_UMRI_benefit=float(pp.delta_UMRI_q83), members=15)
        row.update({f'delta_{k}_UMRI_mean':float(q[f'delta_{k}_UMRI_mean']) for k in ['d','r','g','h','T']})
        rows.append(row)
    return pd.DataFrame(rows)

def verify_published(source_data, evidence):
    published = pd.read_csv(Path(source_data)/PUBLISHED, float_precision='round_trip')
    assert list(published.columns) == list(evidence.columns)
    assert published[['scenario','region_id','region','year']].equals(evidence[['scenario','region_id','region','year']])
    for column in evidence.select_dtypes(include='number').columns:
        np.testing.assert_allclose(published[column], evidence[column], atol=1e-12, rtol=0)
    return published

TABLE_CAPTION='Cumulative construction allocation under protection and resilience benefits at equal construction totals.'
TABLE_COLUMNS=r'@{}>{\raggedright\arraybackslash}p{42mm}>{\raggedleft\arraybackslash}p{19mm}>{\centering\arraybackslash}p{34mm}>{\raggedleft\arraybackslash}p{21mm}>{\raggedleft\arraybackslash}p{17mm}>{\raggedleft\arraybackslash}p{17mm}@{}'
NAMES=['Australia and New Zealand','Central America','Eastern Africa','Eastern Asia','Northern America','South America','South-Eastern Asia','Southern Asia','Western Africa','Western Asia']

def supplementary_table(evidence, ledger):
    
    header=r'Region & \shortstack{Unprotected\\growth} & \shortstack{Within\\source cell} & Relocated & \shortstack{Protected\\realized} & Unserved\\'
    lines=[r'\begingroup\small\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.14}',
        r'\begin{longtable}{'+TABLE_COLUMNS+'}',
        r'\caption{'+TABLE_CAPTION+r'}\label{tab:si_ledger}\\',
        r'\toprule',header,r'\midrule',r'\endfirsthead',
        r'\multicolumn{6}{l}{\small Supplementary Table \thetable\ continued}\\',
        r'\toprule',header,r'\midrule',r'\endhead',r'\bottomrule',r'\endfoot',
        r'\multicolumn{6}{l}{\textbf{Construction allocation during 2026--2100 (km$^2$)}}\\']
    fields=['primary_realized_growth_km2','in_cell_protected_growth_km2','relocated_growth_km2','protected_realized_growth_km2','additional_unmet_growth_km2']
    for scenario in ['SSP2_protected','SSP5_protected']:
        frame=ledger[(ledger.year==2100)&(ledger.scenario==scenario)].sort_values('region_id')
        assert len(frame)==10 and set(frame.region_id)==set(range(1,11))
        label='SSP2--4.5' if scenario=='SSP2_protected' else 'SSP5--8.5'
        lines.extend([r'\addlinespace',r'\multicolumn{6}{l}{'+label+r'}\\'])
        for row in frame.to_dict('records'):
            lines.append(' & '.join([NAMES[int(row['region_id'])-1]]+[f'{row[k]:.3f}' for k in fields])+r'\\')
        lines.append(' & '.join(['Total']+[f'{frame[k].sum():.3f}' for k in fields])+r'\\')
    lines.extend([r'\addlinespace\midrule',
        r'\multicolumn{6}{l}{\textbf{Protection benefits at equal construction totals in 2100}}\\',
        r'Region / pathway & \shortstack{Built in both\\(km$^2$)} & \shortstack{Median benefit\\{[17th--83rd percentiles]}\\(UMRI)} & \shortstack{Mean benefit\\(UMRI)} & \shortstack{Pressure\\share (\%)} & \shortstack{Ecological\\share (\%)}\\',
        r'\midrule'])
    for row in evidence.to_dict('records'):
        label='SSP2--4.5' if row['scenario']=='ssp245' else 'SSP5--8.5'
        lines.append(NAMES[int(row['region_id'])-1]+r'\newline '+label+' & '+f"{row['construction_unprotected_km2']:.2f}"+' & '+
            f"{row['median_UMRI_benefit']:.4f}"+r'\newline ['+f"{row['q17_UMRI_benefit']:.4f}--{row['q83_UMRI_benefit']:.4f}]"+' & '+
            f"{row['mean_UMRI_benefit']:.4f} & {row['urban_pressure_share_percent']:.1f} & {row['ecological_share_percent']:.1f}"+r'\\')
    lines.extend([r'\end{longtable}',
        r'{\small\noindent Construction is cumulative during 2026--2100. Protected realized construction equals within-cell plus relocated growth; unprotected growth equals protected realized construction plus unserved demand. Relocation is restricted to 10 km between source-cell centres within the same macroregion. Zeros are actual ledger zeros. In the equal-construction comparisons, protected and unprotected totals are identical within each row. Benefits are protected-minus-growth differences in the Urban Mangrove Resilience Index (UMRI) across 15 paired climate, sea-level and ecological-model combinations. Medians and 17th--83rd percentiles summarize total member benefits. Component shares divide the mean pressure or combined ecological contribution by the mean total benefit. Ecological contributions comprise recovery, growth, canopy moisture and habitat structure. Shares sum to 100\%; the mean budget and median total are distinct summaries. SSP denotes Shared Socioeconomic Pathway. Full-precision component means remain in Source Data.\par}',
        r'\endgroup',r'\clearpage',''])
    return '\n'.join(lines)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-data',type=Path,default=Path(__file__).resolve().parents[2]/'SourceData')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-published',action='store_true')
    args=parser.parse_args();evidence=derive(args.source_data)
    if args.verify_published:verify_published(args.source_data,evidence)
    args.output.mkdir(parents=True,exist_ok=True)
    evidence.to_csv(args.output/'equal_construction_2100.csv',index=False,float_format='%.17g')
    (args.output/'table7_equal_construction.tex').write_text(supplementary_table(evidence,pd.read_csv(args.source_data/'main_figures/Fig5_allocation_regional_annual.csv',float_precision='round_trip')),'utf-8')
    report={'status':'PASS','table':'Supplementary Table 7','cases':2,'members_per_case':15,
        'construction_equal':True,'component_mean_budget_additive':True,
        'input_sha256':{n:hashlib.sha256((args.source_data/n).read_bytes()).hexdigest() for n in SOURCE_FILES}}
    (args.output/'equal_construction_verification.json').write_text(json.dumps(report,indent=2)+'\n','utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
