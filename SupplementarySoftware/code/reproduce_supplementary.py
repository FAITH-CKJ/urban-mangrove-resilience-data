
from pathlib import Path
import hashlib,json,re,argparse,subprocess,sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
parser=argparse.ArgumentParser()
parser.add_argument('--source-data',type=Path,default=PACKAGE.parent/'SourceData')
parser.add_argument('--output',type=Path,default=PACKAGE/'reproduced')
parser.add_argument('--tables-only',action='store_true')
args=parser.parse_args()
ROOT=args.output
PROJECT=args.source_data
W=PROJECT/'supplementary'
F=ROOT/'figures'
D=ROOT/'source_data/supplementary'
Q=ROOT/'verification'
for p in [F,D,Q]:p.mkdir(parents=True,exist_ok=True)
NAMES=['Australia and New Zealand','Central America','Eastern Africa','Eastern Asia','Northern America','South America','South-Eastern Asia','Southern Asia','Western Africa','Western Asia']
SHORT=['Australia / NZ','Central America','Eastern Africa','Eastern Asia','Northern America','South America','Southeastern Asia','Southern Asia','Western Africa','Western Asia']
ABBR=['ANZ','CAM','EAF','EAS','NAM','SAM','SEA','SAS','WAF','WAS']

def read(p):return pd.read_csv(p,float_precision='round_trip')
def out_csv(name,df):
 source=W/name
 if source.exists():
  import shutil
  shutil.copy2(source,D/name)
 else:df.to_csv(D/name,index=False,float_format='%.16g')
hist=read(PROJECT/'main_figures/Fig3_historical_contributions.csv')
matched=read(PROJECT/'main_figures/Fig2_history_matched2025.csv')
coh=read(W/'historical_fixed_cohorts.csv')
end=read(W/'historical_fixed_cohort_endpoints.csv')
pop=read(W/'regional_population.csv')

blocks=read(PROJECT/'methods/historical_block_annual.csv')
coverage=blocks.groupby('year',as_index=False).valid_centres.sum()
coverage['reference_centres']=934814
coverage['valid_percent']=100*coverage.valid_centres/934814
loo=read(W/'historical_leave_one_out.csv')
floo=read(W/'future_leave_one_out_2100.csv')
checks=json.loads((PROJECT/'methods/classification_calibration.json').read_text(encoding='utf-8'))
cl=checks['classification']
model=json.loads((PROJECT/'methods/model_test_summary.json').read_text(encoding='utf-8'))
future=read(W/'future_absolute_annual.csv')
paired=read(W/'future_paired_annual.csv')
ledger=read(PROJECT/'main_figures/Fig5_allocation_regional_annual.csv')


for name,frame in [('historical_full.csv',hist),('historical_matched2025.csv',matched),('historical_fixed_cohorts.csv',coh),('historical_fixed_cohort_endpoints.csv',end),('historical_coverage.csv',coverage),('historical_leave_one_out.csv',loo),('regional_population.csv',pop)]:
 out_csv(name,frame)
for key in ['actual_raster','original_sample']:
 cm=np.array(cl['predictions'][key]['group_matrix']);pct=100*cm/cm.sum(axis=1)[:,None]
 out_csv('classification_'+key+'.csv',pd.DataFrame({'reference':np.repeat(['urban','vegetation','other'],3),'prediction':np.tile(['urban','vegetation','other'],3),'count':cm.ravel(),'row_percent':pct.ravel()}))
out_csv('classification_record_eligibility.csv',pd.DataFrame(list(cl['status_counts'].items()),columns=['status','records']))
out_csv('classification_record_years.csv',pd.DataFrame(list(cl['by_year_n'].items()),columns=['year','records']))
states=['reference_retention','connected_capacity','confirmed_return_fraction','EVI','NDMI']
mlabels=['m','C','r','EVI','NDMI','S','UMRI']
mrows=[]
for horizon in ['five_year_test','ten_year_terminal_test']:
 for split in ['temporal','spatial']:
  entry=model[horizon][split]
  for method in ['model','persistence']:
   a=entry[method]; row={'horizon':horizon,'split':split,'method':method,'S_MAE':a['region_balanced_five_year_S_MAE'],'UMRI_MAE':a['UMRI_MAE'],'direction_accuracy':a['regional_change_direction_accuracy'],'valid_rows':a['valid_rows'],'centres':a['centres'],'false_absence':a['absence_confusion_counts']['true_present_pred_absent']}
   row.update(a['state_target_MAE']);mrows.append(row)
metrics=pd.DataFrame(mrows);out_csv('forecast_holdout_metrics.csv',metrics)
def tex(s):
 return str(s).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')
def table(caption,label,cols,headers,rows,note=''):
 cols=cols.replace('p{',r'>{\raggedright\arraybackslash}p{')
 x=[r'\begin{longtable}{'+cols+'}',r'\caption{'+caption+r'}\label{tab:si_'+label+r'}\\',r'\toprule',' & '.join(headers)+r'\\',r'\midrule',r'\endfirsthead',r'\multicolumn{'+str(len(headers))+r'}{l}{\small\textit{Supplementary Table \thetable\ continued}}\\',r'\toprule',' & '.join(headers)+r'\\',r'\midrule',r'\endhead',r'\bottomrule',r'\endfoot']
 x.extend(' & '.join(r)+r'\\' for r in rows)
 x.append(r'\end{longtable}')
 if note:x.append(r'{\small '+note+r'\par}')
 return '\n'.join([r'\begingroup\small\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.2}']+x+[r'\endgroup',''])

tables={}
rows=[]
for _,r in pop.iterrows():rows.append([tex(r.region)+' ('+r.abbreviation+')',str(int(r.blocks)),f'{int(r.reference_centres):,}',f'{int(r.valid_centres):,}',f'{int(r.matched2025_centres):,}',f'{r.reference_area_km2:.4f}',f'{100*r.pooled_weight:.3f}'])
rows.append([r'\textbf{Total}','141','934,814','926,220','925,774','841.3326','100.000'])
tables[1]=table('Regional analysis populations and primary forecast weights.','population',r'p{41mm}rrrrrr',['Region (abbreviation)','Blocks',r'\shortstack{Reference\\centres}',r'\shortstack{Forecast\\centres}',r'\shortstack{Matched\\centres}',r'\shortstack{Reference\\area (km$^2$)}',r'\shortstack{Forecast\\weight (\%)}'],rows,'Reference centres form the fixed 2000--2004 habitat union. Forecast centres are valid across all future cases. Matched centres also have a valid historical score in 2025. Area is the reference-centre count multiplied by 900~m$^2$.')
coeff=json.loads((PROJECT/'methods/harmonization_coefficients.json').read_text(encoding='utf-8'))
rows=[[b.upper() if b in ['nir','swir1','swir2'] else b.capitalize(),f'{a:.4f}',f'{bb:.4f}','1.0000','0.0000'] for b,a,bb in zip(coeff['bands'],coeff['sensor_transforms']['8']['slope'],coeff['sensor_transforms']['8']['intercept'])]
tables[2]=table('Band-level transformation to the historical ETM+ reference.','coefficients','lrrrr',['Band','OLI slope','OLI intercept','TM/ETM+ slope','TM/ETM+ intercept'],rows,r'The formula is $\rho_{\mathrm{target}}=a\rho_{\mathrm{source}}+b$ for dimensionless surface reflectance. Landsat 8 and 9 share the OLI-family coefficients; the published fit was estimated for Landsat 8 versus 7\cite{Roy2016}.')
datasets=[
('Historical mangroves','GMW v4.1.12','Annual 30-m maps; 2000--2025 bands; fixed reference union 2000--2004'+r'\cite{GMW2026}.'),
('Urban study boundaries','GUB 2018','Polygons larger than 100 km$^2$ and their 2-km neighbourhoods'+r'\cite{ref086}.'),
('Optical and thermal data','Landsat C2 T1 L2','Annual same-observation maximum-NDVI winners, 2000--2025; 30-m grid.'),
('Static water context','JRC GSW v1.4','1984--2021 occurrence $\geq90\%$; distance in native-grid pixels'+r'\cite{Pekel2016}.'),
('Historical climate','ERA5-Land','0.1-degree native cells; 2000--2025; 252 cells mapped to the analysis'+r'\cite{ERA5Land2021,ERA5LandHourlyData}.'),
('Future climate','NEX-GDDP-CMIP6 v2.0','0.25-degree native daily fields; five GCMs, four SSPs; 2026--2100'+r'\cite{ref054,NEXGDDPCMIP6Data}.'),
('Historical relative sea level','PSMSL RLR; C3S vDT2024','Qualifying gauges and flagged ocean-plus-background proxies; 2000--2014 reference'+r'\cite{Holgate2013}.'),
('Future relative sea level','AR6 regional total; version 20210809','Regional point support; q17/q50/q83; annual interpolation and 2025 anchoring'+r'\cite{AR6SL2021,IPCC2021Chapter9,Kopp2023}.'),
('Local topography','FABDEM v1.2','EGM2008; relative neighbourhood elevation and slope'+r'\cite{Hawker2022,FABDEM12}.'),
('Urban demand','Global 1-km Downscaled Urban Land v1.00','Native land-area denominator; SSP fractions interpolated annually and anchored in 2025'+r'\cite{ref051,GaoUrbanData}.'),
]
tables[3]=table('Source products and the spatial or temporal support used in the analysis.','datasets',r'p{32mm}p{39mm}p{86mm}',['Input','Product','Use'],[list(r) for r in datasets],'Climate, sea-level and urban-demand information retain their native source scales when linked to the 30-m reporting grid and 500-m ecological neighbourhoods. Provider access points are listed in the main Data availability statement.')
features=[
('Dynamic: urban area exposure','fraction','Urban / (urban + vegetation + unknown).'),('Dynamic: urban interface','fraction','Urban contact on the fixed reference interface.'),('Dynamic: historical occupation','fraction','Reference habitat lost and currently urban.'),('Dynamic: mean temperature',r'$^\circ$C','Annual mean of daily mean temperature.'),('Dynamic: high temperature',r'$^\circ$C','Annual 95th percentile of daily maximum temperature.'),('Dynamic: precipitation','mm yr$^{-1}$','Annual precipitation total.'),('Dynamic: dry-season precipitation','mm','Total over three fixed historically driest months.'),('Dynamic: consecutive dry days','days','Longest annual run below 1 mm day$^{-1}$.'),('Dynamic: relative sea level','m','Anomaly relative to the historical reference.'),('Dynamic: sea-level rate','mm yr$^{-1}$','Observed or scenario relative sea-level rate.'),('Dynamic: connected landward space','km$^2$','Available landward connected area.'),('Dynamic: landward urban blockage','fraction','Urban-blocked share of the landward context.'),('Dynamic: landward class coverage','fraction','Known land-cover support.'),('Static: EVI baseline','index','2000--2004 local median; at least three valid years.'),('Static: NDMI baseline','index','2000--2004 local median; at least three valid years.'),('Static: capacity baseline','fraction','2000--2004 connected-capacity median.'),('Static: longitude','degrees','Fixed centre longitude.'),('Static: latitude','degrees','Fixed centre latitude.'),('Static: macroregion','code','Ten-region categorical identity.'),('Static: relative elevation','m','Neighbourhood median minus reference-habitat median.'),('Static: slope','degrees','Local topographic slope.'),('Static: baseline direction coverage','fraction','Known support for baseline landward direction.')]
tables[4]=table('Dynamic and static ecological-model inputs.','features',r'p{52mm}p{23mm}p{82mm}',['Variable','Unit','Definition'],[list(r) for r in features],'The five ecological states and their masks are provided over a five-year history in addition to these inputs. Future dynamic drivers refer to the target year. The tree model therefore receives 137 features.')
rows=[]
for _,r in metrics.iterrows():
 horizon='5-y' if r.horizon=='five_year_test' else '10-y terminal'
 test_label=horizon+' '+r.split if r.horizon=='five_year_test' else horizon+r'\newline '+r.split
 rows.append([test_label,('State model' if r.method=='model' else 'Persistence')]+[f'{r[k]:.5f}' for k in states+['S_MAE']]+[f'{r.UMRI_MAE:.4f}'])
tables[5]=table('Holdout errors and additional evaluation diagnostics.','validation',r'p{26mm}p{21mm}rrrrrrr',['Test','Method','$m$','$C$','$r$','EVI','NDMI','$S$',r'UMRI'],rows,'All numerical entries in the main part of this table are mean absolute errors. $S$ is a computational intermediate; UMRI errors use the final score scale. Both compare annual regional aggregates formed with equal block weights; state errors are first calculated at centres. Evaluation summaries give regions equal weight. Both tests are evaluated against 2021--2025 observations. The temporal split has 55,273 centres and 275,168 valid score rows; the spatial split has 2,625 centres and 12,956 valid score rows.')
rows2=[]
for _,r in metrics.iterrows():rows2.append([('5-y' if r.horizon=='five_year_test' else '10-y terminal')+' '+r.split,'State model' if r.method=='model' else 'Persistence',f'{int(round(r.direction_accuracy*10))}/10',f'{int(r.false_absence):,}'])
tables[5]+= '\n'+r'\begin{center}\small\begin{tabular}{llrr}\toprule Test & Method & Correct regional direction & False absence records\\\midrule'+'\n'+'\n'.join(' & '.join(r)+r'\\' for r in rows2)+r'\bottomrule\end{tabular}\end{center}'+'\n'

rows=[]
for rid in range(11):
 h=hist[hist.region_id==rid].set_index('year')
 rr=future[(future.support=='forecast')&(future.urban_branch=='growth')&(future.year==2100)&(future.region_id==rid)].set_index('scenario')
 rows.append(['All centres' if rid==0 else NAMES[rid-1]]+[f'{h.loc[y,"UMRI"]:.4f}' for y in [2000,2025]]+[f'{rr.loc[s,"UMRI_q50"]:.4f}' for s in ['ssp126','ssp245','ssp370','ssp585']])
tables[6]=table('Historical endpoints and median absolute urban-growth states in 2100 (UMRI).','endpoints',r'p{43mm}rrrrrr',['Region','2000','2025','SSP1--2.6','SSP2--4.5','SSP3--7.0','SSP5--8.5'],rows,'Historical endpoints use the full annual historical support. Absolute future states use the 926,220-centre forecast support. The trajectory panels in Figure 2 use the matched-2025 support for both periods. Full annual member quantiles and paired differences are supplied in Source Data.')
led=ledger[ledger.year==2100].copy()
from figure5_equal_construction import derive,verify_published,supplementary_table
equal_construction=derive(PROJECT);verify_published(PROJECT,equal_construction)
tables[7]=supplementary_table(equal_construction,ledger)
out_csv('equal_construction_2100.csv',equal_construction)
rows=[]
for (s,c),g in floo[floo.support=='forecast'].groupby(['scenario','contrast'],sort=True):
 sign='negative' if c=='urban_growth_minus_fixed' else 'positive'
 assert (g['delta_UMRI_'+sign+'_members']==15).all()
 rows.append([s.replace('ssp','SSP'), 'Urban' if sign=='negative' else 'Protection',str(len(g)),f'{g.delta_UMRI_q50.min():.4f}',f'{g.delta_UMRI_q50.max():.4f}',f'15/15 {sign}'])
tables[8]=table('Leave-one-block and leave-one-region robustness of 2100 paired effects (UMRI).','robustness','llrrrl',['SSP','Contrast','Exclusions','Minimum median','Maximum median','Member direction'],rows,'Each row contains 141 block and ten region exclusions. Ranges refer to medians across the exclusion analyses. The final column holds for every exclusion. Historical endpoint changes are negative in all 151 exclusions, with a range from $-0.2995$ to $-0.1682$ UMRI.')

out_csv('future_absolute_annual.csv',future);out_csv('future_paired_annual.csv',paired);out_csv('future_leave_one_out_2100.csv',floo);out_csv('protection_allocation_annual.csv',ledger)
ledger_amounts=[c for c in led.columns if c.endswith('_km2')]+['regional_growth_preserved']
summary={'historical_reference_area_km2':934814*.0009,'valid_centre_year_fraction':24016461/24305164,'historical_endpoints':hist[(hist.region_id==0)&hist.year.isin([2000,2025])].to_dict('records'),'protection_2100_totals':led.groupby('scenario')[ledger_amounts].sum().to_dict('index'),'source_data_files':len(list(D.glob('*.csv')))}
(Q/'supplementary_build_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
from endpoint_reporting import supplementary_table
tables[9]=supplementary_table(PROJECT,table)


native_endpoints=pd.read_csv(PROJECT/'figure_story/historical_structural_endpoints.csv',float_precision='round_trip')
assert len(native_endpoints)==6 and not native_endpoints.duplicated(['support','variable']).any()
assert np.allclose(native_endpoints.mean_2025_percent-native_endpoints.mean_2000_percent,native_endpoints.change_percentage_points,rtol=0,atol=1e-11)
rows=[]
for support,title in [('available','Endpoint paired'),('complete_history','Complete history')]:
 for var in ['m','C','q']:
  r=native_endpoints[(native_endpoints.support==support)&(native_endpoints.variable==var)].iloc[0]
  rows.append([title,'$'+var+'$',f'{r.mean_2000_percent:.2f}',f'{r.mean_2025_percent:.2f}',f'${r.change_percentage_points:.2f}$',f'{int(r.n):,}'])
tables[10]=table('Paired changes in habitat retention, connected capacity and patch concentration.','native_endpoints','llrrrr',['Support','State',r'2000 (\%)',r'2025 (\%)','Change (pp)','Centres'],rows,r'Values are equal-centre means of the local states multiplied by 100; changes are percentage points (pp). Endpoint-paired $m$ and $C$ use centres with valid scores in both 2000 and 2025. Complete-history rows additionally require a valid score in every year from 2000 to 2025. For $q$, each corresponding population is restricted to neighbourhoods with $m>0$ at both endpoints. Patch concentration is calculated separately at each centre as $q=C/m$ and then averaged. Counts refer to fixed 30-m reference centres whose 500-m neighbourhoods can overlap. The two endpoint means and their difference use the same mask within each row.')

body=(HERE/'supplementary_body.tex').read_text(encoding='utf-8')
for k,t in tables.items():body=body.replace('%%TABLE_'+str(k)+'%%',t)
assert '%%TABLE_' not in body

longtable_pattern = r'\\begin\{longtable\}.*?\\end\{longtable\}'
displayed_endpoint = next(t for t in re.findall(longtable_pattern, body, re.S) if r'\label{tab:si_endpoint_area}' in t)
assert displayed_endpoint == re.search(longtable_pattern, tables[9], re.S)[0], 'Endpoint table does not match calculated values'

expected_table_labels=['tab:si_population', 'tab:si_coefficients', 'tab:si_datasets', 'tab:si_features', 'tab:si_validation', 'tab:si_endpoints', 'tab:si_ledger', 'tab:si_robustness', 'tab:si_endpoint_area', 'tab:si_native_endpoints', 'tab:si_components', 'tab:city_buffer_sensitivity']
table_labels=re.findall(re.escape(r'\label{')+r'(tab:[^}]+)\}',body)
assert table_labels==expected_table_labels,('Supplementary table sequence mismatch',table_labels)
assert body.count(r'\begin{longtable}')==12, 'Expected twelve numbered supplementary tables'
preamble=r'''% !TEX program = xelatex
\documentclass[a4paper,10pt]{article}
\usepackage[margin=19mm]{geometry}
\usepackage{fontspec}
\usepackage{pdflscape}
\usepackage{graphicx,amsmath,amssymb,booktabs,longtable,array,microtype}
\usepackage[super,sort&compress]{natbib}
\renewcommand{\bibnumfmt}[1]{#1.}
\usepackage[hidelinks]{hyperref}
\usepackage{xurl,caption}
\providecommand{\urlprefix}{}
\Urlmuskip=0mu plus 1mu
\captionsetup{font=small,labelfont=bf}
\renewcommand{\figurename}{Supplementary Fig.}
\renewcommand{\tablename}{Supplementary Table}
\renewcommand{\theequation}{\arabic{equation}}
\setcounter{secnumdepth}{0}
\setcounter{tocdepth}{2}
\clubpenalty=10000
\widowpenalty=10000
\displaywidowpenalty=10000
\brokenpenalty=10000
\setlength{\parskip}{3pt}
\raggedbottom
\makeatletter\setlength{\@fptop}{0pt}\makeatother
\title{Urbanization erodes resilience in global mangroves\\[5pt]\large Supplementary Information}
\author{Kangjie Cao, Tao Fang, Xin Luo, Xiaoxue Shen, Guoyu Qiu and Ruili Li}
\date{}
\begin{document}
\maketitle
\noindent This file contains Supplementary Methods 1--10, Supplementary Discussion, Supplementary Figs. 1--6, Supplementary Tables 1--12 and Supplementary References. Source Data accompany the manuscript. The complete numerical observations, source code, fitted models and reproduction instructions are available at \url{https://github.com/FAITH-CKJ/urban-mangrove-resilience-data}, version v1.2.4. Author-generated data are shared under CC BY 4.0 and can be downloaded without registration. The Source Data figure--table index identifies the tables in the attachment and the additional files in the repository.

'''
(ROOT/'supplementary.tex').write_text(preamble+'\n'+body+'\n\\end{document}\n',encoding='utf-8')
print('Built supplementary.tex with six figures and twelve tables; scientific source summaries retained.')

import shutil
for name in ['references.bib','naturemag.bst']:
    shutil.copy2(HERE/name,ROOT/name)


if not args.tables_only:
 subprocess.run([sys.executable,'-X','utf8','-B',str(HERE/'render_supplementary_b.py'),'--source-data',str(PROJECT),'--output',str(F)],check=True)
 subprocess.run([sys.executable,'-X','utf8','-B',str(HERE/'render_future_process.py'),'--source-data',str(PROJECT),'--output',str(ROOT)],check=True)
 subprocess.run([sys.executable,'-X','utf8','-B',str(HERE/'reproduce_compensation_supplement.py'),'--source-data',str(PROJECT/'main_figures'),'--output',str(ROOT)],check=True)

if not args.tables_only:
 subprocess.run([sys.executable,'-X','utf8','-B',str(HERE/'reproduce_figure_story.py'),'--source-data',str(PROJECT),'--output',str(ROOT),'--figures','S5'],check=True)

if not args.tables_only:
 subprocess.run([sys.executable,'-X','utf8','-B',str(HERE/'reproduce_full_histories.py'),'--source-data',str(PROJECT),'--output-dir',str(F)],check=True)
