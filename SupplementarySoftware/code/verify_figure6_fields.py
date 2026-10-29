
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


def verify(source_data):
    source_data = Path(source_data)
    folder = source_data / 'figure_story'
    method = json.loads((folder/'Fig6_theory_methods.json').read_text('utf8'))
    method['settings'] = json.loads((folder.parent/method['settings_file']).read_text('utf8'))
    controls = method['settings']['reconnection']
    w = method['settings']['weights']
    scale = method['settings']['normalization_lambda']
    metadata = json.loads((source_data/'methods/figure6_contours.json').read_text('utf8'))
    checks = []

    def close(name, actual, expected, tol=5e-12):
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
        error = float(np.max(np.abs(actual-expected)))
        assert np.isfinite(error) and error <= tol, (name, error)
        checks.append({'name': name, 'status':'PASS', 'max_abs_error':error})

    def structural(capacity):
        return (capacity-controls['C0'])/(capacity+controls['C0'])

    def gate(capacity, moisture):
        return 1+np.minimum(0,np.minimum(moisture,structural(capacity)))

    def points(score):
        return score/(scale+np.abs(score))

    g = controls.get('g', controls.get('g_positive'))
    response = w['r']*controls['r']+w['g']*g
    initial = 0.5
    pressure = controls['urban_pressure_p']
    moisture = controls['h']
    with np.load(folder/'Fig6_analytical_fields.npz',allow_pickle=False) as data:
        m,q = np.meshgrid(data['m'],data['q'])
        capacity = m*q
        ceiling = (w['h']*moisture+w['T']*structural(capacity)
                   +gate(capacity,moisture)*(w['r']*m+w['g']))/w['d']
        close('Figure 6 full conditional pressure field',data['pressure_ceiling'],ceiling)
        admissible = q >= np.maximum(10**(-.5),(1+30*(1-m))**(-.5))
        assert np.array_equal(data['necessary_support'],admissible)
        checks.append({'name':'Figure 6 necessary continuous support mask','status':'PASS'})
        c,h = np.meshgrid(data['C'],data['h'])
        base = -w['d']*pressure+w['h']*h+w['T']*structural(c)
        dynamic = base+response*gate(c,h)
        fixed = base+response*gate(initial,h)
        extra = points(dynamic)-points(fixed)
        close('Figure 6 full gate-update gain field',data['extra_gain_UMRI'],extra)
        close('Figure 6 gate increment before normalization',data['intermediate_gate_release'],dynamic-fixed)
        assert float(extra.min()) >= -2e-12
        assert np.max(np.abs(extra[h<=structural(initial)])) < 2e-12
        checks.append({'name':'Figure 6 nonnegative gain and moisture-limited zero field','status':'PASS'})
        cx = data['C']
        updated_h = (w['d']*pressure-w['T']*structural(cx)-response*(1+structural(cx)))/w['h']
        fixed_h = (w['d']*pressure-w['T']*structural(cx)-response*(1+structural(initial)))/w['h']
        close('Figure 6 updated-gate balance boundary',data['updated_balance_h'],updated_h)
        close('Figure 6 initial-gate balance boundary',data['fixed_balance_h'],fixed_h)
        close('Figure 6 balance-window width',fixed_h-updated_h,response*(structural(cx)-structural(initial))/w['h'])
        close('Figure 6 gate switch',data['capacity_gate_switch'],np.array(36/55))
        
        close('Figure 6 updated curve has zero score',-w['d']*pressure+w['h']*updated_h+w['T']*structural(cx)+response*gate(cx,updated_h),np.zeros_like(cx))
        close('Figure 6 fixed curve has zero score',-w['d']*pressure+w['h']*fixed_h+w['T']*structural(cx)+response*gate(initial,fixed_h),np.zeros_like(cx))
        counts={'pressure_field_cells':int(ceiling.size),'gain_field_cells':int(extra.size)}

    configurations = pd.read_csv(folder/'Fig6_three_configurations.csv',float_precision='round_trip')
    geometries = metadata['exact_geometries']
    reference = {tuple(x) for x in geometries['B']}
    for row in configurations.itertuples():
        current = {tuple(x) for x in geometries[row.state]}
        remaining = set(current)
        counts_in_reference = []
        while remaining:
            start = remaining.pop()
            component,frontier = {start},[start]
            while frontier:
                x,y=frontier.pop()
                for neighbor in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                    if neighbor in remaining:
                        remaining.remove(neighbor);component.add(neighbor);frontier.append(neighbor)
            counts_in_reference.append(len(component&reference))
        expected_m = len(current&reference)/len(reference)
        expected_c = np.linalg.norm(counts_in_reference)/len(reference)
        close('Figure 6 exact configuration '+row.state,np.array([row.m,row.C]),np.array([expected_m,expected_c]))
    close('Figure 6 exact configuration concentration',configurations.q.to_numpy(),(configurations.C/configurations.m).to_numpy())
    close('Figure 6 exact pressure ceilings',configurations.pressure_ceiling.to_numpy(),np.array([27/130,3/10,857/1500]))
    i,ii,iii = [{tuple(x) for x in geometries[k]} for k in ['I','II','III']]
    assert ii-i == {tuple(x) for x in geometries['external_link']}
    assert (ii-i).isdisjoint(reference)
    assert ii-iii == {tuple(x) for x in geometries['historical_cells_removed_II_to_III']}
    assert len(ii-iii)==2 and ii-iii <= reference
    checks.append({'name':'Figure 6 fragmentation and retention arrows have distinct exact geometry','status':'PASS'})
    sources = ['figure_story/Fig6_analytical_fields.npz','figure_story/Fig6_three_configurations.csv','methods/figure6_contours.json']
    return {'status':'PASS','checks':checks,'grid_counts':counts,
            'input_sha256':{rel:hashlib.sha256((source_data/rel).read_bytes()).hexdigest() for rel in sources}}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-data',type=Path,default=Path(__file__).resolve().parents[2]/'SourceData')
    args=parser.parse_args()
    result=verify(args.source_data)
    print(json.dumps(result,indent=2))
