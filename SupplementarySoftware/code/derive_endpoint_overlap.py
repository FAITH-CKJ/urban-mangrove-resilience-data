
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def summarize(source_data):
    source_data = Path(source_data)
    with np.load(source_data / 'endpoint_2100/pixelwise_median_endpoints_2100.npz',
                 allow_pickle=False) as z:
        scenarios = z['scenarios']
        valid = z['valid']
        matched = z['matched2025']
        regions = z['region_id']
        states = z['growth_median_UMRI']
        effects = z['urban_median_delta_UMRI']
        assert len(valid) == 934814 and int(valid.sum()) == 926220
        assert int(matched.sum()) == 925774 and np.all(matched <= valid)
        assert states.shape == effects.shape == (4, len(valid))
        assert np.isfinite(states[:, valid]).all() and np.isfinite(effects[:, valid]).all()
        assert np.isnan(states[:, ~valid]).all() and np.isnan(effects[:, ~valid]).all()
        assert np.all(np.abs(states[:, valid]) < 1)
        assert np.all(np.abs(effects[:, valid]) < 2)
        np.testing.assert_array_equal(states[:, valid] < 0,
                                      z['state_negative_votes'][:, valid] >= 8)
        np.testing.assert_array_equal(effects[:, valid] < 0,
                                      z['urban_erosion_votes'][:, valid] >= 8)
        identities = pd.MultiIndex.from_arrays([z['block_index'], z['centre_index']])
        assert identities.is_unique
        assert len(z['blocks']) == 141 and int(z['block_index'].max()) == 140
        rows = []
        for j, scenario in enumerate(scenarios):
            for support, support_mask in [('forecast', valid), ('matched2025', matched)]:
                for region in range(11):
                    selected = np.ones(len(valid), dtype=bool) if region == 0 else regions == region
                    use = selected & support_mask
                    nref, nvalid = int(selected.sum()), int(use.sum())
                    negative = states[j, use] < 0
                    loss = effects[j, use] < 0
                    nnegative, nloss = int(negative.sum()), int(loss.sum())
                    njoint = int((negative & loss).sum())
                    assert 0 <= njoint <= min(nnegative, nloss) <= nvalid <= nref
                    rows.append({
                        'scenario': str(scenario), 'support': support, 'region_id': region,
                        'reference_n': nref, 'valid_n': nvalid, 'unknown_n': nref - nvalid,
                        'negative_state_n': nnegative, 'additional_urban_loss_n': nloss,
                        'negative_state_and_urban_loss_n': njoint,
                        'nonnegative_state_and_urban_loss_n': nloss - njoint,
                        'negative_state_without_urban_loss_n': nnegative - njoint,
                        'joint_area_km2': njoint * 0.0009,
                        'joint_pct_reference': 100.0 * njoint / nref,
                        'joint_pct_additional_urban_loss': 100.0 * njoint / nloss if nloss else np.nan,
                        'joint_pct_negative_state': 100.0 * njoint / nnegative if nnegative else np.nan,
                    })
    result = pd.DataFrame(rows)
    previous = pd.read_csv(source_data / 'endpoint_2100/pixelwise_median_negative_area_2100.csv')
    previous = previous.set_index(['scenario', 'support', 'region_id', 'metric'])
    for row in result.itertuples():
        key = (row.scenario, row.support, row.region_id)
        assert row.negative_state_n == previous.loc[key + ('state_negative',), 'negative_n']
        assert row.additional_urban_loss_n == previous.loc[key + ('urban_erosion',), 'negative_n']
        assert row.valid_n == previous.loc[key + ('state_negative',), 'valid_n']
    for _, group in result.groupby(['scenario', 'support']):
        for column in [c for c in result if c.endswith('_n')]:
            assert group.loc[group.region_id == 0, column].iloc[0] == group.loc[group.region_id > 0, column].sum()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-data', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'SourceData')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'reproduced/endpoint_overlap')
    args = parser.parse_args()
    result = summarize(args.source_data)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / 'state_and_urban_loss_overlap_2100.csv'
    result.to_csv(output, index=False, float_format='%.17g', na_rep='')
    expected = args.source_data / 'endpoint_2100/state_and_urban_loss_overlap_2100.csv'
    if expected.exists():
        supplied = pd.read_csv(expected, float_precision='round_trip')
        pd.testing.assert_frame_equal(result, supplied, check_exact=True)
    overall = result.query('support == "forecast" and region_id == 0')
    report = {'status': 'PASS', 'rows': len(result), 'reference_n': 934814,
              'valid_n': 926220, 'unknown_n': 8594,
              'definition': 'Negative median growth-branch UMRI AND negative median within-member growth-minus-fixed2025 UMRI difference at the same reference cell in 2100.',
              'overall': overall.to_dict('records'),
              'existing_marginal_area_counts_reproduced': True,
              'regional_counts_sum_to_overall': True}
    (args.output / 'verification.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'rows': len(result),
                      'overlap_percent_of_additional_loss_area': overall.joint_pct_additional_urban_loss.tolist()}))


if __name__ == '__main__':
    main()
