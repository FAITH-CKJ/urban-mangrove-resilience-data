




from __future__ import annotations

import warnings
import numpy as np

COMPONENTS = ('pressure', 'recovery', 'growth', 'health', 'trajectory')
WEIGHTS = np.array([.25, .15, .10, .25, .25], dtype=np.float64)
LAMBDA = 0.05934161890467785


def ratio(n, d):
    n, d = np.broadcast_arrays(np.asarray(n, float), np.asarray(d, float))
    return np.divide(n, d, out=np.full(n.shape, np.nan), where=d > 0)


def baseline(a):
    a = np.asarray(a, float)[:5]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        value = np.nanmedian(a, axis=0)
    return np.where(np.isfinite(a).sum(axis=0) >= 3, value, np.nan)


def relative_change(a, b):
    return np.where((a == 0) & (b == 0), 0., ratio(a - b, a + b))


def normalize(s, reference=LAMBDA):
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError('The fixed reference must be finite and positive.')
    s = np.asarray(s, dtype=np.float64)
    return s / (reference + np.abs(s))


def compute(data):
    contact = np.where(data['fixed_landward_no_baseline_interface'],
                       data['fixed_all_interface_urban_lower'],
                       data['fixed_landward_urban_lower'])
    pressure = np.maximum.reduce([data['Pvs_lower'], contact, data['direct_B_lower']])
    finite = pressure[np.isfinite(pressure)]
    if np.any((finite < -1e-12) | (finite > 1 + 1e-12)):
        raise ValueError('Urban exposure exceeds its physical range.')
    pressure = np.clip(pressure, 0., 1.)
    evi0 = baseline(data['EVI'])
    ndmi0 = baseline(data['ndmi_corrected'])
    capacity0 = baseline(data['connected_B_capacity'])
    growth = relative_change(data['EVI'], evi0)
    health = relative_change((data['ndmi_corrected'] + 1) / 2, (ndmi0 + 1) / 2)
    trajectory = relative_change(data['connected_B_capacity'], capacity0)
    absent = data['MB_count'] == 0
    components = np.stack([-pressure, np.where(absent, 0., data['r']),
                           np.where(absent, 0., growth),
                           np.where(absent, -1., health), trajectory], axis=-1)
    gate = 1 - np.maximum.reduce([np.zeros(pressure.shape),
                                 -components[..., 3], -components[..., 4]])
    effective = components.copy()
    effective[..., 1] *= gate
    effective[..., 2] = np.minimum(components[..., 2], 0.) + gate * np.maximum(components[..., 2], 0.)
    valid = data['common_support'].astype(bool) & np.isfinite(components).all(axis=-1)
    contributions = np.where(valid[..., None], effective * WEIGHTS, np.nan)
    net = contributions.sum(axis=-1)
    positive = np.maximum(contributions, 0.).sum(axis=-1)
    negative = np.maximum(-contributions, 0.).sum(axis=-1)
    return {'components': components, 'gate': gate, 'contributions': contributions,
            'S': net, 'P': positive, 'N': negative, 'A': positive + negative,
            'UMRI': normalize(net), 'valid': valid,
            'EVI_baseline': evi0, 'NDMI_baseline': ndmi0,
            'capacity_baseline': capacity0}


