#!/usr/bin/env python3
"""Search, verify and export the thesis counterexamples without training a model."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import sys

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asif_reproduction.counterexample import (
    FEATURE_ORDER, coherence_features, density_matrix, entropy,
    exact_endpoint_certificate, find_collision, qutip_check, search_collisions,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/coherence_counterexamples'))
    parser.add_argument('--qutip-check', action='store_true', help='Check using the original thesis extractor.')
    parser.add_argument('--plot', action='store_true', help='Write a PNG figure (requires matplotlib).')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Choose an empty output directory; existing results are never overwritten.')
    canonical = find_collision(0.60, 0.01, upper=0.10)
    if canonical is None or not all(exact_endpoint_certificate().values()):
        raise ArithmeticError('Canonical construction failed.')
    candidates, attempts = search_collisions()
    payload = {
        'schema_version': 1,
        'method': 'Analytically restricted deterministic Bell-diagonal entropy-matching search',
        'provenance': 'The original pair was derived analytically, then root-solved and verified; this grid search was added for reproducible exploration.',
        'scope': 'The chosen basis family; any fixed phase via a matching local rotation. Numerical residuals supplement the exact proof.',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__).resolve(), ROOT/'asif_reproduction/counterexample.py']},
        'feature_order': FEATURE_ORDER, 'entropy_units': 'nats',
        'canonical': canonical.to_dict(), 'exact_endpoint_certificate': exact_endpoint_certificate(),
        'attempts': attempts, 'collisions_found': len(candidates),
        'search_grid': {'pair_sum_percent': list(range(56, 81, 4)), 'margin_percent': list(range(1, 5)),
                        'bracket_upper_fraction_of_tail': 0.95},
        'candidates': [candidate.to_dict() for candidate in candidates],
    }
    if args.qutip_check:
        payload['qutip_checks'] = [qutip_check(canonical, phase) for phase in (0, 0.37, np.pi/2, np.pi)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'results.json').write_text(json.dumps(payload, indent=2)+'\n')
    with (args.output_dir/'search.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['pair_sum', 'margin', 't', 'feature_max_abs_difference', 'entangled_pt_min', 'separable_pt_min'])
        for item in candidates:
            writer.writerow([item.pair_sum, item.margin, item.t, item.feature_max_abs_difference,
                             item.entangled_pt_min, item.separable_pt_min])
    matrices = [density_matrix(w) for w in (canonical.entangled_weights, canonical.separable_weights)]
    np.savez(args.output_dir/'canonical_states.npz', rho_entangled=matrices[0], rho_separable=matrices[1],
             features_entangled=coherence_features(matrices[0]), features_separable=coherence_features(matrices[1]))
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t = np.linspace(0, .1, 401)
        target = entropy(np.array(canonical.entangled_weights))
        y = [entropy(np.array([.49, .11, .2+x, .2-x]))-target for x in t]
        fig, ax = plt.subplots(figsize=(7.2, 4.2), layout='constrained')
        ax.plot(t, y, color='#176b87', linewidth=2, label=r'$H(q_S(t))-H(q_E)$')
        ax.axhline(0, color='0.4', linewidth=.8)
        ax.axvline(canonical.t, color='#a74732', linestyle='--', label=f'Equal features at t = {canonical.t:.6f}')
        ax.set(xlabel='Redistribution parameter t', ylabel='Entropy difference (nats)',
               title='Opposite PPT labels, identical coherence coordinates')
        ax.legend(frameon=False)
        fig.savefig(args.output_dir/'entropy_match.png', dpi=220)
        plt.close(fig)
    print(f'Found {len(candidates)} collisions in {attempts} deterministic candidates.')
    print(f'Canonical t = {canonical.t:.15f}; feature residual = {canonical.feature_max_abs_difference:.3g}')
    print(f'PPT minima = {canonical.entangled_pt_min:+.6f}, {canonical.separable_pt_min:+.6f}')
    print(f'Outputs: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
