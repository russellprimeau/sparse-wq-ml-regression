"""Deterministic bounded subset search, independent of model/data plumbing."""
from __future__ import annotations
import math
import numpy as np
from sparse_wq.fitting import InsufficientCoverage

def search_subsets(features, surrogate_variants, final_variants, measure, allowed, *, budget, beam_width, rounds, swaps, min_features, final_top_k, seed, finalist_seeds, require_profiler=False):
    """Repeat selection under the supplied method, never rank assessment scores.

    budget caps distinct predictor subsets. Representation comparisons and
    finalist seed refits evaluate those same subsets and are recorded separately.
    Only expected coverage failures are skipped; numerical/code failures surface.
    """
    features = tuple(features)
    if len(set(features)) != len(features) or not features:
        raise ValueError('Supply unique offered predictors')
    if budget < 1 or beam_width < 1 or rounds < 0 or (swaps < 0) or (not 1 <= min_features <= len(features)) or (final_top_k < 1):
        raise ValueError('Invalid subset-search settings')
    rng = np.random.default_rng(seed)
    subsets = set()
    cache = {}
    trace = []
    rank = lambda item: (item['objective'], len(item['features']), item['variant'], item['features'])

    def evaluate(selected, variant, seeds=(0,)):
        selected = tuple(selected)
        seeds = tuple(seeds)
        key = (selected, variant, seeds)
        if key in cache:
            return cache[key]
        if not allowed(selected, variant):
            cache[key] = None
            trace.append(dict(features=list(selected), variant=variant, seeds=list(seeds), status='incompatible'))
            return None
        if selected not in subsets:
            if len(subsets) >= budget:
                return None
            subsets.add(selected)
        try:
            info = measure(selected, variant, seeds)
        except InsufficientCoverage as error:
            trace.append(dict(features=list(selected), variant=variant, seeds=list(seeds), status='coverage_unavailable', reason=str(error)))
            cache[key] = None
            return None
        objective = info['objective']
        if objective is None or not math.isfinite(objective):
            trace.append(dict(features=list(selected), variant=variant, seeds=list(seeds), status='undefined_objective'))
            cache[key] = None
            return None
        item = dict(features=selected, variant=variant, seeds=seeds, objective=float(objective), info=info)
        trace.append(dict(features=list(selected), variant=variant, seeds=list(seeds), status='scored', objective=float(objective)))
        cache[key] = item
        return item
    initial = [item for variant in surrogate_variants if (item := evaluate(features, variant)) is not None]
    if not initial:
        raise InsufficientCoverage('No eligible full-pool surrogate; inspect recorded coverage before changing search settings')
    anchor = min(initial, key=rank)
    surrogate = anchor['variant']
    beam = [anchor]
    evaluated = {features: anchor}
    for depth in range(rounds):
        candidates = sorted({item['features'][:i] + item['features'][i + 1:] for item in beam if len(item['features']) > min_features for i in range(len(item['features']))} - set(evaluated))
        rng.shuffle(candidates)
        scored = []
        remaining = budget - len(subsets)
        reductions = min(rounds - depth, max((len(item['features']) - min_features for item in beam)))
        reserve = min(swaps, budget // 4)
        allowance = max(1, (remaining - reserve) // max(1, reductions))
        for selected in candidates[:allowance]:
            if len(subsets) >= budget:
                break
            item = evaluate(selected, surrogate)
            evaluated[selected] = item
            if item is not None:
                scored.append(item)
        if not scored:
            break
        beam = sorted(scored, key=rank)[:beam_width]
        if len(subsets) >= budget:
            break
    best = min((item for item in evaluated.values() if item is not None), key=rank)
    for _ in range(swaps):
        if len(subsets) >= budget:
            break
        excluded = [name for name in features if name not in best['features']]
        candidates = sorted({tuple((name for name in features if name in set(best['features']) - {old} | {new})) for old in best['features'] for new in excluded} - set(evaluated))
        if not candidates:
            break
        selected = candidates[int(rng.integers(len(candidates)))]
        item = evaluate(selected, surrogate)
        evaluated[selected] = item
        if item is not None and rank(item) < rank(best):
            best = item
    retained = [item['features'] for item in sorted((value for value in evaluated.values() if value is not None), key=rank)[:final_top_k]]
    if features not in retained:
        retained.append(features)
    finalists = []
    for selected in retained:
        for variant in final_variants:
            item = evaluate(selected, variant, finalist_seeds(selected, variant))
            if item is None:
                continue
            if require_profiler:
                if not any((name.startswith('Pfl -') for name in selected)):
                    continue
                if variant.startswith('mlr') and (not all((any((name.startswith('Pfl -') for name in detail.get('selected_features', []))) for detail in item['info'].get('details', [])))):
                    continue
            finalists.append(item)
    if not finalists:
        raise InsufficientCoverage('No eligible finalist survived the configured arm requirements')
    return (min(finalists, key=rank), sorted(finalists, key=rank), dict(distinct_subsets=len(subsets), budget=budget, surrogate=surrogate, trace=trace))
