"""Publication figures and tables from the study results bundle."""
from __future__ import annotations
from sparse_wq.evaluate.block_permutation import atomic_text, read_json, clean_json
from sparse_wq.report.names import PARAMS
NAMES = {'color': 'Color', 'total_coliforms_37c': 'Total coliforms'}
UNITS = {}
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import numpy as np
import pandas as pd

def r2_interval(y, p, *, seed=42, iterations=2000):
    """Whole-observation interval; constant resamples do not define R-squared."""
    y, p = (np.asarray(y), np.asarray(p))
    indices = np.random.default_rng(seed).integers(0, len(y), (iterations, len(y)))
    yy, pp = (y[indices], p[indices])
    denominator = np.square(yy - yy.mean(axis=1, keepdims=True)).sum(axis=1)
    valid = (np.ptp(yy, axis=1) > 0) & (denominator > 0)
    values = 1 - np.square(yy[valid] - pp[valid]).sum(axis=1) / denominator[valid]
    return dict(ci05=np.quantile(values, 0.05) if len(values) else None, ci95=np.quantile(values, 0.95) if len(values) else None, n_boot_valid=int(valid.sum()), n_boot=iterations)

def interval_and_tests(table, references, scores, evidence):
    y, p = (table.observed.to_numpy(), table.prediction.to_numpy())
    times = list(map(str, table.index))
    interval = r2_interval(y, p)
    association = evidence.association_permutation_p(y, p, times, n_perm=5000, seed=42)
    prediction = dict(**interval, alignment_p=association, label=evidence.classify_prediction(scores['r2'], interval['ci05'], association, len(y)))
    ref = references[scores['best_baseline']].to_numpy()
    skill = evidence.assess(np.square(y - p), np.square(y - ref), times, n_boot=2000, seed=42, block_len=3, moving_block=True)
    return clean_json(dict(prediction=prediction, skill=skill))

def tex(text):
    return str(text).replace('&', '\\&').replace('_', '\\_').replace('%', '\\%').replace('°', '$^\\circ$').replace('µ', '$\\upmu$').replace('E. coli', r'\textit{E.~coli}')

_HYPHEN_MINUS = re.compile(r'(?<![\w$.\\{^_])-(?=\.?\d)')

def minus(row):
    """A table row with each negative number's hyphen typeset as a minus sign.

    Applied to data rows only: dates (2022-01-04) and ranges (13--50) follow a digit and
    are left alone, as is anything already in math mode.
    """
    return _HYPHEN_MINUS.sub(r'$-$', row)

def species(name):
    """Target name for a figure title, with the species name in italics."""
    return name.replace('E. coli', r'$\it{E.\ coli}$')

def number(value, digits=3):
    if value is None or not np.isfinite(value):
        return '--'
    formatted = f'{value:.{digits}f}'
    return formatted.lstrip('-') if float(formatted) == 0 else formatted

def display_order(cases, laboratory):
    """Targets by descending count of distinct laboratory observations, then by name.

    Every table and figure that lists the targets uses this order, so a target keeps its
    row position across them. The stored results keep the study's configuration order.
    """
    counts = laboratory.groupby('target').size()
    return sorted(cases, key=lambda case: (-int(counts[case['laboratory_target']]), case['target_name'].casefold()))

def p_value(value):
    if value is None or not np.isfinite(value):
        return '--'
    return r'$<0.001$' if value < 0.001 else number(value)

def fitted_predictors(details):
    """Union of inputs actually retained across final fits, in source order.

    MLR's input_features records the candidate subset before screening. Its
    selected_features (including an empty intercept-only set) is authoritative.
    """
    selected = []
    for detail in details:
        features = detail['selected_features'] if detail['family'] == 'mlr' else detail['input_features']
        for name in features:
            if name not in selected:
                selected.append(name)
    return selected


def representation(case):
    family = case['configuration'].split(' (')[0].replace('Gaussian process', 'GP').replace('XGBoost', 'XGB')
    return (family, case['configuration'].split(' (')[-1].rstrip(')'))

def arm_short(case):
    pool = 'profiler' if case['run'].startswith('profiler_') else 'no profiler'
    structure = {'differential': 'change + prior', 'absolute': 'value, prior offered', 'absolute_noprior': 'value, no prior'}[case['run'].split('_', 1)[1]]
    return pool + '; ' + structure

def render_diagnostics(out, cases, *, results=None, models=None):
    """Describe saved predictions on their primary dates, without fitting models."""
    from sparse_wq import paths
    results = Path(results) if results is not None else paths.RESULTS
    model_root = Path(models) if models is not None else results / 'models'
    clipping, seeds = [], []
    for case in cases:
        table = pd.read_csv(out / 'cases' / case['target'] / 'evaluation.csv', float_precision='round_trip')
        y = table.observed.to_numpy()
        difference = table.prediction.to_numpy() - table.prediction_unclipped.to_numpy()
        clipping.append(dict(target=case['target'], target_name=case['target_name'], n=len(table),
                             n_changed=int((np.abs(difference) > 1e-10).sum()),
                             rmse_clipped=float(np.sqrt(np.mean(np.square(y - table.prediction)))),
                             rmse_unclipped=float(np.sqrt(np.mean(np.square(y - table.prediction_unclipped)))),
                             max_prediction_change=float(np.max(np.abs(difference)))))
        assessment = model_root / case['method'] / 'half' / case['run'] / case['target'] / 'assessment' / 'seed_predictions.csv'
        sources = [(0, assessment)]
        horizons = results / 'horizons' / case['target'] if model_root.resolve() == (results / 'models').resolve() else model_root / '__no_inherited_horizons__'
        for path in sorted(horizons.glob('h*/seed_predictions.csv')):
            horizon = int(path.parent.name[1:])
            if horizon:
                sources.append((horizon, path))
        dates = pd.DatetimeIndex(table.timestamp)
        for horizon, path in sources:
            if not path.exists():
                continue
            raw = pd.read_csv(path, parse_dates=['timestamp'], float_precision='round_trip')
            raw = raw.loc[raw.timestamp.isin(dates)]
            for seed, rows in raw.groupby('seed'):
                if len(rows) != len(dates) or rows.timestamp.nunique() != len(dates):
                    raise ValueError(f'Seed sensitivity requires the primary dates: {case["target"]}, {horizon}, {seed}')
                observed = rows.observed.to_numpy()
                error = np.square(rows.prediction.to_numpy() - observed).mean()
                variance = np.var(observed)
                seeds.append(dict(target=case['target'], horizon=horizon, seed=int(seed), n=len(rows),
                                  r2=1 - error / variance if variance > 0 else np.nan,
                                  rmse=float(np.sqrt(error))))
    pd.DataFrame(clipping).to_csv(out / 'clipping_sensitivity.csv', index=False)
    seed_table = pd.DataFrame(seeds, columns=['target', 'horizon', 'seed', 'n', 'r2', 'rmse'])
    seed_table.to_csv(out / 'seed_sensitivity.csv', index=False)
    summary = seed_table.groupby(['target', 'horizon'], as_index=False).agg(
        n=('n', 'first'), seed_r2_min=('r2', 'min'), seed_r2_max=('r2', 'max'),
        seed_rmse_min=('rmse', 'min'), seed_rmse_max=('rmse', 'max'), seeds=('seed', 'count'))
    summary.to_csv(out / 'seed_sensitivity_summary.csv', index=False)
    return clipping


def render_tables(out, cases, *, results=None, models=None):
    tables = out / 'tables'
    tables.mkdir(exist_ok=True)
    clipping = render_diagnostics(out, cases, results=results, models=models)
    lines = [r'\begin{table}[tp]', r'\caption{Validation-selected models on common later evaluation dates. Counts: fitted training / common validation / common evaluation observations. $R^2_{\rm val}$ uses eligible earlier validation dates; $R^2_{\rm eval}$ and errors use primary-model later dates. Target cells give units and evaluated below-DL percentage; Color/pH also give observed range and reporting step (not analytical uncertainty). SS is fractional RMSE reduction against the strongest reference, with a 90\% interval.\label{tab:targets}}', r'\scriptsize', r'\begin{tabularx}{\textwidth}{>{\hsize=0.95\hsize\raggedright\arraybackslash}X>{\hsize=1.4\hsize\raggedright\arraybackslash}Xrrrr>{\hsize=0.65\hsize\centering\arraybackslash}X}', r'\toprule Target & Selected model and arm & Counts & $R^2_{\rm val}$ & $R^2_{\rm eval}$ & RMSE & SS (90\% CI)\\\midrule']
    samples = [r'\begin{table}[H]', r'\caption{Distinct laboratory observations and coverage sensitivity for each reported model. Train: actual fitted observations; Val./Eval.: primary-model validation/evaluation observations; Avail.: all available evaluation observations. Available counts and the last two score columns use all dates predicted by the reported model and its references. SS is percentage RMSE reduction against the strongest reference on those dates.\label{tab:model_samples}}', r'\scriptsize', r'\begin{tabularx}{\textwidth}{Xrrrrrrr}', r'\toprule Target & Train $n$ & Val. $n$ & Eval. $n$ & Avail. $n$ & Below DL (\%) & Avail. $R^2$ & Avail. SS (\%)\\\midrule']
    metrics = [r'\begin{table}[H]', r'\caption{Exploratory support and error summaries for the selected models. Conditional 90\% bootstrap intervals for $R^2$ resample individual measurements. Skill is tested against the strongest reference. Wins/pairs: dates on which the model has the smaller absolute error, out of dates without tied errors.\label{tab:reported_support}}', r'\scriptsize', r'\begin{tabularx}{\textwidth}{XrXrrXX}', r'\toprule Target & MAE & $R^2$ interval & Wins/pairs & $p_{\rm sign}$ & Strongest reference & Prediction / Skill\\\midrule']
    labels = {'not_supported': 'Not supported', 'underpowered': 'Untestable', 'directional': 'Directional', 'supported': 'Supported'}
    rows = []
    for case in cases:
        s, c, support = case['scores'], case['counts'], case['support']
        sk, pr = support['skill'], support['prediction']
        family, rep = representation(case)
        validation_r2 = case['validation']['scores']['r2']
        available = case['all_available_scores']
        available_reference = min(available['baselines'], key=lambda key: available['baselines'][key]['rmse'])
        available_skill = 1 - available['rmse'] / available['baselines'][available_reference]['rmse']
        target = tex(case['target_name']) + r'\newline{\tiny ' + tex(case['unit']) + f"; {c['censored_percent']:.0f}" + r'\% below DL}'
        if case['target'] in ('color', 'ph'):
            observed = pd.read_csv(out / 'cases' / case['target'] / 'evaluation.csv').observed
            increment = 1 if case['target'] == 'color' else 0.1
            target += r'\newline{\tiny ' + f'{observed.min():g}--{observed.max():g}; step {increment:g}' + '}'
        lines.append(target + ' & ' + tex(family + ' (' + rep + ')') + r'\newline ' + tex(arm_short(case)) + f" & {c['n_train']}/{c['n_common_validation']}/{c['n_common_evaluation']} & {number(validation_r2)} & {number(s['r2'])} & {s['rmse']:.4g} & {number(sk['skill'], 2)}" + r'\newline ' + f"[{number(sk['skill_ci05'], 2)}, {number(sk['skill_ci95'], 2)}]" + r'\\\midrule')
        samples.append(tex(case['target_name']) + f" & {c['n_train']} & {c['n_common_validation']} & {c['n_common_evaluation']} & {c['n_available_evaluation']} & {c['censored_percent']:.0f} & {available['r2']:.3f} & {number(100 * available_skill, 1)}" + r'\\')
        metrics.append(tex(case['target_name']) + f" & {s['mae']:.4g} & [{number(pr['ci05'])}, {number(pr['ci95'])}] & {sk['sign_wins']}/{sk['sign_n_pairs']} & {p_value(sk['sign_p'])} & " + tex(s['best_baseline']) + ' & ' + labels[pr['label']] + ' / ' + labels[sk['verdict']] + r'\\')
        rows.append(dict(target=case['target'], arm=arm_short(case), configuration=case['configuration'], unit=case['unit'], **case['counts'], validation_r2=validation_r2, all_available_r2=available['r2'], all_available_skill=available_skill, all_available_best_baseline=available_reference, **{key: s[key] for key in ['r2', 'rmse', 'mae', 'best_baseline']}, **{'skill_' + key: value for key, value in sk.items()}, **{'prediction_' + key: value for key, value in pr.items()}))
    for name, content in [('target_summary.tex', lines), ('review_model_counts.tex', samples), ('reported_support.tex', metrics)]:
        if content[-1].endswith(r'\midrule'):
            content[-1] = content[-1].removesuffix(r'\midrule')
        content.extend([r'\bottomrule', r'\end{tabularx}', r'\end{table}'])
        content = content[:5] + [minus(line) for line in content[5:]]
        atomic_text(tables / name, '\n'.join(content) + '\n')
    pd.DataFrame(rows).to_csv(out / 'reported_scores.csv', index=False)
    unchanged = sum(item['n_changed'] == 0 for item in clipping)
    content = [r'\begin{table}[H]', r'\caption{Effect of the zero floor on primary evaluation forecasts for nonnegative targets. The other UNCHANGED_COUNT reported targets have unchanged ensemble predictions within $10^{-10}$ target units. Each seed is floored after reconstruction and before averaging; pH is unconstrained. Both errors use the same observations and fitted models.\label{tab:clipping_sensitivity}}', r'\small', r'\begin{tabularx}{\textwidth}{Xrrr}', r'\toprule Target & Changed/$n$ & Floored RMSE & Raw RMSE\\\midrule']
    for item in clipping:
        if item['n_changed']:
            content.append(minus(tex(item['target_name']) + f" & {item['n_changed']}/{item['n']} & {item['rmse_clipped']:.4g} & {item['rmse_unclipped']:.4g}" + r'\\'))
    content.extend([r'\bottomrule', r'\end{tabularx}', r'\end{table}'])
    content = [line.replace('UNCHANGED_COUNT', str(unchanged)) for line in content]
    atomic_text(tables / 'review_clipping.tex', '\n'.join(content) + '\n')

    from sparse_wq.report.permutation_results import copy_supplied
    copy_supplied(out, cases, results=results, models=models)


def save_figure(fig, directory, name):
    fig.savefig(directory / (name + '.png'), dpi=240, bbox_inches='tight')
    fig.savefig(directory / (name + '.pdf'), bbox_inches='tight')

def render(out, *, results=None, models=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    out = Path(out)
    cases = display_order(read_json(out / 'results.json')['targets'], pd.read_csv(out / 'laboratory.csv'))
    scores = read_json(out / 'all_arm_scores.json')
    render_tables(out, cases, results=results, models=models)
    from .two_arm import focused_outputs
    focused_outputs(out,cases,results=results,models=models)
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    from sparse_wq.design import ARM_KEYS
    arms = list(ARM_KEYS)
    matrix = np.array([[next((r for r in scores if r['target'] == case['target'] and r['period'] == 'evaluation' and ((r['run'], r['method']) == arm)))['common']['r2'] for arm in arms] for case in cases])
    fig, ax = plt.subplots(figsize=(7.2, 7.5))
    im = ax.imshow(np.maximum(matrix,0), cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for i, case in enumerate(cases):
        for j, arm in enumerate(arms):
            ax.text(j, i, (f'{matrix[i,j]:.2f}' if matrix[i,j]>0 else '$\\leq 0$'), ha='center', va='center', fontsize=10, color='white' if matrix[i, j] > 0.65 else 'black')
            if arm == (case['run'], case['method']):
                ax.add_patch(Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, fill=False, edgecolor='black', linewidth=2.3))
    labels = ['No profiler\nchange from prior','Profiler\nchange from prior']
    ax.set_xticks(range(len(arms)), labels, rotation=48, ha='right', fontsize=9)
    ax.set_yticks(range(len(cases)), [case['target_name'] + f" (paired n={case['counts']['n_paired_evaluation']})" for case in cases])
    ax.set_xticks(np.arange(-0.5, len(arms)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cases)), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    fig.colorbar(im, ax=ax, pad=0.02, label='Evaluation R² (nonpositive values shown at zero)')
    save_figure(fig, figures, 'structure_r2')
    plt.close(fig)
    references = list(cases[0]['scores']['baselines'])
    normalized = np.array([[case['scores']['nrmse']] + [case['scores']['baselines'][r]['nrmse'] for r in references] for case in cases])
    fig, ax = plt.subplots(figsize=(11, 7.6))
    im = ax.imshow(normalized, cmap='YlOrRd', vmin=0, vmax=3, aspect='auto')
    for i in range(len(cases)):
        for j in range(7):
            ax.text(j, i, f'{normalized[i, j]:.2f}', ha='center', va='center', color='white' if normalized[i, j] > 2 else 'black')
    ax.set_yticks(range(len(cases)), [case['target_name'] for case in cases])
    ax.set_xticks(range(7), ['Selected model'] + references, rotation=35, ha='right')
    fig.colorbar(im, ax=ax, pad=0.02, label='RMSE / scored target standard deviation (lower is better)')
    save_figure(fig, figures, 'method_comparison')
    plt.close(fig)
    fig, axes = plt.subplots(7, 2, figsize=(13, 17), squeeze=False)
    for ax, case in zip(axes.flat, cases):
        table = pd.read_csv(out / 'cases' / case['target'] / 'evaluation.csv', parse_dates=['timestamp']).set_index('timestamp')
        for column, label, style in [('observed', 'Measured', 'o-'), ('prediction', 'Selected model', 's-'), ('reference: ' + case['scores']['best_baseline'], 'Best reference', '^--')]:
            ax.plot(table.index, table[column], style, ms=3, lw=1, label=label)
        ax.set_title(case['target_name'] + ' — ' + case['scores']['best_baseline'], fontsize=10)
        ax.set_ylabel(case['unit'])
        ax.tick_params(axis='x', rotation=25, labelsize=8)
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8, ncol=3, loc='upper left')
    fig.tight_layout()
    save_figure(fig, figures, 'reported_predictions')
    plt.close(fig)
    channels = read_json(out / 'results.json')['predictor_channels']
    prior = 'Prior laboratory value'
    channels.append(prior)
    values = np.zeros((len(cases), len(channels)))
    for i, case in enumerate(cases):
        for j, f in enumerate(channels):
            is_prior = f == prior
            unavailable = is_prior and case['run'].endswith('noprior') or (not is_prior and f.startswith('Pfl - ') and case['run'].startswith('noprofiler'))
            values[i, j] = -1 if unavailable else int(any((x.endswith('_state') for x in case['fitted_features'])) if is_prior else f in case['fitted_features'])
    from matplotlib.colors import ListedColormap, BoundaryNorm
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(values, cmap=ListedColormap(['#606060', '#eeeeee', '#0072b2']), norm=BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3), aspect='auto')

    def short(f):
        return f.replace('Longwave (IR) radiation (W/m2)', 'Longwave radiation').replace('Shortwave (solar) radiation (W/m2)', 'Solar radiation').replace('Air temperature (°C)', 'Air temperature').replace('Atmospheric pressure (mBar)', 'Air pressure').replace('24hr precipitation total (mm)', 'Precipitation (24 h)').replace('Humidity (%)', 'Humidity').replace('Pfl - ', '').replace(' (microS_cm)', '').replace(' (°C)', '').replace(' [35 m]', ' (~35 m)') + (' (~3 m)' if f.startswith('Pfl - ') and (not f.endswith('[35 m]')) else '')
    ax.set_xticks(range(len(channels)), [short(f) for f in channels], rotation=65, ha='right', fontsize=8)
    ax.set_yticks(range(len(cases)), [case['target_name'] for case in cases])
    ax.legend(handles=[Patch(color='#0072b2', label='Selected'), Patch(color='#eeeeee', label='Offered, not selected'), Patch(color='#606060', label='Withheld')], loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=3, frameon=False)
    save_figure(fig, figures, 'selected_predictors')
    plt.close(fig)
    pd.DataFrame(values, index=[case['target'] for case in cases], columns=channels).to_csv(out / 'selected_predictors.csv')
    if (out / 'horizon_scores.csv').exists():
        horizons = pd.read_csv(out / 'horizon_scores.csv')
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
        for ax in axes:
            ax.tick_params(labelsize=12)
        plotted = 0
        for case in cases:
            if not case['eligible_for_followup']:
                continue
            plotted += 1
            frame = horizons.loc[horizons.target.eq(case['target'])]
            if not frame.fixed_set_complete.all():
                raise ValueError('Horizon plot needs explicit partial-date handling')
            label = case['target_name'] + f" (n={case['counts']['n_common_evaluation']})"
            axes[0].plot(frame.horizon, frame.r2, 'o-', label=label)
            axes[1].plot(frame.horizon, 100 * frame.skill, 'o-', label=label)
        for ax in axes:
            ax.axhline(0, color='black', lw=0.7)
            ax.set_xlabel('Lead time (hours)')
            ax.set_xticks([0, 12, 24, 48, 96, 144, 168])
            ax.grid(alpha=0.2)
            if plotted > 1:
                ax.legend()
        axes[0].set_ylabel('R²')
        axes[1].set_ylabel('RMSE reduction vs strongest reference (%)')
        fig.tight_layout()
        save_figure(fig, figures, 'horizon_r2')
        plt.close(fig)
    descriptive(out, cases)
    if results is not None:
        from sparse_wq.evaluate.targeted_checks import render_targeted_tables
        render_targeted_tables(Path(results) / 'targeted_checks', out, cases)

def descriptive(out, cases):
    """Tables and laboratory plots use the same qualified record as the fits."""
    import matplotlib.pyplot as plt
    from sparse_wq.data.qualified import chronological_split
    lab = pd.read_csv(out / 'laboratory.csv', parse_dates=['TIMESTAMP'], float_precision='round_trip')
    tables = out / 'tables'
    figures = out / 'figures'
    metadata = ['\\begin{table}[tp]', '\\caption{Laboratory records, all ending on 10 February 2025, ordered by $N$ and then alphabetically; other tables and figures listing all targets follow this order. $N$: number of laboratory measurements; interval: median sampling interval in days; DL: detection limit reported by the laboratory, in target units; below DL: share of results reported as below the detection limit. Reported zeros are not counted as below DL, and no DL is inferred for them.\\label{tab:target_metadata}}', '\\scriptsize', '\\begin{tabularx}{\\textwidth}{Xllrrrr}', '\\toprule Target & Unit & First date & $N$ & Interval & DL & Below DL (\\%)\\\\\\midrule']
    censoring = ['\\begin{table}[H]', '\\caption{Laboratory results reported below a detection limit (DL), and reported zeros. Each target\'s below-DL results share a single limit; limits are not reported for other results, so whether they changed over time cannot be established.\\label{tab:lab_censoring}}', '\\scriptsize', '\\begin{tabularx}{\\textwidth}{Xrrrr}', '\\toprule Target & $N$ & Below DL & Below DL (\\%) & Reported zero\\\\\\midrule']
    fig, axes = plt.subplots(7, 2, figsize=(12, 15))
    figd, axesd = plt.subplots(7, 2, figsize=(12, 15), sharex=True)
    for ax, axd, case in zip(axes.flat, axesd.flat, cases):
        data = lab.loc[lab.target.eq(case['laboratory_target'])].set_index('TIMESTAMP').sort_index()
        first, last = (data.index.min(), data.index.max())
        limits = ','.join((f'{x:g}' for x in sorted(data.limit.dropna().unique()))) or '--'
        median = float(np.median((data.index[1:] - data.index[:-1]).total_seconds()) / 86400)
        metadata.append(tex(case['target_name']) + ' & ' + tex(case['unit']) + f' & {first:%Y-%m-%d} & {len(data)} & {median:g} & {limits} & {100 * data.is_left_censored.mean():.1f}' + '\\\\')
        censoring.append(tex(case['target_name']) + f' & {len(data)} & {data.is_left_censored.sum()} & {100 * data.is_left_censored.mean():.1f} & {(~data.is_left_censored & data.reported_number.eq(0)).sum()}' + '\\\\')
        train, test = chronological_split(data.index)
        ax.plot(data.index, data.value, 'o-', ms=3, lw=0.8, color='#0072b2')
        censored = data.loc[data.is_left_censored]
        ax.plot(censored.index, censored.value, 'v', ms=4, color='#d55e00', label='Below DL (DL/2)')
        ax.axvline(test.min(), color='grey', ls='--', lw=1)
        ax.set_title(species(case['target_name']), fontsize=12)
        ax.set_ylabel(case['unit'])
        ax.tick_params(axis='x', labelsize=9, rotation=20)
        ax.grid(alpha=0.2)
        from sparse_wq.evaluate.forecasts import prior_values
        delta = data.value - prior_values(data.value, data.index, case['rows'])
        span = delta.max() - delta.min()
        normalized = (delta - delta.min()) / span if span else delta * 0
        axd.plot(data.index, normalized, 'o-', ms=3, lw=0.8)
        axd.axvline(test.min(), color='grey', ls='--', lw=1)
        axd.set_title(species(case['target_name']), fontsize=12)
        axd.set_ylim(-0.05, 1.05)
        axd.tick_params(axis='x', labelsize=9, rotation=20)
        axd.grid(alpha=0.2)
        if case['target'] in ['color', 'total_coliforms_37c']:
            pd.DataFrame(dict(value=data.value, difference=delta, normalized_difference=normalized, nominal_training=data.index.isin(train))).to_csv(out / 'cases' / case['target'] / 'descriptive.csv', index_label='timestamp')
    for name, content in [('target_metadata.tex', metadata), ('review_censoring.tex', censoring)]:
        content.extend(['\\bottomrule', '\\end{tabularx}', '\\end{table}'])
        atomic_text(tables / name, '\n'.join(content) + '\n')
    fig.tight_layout()
    figd.tight_layout()
    save_figure(fig, figures, 'Target_timeseries_no_raster')
    save_figure(figd, figures, 'Target_diff_timeseries_overlay_norm')
    plt.close(fig)
    plt.close(figd)
    examples=[c['target'] for c in cases if c['target'] in ('color','total_coliforms_37c')]
    if not examples:return
    fig, axes = plt.subplots(len(examples), 1, figsize=(10, 2.4*len(examples)), sharex=True,squeeze=False)
    for ax, target in zip(axes.flat, examples):
        table = pd.read_csv(out / 'cases' / target / 'descriptive.csv', parse_dates=['timestamp']).set_index('timestamp')
        ax.plot(table.index, table.normalized_difference, 'o-', ms=3, lw=1)
        ax.axvline(table.loc[~table.nominal_training].index.min(), color='grey', ls='--')
        ax.set_ylabel(NAMES[target] + '\nrescaled change')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figures, 'Target_diff_timeseries_overlay_norm_abbrev')
    plt.close(fig)
