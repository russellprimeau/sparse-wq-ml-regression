"""Focused two-arm publication outputs; all numbers derive from supplied fits."""
from pathlib import Path
import numpy as np
import pandas as pd
from .publication import tex,number,save_figure,minus
from ..evaluate.block_permutation import read_json,save_json,atomic_text


def paired_interval(first,second,draws=4000,block=3):
    n=len(first);block=min(block,n)
    starts=np.random.default_rng(42).integers(0,n,size=(draws,int(np.ceil(n/block))))
    idx=((starts[:,:,None]+np.arange(block))%n).reshape(draws,-1)[:,:n]
    denom=np.sqrt(first[idx].mean(axis=1));numer=np.sqrt(second[idx].mean(axis=1))
    values=100*(1-numer[denom>0]/denom[denom>0])
    return np.quantile(values,[.05,.95]).tolist()


def table(path,caption,columns,header,rows,placement='tp'):
    text=[r'\begin{table}['+placement+']',r'\caption{'+caption+'}',r'\small',
          r'\begin{tabularx}{\textwidth}{'+columns+'}',r'\toprule '+header+r'\\\midrule',
          *map(minus,rows),r'\bottomrule\end{tabularx}\end{table}']
    atomic_text(path,'\n'.join(text)+'\n')


def focused_outputs(out,cases,*,results=None,models=None):
    import matplotlib.pyplot as plt
    from .. import workflow
    out=Path(out);results=Path(results);models=Path(models)
    figures=out/'figures';figures.mkdir(exist_ok=True)
    scores=read_json(out/'all_arm_scores.json')
    predictions=pd.read_csv(out/'all_arm_predictions.csv',parse_dates=['timestamp'],float_precision='round_trip')
    main=[];paired=[];records=[];coverage=[]
    for case in cases:
        s=case['scores'];n=case['counts']['n_common_evaluation']
        r2=number(s['r2']) if s['r2']>0 else r'$\leq0$'
        family=case['configuration'].split(' (')[0].replace('Gaussian process','GP').replace('Multiple linear regression','MLR')
        # Errors of nonpositive-R² forecasts remain in the numerical release.
        error=f"{s['rmse']:.4g}" if s['r2']>0 else '--'
        gain=f"{100*(1-s['rmse']/s['baselines'][s['best_baseline']]['rmse']):.2f}" if s['r2']>0 else '--'
        main.append(tex(case['target_name'])+' & '+tex(family)+' & '+str(n)+' & '+r2+' & '+error+' & '+gain+r'\\')
        arms=[next(r for r in scores if r['target']==case['target'] and r['period']=='evaluation' and r['run']==arm)
              for arm in ['noprofiler_differential','profiler_differential']]
        a,b=(r['common'] for r in arms)
        prof=read_json(models/'off/half/profiler_differential'/case['target']/'assessment/fit.json')
        prof_training=len({m for d in prof['details'] for m in d['training_observations']})
        val=[next(r for r in scores if r['target']==case['target'] and r['period']=='validation' and r['run']==arm)['all_available']['n'] for arm in ['noprofiler_differential','profiler_differential']]
        coverage.append(tex(case['target_name'])+f" & {case['counts']['n_nominal_train']}/{case['counts']['n_nominal_evaluation']} & {case['counts']['n_train']}/{prof_training} & {val[0]}/{val[1]} & {arms[0]['all_available']['n']}/{arms[1]['all_available']['n']} & {a['n']}"+r'\\')
        times=predictions.loc[predictions.target.eq(case['target'])&predictions.period.eq('evaluation')&predictions.primary_comparison]
        left=times.loc[times.run.eq('noprofiler_differential')].set_index('timestamp').sort_index()
        right=times.loc[times.run.eq('profiler_differential')].set_index('timestamp').sort_index()
        pd.testing.assert_index_equal(left.index,right.index)
        np.testing.assert_allclose(left.observed,right.observed)
        loss_a=(left.prediction-left.observed).to_numpy()**2;loss_b=(right.prediction-right.observed).to_numpy()**2
        ci=paired_interval(loss_a,loss_b)
        change=100*(1-b['rmse']/a['rmse'])
        interpreted=case['target'] in {'total_coliforms_37c','ph','zinc'}
        # Report every positive R²; interpretation of the weak results stays in prose.
        label=lambda v:number(v) if v>0 else r'$\leq0$'
        effect=f'{change:.1f} [{ci[0]:.1f}, {ci[1]:.1f}]' if interpreted else '--'
        paired.append(tex(case['target_name'])+f" & {len(left)} & {arms[0]['all_available']['n']}/{arms[1]['all_available']['n']} & "+label(a['r2'])+' & '+label(b['r2'])+' & '+effect+r'\\')
        records.append(dict(target=case['target'],n=len(left),no_profiler_r2=a['r2'],profiler_r2=b['r2'],
            profiler_rmse_reduction_percent=change,ci05=ci[0],ci95=ci[1],
            no_profiler_scores=a,profiler_scores=b))
    table(out/'tables/target_summary.tex',
        r'Main-analysis models, using weather and intake inputs without profiler measurements, scored on all evaluation measurements they can predict. Model type, inputs and input representation are selected using the training period only. RMSE uses the units in Table~\ref{tab:target_metadata}; SS is the percentage RMSE reduction against the strongest of the six reference forecasts. Where $R^2\leq0$, the model is no more accurate than the mean of the evaluation measurements, and its values are not shown; complete errors, configurations and support intervals are supplied with the reproduction code.\label{tab:targets}',
        'Xlrrrr',r'Target & Family & $n$ & $R^2$ & RMSE & SS (\%)',main)
    table(out/'tables/profiler_comparison.tex',
        r'Effect of offering profiler measurements as inputs. Models with and without them are compared on the evaluation dates both can predict (paired $n$); available $n$ gives the number each can predict, without/with profiler inputs. Positive RMSE reduction favors the model with profiler inputs. Dashes mark contrasts not interpreted further because both predictions explain little or none of the observed variation. Central 90\% paired circular block-bootstrap intervals use 4,000 resamples of three measurements and condition on the fitted models.\label{tab:profiler_comparison}',
        r'>{\raggedright\arraybackslash}Xrrrr>{\raggedright\arraybackslash}X',r'Target & Paired $n$ & Available $n$ & \makecell{$R^2$\\No profiler} & \makecell{$R^2$\\Profiler} & \makecell{Profiler RMSE\\reduction (\%)}',paired)
    save_json(out/'profiler_comparison.json',records)
    table(out/'tables/review_model_counts.tex',
        r'Numbers of laboratory measurements used by the models without/with profiler inputs. Nominal counts give the chronological training/evaluation split before measurements lacking inputs are dropped. Paired evaluation counts the dates both models can predict. Fitted training counts distinct measurements used by any retained seed.\label{tab:model_samples}',
        'Xrrrrr',r'Target & Nominal train/eval. & Fitted train & Validation & Evaluation & Paired eval.',coverage,
        placement='H')  # appendix table: keep it under its own heading
    # Detailed case-study figure retains physical units and the same 50 dates.
    case=next((c for c in cases if c['target']=='total_coliforms_37c'),None)
    if case is None:return
    frame=pd.read_csv(out/'cases/total_coliforms_37c/evaluation.csv',parse_dates=['timestamp'],float_precision='round_trip').set_index('timestamp')
    profiler=predictions.loc[predictions.target.eq(case['target'])&predictions.period.eq('evaluation')&predictions.run.eq('profiler_differential')].set_index('timestamp')
    fig,axes=plt.subplots(2,1,figsize=(11,6),gridspec_kw={'height_ratios':[2,1]},sharex=True)
    ax=axes[0]
    for name,label,color,style in [('observed','Measured','#222222','o-'),('prediction','Selected GP (no profiler inputs)','#0072b2','s-'),
        ('reference: '+case['scores']['best_baseline'],case['scores']['best_baseline'],'#d55e00','^--')]:
        ax.plot(frame.index,frame[name],style,color=color,ms=3,lw=.9,label=label)
    ax.set_ylabel('Total coliforms (CFU/100 mL)');ax.legend(fontsize=9,ncol=3);ax.grid(alpha=.2)
    axes[1].plot(frame.index,frame.prediction-frame.observed,'o-',ms=3,lw=.8,color='#0072b2',label='Selected GP (no profiler inputs)')
    axes[1].plot(frame.index,profiler.loc[frame.index,'prediction']-frame.observed,'s-',ms=3,lw=.8,color='#009e73',label='XGBoost with profiler inputs')
    from matplotlib.ticker import MultipleLocator
    axes[1].yaxis.set_major_locator(MultipleLocator(10))
    axes[1].axhline(0,color='black',lw=.7);axes[1].set_ylabel('Prediction error\n(CFU/100 mL)');axes[1].legend(fontsize=9);axes[1].grid(alpha=.2)
    for ax in axes:
        ax.tick_params(labelsize=12)
        ax.yaxis.label.set_size(12)
        ax.legend(fontsize=11,ncol=3 if ax is axes[0] else 2)
    fig.autofmt_xdate();fig.tight_layout();save_figure(fig,figures,'total_coliform_case');plt.close(fig)
    controls=results/'gp_controls/summary.json'
    if controls.exists():
        rows=[];control_stats=[]
        for row in read_json(controls):
            table_=pd.read_csv(results/'gp_controls/controls'/row['control']/'predictions_and_references.csv',parse_dates=['timestamp'],float_precision='round_trip').set_index('timestamp').loc[frame.index]
            ci=paired_interval((table_.prediction-table_.observed).to_numpy()**2,(frame.prediction-frame.observed).to_numpy()**2)
            gain=100*(1-case['scores']['rmse']/row['scores']['rmse'])
            title='Prior value only' if row['control']=='prior_only' else 'Prior value + calendar'
            rows.append(title+f" & {row['scores']['r2']:.3f} & {row['scores']['rmse']:.3f} & {gain:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"+r'\\')
            control_stats.append(dict(control=row['control'],environmental_rmse_reduction_percent=gain,ci05=ci[0],ci95=ci[1]))
        table(out/'tables/gp_controls.tex',r'Control GP models for total coliforms, without environmental inputs, on the same 112 training and 50 evaluation measurements as the selected GP. Calendar: annual sine and cosine terms of the day of year. All use a Mat\'ern-5/2 kernel; optimization duration is selected within the training period. The last column gives the RMSE reduction of the selected GP relative to each control; positive values favor the environmental inputs. Central 90\% intervals use 4,000 paired circular block-bootstrap resamples of three measurements and condition on the fitted models.\label{tab:gp_controls}',
              'XrrX',r'GP inputs & $R^2$ & RMSE & Environmental RMSE reduction (\%)',rows)
        save_json(out/'gp_control_comparisons.json',control_stats)
