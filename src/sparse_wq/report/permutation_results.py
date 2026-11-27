"""Report the retained block-permutation experiment for verified source models."""
from pathlib import Path
import pandas as pd
from sparse_wq.evaluate.block_permutation import atomic_text,digest,read_json,save_json


def copy_supplied(out,cases,*,results=None,models=None):
    from sparse_wq import paths
    out=Path(out);results=Path(results) if results is not None else paths.RESULTS
    models=Path(models) if models is not None else results/'models'
    source=results/'permutations'
    def unavailable():
        for name in ['tables/block_permutation.tex','block_permutation_summary.json','block_permutation_comparisons.csv']:
            (out/name).unlink(missing_ok=True)
        return False
    if not (source/'source_models.json').exists():return unavailable()
    expected=read_json(source/'source_models.json');current={c['target']:c for c in cases}
    retained=[]
    for target,original in expected['cases'].items():
        case=current.get(target)
        if case is None:continue
        if any(case[key]!=original[key] for key in expected['case_fields']):return unavailable()
        if case['evaluation']['timestamps']!=original['timestamps']:return unavailable()
        directory=models/case['method']/'half'/case['run']/target
        for relative,sha in original['files'].items():
            if not (directory/relative).exists() or digest(directory/relative)!=sha:return unavailable()
        retained.append(target)
    if not retained:return unavailable()
    for relative,sha in read_json(source/'release_manifest.json').items():
        path=(source/relative).resolve()
        if not path.is_relative_to(source.resolve()) or digest(path)!=sha:
            raise ValueError('Supplied block-permutation artifact changed: '+relative)
    (out/'tables').mkdir(parents=True,exist_ok=True)
    report=source/'report'
    summary=read_json(report/'summary.json')
    summary['targets']=[row for row in summary['targets'] if row['target'] in retained]
    save_json(out/'block_permutation_summary.json',summary)
    comparisons=pd.read_csv(report/'comparisons.csv')
    comparisons.loc[comparisons.target.isin(retained)].to_csv(out/'block_permutation_comparisons.csv',index=False)
    atomic_text(out/'tables/block_permutation.tex',(report/'block_permutation.tex').read_text(encoding='utf-8'))
    return True
