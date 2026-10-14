"""Evidence-backed sentences. No model calls, invented thresholds, or causes."""
import numpy as np
import pandas as pd


def number(value, percent=False):
    if value is None or not np.isfinite(value):
        return 'not estimable'
    return f'{value:.1%}' if percent else f'{value:,.4g}'


def explain_feature(name, result, manifest):
    """Return readable paragraphs and a plain evidence dictionary."""
    source = result.get('source', {})
    paragraphs = [f'{name}: {source.get("n_known", 0):,} of '
                  f'{source.get("n_items", manifest["n_rows"]):,} rows have known values '
                  f'({number(source.get("coverage"), True)} coverage).']
    evidence = {'feature': name, 'kind': result['kind'], 'source': source,
                'metric': manifest['config']['metric'], 'graph_source': manifest['graph_source'],
                'reference_scope': manifest['reference_scope'],
                'aggregation': ('known directed edges for mean absolute difference; per-anchor neighbor variance'
                                if result['kind']=='continuous' else 'positive anchor–label pairs for micro; equal labels for macro'),
                'neighborhood_missingness': manifest['neighborhood_missingness']}
    geometry = result.get('geometry', pd.DataFrame())
    if geometry.empty:
        paragraphs.append('Geometry is not estimable: ' + result.get('error', 'no usable anchors.'))
        return {'paragraphs': paragraphs, 'evidence': evidence}
    row = geometry.iloc[(geometry.k - 50).abs().argmin()]
    evidence['geometry_at_primary_k'] = row.to_dict()
    k = int(row.k)
    if result['kind'] == 'continuous':
        paragraphs.append(f'At k = {k}, known anchor–neighbor pairs differ by '
                          f'{number(row.mean_neighbor_abs_difference)} units on average. '
                          f'The local/global variance ratio is {number(row.variance_ratio_to_global)} '
                          f'and Moran’s I is {number(row.morans_i)}. '
                          'These describe similarity of nearby values, not prediction accuracy.')
    else:
        paragraphs.append(f'At k = {k}, mean shared-label agreement is '
                          f'{number(row.micro_purity, True)} when weighting positive anchor–label pairs, '
                          f'and {number(row.macro_purity, True)} when weighting labels equally. '
                          f'Mean individual label lift is {number(row.micro_lift)} and '
                          f'equal-label lift is {number(row.macro_lift)}. '
                          'Mean lift is not aggregate purity divided by aggregate chance.')
        labels = result.get('per_label', pd.DataFrame())
        selected = labels[labels.k == k].sort_values(['count', 'label'], ascending=[False, True]).head(3)
        evidence['example_labels'] = selected.to_dict('records')
        for label in selected.to_dict('records'):
            display = label.get('label_display', label['label'])
            paragraphs.append(f'{display}: {label["count"]:,} known positives; '
                              f'{label["n_anchors"]:,} evaluable positive anchors have '
                              f'{number(label["purity"], True)} mean agreement, versus '
                              f'{number(label["baseline_prevalence"], True)} reference prevalence '
                              f'(lift {number(label["lift"])}). '
                              f'An average of {number(label["effective_neighbors"])} of {k} neighbors '
                              'have known annotations. Large lift for a rare label can coexist with low agreement.')
    paragraphs.append(f'Known-neighbor coverage at this k is {number(row.known_neighbor_share, True)}. '
                      'Unknown neighbors are excluded from agreement denominators; missing annotations can bias the result.')
    nulls = result.get('null_comparison', pd.DataFrame())
    if not nulls.empty:
        null_row = nulls.iloc[(nulls.k - k).abs().argmin()]
        metric = 'mean_neighbor_abs_difference' if result['kind'] == 'continuous' else 'macro_lift'
        if metric + '_p' in null_row:
            evidence['null_comparison'] = null_row.to_dict()
            paragraphs.append(f'The saved null comparison gives an empirical one-sided p-value of '
                              f'{number(null_row[metric + "_p"])} for {metric}. '
                              'This tests the specified shuffle, not semantic understanding; use the combined '
                              'multiple-testing table and practical effect sizes when judging the finding.')
    probes = result.get('probes', pd.DataFrame())
    if not probes.empty:
        evidence['probes'] = probes.to_dict('records')
        paragraphs.append('The probe panel measures recoverability on a held-out split. Compare each model '
                          'with its baseline; recoverability and nearest-neighbor usefulness answer different questions.')
        for probe in probes.to_dict('records'):
            if probe.get('status','ok') != 'ok':
                continue
            metrics = ['mae','r2'] if result['kind']=='continuous' else ['balanced_accuracy','macro_f1']
            values = ', '.join(f'{metric} {number(probe[metric])}' for metric in metrics if metric in probe)
            paragraphs.append(f'{probe["model"]}: {values}; '
                              f'{number(probe.get("n_test"))} held-out test rows '
                              '(or the macro summary of supported binary tags).')
    prediction = result.get('prediction_metrics')
    if prediction:
        evidence['external_predictions'] = prediction
        metrics = ['mae','r2'] if result['kind']=='continuous' else ['macro_f1','micro_f1','jensen_shannon_divergence']
        values = ', '.join(f'{metric} {number(prediction[metric])}' for metric in metrics if metric in prediction)
        paragraphs.append(f'External predictions were scored on {prediction["n_compared"]:,} matched rows with known truth and predictions: '
                          f'{values}. No accuracy is measured on naturally missing truth. Prediction distributions over other rows have a separate scope.')
    missingness = result.get('missingness', pd.DataFrame())
    if not missingness.empty:
        evidence['missingness'] = missingness.to_dict('records')
        for trial in missingness.to_dict('records'):
            if trial.get('status') != 'ok':
                paragraphs.append(f'{trial["mode"]} missingness at {number(trial["missing_fraction"],True)}: skipped ({trial.get("reason","not estimable")}).')
                continue
            metric='mae' if result['kind']=='continuous' else 'macro_f1'
            paragraphs.append(f'With {number(trial["missing_fraction"],True)} of known rows hidden under {trial["mode"]}, '
                              f'{number(trial.get("n_hidden"))} hidden rows give {metric} {number(trial.get(metric))}. '
                              'This is an artificial recovery experiment, not a score on naturally unannotated rows.')
    comparison = result.get('paired_comparison', pd.DataFrame())
    if not comparison.empty:
        pair=comparison.iloc[(comparison.k-k).abs().argmin()]
        evidence['paired_comparison_at_primary_k']=pair.to_dict()
        paragraphs.append(f'For {pair.metric} at k = {int(pair.k)}, A is {number(pair.a)} and B is {number(pair.b)}, '
                          f'with B−A {number(pair.delta)} across {int(pair.n_paired_anchors):,} paired anchors. '
                          f'Conditional 95% interval: {number(pair.ci_low)} to {number(pair.ci_high)}; '
                          f'{pair.better_direction} is better for this measure. This does not rank overall embedding quality.')
    normalization=result.get('normalization_scores',pd.DataFrame())
    if len(normalization):
        evidence['normalization_scores']=normalization.to_dict('records')
        metric='r2' if result['kind']=='continuous' else 'balanced_accuracy'
        for model in ['A','B']:
            rows=normalization[(normalization.embedding==model)&(normalization.model=='linear')]
            messages=[]
            for variant in ['raw_standardized','unit','unit_standardized','norm_only']:
                match=rows[rows.variant==variant]
                if len(match): messages.append(f'{variant}: {metric} {number(match.iloc[0].get(metric))}')
            if messages: paragraphs.append(f'Embedding {model} normalization comparison on identical held-out rows: '+', '.join(messages)+'. Direction-only and magnitude-only scores describe accessible information for these pipelines; fit telemetry does not establish encoder-training stability.')
    topology=result.get('topology_summary',pd.DataFrame())
    if len(topology):
        evidence['topology_summary']=topology.to_dict('records')
        margin=manifest['config'].get('overview',{}).get('neutral_delta',.02)
        for (space,model),rows in topology.groupby(['space','embedding'],sort=False):
            finite=rows[rows.organization_gap.notna()]
            count=int((finite.organization_gap>margin).sum())
            if len(finite): paragraphs.append(f'In the {space} filtration for embedding {model}, the shuffled-minus-observed component variation gap exceeds {number(margin)} at {count} of {len(finite)} evaluated radii, across {int(rows.n_samples.max())} sampled cohorts. This is descriptive annotation organization across correlated scales, not a topology significance test or a semantic quality grade.')
    if manifest['config'].get('embedding_normalization')=='l2':
        paragraphs.append('The main analysis uses unit-normalized embedding vectors. Standardized linear-probe inputs can leave the unit sphere; original vector norms are saved separately.')
    paragraphs.append('Next: compare feature measurements across scales, check probe gains against baselines, or summarize a selected region. '
                      'Shared labels and correlated features can suggest patterns, but do not establish why the model retrieved an item.')
    return {'paragraphs': paragraphs, 'evidence': evidence}
