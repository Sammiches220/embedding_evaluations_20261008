/* Aggregate normalization and filtration evidence; no single-radius or tag gate. */
(function(){
'use strict';
const finite=x=>typeof x==='number'&&Number.isFinite(x),fmt=x=>finite(x)?x.toLocaleString(undefined,{maximumFractionDigits:3}):'—';
const signed=x=>finite(x)?(x>0?'+':'')+fmt(x):'—';
function scoreGain(linear,baseline,kind){return kind==='continuous'?(baseline.mae>0?(baseline.mae-linear.mae)/baseline.mae:null):(finite(linear.balanced_accuracy)&&finite(baseline.balanced_accuracy)?linear.balanced_accuracy-baseline.balanced_accuracy:null);}
function gapDescription(rows,margin=.02){const usable=rows.filter(r=>finite(r.organization_gap));if(!usable.length)return 'No estimable shuffle comparison.';const positive=usable.filter(r=>r.organization_gap>margin).length;return `Less within-component variation than shuffled annotations at ${positive}/${usable.length} measured radii. These scales are correlated; this is descriptive evidence.`;}
function changeDescription(values,margin=.02){const usable=values.filter(finite);if(!usable.length)return 'No estimable paired model changes.';return `B has a larger organization gap at ${usable.filter(x=>x>margin).length}/${usable.length} radii and a smaller gap at ${usable.filter(x=>x< -margin).length}/${usable.length}. These correlated scale comparisons do not rank overall quality.`;}
const Core={scoreGain,gapDescription,changeDescription};
if(typeof module!=='undefined'&&module.exports){module.exports=Core;return;}
const node=document.getElementById('explorer-payload');if(!node)return;
const payload=window.EmbeddingAuditData||JSON.parse(node.textContent),data=payload.enhancements;if(!data)return;
const $=id=>document.getElementById('ea-'+id),settings=payload.overview.settings;
let selected=new Set(Object.keys(payload.overview.features));
function element(tag,text,cls){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;}
function classify(value,margin,supported,direction='up'){if(!supported||!finite(value))return 'warn';if(Math.abs(value)<=margin)return 'neutral';return (value>0)===(direction==='up')?'good':'bad';}
function badge(text,cls){return element('span',text,'ex-badge ex-'+cls);}
function options(select,values){select.replaceChildren();for(const value of values){const label=value==='A'?payload.overview.embedding_name:value==='B'?(payload.overview.comparison_name||'Embedding B'):value;const item=element('option',label);item.value=value;select.append(item);}}
const norm=data.normalization,topo=data.topology,metricTools=window.AuditMetrics;
const labelFor=key=>key==='A'?payload.overview.embedding_name:payload.overview.comparison_name;
const normMetrics=new Set(metricTools.available(norm.scores||[]));
const topologyMeta={organization_gap:'Shuffled − observed organization gap',observed:'Observed entropy / variance ratio',null_mean:'Shuffled entropy / variance ratio',model_delta:'Other embedding − selected organization gap'};
const topologyMetrics=new Set(Object.keys(topologyMeta));
function toggles(id,keys,labels,state,render){const target=$(id);target.replaceChildren();for(const key of keys){const label=element('label',undefined,'ex-metric-option'),box=element('input');box.type='checkbox';box.checked=state.has(key);box.addEventListener('change',()=>{box.checked?state.add(key):state.delete(key);render();});label.append(box,document.createTextNode(labels[key]));target.append(label);}}
options($('normalization-probe'),[...new Set((norm.scores||[]).map(r=>r.model).filter(x=>x!=='baseline'))]);
toggles('normalization-metrics',[...normMetrics],Object.fromEntries(Object.entries(metricTools.definitions).map(([k,v])=>[k,v[0]])),normMetrics,normalization);
if(!(topo.structure||[]).some(r=>r.embedding==='B')){topologyMetrics.delete('model_delta');delete topologyMeta.model_delta;}
toggles('topology-metrics',Object.keys(topologyMeta),topologyMeta,topologyMetrics,topology);
options($('normalization-model'),[...new Set((norm.scores||[]).map(x=>x.embedding))]);
options($('topology-model'),[...new Set((topo.structure||[]).map(x=>x.embedding))]);
options($('topology-space'),Object.keys(topo.radii||{}));

function normalization(){
 const body=$('normalization-table');body.replaceChildren();
 $('normalization-scope').textContent=norm.status==='ok'?norm.scope+' Global experiment; row brushing does not change these scores.':`Normalization analysis ${norm.status}. Enable normalization.enabled in audit_config.json. ${norm.reason||''}`;
 if(norm.status!=='ok')return;
 const embedding=$('normalization-model').value,probe=$('normalization-probe').value,variants=['raw_standardized','unit','unit_standardized','norm_only'];
 for(const name of selected){
  const featureRows=norm.scores.filter(r=>r.feature===name&&r.embedding===embedding),status=(norm.feature_status||{})[name+':'+embedding];if(!featureRows.length&&!status)continue;
  const row=element('tr'),feature=element('td');feature.append(element('strong',name));row.append(feature);
  const results={};
  for(const variant of variants){
   const cell=element('td'),score=featureRows.find(r=>r.variant===variant&&r.model===probe),base=featureRows.find(r=>r.variant===variant&&r.model==='baseline');
   if(!score||!base||score.status==='skipped'){const error=(status?.errors||[]).find(r=>r.variant===variant);cell.append(element('p',score?.reason||error?.reason||status?.reason||'Not evaluated','muted'));row.append(cell);continue;}
   const supported=(score.n_test??0)>=settings.min_test&&(score.n_classes===undefined||score.n_classes===null||score.n_classes>1)&&(score.n_labels===undefined||score.n_labels===null||score.n_supported_labels===score.n_labels);
   let count=0;
   for(const key of normMetrics){
    if(!finite(score[key]))continue;
    const gain=metricTools.gain(score[key],base[key],key),chip=element('div',undefined,'ex-probe-metric');
    chip.append(badge(`${metricTools.definitions[key][0]} ${fmt(score[key])}`,classify(gain,settings.probe_neutral_delta,supported&&finite(gain))));
    chip.append(element('div',`baseline ${fmt(base[key])} · ${metricTools.referenceText(key)} ${key==='mae'&&finite(gain)?fmt(100*gain)+'%':signed(gain)}`,'muted'));cell.append(chip);count++;
   }
   if(!count)cell.append(element('p','No selected metrics are estimable.','muted'));
   cell.append(element('div',`${probe.replaceAll('_',' ')} · ${fmt(score.n_test)} held-out rows · fit ${fmt(score.fit_seconds)} s`,'muted'));
   cell.append(element('div',`${score.tuning||'fixed'} settings; ${variant==='unit'?'unit inputs':variant==='norm_only'?'one magnitude coordinate':'coordinate standardized'}`,'muted'));
   results[variant]=score;row.append(cell);
  }
  const interpretation=element('td'),raw=results.raw_standardized,unit=results.unit;
  for(const key of normMetrics){if(finite(raw?.[key])&&finite(unit?.[key]))interpretation.append(element('p',`${metricTools.definitions[key][0]}: direction-only minus raw standardized ${signed(unit[key]-raw[key])}.`));}
  interpretation.append(element('p','Identical held-out rows across variants. Color compares each metric with its own dummy baseline. Error and divergence decrease; signed distribution differences improve toward zero. Distribution matching alone does not establish correct item predictions.','muted'));
  if(results.norm_only)interpretation.append(element('p','Magnitude-only results test information recoverable from vector length. Unit normalization removes that length.'));
  row.append(interpretation);body.append(row);
 }
}
function topology(){
 const body=$('topology-table');body.replaceChildren();
 $('topology-scope').textContent=topo.status==='ok'?`${topo.n_sample} of ${topo.n_population} evaluated rows per sample; ${topo.actual_repetitions} actual samples. ${topo.sample_scope}`:`Topology ${topo.status}. Enable topology.enabled in config. ${topo.reason||''}`;
 if(topo.status!=='ok')return;
 const space=$('topology-space').value,model=$('topology-model').value,other=model==='A'?'B':'A';
 $('topology-readout').textContent=`${labelFor(model)}. Each selected measurement is shown at every radius. Blue describes composition, green/red compares organization against shuffling. Paired changes are ${labelFor(other)||'other embedding'} minus ${labelFor(model)}. These are descriptive contrasts.`;
 for(const name of selected){
  const rows=(topo.features||[]).filter(r=>r.feature===name&&r.space===space&&r.embedding===model);if(!rows.length)continue;
  const row=element('tr'),info=element('td'),curve=element('td'),text=element('td');info.append(element('strong',name),element('div',rows[0].measure.replaceAll('_',' '),'muted'));
  info.append(element('div',`Mean known sample rows ${fmt(rows[0].n_known)}; ${rows[0].n_samples} samples`));
  for(const mode of topologyMetrics){
   curve.append(element('div',topologyMeta[mode],'ex-short'));const heat=element('div',undefined,'ex-heat'),changes=[];
   for(const record of rows){
    const counterpart=(topo.features||[]).find(r=>r.feature===name&&r.space===space&&r.embedding===other&&r.radius===record.radius);
    const value=mode==='model_delta'?(finite(record.organization_gap)&&finite(counterpart?.organization_gap)?counterpart.organization_gap-record.organization_gap:null):record[mode];
    changes.push(value);
    const supported=record.retained_known>=settings.min_known&&record.retained_share>=topo.options.min_retained_share;
    const descriptive=['observed','null_mean'].includes(mode),cls=descriptive?(supported&&finite(value)?'diversity':'warn'):classify(value,settings.neutral_delta,supported);
    const tile=element('span',undefined,'ex-tile ex-'+cls);tile.append(element('small',`ε=${fmt(record.radius)}`),element('strong',descriptive?fmt(value):signed(value)));
    tile.title=`Retained known share ${fmt(record.retained_share)}; eligible components ${fmt(record.n_components_measured)}; per-sample gap range ${fmt(record.sample_min)} to ${fmt(record.sample_max)}. Descriptive range, not a confidence interval.`;heat.append(tile);
   }
   curve.append(heat);
   if(mode==='model_delta')text.append(element('p',`${labelFor(other)} has a larger gap at ${changes.filter(v=>finite(v)&&v>settings.neutral_delta).length}/${changes.filter(finite).length} radii relative to ${labelFor(model)}.`));
  }
  const supportedRows=rows.filter(r=>r.retained_known>=settings.min_known&&r.retained_share>=topo.options.min_retained_share);
  text.append(element('p',supportedRows.length?gapDescription(supportedRows,settings.neutral_delta):'Limited known annotation support.'),element('p','Observed and shuffled entropy/variance are descriptive. Feature annotations explain organization; H0/H1 persistence itself does not identify semantics.','muted'));row.append(info,curve,text);body.append(row);
 }

 const h1=(topo.h1_status||[]).filter(r=>r.space===space&&r.embedding===model);$('h1-status').textContent=h1.length?h1.map(r=>r.status==='ok'?`H1 ${labelFor(r.embedding)}: ${r.n_points} points, ${r.n_simplices} simplices, cutoff ${fmt(r.cutoff)}; exact GF(2) within this sample.`:`H1 ${labelFor(r.embedding)}: skipped (${r.reason}).`).join(' '):'H1 disabled. H0 component and feature filtrations remain available.';
 const plots=$('topology-plots');plots.replaceChildren();const index=Object.keys(topo.radii).indexOf(space),path=topo.plots_by_embedding?.[space+':'+model]||(topo.plots||[])[index];if(path){const img=element('img');img.src=path;img.alt=space+' component curves and bounded persistence barcodes';plots.append(img);}
}
for(const id of ['normalization-model','normalization-probe','topology-space','topology-model'])$(id).addEventListener('change',()=>{normalization();topology();});
window.addEventListener('audit-feature-selection',event=>{selected=new Set(event.detail.features);normalization();topology();});
normalization();topology();
})();
