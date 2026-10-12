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
function options(select,values){select.replaceChildren();for(const value of values){const item=element('option',value);item.value=value;select.append(item);}}
const norm=data.normalization,topo=data.topology;
options($('normalization-model'),[...new Set((norm.scores||[]).map(x=>x.embedding))]);
options($('topology-model'),[...new Set((topo.structure||[]).map(x=>x.embedding))]);
options($('topology-space'),Object.keys(topo.radii||{}));
$('topology-measure').options[2].disabled=!(topo.structure||[]).some(r=>r.embedding==='B');
function normalization(){
 const body=$('normalization-table');body.replaceChildren();
 $('normalization-scope').textContent=norm.status==='ok'?norm.scope+' Global experiment; row brushing does not change these scores.':`Normalization analysis ${norm.status}. Enable normalization.enabled in audit_config.json. ${norm.reason||''}`;
 if(norm.status!=='ok')return;
 const model=$('normalization-model').value,variants=['raw_standardized','unit','unit_standardized','norm_only'];
 for(const name of selected){
  const featureRows=norm.scores.filter(r=>r.feature===name&&r.embedding===model);const status=(norm.feature_status||{})[name+':'+model];if(!featureRows.length&&!status)continue;
  const row=element('tr'),feature=element('td');feature.append(element('strong',name));row.append(feature);
  const kind=payload.overview.features[name].kind,key=kind==='continuous'?'r2':'balanced_accuracy';const results={};
  for(const variant of variants){
   const cell=element('td'),linear=featureRows.find(r=>r.variant===variant&&r.model==='linear'),base=featureRows.find(r=>r.variant===variant&&r.model==='baseline');
   if(!linear||!base||linear.status==='skipped'){const error=(status?.errors||[]).find(r=>r.variant===variant);cell.append(element('p',linear?.reason||error?.reason||status?.reason||'Not evaluated','muted'));row.append(cell);continue;}
   const gain=scoreGain(linear,base,kind),supported=(linear.n_test??0)>=settings.min_test&&(kind==='continuous'||(linear.n_classes??2)>1)&&(linear.n_labels===undefined||linear.n_supported_labels===linear.n_labels);
   cell.append(badge(`${kind==='continuous'?'R²':'Balanced accuracy'} ${fmt(linear[key])}`,classify(gain,settings.probe_neutral_delta,supported)));
   cell.append(element('div',`Baseline ${fmt(base[key])}; test ${fmt(linear.n_test)}`));
   cell.append(element('div',kind==='continuous'?`MAE ${fmt(linear.mae)}; improvement ${finite(gain)?fmt(100*gain)+'%':'—'}`:`Gain ${signed(gain)}; macro F1 ${fmt(linear.macro_f1)}`,'muted'));
   cell.append(element('div',`Fit ${fmt(linear.fit_seconds)} s; warnings ${fmt(linear.convergence_warnings)}; iterations ${fmt(linear.max_iterations)}`,'muted'));
   cell.append(element('div',`${linear.tuning||'fixed'} regularization; ${variant==='unit'?'unit inputs':variant==='norm_only'?'one magnitude coordinate':'coordinate standardized'}`,'muted'));
   if(linear.n_labels!==undefined)cell.append(element('div',`${linear.n_labels} binary tags tested`,'muted'));
   results[variant]=linear;row.append(cell);
  }
  const interpretation=element('td'),raw=results.raw_standardized,unit=results.unit;
  if(raw&&unit&&finite(raw[key])&&finite(unit[key])){
   const change=unit[key]-raw[key];interpretation.append(element('p',`Direction-only ${key.replaceAll('_',' ')} is ${Math.abs(change)<=settings.probe_neutral_delta?'similar to':change>0?'higher than':'lower than'} the raw standardized pipeline (${signed(change)}). Same held-out rows; access to information and regularization differ.`));
  }else interpretation.append(element('p','Insufficient comparable linear-probe evidence.'));
  if(results.norm_only)interpretation.append(element('p','Magnitude-only results show what can be recovered from vector length. Its baseline comparison determines whether this is useful evidence.'));
  if(model==='B'&&raw){const a=norm.scores.find(r=>r.feature===name&&r.embedding==='A'&&r.variant==='raw_standardized'&&r.model==='linear');if(a)interpretation.append(element('p',`Raw pipeline B−A: ${signed(raw[key]-a[key])}.`));}
  interpretation.append(element('p','A higher score does not establish a more stable encoder-training process.','muted'));row.append(interpretation);body.append(row);
 }
}
function topology(){
 const body=$('topology-table');body.replaceChildren();
 $('topology-scope').textContent=topo.status==='ok'?`${topo.n_sample} of ${topo.n_population} evaluated rows per sample; ${topo.actual_repetitions} actual samples. ${topo.sample_scope}`:`Topology ${topo.status}. Enable topology.enabled in config. ${topo.reason||''}`;
 if(topo.status!=='ok')return;
 const space=$('topology-space').value,model=$('topology-model').value,mode=$('topology-measure').value;
 $('topology-readout').textContent=mode==='observed'?'Observed component diversity: lower means more concentrated information; it is not automatically better.':mode==='model_delta'?'B minus A organization gap on identical samples and shared radii. Positive suggests stronger annotation organization for this null, not overall quality.':'Positive gap means less within-component variation than shuffled annotations on the same geometry. Singleton and fully merged extremes can give zero mechanically.';
 for(const name of selected){
  const rows=(topo.features||[]).filter(r=>r.feature===name&&r.space===space&&r.embedding===model);if(!rows.length)continue;
  const row=element('tr'),info=element('td'),curve=element('td'),text=element('td');info.append(element('strong',name),element('div',rows[0].measure.replaceAll('_',' '),'muted'));
  info.append(element('div',`Mean known sample rows ${fmt(rows[0].n_known)}; ${rows[0].n_samples} samples`));
  const heat=element('div',undefined,'ex-heat'),changes=[];
  for(const record of rows){
   const a=(topo.features||[]).find(r=>r.feature===name&&r.space===space&&r.embedding==='A'&&r.radius===record.radius),b=(topo.features||[]).find(r=>r.feature===name&&r.space===space&&r.embedding==='B'&&r.radius===record.radius);
   const value=mode==='model_delta'?(finite(a?.organization_gap)&&finite(b?.organization_gap)?b.organization_gap-a.organization_gap:null):record[mode];
   changes.push(value);
   const supported=record.retained_known>=settings.min_known&&record.retained_share>=topo.options.min_retained_share;
   const cls=mode==='observed'?(supported&&finite(value)?'neutral':'warn'):classify(value,settings.neutral_delta,supported);
   const tile=element('span',undefined,'ex-tile ex-'+cls);tile.append(element('small',`ε=${fmt(record.radius)}`),element('strong',mode==='observed'?fmt(value):signed(value)));
   tile.title=`Retained known share ${fmt(record.retained_share)}; eligible components ${fmt(record.n_components_measured)}; per-sample gap range ${fmt(record.sample_min)} to ${fmt(record.sample_max)}. This range is descriptive, not a confidence interval.`;heat.append(tile);
  }
  const supportedRows=rows.filter(r=>r.retained_known>=settings.min_known&&r.retained_share>=topo.options.min_retained_share);
  const description=mode==='model_delta'?changeDescription(changes,settings.neutral_delta):supportedRows.length?gapDescription(supportedRows,settings.neutral_delta):'Limited known annotation support; interpret these measurements descriptively.';
  curve.append(heat);text.append(element('p',description),element('p','Feature annotations explain organization; H0/H1 persistence itself does not identify semantics.','muted'));row.append(info,curve,text);body.append(row);
 }
 const h1=(topo.h1_status||[]).filter(r=>r.space===space);$('h1-status').textContent=h1.length?h1.map(r=>r.status==='ok'?`H1 ${r.embedding}: ${r.n_points} points, ${r.n_simplices} simplices, cutoff ${fmt(r.cutoff)}; exact GF(2) within this sample.`:`H1 ${r.embedding}: skipped (${r.reason}).`).join(' '):'H1 disabled. H0 component and feature filtrations remain available.';
 const plots=$('topology-plots');plots.replaceChildren();const index=Object.keys(topo.radii).indexOf(space),path=(topo.plots||[])[index];if(path){const img=element('img');img.src=path;img.alt=space+' component curves and bounded persistence barcodes';plots.append(img);}
}
for(const id of ['normalization-model','topology-space','topology-model','topology-measure'])$(id).addEventListener('change',()=>{normalization();topology();});
window.addEventListener('audit-feature-selection',event=>{selected=new Set(event.detail.features);normalization();topology();});
normalization();topology();
})();
