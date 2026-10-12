/* Offline feature-level summaries. Pure functions are exported for numerical tests. */
(function () {
'use strict';
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const mean=a=>{const v=a.filter(finite);return v.length?v.reduce((s,x)=>s+x,0)/v.length:null;};
const h=p=>p>0?-p*Math.log2(p):0;
const bh=p=>h(p)+h(1-p);
const delta=(a,b)=>finite(a)&&finite(b)?a-b:null;
const fmt=x=>finite(x)?x.toLocaleString(undefined,{maximumFractionDigits:3}):'—';
const signed=x=>finite(x)?(x>0?'+':'')+fmt(x):'—';
function source(feature,rows,labels) {
  const v=rows.map(i=>feature.values[i]).filter(x=>x!==null), n=v.length;
  const out={n_known:n,coverage:rows.length?n/rows.length:null,diversity:null,mean:null,variance:null};
  if(!n)return out;
  if(feature.kind==='continuous') {out.mean=mean(v);out.variance=mean(v.map(x=>(x-out.mean)**2));return out;}
  if(!labels.length)return out;
  const counts=labels.map(label=>v.filter(x=>feature.kind==='multilabel'?x.includes(label):String(x)===String(label)).length);
  out.diversity=feature.kind==='multilabel'?mean(counts.map(c=>bh(c/n))):counts.reduce((s,c)=>s+h(c/n),0)/(labels.length>1?Math.log2(labels.length):1);
  return out;
}
function geometry(feature,labels,graph,rows,k,globalSource,baseline) {
  const values=feature.values, numeric=feature.kind==='continuous', tags=feature.kind==='multilabel';
    let knownAnchors=0,knownEdges=0,totalEdges=0,abs=0,varSum=0,nVar=0,cross=0,entropySum=0,entropyN=0;
  const positive=labels.map(()=>({sum:0,n:0})), labelIndex=new Map(labels.map((label,c)=>[String(label),c]));
  for(const i of rows) {
    const value=values[i]; if(value===null)continue;
    knownAnchors++;totalEdges+=k;
    const neighbors=graph[i].slice(0,k).map(j=>values[j]).filter(v=>v!==null), n=neighbors.length;
    knownEdges+=n; if(!n)continue;
    if(numeric) {
      const m=mean(neighbors);varSum+=mean(neighbors.map(v=>(v-m)**2));nVar++;
      for(const v of neighbors){abs+=Math.abs(value-v);cross+=(value-globalSource.mean)*(v-globalSource.mean);}
    } else {
      let local=0;const counts=new Map();
      for(const v of neighbors)for(const label of (tags?v:[String(v)])){const c=labelIndex.get(String(label));if(c!==undefined)counts.set(c,(counts.get(c)||0)+1);}
      for(const count of counts.values())local+=tags?bh(count/n):h(count/n);
      for(const label of (tags?value:[String(value)])){const c=labelIndex.get(String(label));if(c!==undefined){positive[c].sum+=(counts.get(c)||0)/n;positive[c].n++;}}
      if(labels.length){entropySum+=local/(tags?labels.length:(labels.length>1?Math.log2(labels.length):1));entropyN++;}
    }
  }
  const out={k,known_neighbor_share:totalEdges?knownEdges/totalEdges:null,n_known:knownAnchors,n_edges:knownEdges,entropy:entropyN?entropySum/entropyN:null};
  if(numeric) {
    out.mean_neighbor_abs_difference=knownEdges?abs/knownEdges:null;
    out.mean_local_variance=nVar?varSum/nVar:null;
    out.variance_ratio_to_global=globalSource.variance>0?out.mean_local_variance/globalSource.variance:null;
    // Subset anchors retain full-source centering and variance for comparability.
    out.morans_i=knownEdges&&globalSource.variance>0?cross/(knownEdges*globalSource.variance):null;
  } else {
    const valid=positive.map((r,c)=>({...r,c})).filter(r=>r.n), total=valid.reduce((s,r)=>s+r.n,0);
    out.micro_purity=total?valid.reduce((s,r)=>s+r.sum,0)/total:null;
    out.macro_purity=mean(valid.map(r=>r.sum/r.n));
    out.micro_lift=total?valid.reduce((s,r)=>s+(baseline[r.c]>0?r.sum/baseline[r.c]:0),0)/total:null;
    out.macro_lift=mean(valid.map(r=>baseline[r.c]>0?r.sum/r.n/baseline[r.c]:null));
    out.n_anchors=total;
  }
  return out;
}
function classify(change,threshold=.02,direction='up',supported=true) {
  if(!supported||!finite(change))return 'warn';
  if(Math.abs(change)<=threshold)return 'neutral';
  if(direction==='descriptive')return change>0?'up':'down';
  return (change>0)===(direction==='up')?'good':'bad';
}
function metrics(truth,predicted,kind) {
  const n=truth.length;if(!n)return {n_compared:0};
  if(kind==='continuous') {
    const m=mean(truth), sst=truth.reduce((s,x)=>s+(x-m)**2,0), sse=truth.reduce((s,x,i)=>s+(x-predicted[i])**2,0);
    return {n_compared:n,mae:mean(truth.map((x,i)=>Math.abs(x-predicted[i]))),r2:n<2?null:sst>0?1-sse/sst:(sse===0?1:0),n_classes:null};
  }
  truth=truth.map(String);predicted=predicted.map(String);
  const observed=[...new Set(truth)], labels=[...new Set([...truth,...predicted])];
  const recall=observed.map(label=>{let total=0,hits=0;truth.forEach((v,i)=>{if(v===label){total++;if(predicted[i]===v)hits++;}});return hits/total;});
  const f1=labels.map(label=>{let tp=0,fp=0,fn=0;truth.forEach((v,i)=>{if(v===label&&predicted[i]===label)tp++;else if(predicted[i]===label)fp++;else if(v===label)fn++;});return 2*tp+fp+fn?2*tp/(2*tp+fp+fn):0;});
  return {n_compared:n,n_classes:observed.length,balanced_accuracy:mean(recall),macro_f1:mean(f1),accuracy:mean(truth.map((v,i)=>+(v===predicted[i])))};
}
function probe(records,rows,kind,model) {
  const selected=rows===null?null:new Set(rows), rec=records.filter(r=>r.model===model&&(!selected||selected.has(r.row_position)));
  if(kind!=='multilabel')return metrics(rec.map(r=>r.true),rec.map(r=>r.predicted),kind);
  const labels=[...new Set(rec.map(r=>r.label))], scores=labels.map(label=>{const r=rec.filter(x=>x.label===label);return metrics(r.map(x=>x.true),r.map(x=>x.predicted),'categorical');});
  return {n_compared:new Set(rec.map(r=>r.row_position)).size,n_labels:labels.length,
    n_supported_labels:scores.filter(x=>x.n_classes===2).length,balanced_accuracy:mean(scores.map(x=>x.balanced_accuracy)),macro_f1:mean(scores.map(x=>x.macro_f1))};
}
function external(feature,rows,labels) {
  const selected=rows.filter(i=>feature.values[i]!==null&&feature.predictions[i]!==null);
  if(feature.kind!=='multilabel')return metrics(selected.map(i=>feature.values[i]),selected.map(i=>feature.predictions[i]),feature.kind);
  const f1=labels.map(label=>{let tp=0,fp=0,fn=0;for(const i of selected){const t=feature.values[i].includes(label),p=feature.predictions[i].includes(label);if(t&&p)tp++;else if(p)fp++;else if(t)fn++;}return 2*tp+fp+fn?2*tp/(2*tp+fp+fn):0;});
  return {n_compared:selected.length,macro_f1:selected.length?mean(f1):null};
}
function curveMeaning(values,ks,trust,threshold=.02) {
  const v=values.map((value,i)=>({value,k:ks[i]})).filter(x=>trust.includes(x.k)&&finite(x.value));
  if(v.length<2)return 'Too few trusted, estimable k values to describe a trend.';
  const change=v[v.length-1].value-v[0].value;
  return `From k=${v[0].k} to ${v[v.length-1].k}, ${Math.abs(change)<=threshold?'little net change':change>0?'increases':'decreases'} (${signed(change)}).`;
}
function comparisonMeaning(values,ks,trust,direction,threshold=.02) {
  const usable=values.filter((x,i)=>trust.includes(ks[i])&&finite(x));if(!usable.length)return 'No comparable evidence at trusted scales.';
  const positive=usable.filter(x=>x>threshold).length,negative=usable.filter(x=>x< -threshold).length;
  if(direction==='descriptive')return `Increases at ${positive}/${usable.length} trusted scales; decreases at ${negative}/${usable.length}. These describe composition, not quality.`;
  const better=direction==='up'?positive:negative,worse=direction==='up'?negative:positive;
  return `Stronger organization at ${better}/${usable.length} trusted scales; weaker at ${worse}/${usable.length}. These scales are correlated.`;
}
const Core={source,geometry,metrics,probe,external,classify,curveMeaning,comparisonMeaning,mean,delta};
if(typeof module!=='undefined'&&module.exports){module.exports=Core;return;}
const payload=document.getElementById('explorer-payload');if(!payload)return;
const data=JSON.parse(payload.textContent), view=data.overview;window.EmbeddingAuditData=data;if(!view)return;
const $=id=>document.getElementById('ex-'+id), settings=view.settings, names=Object.keys(view.features), ks=view.ks;
let rowVersion=0;const curveCache=new Map();
let rows=data.available?data.ids.map((_,i)=>i):null, selectedFeatures=new Set(names), displayed=[], brush=null;
const allRows=rows?rows.slice():null, isAll=()=>rows===null||rows.length===data.ids.length;
function element(tag,text,cls){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;}
function badge(text,cls){return element('span',text,'ex-badge ex-'+cls);}
function line(parent,text,cls){parent.append(element('div',text,cls));}
function referenceSource(name,record) {
  if(data.available)return source(data.features[name],allRows,record.labels);
  return {n_known:record.source.n_known,coverage:record.source.coverage,mean:record.source.mean,variance:record.source.variance,diversity:record.diversity};
}
function getCurves(name,record,which,src) {
  const cacheKey=name+':'+which+':'+rowVersion;if(curveCache.has(cacheKey))return curveCache.get(cacheKey);
  if(isAll()) {
    const table=which==='B'?record.comparison_geometry:record.geometry, entropy=which==='B'?record.comparison_entropy:record.entropy;
    return ks.map(k=>({...table.find(x=>x.k===k),k,entropy:(entropy.find(x=>x.k===k)||{}).normalized_entropy}));
  }
  const feature=data.features[name], graph=which==='B'?data.comparison.graph:data.graph;
  const known=allRows.filter(i=>feature.values[i]!==null), has=(v,l)=>feature.kind==='multilabel'?v.includes(l):String(v)===String(l);
  const baseline=record.labels.map(label=>record.geometry.length&&feature.kind!=='continuous'?
    ((data.features[name].per_label.find(x=>String(x.label)===String(label))||{}).baseline_prevalence??known.filter(i=>has(feature.values[i],label)).length/known.length):null);
  const result=ks.map(k=>geometry(feature,record.labels,graph,rows,k,src,baseline));curveCache.set(cacheKey,result);return result;
}
function getProbe(record,which,model) {
  const predictions=which==='B'?record.comparison_probe_predictions:record.probe_predictions;
  if(predictions&&predictions.length)return probe(predictions,rows,record.kind,model);
  if(!isAll())return {n_compared:0};
  const table=which==='B'?record.comparison_probes:record.probes;
  return table.find(x=>x.model===model)||{n_compared:0};
}
function measureKey(kind) {
  const choice=$('measure').value, numeric=kind==='continuous';
  return ({diversity:numeric?'variance_ratio_to_global':'entropy',agreement:numeric?'mean_neighbor_abs_difference':'micro_purity',lift:numeric?'morans_i':'micro_lift',macro:numeric?'mean_local_variance':'macro_purity',macro_lift:numeric?'mean_neighbor_abs_difference':'macro_lift',support:'known_neighbor_share'})[choice];
}
function metricDirection(key) {return ['entropy','known_neighbor_share'].includes(key)?'descriptive':['mean_neighbor_abs_difference','mean_local_variance','variance_ratio_to_global'].includes(key)?'down':'up';}
function probeCell(record,src) {
  const cell=element('td'), linear=getProbe(record,'A','linear'), base=getProbe(record,'A','baseline'), numeric=record.kind==='continuous';
  const key=numeric?'r2':'balanced_accuracy', n=linear.n_compared??linear.n_test??0;
  let gain=numeric&&base.mae>0?(base.mae-linear.mae)/base.mae:!numeric?delta(linear[key],base[key]):null;
  const support=n>=settings.min_test&&(!numeric||base.mae>0)&&(numeric||(linear.n_classes===undefined?true:linear.n_classes>1))&&(linear.n_supported_labels===undefined||linear.n_supported_labels===linear.n_labels);
  if(!finite(linear[key])){line(cell,record.probe_status||'Not evaluated','muted');return {cell,text:'No estimable linear-probe evidence.'};}
  const cls=classify(gain,settings.probe_neutral_delta,'up',support);
  cell.append(badge(`${numeric?'R²':'Balanced accuracy'} ${fmt(linear[key])}`,cls));
  line(cell,`Baseline ${fmt(base[key])}; n test ${n}`);
  line(cell,numeric?`MAE ${fmt(linear.mae)}; baseline ${fmt(base.mae)}; improvement ${finite(gain)?fmt(100*gain)+'%':'—'}`:`Gain ${signed(gain)}; macro F1 ${fmt(linear.macro_f1)}`,'muted');
  if(linear.n_labels!==undefined)line(cell,`${linear.n_labels} tested tags; ${linear.n_supported_labels} with both test classes`,'muted');
  else if(record.kind==='multilabel')line(cell,`${record.probe_labels.length} selected binary tags`,'muted');
  const global=(record.probes||[]).find(x=>x.model==='linear');
  if(!isAll()&&global)line(cell,`Subset − all: ${signed(delta(linear[key],global[key]))}`,'muted');
  if(data.comparison||record.comparison_probes.length) {
    const b=getProbe(record,'B','linear'), difference=delta(b[key],linear[key]);
    cell.append(badge(`B ${fmt(b[key])}; B−A ${signed(difference)}`,classify(difference,settings.probe_neutral_delta,'up',support&&(b.n_compared??b.n_test??0)>=settings.min_test)));
    if(numeric)line(cell,`B MAE ${fmt(b.mae)}`,'muted');
  }
  let text=!support?'Probe evidence is limited by test support.':cls==='good'?'Linear prediction beats its baseline for this pipeline.':cls==='bad'?'Linear prediction falls below its baseline for this pipeline.':'Linear prediction is near baseline at the configured margin.';
  if(record.comparison_probes.length){const b=getProbe(record,'B','linear'), difference=delta(b[key],linear[key]);if(finite(difference))text+=` B has ${Math.abs(difference)<=settings.probe_neutral_delta?'similar':difference>0?'higher':'lower'} ${numeric?'R²':'balanced accuracy'} on the same split.`;}
  return {cell,text};
}
function render() {
  const body=$('overview');body.replaceChildren();displayed=[];
  const mode=$('mode').value;
  $('scope').textContent=(view.embedding_name+(view.comparison_name?' / '+view.comparison_name:'')+' · ')+(data.available?`${rows.length.toLocaleString()} of ${data.ids.length.toLocaleString()} evaluated rows · ${selectedFeatures.size} features · ${ks.length} neighborhood scales · ${data.excluded_rows} invalid embedding rows excluded.`:`Aggregate view · ${data.reason||'Row controls unavailable'} · ${selectedFeatures.size} features`);
  $('readout').textContent=mode==='values'?'Compare each feature across k. Blue shows diversity; prediction colors compare with the baseline.':mode==='selection'?'Changes are selected rows minus all evaluated rows. Blue arrows describe diversity; green means stronger organization for the displayed metric.':'Changes are embedding B minus A. Features, candidate pool and k match; positive changes are not always improvements.';
  $('curve-heading').textContent=$('measure').selectedOptions[0].text;
  for(const name of names) {
    if(!selectedFeatures.has(name))continue;
    const record=view.features[name], reference=referenceSource(name,record), src=data.available?source(data.features[name],rows,record.labels):reference;
    const curves=getCurves(name,record,'A',reference), full=ks.map(k=>({...record.geometry.find(x=>x.k===k),k,entropy:(record.entropy.find(x=>x.k===k)||{}).normalized_entropy}));
    const curvesB=record.comparison_geometry.length?getCurves(name,record,'B',reference):null;
    const key=measureKey(record.kind), direction=metricDirection(key), margin=settings.metric_thresholds[key]??settings.neutral_delta;
    const values=curves.map((r,i)=>{
      if(mode==='values')return r[key];if(mode==='selection')return delta(r[key],full[i][key]);if(!curvesB)return null;
      if(isAll()){
        if(key==='entropy')return (record.paired_entropy_delta.find(x=>x.k===ks[i])||{}).delta;
        return delta((record.paired_geometry_b.find(x=>x.k===ks[i])||{})[key],(record.paired_geometry_a.find(x=>x.k===ks[i])||{})[key]);
      }
      const feature=data.features[name], shared=rows.filter(row=>feature.values[row]!==null&&data.graph[row].slice(0,ks[i]).some(j=>feature.values[j]!==null)&&data.comparison.graph[row].slice(0,ks[i]).some(j=>feature.values[j]!==null));
      const known=allRows.filter(row=>feature.values[row]!==null);
      const baseline=record.labels.map(label=>(feature.kind==='continuous'?null:(data.features[name].per_label.find(x=>String(x.label)===String(label))||{}).baseline_prevalence)??known.filter(row=>feature.kind==='multilabel'?feature.values[row].includes(label):String(feature.values[row])===String(label)).length/known.length);
      return delta(geometry(feature,record.labels,data.comparison.graph,shared,ks[i],reference,baseline)[key],geometry(feature,record.labels,data.graph,shared,ks[i],reference,baseline)[key]);
    });
    const tr=element('tr'), info=element('td');line(info,name,'ex-feature-name');line(info,record.kind,'muted');
    line(info,`Known ${src.n_known}; coverage ${finite(src.coverage)?fmt(src.coverage*100)+'%':'—'}`);
    const diversityKey=record.kind==='continuous'?'mean':'diversity', change=delta(src[diversityKey],reference[diversityKey]);
    info.append(badge(`${record.kind==='continuous'?'Mean':'Source entropy'} ${fmt(src[diversityKey])}; Δ ${signed(change)}`,classify(change,settings.neutral_delta,'descriptive',src.n_known>=settings.min_known)));
    if(record.kind==='continuous')line(info,`Source variance ${fmt(src.variance)}; Δ ${signed(delta(src.variance,reference.variance))}`,'muted');
    line(info,`Coverage Δ ${signed(delta(src.coverage,reference.coverage))}`,'muted');
    if(record.kind!=='continuous')line(info,`${record.labels.length} fixed audited ${record.kind==='multilabel'?'tags':'labels'}`,'muted');
    const curveCell=element('td');line(curveCell,key.replaceAll('_',' '),'muted');const heat=element('div',undefined,'ex-heat');
    values.forEach((value,i)=> {
      const trusted=settings.trusted_ks.includes(ks[i]), supported=src.n_known>=settings.min_known&&(curves[i].known_neighbor_share??0)>=settings.min_known_neighbor_share;
      let cls=mode==='values'?(supported&&finite(value)?'neutral':'warn'):classify(value,margin,direction,supported);
      if(mode==='values'&&key==='entropy'&&finite(value)&&supported)cls='diversity';
      const tile=element('span',undefined,'ex-tile ex-'+cls+(trusted?'':' ex-untrusted'));
      tile.append(element('small',`k=${ks[i]}`),element('strong',mode==='values'?fmt(value):signed(value)));
      if(mode==='values'&&key==='entropy'&&finite(value)&&supported)tile.style.backgroundColor=`hsl(207 62% ${96-34*Math.min(1,Math.max(0,value))}%)`;
      tile.title=`Known-neighbor share ${fmt(curves[i].known_neighbor_share)}. ${trusted?'Included':'Excluded'} in trusted-scale narrative. ${supported?'Descriptive evidence':'Limited known support'}.`;
      heat.append(tile);
      displayed.push({feature:name,kind:record.kind,k:ks[i],measure:key,mode,value,known_rows:src.n_known,known_neighbor_share:curves[i].known_neighbor_share,trusted,source_entropy:src.diversity,source_mean:src.mean});
    });curveCell.append(heat);
    const narrativeKs=ks.filter((k,i)=>settings.trusted_ks.includes(k)&&src.n_known>=settings.min_known&&(curves[i].known_neighbor_share??0)>=settings.min_known_neighbor_share);
    line(curveCell,curveMeaning(values,ks,narrativeKs,margin),'muted');
    const p=probeCell(record,src), externalCell=element('td');
    if(record.external) {
      const score=!isAll()&&data.features[name].predictions?external(data.features[name],rows,record.labels):record.external;
      const metric=record.kind==='continuous'?'r2':record.kind==='multilabel'?'macro_f1':'balanced_accuracy';
      externalCell.append(badge(`${metric.replaceAll('_',' ')} ${fmt(score[metric])}`,score.n_compared>=settings.min_test?'neutral':'warn'));
      line(externalCell,`Matched truth ${score.n_compared??0}`);line(externalCell,`Uses embedding: ${String(record.external.predictor_inputs)}`,'muted');
      if(record.kind==='continuous')line(externalCell,`MAE ${fmt(score.mae)}; lower error is better`,'muted');
      else line(externalCell,record.kind==='multilabel'?'Macro F1 balances precision and recall across audited tags.':'Balanced accuracy is average recall across observed classes.','muted');
    } else line(externalCell,'Not supplied','muted');
    const interpretation=element('td'), measured=curves.map(r=>r[key]), first=measured.find(finite), last=[...measured].reverse().find(finite), trend=delta(last,first);
    let text=key==='entropy'?'Higher entropy means a more mixed neighborhood; lower means greater label concentration.':key==='known_neighbor_share'?'Higher means more annotated neighbors available.':direction==='down'?'Lower means closer numeric values or less neighborhood variation.':'Higher means more shared feature information among neighbors.';
    line(interpretation,text);line(interpretation,p.text,'ex-short');
    if(!isAll()&&finite(change))line(interpretation,record.kind==='continuous'?`The subset mean is ${signed(change)} units relative to all rows.`:`Source diversity is ${Math.abs(change)<=settings.neutral_delta?'similar to':change>0?'higher than':'lower than'} all rows (${signed(change)}).`);
    if(mode!=='values')line(interpretation,comparisonMeaning(values,ks,narrativeKs,direction,margin),'muted');
    if(record.kind==='multilabel')line(interpretation,'Tag absence is an operational negative among known annotations.','muted');
    const details=element('details'), summary=element('summary','All measurements and explanation');details.append(summary);
    const table=element('table'), headings=['k','Entropy','Micro purity','Macro purity','Micro lift','Macro lift','Abs difference','Local variance','Variance ratio','Moran’s I','Known share'];
    const header=element('tr');headings.forEach(t=>header.append(element('th',t)));table.append(header);
    for(const r of curves){const row=element('tr');[r.k,r.entropy,r.micro_purity,r.macro_purity,r.micro_lift,r.macro_lift,r.mean_neighbor_abs_difference,r.mean_local_variance,r.variance_ratio_to_global,r.morans_i,r.known_neighbor_share].forEach(v=>row.append(element('td',fmt(v))));table.append(row);}details.append(table);
    if(data.available&&data.features[name].explanation){details.append(element('h4','Full-run explanation (all evaluated rows)'));for(const paragraph of data.features[name].explanation.paragraphs)details.append(element('p',paragraph));}
    const link=element('a','Full feature plots, nulls and tables');link.href='#feature-'+names.indexOf(name);link.addEventListener('click',()=>{const select=document.getElementById('feature');select.value='feature-'+names.indexOf(name);window.showFeature();});details.append(link);interpretation.append(details);
    tr.append(info,curveCell,p.cell,externalCell,interpretation);body.append(tr);
    const probeMetric=record.kind==='continuous'?'r2':'balanced_accuracy', aProbe=getProbe(record,'A','linear'), bProbe=getProbe(record,'B','linear'), baseProbe=getProbe(record,'A','baseline');
    for(const row of displayed.filter(x=>x.feature===name))Object.assign(row,{probe_metric:probeMetric,probe_a:aProbe[probeMetric]??null,probe_baseline:baseProbe[probeMetric]??null,probe_b:bProbe[probeMetric]??null,probe_n_test:aProbe.n_compared??aProbe.n_test??0});
  }
  if(!selectedFeatures.size){const tr=element('tr'),td=element('td','Select features to compare.');td.colSpan=5;tr.append(td);body.append(tr);}
  if(data.available)drawMap();
  window.dispatchEvent(new CustomEvent('audit-feature-selection',{detail:{features:[...selectedFeatures]}}));
}
function renderFeatureChoices(){const list=$('features');list.replaceChildren();const query=$('search').value.toLowerCase();for(const name of names){if(!name.toLowerCase().includes(query))continue;const label=element('label'),box=element('input');box.type='checkbox';box.checked=selectedFeatures.has(name);box.addEventListener('change',()=>{box.checked?selectedFeatures.add(name):selectedFeatures.delete(name);render();});label.append(box,document.createTextNode(name));list.append(label);}}
$('search').addEventListener('input',renderFeatureChoices);
$('all-features').addEventListener('click',()=>{selectedFeatures=new Set(names);renderFeatureChoices();render();});
$('no-features').addEventListener('click',()=>{selectedFeatures.clear();renderFeatureChoices();render();});
$('measure').addEventListener('change',render);$('mode').addEventListener('change',render);
$('mode').options[1].disabled=!data.available;$('mode').options[2].disabled=!Object.values(view.features).some(x=>x.comparison_geometry.length);
$('row-tools').hidden=!data.available;
let mapPoints=[];
function drawMap(){const canvas=$('map'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);if(!data.coordinates){$('map-status').textContent='PCA coordinates were not generated.';return;}const x=data.coordinates.map(p=>p[0]),y=data.coordinates.map(p=>p[1]),xmin=x.reduce((a,b)=>Math.min(a,b),Infinity),xmax=x.reduce((a,b)=>Math.max(a,b),-Infinity),ymin=y.reduce((a,b)=>Math.min(a,b),Infinity),ymax=y.reduce((a,b)=>Math.max(a,b),-Infinity),selection=new Set(rows);mapPoints=data.coordinates.map(p=>[20+(p[0]-xmin)/(xmax-xmin||1)*(canvas.width-40),20+(p[1]-ymin)/(ymax-ymin||1)*(canvas.height-40)]);mapPoints.forEach((p,i)=>{ctx.fillStyle=selection.has(i)?'#146f9e':'#d4dde3';ctx.beginPath();ctx.arc(p[0],p[1],2.5,0,Math.PI*2);ctx.fill();});if(brush){ctx.strokeStyle='#17303f';ctx.strokeRect(brush[0],brush[1],brush[2]-brush[0],brush[3]-brush[1]);} $('map-status').textContent=`${rows.length} selected rows. Drag to select; All rows resets the selection.`;}
if(data.available){const cohorts={'All evaluated rows':data.ids,...data.cohorts}, select=$('cohort');for(const name of Object.keys(cohorts)){const opt=element('option',name);opt.value=name;select.append(opt);}const custom=element('option','Brushed region');custom.value='__brush__';custom.disabled=true;select.append(custom);
const byId=new Map(data.ids.map((id,i)=>[id,i]));select.addEventListener('change',()=>{rowVersion++;curveCache.clear();rows=cohorts[select.value].map(id=>byId.get(String(id)));brush=null;render();});
$('reset').addEventListener('click',()=>{rowVersion++;curveCache.clear();rows=allRows.slice();brush=null;select.value='All evaluated rows';render();});
const canvas=$('map'),point=e=>{const b=canvas.getBoundingClientRect();return [(e.clientX-b.left)*canvas.width/b.width,(e.clientY-b.top)*canvas.height/b.height];};let start=null;
canvas.addEventListener('pointerdown',e=>{if(!mapPoints.length)return;start=point(e);canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(!start)return;const end=point(e);brush=[...start,...end];drawMap();});
canvas.addEventListener('pointerup',e=>{if(!start)return;const end=point(e),x0=Math.min(start[0],end[0]),x1=Math.max(start[0],end[0]),y0=Math.min(start[1],end[1]),y1=Math.max(start[1],end[1]);rowVersion++;curveCache.clear();rows=mapPoints.map((p,i)=>p[0]>=x0&&p[0]<=x1&&p[1]>=y0&&p[1]<=y1?i:-1).filter(i=>i>=0);start=null;brush=[x0,y0,x1,y1];select.value='__brush__';render();});
canvas.addEventListener('pointercancel',()=>{start=null;brush=null;drawMap();});}
$('export').addEventListener('click',()=>{const blob=new Blob([JSON.stringify({run_id:data.run_id,selected_ids:rows?rows.map(i=>data.ids[i]):null,settings,measurements:displayed},null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=element('a');a.href=url;a.download='feature_measurements_selection.json';a.click();URL.revokeObjectURL(url);});
renderFeatureChoices();render();
})();
