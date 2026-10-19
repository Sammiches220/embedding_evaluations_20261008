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
function rankAuc(truth,scores) {
  const n=truth.length,pos=truth.reduce((s,x)=>s+(x?1:0),0),neg=n-pos;if(!pos||!neg)return null;
  const order=scores.map((score,i)=>({score,i})).sort((a,b)=>a.score-b.score),ranks=Array(n);let start=0;
  while(start<n){let end=start+1;while(end<n&&order[end].score===order[start].score)end++;const rank=(start+1+end)/2;for(let i=start;i<end;i++)ranks[order[i].i]=rank;start=end;}
  const rankSum=ranks.reduce((s,r,i)=>s+(truth[i]?r:0),0);return (rankSum-pos*(pos+1)/2)/(pos*neg);
}
function averagePrecision(truth,scores) {
 const positives=truth.reduce((s,x)=>s+(x?1:0),0);if(!positives)return null;
 const order=scores.map((score,i)=>({score,i})).sort((a,b)=>b.score-a.score);let hits=0,total=0,ap=0,start=0;
 while(start<order.length){let end=start,groupHits=0;while(end<order.length&&order[end].score===order[start].score){if(truth[order[end].i])groupHits++;end++;}hits+=groupHits;total=end;ap+=groupHits/positives*hits/total;start=end;}return ap;
}
function ranks(values){const sorted=values.map((v,i)=>({v,i})).sort((a,b)=>a.v-b.v),out=Array(values.length);let start=0;while(start<sorted.length){let end=start+1;while(end<sorted.length&&sorted[end].v===sorted[start].v)end++;for(let j=start;j<end;j++)out[sorted[j].i]=(start+end-1)/2;start=end;}return out;}
function spearman(a,b){const x=ranks(a),y=ranks(b),xm=mean(x),ym=mean(y),xx=x.map(v=>v-xm),yy=y.map(v=>v-ym),den=Math.sqrt(xx.reduce((s,v)=>s+v*v,0)*yy.reduce((s,v)=>s+v*v,0));return den>0?xx.reduce((s,v,i)=>s+v*yy[i],0)/den:null;}
function distributionMetrics(truth,predicted){
 const labels=[...new Set([...truth,...predicted])],tc=labels.map(l=>truth.filter(v=>v===l).length),pc=labels.map(l=>predicted.filter(v=>v===l).length),n=truth.length;
 const t=tc.map(x=>x/n),p=pc.map(x=>x/n),ht=t.reduce((s,v)=>s+h(v),0),hp=p.reduce((s,v)=>s+h(v),0),nt=tc.filter(x=>x>0).length,np=pc.filter(x=>x>0).length;
 const norm=(entropy,count)=>count>1?entropy/Math.log2(count):0,m=t.map((v,i)=>(v+p[i])/2),kl=(a,b)=>a.reduce((s,v,i)=>s+(v>0?v*Math.log2(v/b[i]):0),0);
 return {entropy_delta:(hp-ht)*Math.LN2,normalized_entropy_delta:norm(hp,np)-norm(ht,nt),largest_label_share_delta:Math.max(...p)-Math.max(...t),n_labels_delta:np-nt,jensen_shannon_divergence:(kl(t,m)+kl(p,m))/2};
}
function metrics(truth,predicted,kind,records=[]) {
  const n=truth.length;if(!n)return {n_compared:0};
  if(kind==='continuous') {
    const m=mean(truth), sst=truth.reduce((s,x)=>s+(x-m)**2,0), sse=truth.reduce((s,x,i)=>s+(x-predicted[i])**2,0);
    return {n_compared:n,spearman:spearman(truth,predicted),mae:mean(truth.map((x,i)=>Math.abs(x-predicted[i]))),r2:n<2?null:sst>0?1-sse/sst:(sse===0?1:0),n_classes:null};
  }
  truth=truth.map(String);predicted=predicted.map(String);
  const observed=[...new Set(truth)], labels=[...new Set([...truth,...predicted])];
  const recall=observed.map(label=>{let total=0,hits=0;truth.forEach((v,i)=>{if(v===label){total++;if(predicted[i]===v)hits++;}});return hits/total;});
  const f1=labels.map(label=>{let tp=0,fp=0,fn=0;truth.forEach((v,i)=>{if(v===label&&predicted[i]===label)tp++;else if(predicted[i]===label)fp++;else if(v===label)fn++;});return 2*tp+fp+fn?2*tp/(2*tp+fp+fn):0;});
  const out={n_compared:n,n_classes:observed.length,balanced_accuracy:mean(recall),macro_f1:mean(f1),accuracy:mean(truth.map((v,i)=>+(v===predicted[i])))};
  const probabilityRows=observed.map(label=>{
    const p=records.map(r=>r.probabilities?.[String(label)]??(finite(r.probability)?(String(label)==='1'?r.probability:1-r.probability):null));if(p.some(x=>!finite(x)))return null;
    const binary=truth.map(v=>String(v)===String(label));return {auc:rankAuc(binary,p),ap:averagePrecision(binary,p)};
  }).filter(Boolean);
  Object.assign(out,distributionMetrics(truth,predicted));out.micro_f1=out.accuracy;out.roc_auc=mean(probabilityRows.map(x=>x.auc));out.average_precision=mean(probabilityRows.map(x=>x.ap));return out;
}
function probe(records,rows,kind,model) {
  const selected=rows===null?null:new Set(rows), rec=records.filter(r=>r.model===model&&(!selected||selected.has(r.row_position)));
  if(kind!=='multilabel')return metrics(rec.map(r=>r.true),rec.map(r=>r.predicted),kind,rec);
  const labels=[...new Set(rec.map(r=>r.label))], scores=labels.map(label=>{const r=rec.filter(x=>x.label===label),score=metrics(r.map(x=>x.true),r.map(x=>x.predicted),'categorical',r),truth=r.map(x=>!!x.true),probability=r.map(x=>x.probability);score.roc_auc=probability.every(finite)?rankAuc(truth,probability):null;score.average_precision=probability.every(finite)?averagePrecision(truth,probability):null;return score;});
  return {n_compared:new Set(rec.map(r=>r.row_position)).size,n_labels:labels.length,
    n_supported_labels:scores.filter(x=>x.n_classes===2).length,
    balanced_accuracy:mean(scores.map(x=>x.balanced_accuracy)),macro_f1:mean(scores.map(x=>x.macro_f1)),
    accuracy:mean(scores.map(x=>x.accuracy)),micro_f1:mean(scores.map(x=>x.micro_f1)),entropy_delta:mean(scores.map(x=>x.entropy_delta)),normalized_entropy_delta:mean(scores.map(x=>x.normalized_entropy_delta)),largest_label_share_delta:mean(scores.map(x=>x.largest_label_share_delta)),n_labels_delta:mean(scores.map(x=>x.n_labels_delta)),jensen_shannon_divergence:mean(scores.map(x=>x.jensen_shannon_divergence)),roc_auc:mean(scores.map(x=>x.roc_auc)),average_precision:mean(scores.map(x=>x.average_precision))};
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
const Core={source,geometry,metrics,probe,classify,curveMeaning,comparisonMeaning,mean,delta};
if(typeof module!=='undefined'&&module.exports){module.exports=Core;return;}
const payload=document.getElementById('explorer-payload');if(!payload)return;
const data=JSON.parse(payload.textContent), view=data.overview;window.EmbeddingAuditData=data;if(!view)return;
const $=id=>document.getElementById('ex-'+id), settings=view.settings, names=Object.keys(view.features), ks=view.ks;
let rowVersion=0;const curveCache=new Map();
let rows=data.available?data.ids.map((_,i)=>i):null, selectedFeatures=new Set(names), displayed=[], brush=null;
const allRows=rows?rows.slice():null, isAll=()=>rows===null||rows.length===data.ids.length;
const metricTools=window.AuditMetrics,metricsMeta=metricTools.definitions;
let selectedProbeMetrics=new Set(), initializingControls=true;
function element(tag,text,cls){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;}
function badge(text,cls){return element('span',text,'ex-badge ex-'+cls);}
function line(parent,text,cls){parent.append(element('div',text,cls));}
function referenceSource(name,record) {
  if(data.available&&record.kind!=='summary')return source(data.features[name],allRows,record.labels);
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
  const table=which==='B'?record.comparison_probes:record.probes;
  if(isAll())return table.find(x=>x.model===model)||{n_compared:0};
  const predictions=which==='B'?record.comparison_probe_predictions:record.probe_predictions;
  if(predictions&&predictions.length)return probe(predictions,rows,record.kind,model);
  if(!isAll())return {n_compared:0};
  return {n_compared:0};
}
function currentEmbedding(){return $('embedding').value||'A';}
function otherEmbedding(){return currentEmbedding()==='A'?'B':'A';}
function embeddingLabel(key){return key==='A'?view.embedding_name:(view.comparison_name||'Embedding B');}
function selectedEmbeddingLabel(){return embeddingLabel(currentEmbedding());}
function availableProbes(){return [...new Set(Object.values(view.features).flatMap(r=>[...(r.probes||[]),...(r.comparison_probes||[])].map(x=>x.model).filter(x=>x&&x!=='baseline')))];}
function availableProbeMetrics(){return metricTools.available(Object.values(view.features).flatMap(r=>[...(r.probes||[]),...(r.comparison_probes||[])]));}
function initializeControls(){
  const embedding=$('embedding');embedding.replaceChildren();[['A',view.embedding_name],...(view.comparison_name?[['B',view.comparison_name]]:[])].forEach(([key,label])=>{const o=element('option',label);o.value=key;embedding.append(o);});
  const probes=$('probe');probes.replaceChildren();availableProbes().forEach(name=>{const o=element('option',name.replaceAll('_',' '));o.value=name;probes.append(o);});if(!availableProbes().length){const o=element('option','No configured probe');o.value='';probes.append(o);probes.disabled=true;}
  selectedProbeMetrics=new Set(availableProbeMetrics());renderMetricToggles();
}
function renderMetricToggles(){const box=$('probe-metrics');box.replaceChildren();for(const key of availableProbeMetrics()){const label=element('label',undefined,'ex-metric-option'),input=element('input');input.type='checkbox';input.checked=selectedProbeMetrics.has(key);input.addEventListener('change',()=>{input.checked?selectedProbeMetrics.add(key):selectedProbeMetrics.delete(key);render();});label.append(input,document.createTextNode(metricsMeta[key][0]));box.append(label);}}
function measureKey(kind) {
  const choice=$('measure').value, numeric=kind==='continuous';
  return ({diversity:numeric?'variance_ratio_to_global':'entropy',agreement:numeric?'mean_neighbor_abs_difference':'micro_purity',lift:numeric?'morans_i':'micro_lift',macro:numeric?'mean_local_variance':'macro_purity',macro_lift:numeric?'mean_neighbor_abs_difference':'macro_lift',support:'known_neighbor_share'})[choice];
}
function metricDirection(key) {return ['entropy','known_neighbor_share'].includes(key)?'descriptive':['mean_neighbor_abs_difference','mean_local_variance','variance_ratio_to_global'].includes(key)?'down':'up';}
function probeCell(record) {
  const cell=element('td'),embedding=currentEmbedding(),model=$('probe').value||'linear';
  const score=getProbe(record,embedding,model),base=getProbe(record,embedding,'baseline'),n=score.n_compared??score.n_test??0;
  const support=n>=settings.min_test&&(score.n_supported_labels===undefined||score.n_supported_labels===score.n_labels),shown=[];
  for(const key of selectedProbeMetrics){
    const value=score[key],reference=base[key];if(!finite(value))continue;
    const change=metricTools.gain(value,reference,key);
    const cls=classify(change,settings.probe_neutral_delta,'up',support&&finite(change));
    const chip=element('span',undefined,'ex-probe-metric');chip.append(badge(`${metricsMeta[key][0]} ${fmt(value)}`,cls));
    line(chip,`baseline ${fmt(reference)} · ${metricTools.referenceText(key)} ${key==='mae'&&finite(change)?fmt(100*change)+'%':signed(change)}`,'muted');cell.append(chip);shown.push(key);
    if(!isAll()){const overall=(embedding==='B'?record.comparison_probes:record.probes).find(x=>x.model===model);if(overall&&finite(overall[key]))line(chip,`selected rows ${signed(delta(value,overall[key]))} vs all`,'muted');}
  }
  if(!shown.length)line(cell,score.status==='skipped'?(score.reason||'Not estimable for this probe and feature.'):'No selected metrics are estimable for this probe.','muted');
  else line(cell,`${model.replaceAll('_',' ')} · ${n} held-out rows${score.n_labels!==undefined?` · ${score.n_supported_labels}/${score.n_labels} supported tags`:''}`,'muted');
  return {cell,text:shown.length?`${model.replaceAll('_',' ')} compared with the dummy baseline on ${n} held-out rows; color follows each metric’s direction.`:`No estimable ${model.replaceAll('_',' ')} metrics.`};
}
function render() {
  const body=$('overview');body.replaceChildren();displayed=[];
  const mode=$('mode').value, selected=currentEmbedding(), other=otherEmbedding(), probeName=$('probe').value||availableProbes()[0]||'linear';
  $('scope').textContent=(embeddingLabel(selected)+(view.comparison_name?' / '+embeddingLabel(other):'')+' · ')+(data.available?`${rows.length.toLocaleString()} of ${data.ids.length.toLocaleString()} evaluated rows · ${selectedFeatures.size} features · ${ks.length} neighborhood scales · ${data.excluded_rows} invalid embedding rows excluded.`:`Aggregate view · ${data.reason||'Row controls unavailable'} · ${selectedFeatures.size} features`);
  $('readout').textContent=mode==='values'?'Compare each feature across k. Neighborhood colors use the chosen metric; probe colors compare against the baseline.':mode==='selection'?'Changes are selected rows minus all evaluated rows. Blue arrows describe diversity; green means stronger organization for the displayed metric.':`${embeddingLabel(other)} minus ${embeddingLabel(selected)}. Features, candidate pool and k match; positive changes are not always improvements.`;
  $('curve-heading').textContent=$('measure').selectedOptions[0].text;
  for(const name of names) {
    if(!selectedFeatures.has(name))continue;
    const record=view.features[name], reference=referenceSource(name,record), src=data.available&&record.kind!=='summary'?source(data.features[name],rows,record.labels):reference;
    const curves=getCurves(name,record,selected,reference), full=ks.map(k=>({...((selected==='B'?record.comparison_geometry:record.geometry).find(x=>x.k===k)),k,entropy:(((selected==='B'?record.comparison_entropy:record.entropy).find(x=>x.k===k)||{}).normalized_entropy)}));
    const otherTable=other==='B'?record.comparison_geometry:record.geometry, curvesOther=otherTable.length?getCurves(name,record,other,reference):null;
    const key=measureKey(record.kind), direction=metricDirection(key), margin=settings.metric_thresholds[key]??settings.neutral_delta;
    const values=curves.map((r,i)=>{
      if(mode==='values')return r[key];if(mode==='selection')return delta(r[key],full[i][key]);if(!curvesOther)return null;
      if(isAll()){
        if(key==='entropy'){const d=(record.paired_entropy_delta.find(x=>x.k===ks[i])||{}).delta;return selected==='A'?d:-d;}
        const a=(record.paired_geometry_a.find(x=>x.k===ks[i])||{})[key],b=(record.paired_geometry_b.find(x=>x.k===ks[i])||{})[key];return selected==='A'?delta(b,a):delta(a,b);
      }
      const feature=data.features[name], graphSelected=selected==='B'?data.comparison.graph:data.graph, graphOther=other==='B'?data.comparison.graph:data.graph;
      const shared=rows.filter(row=>feature.values[row]!==null&&graphSelected[row].slice(0,ks[i]).some(j=>feature.values[j]!==null)&&graphOther[row].slice(0,ks[i]).some(j=>feature.values[j]!==null));
      const known=allRows.filter(row=>feature.values[row]!==null);
      const baseline=record.labels.map(label=>(feature.kind==='continuous'?null:(data.features[name].per_label.find(x=>String(x.label)===String(label))||{}).baseline_prevalence)??known.filter(row=>feature.kind==='multilabel'?feature.values[row].includes(label):String(feature.values[row])===String(label)).length/known.length);
      return delta(geometry(feature,record.labels,graphOther,shared,ks[i],reference,baseline)[key],geometry(feature,record.labels,graphSelected,shared,ks[i],reference,baseline)[key]);
    });
    const tr=element('tr'), info=element('td');line(info,name,'ex-feature-name');line(info,record.semantic_kind||record.kind,'muted');if(record.units)line(info,'Units: '+record.units,'muted');
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
      const tile=element('span',undefined,'ex-tile ex-'+cls+(trusted?'':' ex-untrusted'));tile.append(element('small',`k=${ks[i]}`),element('strong',mode==='values'?fmt(value):signed(value)));
      if(mode==='values'&&key==='entropy'&&finite(value)&&supported)tile.style.backgroundColor=`hsl(207 62% ${96-34*Math.min(1,Math.max(0,value))}%)`;
      tile.title=`Known-neighbor share ${fmt(curves[i].known_neighbor_share)}. ${trusted?'Included':'Excluded'} in trusted-scale narrative. ${supported?'Descriptive evidence':'Limited known support'}.`;heat.append(tile);
      displayed.push({feature:name,kind:record.semantic_kind||record.kind,units:record.units||'',k:ks[i],measure:key,mode,value,known_rows:src.n_known,known_neighbor_share:curves[i].known_neighbor_share,trusted,source_entropy:src.diversity,source_mean:src.mean});
    });curveCell.append(heat);
    const narrativeKs=ks.filter((k,i)=>settings.trusted_ks.includes(k)&&src.n_known>=settings.min_known&&(curves[i].known_neighbor_share??0)>=settings.min_known_neighbor_share);
    line(curveCell,curveMeaning(values,ks,narrativeKs,margin),'muted');
    const p=probeCell(record);
    if(record.population_source) {const pop=record.population_source;line(info,`Source population: ${pop.n_known} known / ${pop.n_items} rows; ${pop.scope||'declared scope'}`,'muted');}
    if(record.kind==='summary')line(curveCell,'Unavailable for this source-only kind','muted');
    const interpretation=element('td'), measured=curves.map(r=>r[key]), first=measured.find(finite), last=[...measured].reverse().find(finite);
    let text=record.kind==='summary'?'Source-only information; geometry and direct probes are unavailable.':key==='entropy'?'Higher entropy means a more mixed neighborhood; lower means greater label concentration.':key==='known_neighbor_share'?'Higher means more annotated neighbors available.':direction==='down'?'Lower means closer numeric values or less neighborhood variation.':'Higher means more shared feature information among neighbors.';
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
    tr.append(info,curveCell,p.cell,interpretation);body.append(tr);
    const metricKeys=[...selectedProbeMetrics], base=getProbe(record,selected,'baseline'), score=getProbe(record,selected,probeName), otherScore=getProbe(record,other,probeName);
    for(const row of displayed.filter(x=>x.feature===name))Object.assign(row,{probe:probeName,embedding:selected,probe_metrics:Object.fromEntries(metricKeys.map(m=>[m,score[m]??null])),probe_baseline:Object.fromEntries(metricKeys.map(m=>[m,base[m]??null])),probe_other:Object.fromEntries(metricKeys.map(m=>[m,otherScore[m]??null])),probe_n_test:score.n_compared??score.n_test??0});
  }
  if(!selectedFeatures.size){const tr=element('tr'),td=element('td','Select features to compare.');td.colSpan=4;tr.append(td);body.append(tr);}
  if(data.available)drawMap();
  window.dispatchEvent(new CustomEvent('audit-feature-selection',{detail:{features:[...selectedFeatures]}}));
}
function renderFeatureChoices(){const list=$('features');list.replaceChildren();const query=$('search').value.toLowerCase();for(const name of names){if(!name.toLowerCase().includes(query))continue;const label=element('label'),box=element('input');box.type='checkbox';box.checked=selectedFeatures.has(name);box.addEventListener('change',()=>{box.checked?selectedFeatures.add(name):selectedFeatures.delete(name);render();});label.append(box,document.createTextNode(name));list.append(label);}}
$('search').addEventListener('input',renderFeatureChoices);
$('all-features').addEventListener('click',()=>{selectedFeatures=new Set(names);renderFeatureChoices();render();});
$('no-features').addEventListener('click',()=>{selectedFeatures.clear();renderFeatureChoices();render();});
$('measure').addEventListener('change',render);$('mode').addEventListener('change',render);$('embedding').addEventListener('change',()=>{render();if(data.available)drawMap();});$('probe').addEventListener('change',render);
initializeControls();
$('mode').options[1].disabled=!data.available;$('mode').options[2].disabled=!Object.values(view.features).some(x=>x.comparison_geometry.length);
$('row-tools').hidden=!data.available;
let mapPoints=[];
function drawMap(){const canvas=$('map'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);const coordinates=currentEmbedding()==='B'?(data.comparison?.coordinates||null):data.coordinates;if(!coordinates){$('map-status').textContent=`PCA coordinates are unavailable for ${embeddingLabel(currentEmbedding())}.`;return;}const x=coordinates.map(p=>p[0]),y=coordinates.map(p=>p[1]),xmin=x.reduce((a,b)=>Math.min(a,b),Infinity),xmax=x.reduce((a,b)=>Math.max(a,b),-Infinity),ymin=y.reduce((a,b)=>Math.min(a,b),Infinity),ymax=y.reduce((a,b)=>Math.max(a,b),-Infinity),selection=new Set(rows);mapPoints=coordinates.map(p=>[20+(p[0]-xmin)/(xmax-xmin||1)*(canvas.width-40),20+(p[1]-ymin)/(ymax-ymin||1)*(canvas.height-40)]);mapPoints.forEach((p,i)=>{ctx.fillStyle=selection.has(i)?'#146f9e':'#d4dde3';ctx.beginPath();ctx.arc(p[0],p[1],2.5,0,Math.PI*2);ctx.fill();});if(brush){ctx.strokeStyle='#17303f';ctx.strokeRect(brush[0],brush[1],brush[2]-brush[0],brush[3]-brush[1]);} $('map-status').textContent=`${rows.length} selected rows in ${embeddingLabel(currentEmbedding())}. Drag to select; All rows resets the selection.`;}
if(data.available){const cohorts={'All evaluated rows':data.ids,...data.cohorts}, select=$('cohort');for(const name of Object.keys(cohorts)){const opt=element('option',name);opt.value=name;select.append(opt);}const custom=element('option','Brushed region');custom.value='__brush__';custom.disabled=true;select.append(custom);
const byId=new Map(data.ids.map((id,i)=>[id,i]));select.addEventListener('change',()=>{rowVersion++;curveCache.clear();rows=cohorts[select.value].map(id=>byId.get(String(id)));brush=null;render();});
$('reset').addEventListener('click',()=>{rowVersion++;curveCache.clear();rows=allRows.slice();brush=null;select.value='All evaluated rows';render();});
const canvas=$('map'),point=e=>{const b=canvas.getBoundingClientRect();return [(e.clientX-b.left)*canvas.width/b.width,(e.clientY-b.top)*canvas.height/b.height];};let start=null;
canvas.addEventListener('pointerdown',e=>{if(!mapPoints.length)return;start=point(e);canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(!start)return;const end=point(e);brush=[...start,...end];drawMap();});
canvas.addEventListener('pointerup',e=>{if(!start)return;const end=point(e),x0=Math.min(start[0],end[0]),x1=Math.max(start[0],end[0]),y0=Math.min(start[1],end[1]),y1=Math.max(start[1],end[1]);rowVersion++;curveCache.clear();rows=mapPoints.map((p,i)=>p[0]>=x0&&p[0]<=x1&&p[1]>=y0&&p[1]<=y1?i:-1).filter(i=>i>=0);start=null;brush=[x0,y0,x1,y1];select.value='__brush__';render();});
canvas.addEventListener('pointercancel',()=>{start=null;brush=null;drawMap();});}
$('export').addEventListener('click',()=>{const blob=new Blob([JSON.stringify({run_id:data.run_id,selected_ids:rows?rows.map(i=>data.ids[i]):null,settings,scope:{mode:$('mode').value,measure:$('measure').value,row_scope:isAll()?'all evaluated rows':'selected anchors',candidate_pool:'fixed evaluation cohort',selected_features:[...selectedFeatures],embedding:selectedEmbeddingLabel(),embedding_key:currentEmbedding(),probe:$('probe').value,probe_metrics:[...selectedProbeMetrics],source_scope:view.scope},measurements:displayed},null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=element('a');a.href=url;a.download='feature_measurements_selection.json';a.click();URL.revokeObjectURL(url);});
renderFeatureChoices();render();
})();
