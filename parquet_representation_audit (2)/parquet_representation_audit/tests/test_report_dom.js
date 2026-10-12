/* Actual report controls executed against its serialized run data in jsdom.
   npm install --prefix /tmp/parquet-audit-ui jsdom
   NODE_PATH=/tmp/parquet-audit-ui/node_modules node tests/test_report_dom.js */
const {JSDOM}=require('jsdom');const fs=require('fs');const path=require('path');const assert=require('assert');
const root=path.resolve(__dirname,'..');
async function check(folder){
 const file=path.join(folder,'report.html'),html=fs.readFileSync(file,'utf8');
 const dom=new JSDOM(html,{runScripts:'outside-only',url:'file://'+file,pretendToBeVisual:true});const w=dom.window,d=w.document;
 const context=new Proxy({}, {get:(_,key)=>key==='measureText'?()=>({width:10}):()=>{}});
 w.HTMLCanvasElement.prototype.getContext=()=>context;w.HTMLCanvasElement.prototype.setPointerCapture=()=>{};
 w.HTMLCanvasElement.prototype.getBoundingClientRect=function(){return {left:0,top:0,width:this.width,height:this.height};};
 let captured=null;w.Blob=globalThis.Blob;w.URL.createObjectURL=blob=>{captured=blob;return 'blob:test'};w.URL.revokeObjectURL=()=>{};w.HTMLAnchorElement.prototype.click=()=>{};
 // The page's inline selector functions and both original adapted report assets.
 for(const script of [...d.scripts])if(!script.src&&script.type!=='application/json')w.eval(script.textContent);
 for(const name of ['explorer.js','enhancements.js'])w.eval(fs.readFileSync(path.join(folder,'assets',name),'utf8'));
 const query=id=>d.getElementById(id),data=w.EmbeddingAuditData,names=Object.keys(data.overview.features),
       change=id=>query(id).dispatchEvent(new w.Event('change'));
 assert.equal(query('ex-overview').children.length,names.length);
 const headings=[...query('ex-overview').closest('table').querySelectorAll(':scope > thead > tr > th')].map(x=>x.textContent);
 assert.equal(headings.length,5);assert(headings.includes('External pipeline'));
 assert(query('ea-workspace'));assert(d.querySelector('.feature-workspace'));
 const sections=[...d.querySelectorAll('main>section,main>div,main>details')];
 assert(sections.indexOf(d.querySelector('.feature-workspace'))<sections.indexOf(query('ea-workspace')));
 // Feature subset applies consistently to main, normalization and filtration tables.
 query('ex-no-features').click();assert.equal(query('ex-overview').children.length,1);assert(query('ex-overview').textContent.includes('Select features'));
 assert.equal(query('ea-normalization-table').children.length,0);assert.equal(query('ea-topology-table').children.length,0);
 query('ex-all-features').click();query('ex-search').value=names[0];query('ex-search').dispatchEvent(new w.Event('input'));assert(query('ex-features').children.length>=1);
 query('ex-search').value='';query('ex-search').dispatchEvent(new w.Event('input'));
 const boxes=query('ex-features').querySelectorAll('input');for(let i=1;i<boxes.length;i++){boxes[i].checked=false;boxes[i].dispatchEvent(new w.Event('change'));}
 assert.equal(query('ex-overview').children.length,1);assert(query('ex-overview').textContent.includes(names[0]));
 if(data.enhancements.normalization.status==='ok')assert.equal(query('ea-normalization-table').children.length,1);
 if(data.enhancements.topology.status==='ok')assert.equal(query('ea-topology-table').children.length,1);
 query('ex-measure').value='agreement';change('ex-measure');assert(query('ex-curve-heading').textContent.includes('purity'));
 if(data.available){
  assert.equal(query('ex-overview').querySelectorAll('.ex-tile').length,data.overview.ks.length);
  query('ex-mode').value='selection';change('ex-mode');assert(query('ex-readout').textContent.includes('selected rows minus'));
  if(data.overview.comparison_name){query('ex-mode').value='models';change('ex-mode');assert(query('ex-readout').textContent.includes('B minus A'));}
  // Brushing changes anchors only; topology/normalization remain fixed.
  query('ex-mode').value='values';change('ex-mode');const normBefore=query('ea-normalization-table').textContent;
  const topoBefore=query('ea-topology-table').textContent,canvas=query('ex-map');
  canvas.dispatchEvent(new w.MouseEvent('pointerdown',{clientX:0,clientY:0}));canvas.dispatchEvent(new w.MouseEvent('pointerup',{clientX:canvas.width/2,clientY:canvas.height}));
  assert.equal(query('ea-normalization-table').textContent,normBefore);assert.equal(query('ea-topology-table').textContent,topoBefore);
  query('ex-export').click();assert(captured);const exported=JSON.parse(await captured.text());
  assert.equal(exported.scope.candidate_pool,'fixed evaluation cohort');assert.deepEqual(exported.scope.selected_features,[names[0]]);
  assert(exported.selected_ids.length<data.ids.length);assert(exported.measurements.every(x=>x.feature===names[0]&&x.measure==='micro_purity'));
  query('ex-reset').click();assert(query('ex-map-status').textContent.includes(data.ids.length+' selected'));
 }else{
  assert(query('ex-row-tools').hidden);assert(query('ex-mode').options[1].disabled);
  assert(query('ex-overview').textContent.includes('Unavailable'));
 }
 // Model/measurement controls update explanations along with displayed topology values.
 if(data.enhancements.topology.status==='ok'){
  query('ea-topology-measure').value='observed';change('ea-topology-measure');assert(query('ea-topology-readout').textContent.includes('Observed component diversity'));
  if(data.overview.comparison_name){query('ea-topology-measure').value='model_delta';change('ea-topology-measure');assert(query('ea-topology-readout').textContent.includes('B minus A'));}
 }
 // Every same-run link points at a shipped local file or valid feature anchor.
 for(const a of d.querySelectorAll('a[href]')){
  const href=a.getAttribute('href');if(href.startsWith('#'))continue;if(/^(https?:|data:|blob:)/.test(href))continue;
  assert(fs.existsSync(path.resolve(folder,href)),'Missing local report link '+href);
 }
 dom.window.close();return names.length;
}
(async()=>{for(const folder of ['demo_generic_audit','demo_telemetry_ab','demo_source_only'])console.log(folder,await check(path.join(root,folder)),'features: report DOM, controls, exports, scopes and local links passed.');})().catch(error=>{console.error(error);process.exit(1)});
