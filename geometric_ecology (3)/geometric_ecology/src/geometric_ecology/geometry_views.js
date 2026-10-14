/* Reusable, data-driven geometry views. No synthetic peaks or icon substitutes. */
window.GeometryViews=(()=>{
 const color={ink:'#21443d',muted:'#6b7e74',green:'#397f66',orange:'#c47942',purple:'#82658e',blue:'#638ba5'};
 function setup(canvas){const r=canvas.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);canvas.width=Math.max(1,r.width*d);canvas.height=Math.max(1,r.height*d);const c=canvas.getContext('2d');c.setTransform(d,0,0,d,0,0);c.clearRect(0,0,r.width,r.height);return[c,r.width,r.height]}
 function text(c,s,x,y,size=11,ink=color.ink){c.fillStyle=ink;c.font=`${size}px system-ui`;c.fillText(s,x,y)}
 function lens(canvas,data,condition='original'){
  const[c,w,h]=setup(canvas),compact=h<300,left=Math.max(150,w*.50),top=35,scatterHeight=compact?h-65:Math.min(240,h*.48);
  text(c,'Actual neighbor offsets · first two PCA axes',14,19,11);
  text(c,`Variance by independent PCA direction`,left+20,19,11);
  const radius=Math.max(...data.offsets.flatMap(p=>p.slice(0,2).map(Math.abs)),1e-8);
  const scale=.43*Math.min(left-45,scatterHeight)/radius,cx=left/2,cy=top+scatterHeight/2;
  c.strokeStyle='#b9c9ba';c.lineWidth=1;c.strokeRect(12,top,left-25,scatterHeight);
  c.beginPath();c.moveTo(14,cy);c.lineTo(left-15,cy);c.moveTo(cx,top);c.lineTo(cx,top+scatterHeight);c.stroke();
  const original=new Set(data.neighbor_ids);
  data.offsets.forEach((p,i)=>{const x=cx+p[0]*scale,y=cy-p[1]*scale;c.strokeStyle='#52857533';c.beginPath();c.moveTo(cx,cy);c.lineTo(x,y);c.stroke();c.fillStyle=color.green;c.beginPath();c.arc(x,y,2.7,0,Math.PI*2);c.fill()});
  if(condition!=='original'&&data[condition]){const v=data[condition];v.offsets.forEach((p,i)=>{if(original.has(v.neighbor_ids[i]))return;const x=cx+p[0]*scale,y=cy-p[1]*scale;c.strokeStyle=condition==='compression'?'#906498b0':'#578ea7b0';c.setLineDash([3,2]);c.beginPath();c.moveTo(cx,cy);c.lineTo(x,y);c.stroke();c.setLineDash([]);c.beginPath();c.arc(x,y,3.5,0,Math.PI*2);c.stroke()})}
  text(c,`PCA 1 / PCA 2 · chord radius ${data.radius.toFixed(4)}`,14,top+scatterHeight+17,10,color.muted);
  const values=data.spectrum.slice(0,Math.min(16,data.spectrum.length)),bw=(w-left-45)/Math.max(values.length,1),bh=scatterHeight-38,base=top+scatterHeight-10;
  values.forEach((v,i)=>{c.fillStyle=i<2?color.green:color.orange;c.fillRect(left+22+i*bw,base-v*bh*1.35,Math.max(3,bw-3),v*bh*1.35);if(i<12)text(c,String(i+1),left+23+i*bw,base+14,9,color.muted)});
  text(c,`${Math.round(100*data.visible_two_fraction)}% visible in two axes`,left+21,top+12,12);
  text(c,`${data.components90} directions retain 90%`,left+21,top+29,11,color.muted);
  if(!compact){
   const y0=top+scatterHeight+60,height=Math.max(60,h-y0-35),count=Math.min(12,data.spectrum.length),gap=(w-45)/Math.max(1,count-1);
   text(c,'Unfolded direction ribbons · each line is one measured neighbor',14,y0-15,11);
   const max=Math.max(...data.offsets.flatMap(p=>p.map(Math.abs)),1e-8);
   // Reconstructing directions beyond PCA 3 uses stored normalized component
   // scores, not invented spokes or decorative noise.
   const scores=data.direction_scores||data.offsets;
   const amplitude=Math.max(...scores.flatMap(p=>p.slice(0,count).map(Math.abs)),max);
   for(let j=0;j<count;j++){const x=20+j*gap;c.strokeStyle='#d8e1d6';c.beginPath();c.moveTo(x,y0);c.lineTo(x,y0+height);c.stroke();text(c,`PC${j+1}`,x-7,y0+height+18,9,color.muted)}
   scores.slice(0,80).forEach(row=>{c.strokeStyle='#3b7c6540';c.lineWidth=.8;c.beginPath();for(let j=0;j<count;j++){const y=y0+height/2-(row[j]||0)/amplitude*height*.46;j?c.lineTo(20+j*gap,y):c.moveTo(20+j*gap,y)}c.stroke()});
  }
 }
 function surface(ctx,field,project,heightChannel='height',options={}){
  const n=field.size,b=field.bounds,v=field.values,dx=(b[2]-b[0])/(n-1),dy=(b[3]-b[1])/(n-1),tiles=[];
  for(let r=0;r<n-1;r++)for(let c=0;c<n-1;c++){
   if(!field.valid[r][c]||!field.valid[r+1][c]||!field.valid[r][c+1]||!field.valid[r+1][c+1])continue;
   const z=(v[heightChannel]||v.height),shade=Math.max(.55,Math.min(1.22,.94-1.4*(z[r][c+1]-z[r][c])+.8*(z[r+1][c]-z[r][c])));
   const dim=Math.min(1,Math.max(0,z[r][c])),density=v.density[r][c];
   const low=[148,183,139],high=heightChannel==='ambiguity'?[133,113,153]:[188,139,90];
   let rgb=low.map((x,i)=>Math.round(Math.max(0,Math.min(255,(x*(1-dim)+high[i]*dim)*shade+(1-density)*8))));if(options.fog){const fog=Math.min(.6,Math.max(0,(v.fog[r][c]-.15)*.85)),mist=[174,187,198];rgb=rgb.map((x,i)=>Math.round(x*(1-fog)+mist[i]*fog))}
   const vertices=[[c,r],[c+1,r],[c+1,r+1],[c,r+1]].map(([cc,rr])=>project([b[0]+cc*dx,b[1]+rr*dy,z[rr][cc]]));
   tiles.push({vertices,fill:`rgb(${rgb.join(',')})`,order:r});
  }
  tiles.sort((a,b)=>b.order-a.order);
  tiles.forEach(tile=>{ctx.fillStyle=tile.fill;ctx.strokeStyle=tile.fill;ctx.lineWidth=.35;ctx.beginPath();tile.vertices.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.fill();ctx.stroke()});
 }
 function graph(canvas,scene,{compare=false}={}){
  const[c,w,h]=setup(canvas),xy=scene.xy,all=xy.flat();
  const xs=xy.map(p=>p[0]),ys=xy.map(p=>p[1]),minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys);
  const unit=.72*Math.min(w/Math.max(1,maxx-minx),h/Math.max(1,maxy-miny));
  const pos=i=>[w/2+(xy[i][0]-(minx+maxx)/2)*unit,h/2+10-(xy[i][1]-(miny+maxy)/2)*unit];
  const gates=new Set(scene.gates.map(([a,b])=>[a,b].sort((a,b)=>a-b).join(':')));
  if(scene.kind==='ridge'){
   const left=xy.filter((_,i)=>i<xy.length/2),boundary=(Math.max(...left.map(p=>p[0]))+Math.min(...xy.slice(xy.length/2).map(p=>p[0])))/2;
   const xx=w/2+(boundary-(minx+maxx)/2)*unit;c.fillStyle='#8b739e30';c.fillRect(xx-10,h*.12,20,h*.77);c.strokeStyle='#8a6d9e';c.setLineDash([5,4]);c.beginPath();c.moveTo(xx,h*.12);c.lineTo(xx,h*.89);c.stroke();c.setLineDash([]);
  }
  scene.edges.forEach(([a,b],i)=>{const[x,y]=pos(a),[xx,yy]=pos(b),gate=gates.has([a,b].sort((a,b)=>a-b).join(':'));c.strokeStyle=gate?(scene.kind==='cliff'?'#cc7655':'#b27746'):'#a4b7a6';c.lineWidth=gate?3.5:1.7;c.beginPath();c.moveTo(x,y);c.lineTo(xx,yy);c.stroke()});
  const routes=scene.routes;
  routes.paths.forEach((path,j)=>{c.strokeStyle=`hsla(${145+j*9},35%,${34+j*1.7}%,${.32+(routes.probabilities[j]||0)*.8})`;c.lineWidth=2.2;c.beginPath();path.forEach((i,k)=>{const[x,y]=pos(i);k?c.lineTo(x,y):c.moveTo(x,y)});c.stroke()});
  xy.forEach((_,i)=>{const[x,y]=pos(i);c.fillStyle='#4c7967';c.beginPath();c.arc(x,y,3.5,0,Math.PI*2);c.fill()});
  [[scene.source,'Start','#4b7f9e'],[scene.target,'Destination','#c47644']].forEach(([i,name,fill])=>{const[x,y]=pos(i);c.fillStyle=fill;c.beginPath();c.arc(x,y,7,0,Math.PI*2);c.fill();text(c,name,x-18,y-15,11,fill)});
  if(scene.kind==='cliff')text(c,`Crossing step = ${scene.gap_ratio.toFixed(1)} × typical edge`,Math.max(20,w*.35),h*.14,12,'#a65f43');
  if(scene.kind==='island')text(c,'No graph crossing',w/2-52,h*.30,12,'#5d8390');
  if(scene.kind==='pass')text(c,'Approach paths funnel through one gate',Math.max(18,w*.28),h*.14,12,'#a37046');
  if(scene.kind==='ridge')text(c,'Persistent separator · three crossing gates',Math.max(18,w*.28),h*.11,12,'#82618e');
  text(c,routes.reachable?`${routes.capped?'At least ':''}${routes.paths.length} near-shortest routes enumerated · effective ${routes.effective_routes.toFixed(2)}`:'Disconnected · no route',18,h-16,11,color.muted);
 }
 function inlineLens(canvas,data,condition='original'){
  const[c,w,h]=setup(canvas),small=w<560,scatter=small?0:125,bars=small?w*.42:Math.min(230,w*.27),x0=scatter+12,base=h-20,top=18;
  if(!small){text(c,'Actual PCA-2 offsets',4,12,9);const radius=Math.max(...data.offsets.flatMap(p=>p.slice(0,2).map(Math.abs)),1e-9),scale=32/radius,center=[54,h/2];c.strokeStyle='#c9d9d1';c.strokeRect(2,20,106,h-27);data.offsets.forEach(p=>{const x=center[0]+p[0]*scale,y=center[1]-p[1]*scale;c.strokeStyle='#3d827233';c.beginPath();c.moveTo(...center);c.lineTo(x,y);c.stroke();c.fillStyle='#318575';c.beginPath();c.arc(x,y,1.7,0,Math.PI*2);c.fill()});if(condition!=='original')data[condition].offsets.forEach((p,i)=>{if(data.neighbor_ids.includes(data[condition].neighbor_ids[i]))return;c.strokeStyle='#9264a1';c.beginPath();c.moveTo(...center);c.lineTo(center[0]+p[0]*scale,center[1]-p[1]*scale);c.stroke()})}
  text(c,'Variance per direction',x0,12,9);const values=data.spectrum.slice(0,16),bw=(bars-14)/Math.max(1,values.length),max=values[0]||1;values.forEach((v,j)=>{c.fillStyle=j<2?'#257c74':'#a061a4';c.fillRect(x0+j*bw,base-v/max*(h-42),Math.max(2,bw-2),v/max*(h-42));if(j<12)text(c,j+1,x0+j*bw,base+12,7,color.muted)});
  const start=x0+bars+8,width=w-start-8,scores=data.direction_scores,count=Math.min(12,scores[0]?.length||0),amplitude=Math.max(1e-9,...scores.flatMap(row=>row.slice(0,count).map(Math.abs)));text(c,'Unfolded directions · one ribbon per neighbor',start,12,9);if(width<40)return;for(let j=0;j<count;j++){const x=start+j*width/Math.max(1,count-1);c.strokeStyle='#d5e0d8';c.beginPath();c.moveTo(x,top+5);c.lineTo(x,base);c.stroke();text(c,j+1,x-3,base+12,7,color.muted)}scores.forEach(row=>{c.strokeStyle='#357e6b38';c.beginPath();for(let j=0;j<count;j++){const x=start+j*width/Math.max(1,count-1),y=(top+base)/2-row[j]/amplitude*(base-top)*.46;j?c.lineTo(x,y):c.moveTo(x,y)}c.stroke()});
 }
 function spectralRing(ctx,x,y,lens,radius=16){const p=lens.spectrum,sum=p.reduce((a,b)=>a+b,0)||1;ctx.save();ctx.fillStyle='#fffff4eb';ctx.beginPath();ctx.arc(x,y,radius+2,0,Math.PI*2);ctx.fill();let angle=-Math.PI/2;const pieces=p.slice(0,12);if(p.length>12)pieces.push(p.slice(12).reduce((a,b)=>a+b,0));pieces.forEach((v,i)=>{const arc=2*Math.PI*v/sum;if(arc>.002){ctx.strokeStyle=i<2?'#247f78':i===12?'#a3a9a3':'#a25da5';ctx.lineWidth=3;ctx.beginPath();ctx.arc(x,y,radius,angle+.012,angle+Math.max(.012,arc-.012));ctx.stroke()}angle+=arc});const ratio=Math.sqrt((p[1]||0)/(p[0]||1));ctx.fillStyle='#d0e4dc';ctx.strokeStyle='#376c67';ctx.lineWidth=1;ctx.beginPath();ctx.ellipse(x,y,radius*.63,Math.max(.6,radius*.63*ratio),0,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.restore()}
 return{setup,lens,inlineLens,surface,graph,spectralRing};
})();
