/* Small offline WebGL renderer. The caller supplies measured triangles/edges. */
window.MeshGL=(()=>{
 function create(canvas){
  const gl=canvas.getContext('webgl',{antialias:true,alpha:false,preserveDrawingBuffer:true});
  if(!gl)return null;
  function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s}
  const program=gl.createProgram();
  gl.attachShader(program,shader(gl.VERTEX_SHADER,`attribute vec3 a_position;attribute vec3 a_normal;attribute vec4 a_color;uniform mat4 u_matrix;uniform float u_size;uniform float u_shade;varying vec4 v_color;void main(){gl_Position=u_matrix*vec4(a_position,1.);gl_PointSize=u_size;float light=.42+.58*abs(dot(normalize(a_normal+vec3(0.,0.,.0001)),normalize(vec3(-.4,.6,1.))));v_color=vec4(a_color.rgb*mix(1.,light,u_shade),a_color.a);}`));
  gl.attachShader(program,shader(gl.FRAGMENT_SHADER,`precision mediump float;varying vec4 v_color;uniform float u_round;void main(){if(u_round>.5&&distance(gl_PointCoord,vec2(.5))>.5)discard;gl_FragColor=v_color;}`));
  gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
  const attrs=['a_position','a_normal','a_color'].map(n=>gl.getAttribLocation(program,n));
  const uniforms=Object.fromEntries(['matrix','size','shade','round'].map(n=>[n,gl.getUniformLocation(program,'u_'+n)]));
  let buffers=new Map();
  function clear(){for(const buffer of buffers.values())gl.deleteBuffer(buffer);buffers.clear()}
  function render(batches,matrix){
   const d=Math.min(devicePixelRatio||1,2),w=Math.max(1,Math.round(canvas.clientWidth*d)),h=Math.max(1,Math.round(canvas.clientHeight*d));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}
   gl.viewport(0,0,w,h);gl.clearColor(.91,.95,.955,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.useProgram(program);gl.uniformMatrix4fv(uniforms.matrix,false,matrix);
   for(const b of batches){if(!b.data.length)continue;let buffer=buffers.get(b);if(!buffer){buffer=gl.createBuffer();buffers.set(b,buffer);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(b.data),gl.STATIC_DRAW)}else gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
    attrs.forEach((a,i)=>{gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,i===2?4:3,gl.FLOAT,false,40,[0,12,24][i])});
    gl.uniform1f(uniforms.size,(b.size||1.5)*d);gl.uniform1f(uniforms.shade,b.shade?1:0);gl.uniform1f(uniforms.round,b.kind==='points'?1:0);gl.depthMask(b.kind==='triangles');gl.drawArrays(b.kind==='triangles'?gl.TRIANGLES:b.kind==='points'?gl.POINTS:gl.LINES,0,b.data.length/10);
   }gl.depthMask(true);
  }
  function release(batch){const b=buffers.get(batch);if(b){gl.deleteBuffer(b);buffers.delete(batch)}}
  return{render,clear,release,error:()=>gl.getError(),context:gl};
 }
 function vertex(batch,xyz,color,normal=[0,0,1]){batch.data.push(...xyz,...normal,...color)}
 function triangle(batch,points,color){const[a,b,c]=points,u=b.map((v,i)=>v-a[i]),v=c.map((v,i)=>v-a[i]),normal=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]],length=Math.hypot(...normal)||1;points.forEach(p=>vertex(batch,p,color,normal.map(v=>v/length)))}
 function edge(batch,a,b,color){vertex(batch,a,color);vertex(batch,b,color)}
 function color(hidden,alpha=1){const t=Math.max(0,Math.min(1,hidden/.85)),low=[.16,.63,.57],mid=[.94,.64,.27],high=[.66,.35,.68],a=t<.5?low:mid,b=t<.5?mid:high,u=t<.5?t*2:(t-.5)*2;return[...a.map((v,i)=>v*(1-u)+b[i]*u),alpha]}
 // Orthographic orbit camera, shared by the main view and pinned shape key.
 function camera({width,height,bounds,yaw=-.18,tilt=.82,zoom=1,pan=[0,0],dock=0}){
  const cx=(bounds[0]+bounds[2])/2,cy=(bounds[1]+bounds[3])/2,cz=1.1,c=Math.cos(yaw),s=Math.sin(yaw),ct=Math.cos(tilt),st=Math.sin(tilt);
  const sx=Math.abs(c)*(bounds[2]-bounds[0])+Math.abs(s)*(bounds[3]-bounds[1]),sy=(Math.abs(s)*(bounds[2]-bounds[0])+Math.abs(c)*(bounds[3]-bounds[1]))*ct+3*st;
  const unit=.81*Math.min(width/Math.max(sx,1),(height-dock-90)/Math.max(sy,1))*zoom,ux=2*unit/width,uy=2*unit/height,depth=1/Math.max(100,sx+sy),tx=2*pan[0]/width,ty=(dock-35-2*pan[1])/height;
  const m=new Float32Array([ux*c,uy*ct*s,depth*st*s,0,-ux*s,uy*ct*c,depth*st*c,0,0,uy*st,-depth*ct,0,tx-ux*(c*cx-s*cy),ty-uy*(ct*(s*cx+c*cy)+st*cz),depth*(ct*cz-st*(s*cx+c*cy)),1]);
  function project(p){return[(1+m[0]*p[0]+m[4]*p[1]+m[8]*p[2]+m[12])*width/2,(1-m[1]*p[0]-m[5]*p[1]-m[9]*p[2]-m[13])*height/2,m[2]*p[0]+m[6]*p[1]+m[10]*p[2]+m[14]]}
  return{matrix:m,project,unit};
 }
 return{create,vertex,triangle,edge,color,camera};
})();
