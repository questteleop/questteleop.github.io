// ===== 矩阵与角度工具（原样） =====
export function deg2rad(d){ return d*Math.PI/180; }
export function mat4Identity(){
  return new Float32Array([1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1]);
}
export function mat4Multiply(a,b){
  const out=new Float32Array(16);
  for(let r=0;r<4;r++){
    for(let c=0;c<4;c++){
      out[r*4+c]=a[r*4+0]*b[0*4+c]+a[r*4+1]*b[1*4+c]+a[r*4+2]*b[2*4+c]+a[r*4+3]*b[3*4+c];
    }
  }
  return out;
}
export function mat4Translate(tx,ty,tz){
  const m=mat4Identity(); m[12]=tx; m[13]=ty; m[14]=tz; return m;
}
export function mat4RotateX(rad){
  const c=Math.cos(rad), s=Math.sin(rad);
  return new Float32Array([1,0,0,0,  0,c,s,0,  0,-s,c,0,  0,0,0,1]);
}
