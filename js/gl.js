// ===== WebGL 变量（与原逻辑一致） =====
export let gl, texture, program3D, pos3DBuffer, uvBuffer;

import { ORIENT, FLIP_X, PANEL_WIDTH, PANEL_HEIGHT } from './config.js';

// 屏幕四边形的顶点（3D）
function buildPanelVertices(){
  const hw = PANEL_WIDTH/2, hh = PANEL_HEIGHT/2;
  return new Float32Array([
    -hw,-hh,0,   hw,-hh,0,   -hw, hh,0,
    -hw, hh,0,   hw,-hh,0,    hw, hh,0
  ]);
}
// 纹理坐标生成（考虑旋转/翻转）
function makeTexcoords(orient, flipX, flipY){
  let uv = [ 0,0, 1,0, 0,1,  0,1, 1,0, 1,1 ];
  function rot90(arr){
    const out = [];
    for(let i=0;i<arr.length;i+=2){ const u=arr[i], v=arr[i+1]; out.push(v, 1-u); }
    return out;
  }
  let arr = uv;
  if(orient===90)  arr = rot90(arr);
  if(orient===180) arr = rot90(rot90(arr));
  if(orient===270) arr = rot90(rot90(rot90(arr)));
  for(let i=0;i<arr.length;i+=2){
    if(flipX) arr[i]   = 1 - arr[i];
    if(flipY) arr[i+1] = 1 - arr[i+1];
  }
  return new Float32Array(arr);
}

export function applyTexcoords(){
  const tc = makeTexcoords(ORIENT, FLIP_X, true /*与原代码相同：默认翻转Y*/);
  gl.bindBuffer(gl.ARRAY_BUFFER, uvBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, tc, gl.STATIC_DRAW);
}

export function initGLResources(glCtx){
  gl = glCtx;

  const vs3D=`
  attribute vec3 a_position;
  attribute vec2 a_texcoord;
  uniform mat4 u_viewProj;
  uniform mat4 u_model;
  varying vec2 v_uv;
  void main(){
    gl_Position = u_viewProj * u_model * vec4(a_position,1.0);
    v_uv = a_texcoord;
  }`;
  const fs=`
  precision mediump float;
  varying vec2 v_uv;
  uniform sampler2D u_texture;
  void main(){ gl_FragColor = texture2D(u_texture, v_uv); }`;

  function compile(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);return s;}
  const vsObj=compile(gl.VERTEX_SHADER,vs3D), fsObj=compile(gl.FRAGMENT_SHADER,fs);
  program3D=gl.createProgram();gl.attachShader(program3D,vsObj);gl.attachShader(program3D,fsObj);gl.linkProgram(program3D);

  // 顶点缓冲：面板 3D 顶点
  pos3DBuffer=gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER,pos3DBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, buildPanelVertices(), gl.STATIC_DRAW);

  // 纹理坐标缓冲
  uvBuffer=gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER,uvBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, makeTexcoords(ORIENT, FLIP_X, true), gl.STATIC_DRAW);

  // 纹理
  texture=gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D,texture);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);

  gl.useProgram(program3D);
  const uTexLoc = gl.getUniformLocation(program3D, "u_texture");
  if (uTexLoc) gl.uniform1i(uTexLoc, 0);
}
