// xr_sender.js
import { setupSockets, wsSend, frameImageBitmap } from './ws.js';
import {
  MIN_JOINTS, gateTrack,
  quatMul, quatNorm,
  JOINTS, PANEL_DISTANCE, PANEL_RX_DEG
} from './config.js';
import { deg2rad, mat4Multiply, mat4Translate, mat4RotateX } from './math.js';
import { gl, texture, program3D, pos3DBuffer, uvBuffer, initGLResources } from './gl.js';

// =================== 可调参数 ===================
const CALIB_SECS = 2.0;                 // 静止标定秒数
const MAX_RADIUS = 0.02;                // 关键点半径阈值（米）
const TIP_RADIUS = 0.012;               // 指尖更严
const MIN_VALID_RATIO_FINGER = 0.8;     // 每指有效骨段比例下限
const DEAD_BAND_DEG = 1.0;              // 角度死区
const MAX_SPEED_DEG_S = 40;             // 角速度限速
const EMA_ALPHA_SLOW = 0.12;            // 小运动平滑
const EMA_ALPHA_FAST = 0.55;            // 快运动响应
const SPEED_SWITCH_DEG_S = 25;          // 自适应切换阈

const POS_DEAD_BAND = 0.0015;           // 位移死区（米）
const MAX_SPEED_M_S = 0.25;             // 位移限速（米/秒）
const EMA_POS_SLOW = 0.12;
const EMA_POS_FAST = 0.5;
const SPEED_SWITCH_M_S = 0.10;

// =============== 坐标系统一：X前/Y右/Z上 =================
// 位置基变换：x' = -z, y' = x, z' = y
function remapPositionToFwdRightUp(p)   { return { x:-p.z, y:p.x, z:p.y }; }
// 对应旋转基的四元数（把 XR 的 world/floor 旋到 前/右/上）
const rBasis = { x: 0.5, y: 0.5, z: -0.5, w: 0.5 };
function quatConj(q){ return { x:-q.x, y:-q.y, z:-q.z, w:q.w }; }
const rBasisInv = quatConj(rBasis);

// =============== 数学小工具 ===============
function qMul(a,b){ return quatMul(a,b); }
function qNorm(a){ return quatNorm(a); }
function qToMatrix(q){ // 旋转矩阵 3x3
  const {x,y,z,w} = q;
  const xx=x*x, yy=y*y, zz=z*z, xy=x*y, xz=x*z, yz=y*z, wx=w*x, wy=w*y, wz=w*z;
  return [
    1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy),
    2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx),
    2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)
  ];
}
function vSub(a,b){ return {x:a.x-b.x, y:a.y-b.y, z:a.z-b.z}; }
function vAdd(a,b){ return {x:a.x+b.x, y:a.y+b.y, z:a.z+b.z}; }
function vDot(a,b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
function vLen(a){ return Math.sqrt(Math.max(vDot(a,a),1e-12)); }
function vNorm(a){ const L=vLen(a); return {x:a.x/L,y:a.y/L,z:a.z/L}; }
function vCross(a,b){ return {x:a.y*b.z-a.z*b.y, y:a.z*b.x-a.x*b.z, z:a.x*b.y-a.y*b.x}; }
function vScale(a,s){ return {x:a.x*s,y:a.y*s,z:a.z*s}; }
function projOnPlane(v,n){ const nn=vNorm(n); return vSub(v, vScale(nn, vDot(v,nn))); }
function clamp(x,lo,hi){ return x<lo?lo:(x>hi?hi:x); }
function acosClamp(x){ return Math.acos(clamp(x,-1,1)); }
function angleBetween(a,b){ const an=vNorm(a), bn=vNorm(b); return acosClamp(vDot(an,bn)); } // rad
function signedAngleOnPlane(ref, vec, n){ // rad
  const refp=vNorm(projOnPlane(ref,n));
  const vecp=vNorm(projOnPlane(vec,n));
  const ang=acosClamp(vDot(refp,vecp));
  const s=vDot(vCross(refp,vecp), vNorm(n));
  return s>=0?ang:-ang;
}
function rad2deg(x){ return x*180/Math.PI; }
function deg2rad_(x){ return x*Math.PI/180; }

// =============== 一阶滤波器（自适应 EMA + 死区 + 限速） ===============
class Filter1D {
  constructor({dead=0, maxSpeed=Infinity, alphaSlow=0.2, alphaFast=0.6, speedSwitch=Infinity}){
    this.dead=dead; this.maxSpeed=maxSpeed;
    this.alphaSlow=alphaSlow; this.alphaFast=alphaFast; this.speedSwitch=speedSwitch;
    this.v = 0; this.ok=false;
  }
  step(raw, dt){
    if(!this.ok){ this.v=raw; this.ok=true; return this.v; }
    // 死区（对输入增量）
    const dv = raw - this.v;
    const mag = Math.abs(dv);
    const dvDead = mag<this.dead ? 0 : dv;
    // 限速
    const maxStep = this.maxSpeed * dt;
    const dvClamped = clamp(dvDead, -maxStep, maxStep);
    // 自适应 α
    const alpha = (Math.abs(dvClamped)/Math.max(dt,1e-3)) > this.speedSwitch ? this.alphaFast : this.alphaSlow;
    this.v = alpha*(this.v + dvClamped) + (1-alpha)*this.v; // = this.v + alpha*dvClamped
    return this.v;
  }
}

// =============== 手指链定义（WebXR 25 关节名） ===============
const CHAINS = {
  thumb:  ["thumb-metacarpal","thumb-phalanx-proximal","thumb-phalanx-distal","thumb-tip"],
  index:  ["index-finger-metacarpal","index-finger-phalanx-proximal","index-finger-phalanx-intermediate","index-finger-phalanx-distal","index-finger-tip"],
  middle: ["middle-finger-metacarpal","middle-finger-phalanx-proximal","middle-finger-phalanx-intermediate","middle-finger-phalanx-distal","middle-finger-tip"],
  ring:   ["ring-finger-metacarpal","ring-finger-phalanx-proximal","ring-finger-phalanx-intermediate","ring-finger-phalanx-distal","ring-finger-tip"],
  pinky:  ["pinky-finger-metacarpal","pinky-finger-phalanx-proximal","pinky-finger-phalanx-intermediate","pinky-finger-phalanx-distal","pinky-finger-tip"],
};

// =============== 关节角限幅（度） ===============
const LIM = {
  thumb_abd: [-35,35], thumb_mcp:[0,90], thumb_ip:[0,90], thumb_cmc:[0,60],
  mcp:[0,90], pip:[0,100], dip:[0,90]
};

// =============== Baseline/滤波状态 ===============
let baseline = {
  palm: {pos0:null, quat0:null}, // world 统一坐标
  fingers: { thumb:{abd:0,mcp:0,ip:0,cmc:0}, index:{mcp:0,pip:0,dip:0},
             middle:{mcp:0,pip:0,dip:0}, ring:{mcp:0,pip:0,dip:0}, pinky:{mcp:0,pip:0,dip:0} }
};

const filt = { // 每个 DOF 一个 1D 滤波器（角度：rad；位置：m）
  // palm delta
  dx:new Filter1D({dead:POS_DEAD_BAND, maxSpeed:MAX_SPEED_M_S, alphaSlow:EMA_POS_SLOW, alphaFast:EMA_POS_FAST, speedSwitch:SPEED_SWITCH_M_S}),
  dy:new Filter1D({dead:POS_DEAD_BAND, maxSpeed:MAX_SPEED_M_S, alphaSlow:EMA_POS_SLOW, alphaFast:EMA_POS_FAST, speedSwitch:SPEED_SWITCH_M_S}),
  dz:new Filter1D({dead:POS_DEAD_BAND, maxSpeed:MAX_SPEED_M_S, alphaSlow:EMA_POS_SLOW, alphaFast:EMA_POS_FAST, speedSwitch:SPEED_SWITCH_M_S}),
  rx:new Filter1D({dead:deg2rad_(DEAD_BAND_DEG), maxSpeed:deg2rad_(MAX_SPEED_DEG_S), alphaSlow:EMA_ALPHA_SLOW, alphaFast:EMA_ALPHA_FAST, speedSwitch:deg2rad_(SPEED_SWITCH_DEG_S)}),
  ry:new Filter1D({dead:deg2rad_(DEAD_BAND_DEG), maxSpeed:deg2rad_(MAX_SPEED_DEG_S), alphaSlow:EMA_ALPHA_SLOW, alphaFast:EMA_ALPHA_FAST, speedSwitch:deg2rad_(SPEED_SWITCH_DEG_S)}),
  rz:new Filter1D({dead:deg2rad_(DEAD_BAND_DEG), maxSpeed:deg2rad_(MAX_SPEED_DEG_S), alphaSlow:EMA_ALPHA_SLOW, alphaFast:EMA_ALPHA_FAST, speedSwitch:deg2rad_(SPEED_SWITCH_DEG_S)}),
};
function mkFingerFilters(){
  return {
    mcp:new Filter1D({dead:deg2rad_(DEAD_BAND_DEG), maxSpeed:deg2rad_(MAX_SPEED_DEG_S), alphaSlow:EMA_ALPHA_SLOW, alphaFast:EMA_ALPHA_FAST, speedSwitch:deg2rad_(SPEED_SWITCH_DEG_S)}),
    pip:new Filter1D({dead:deg2rad_(DEAD_BAND_DEG), maxSpeed:deg2rad_(MAX_SPEED_DEG_S), alphaSlow:EMA_ALPHA_SLOW, alphaFast:EMA_ALPHA_FAST, speedSwitch:deg2rad_(SPEED_SWITCH_DEG_S)}),
    dip:new Filter1D({dead:deg2rad_(DEAD_BAND_DEG), maxSpeed:deg2rad_(MAX_SPEED_DEG_S), alphaSlow:EMA_ALPHA_SLOW, alphaFast:EMA_ALPHA_FAST, speedSwitch:deg2rad_(SPEED_SWITCH_DEG_S)}),
    abd:new Filter1D({dead:deg2rad_(DEAD_BAND_DEG), maxSpeed:deg2rad_(MAX_SPEED_DEG_S), alphaSlow:EMA_ALPHA_SLOW, alphaFast:EMA_ALPHA_FAST, speedSwitch:deg2rad_(SPEED_SWITCH_DEG_S)}),
    cmc:new Filter1D({dead:deg2rad_(DEAD_BAND_DEG), maxSpeed:deg2rad_(MAX_SPEED_DEG_S), alphaSlow:EMA_ALPHA_SLOW, alphaFast:EMA_ALPHA_FAST, speedSwitch:deg2rad_(SPEED_SWITCH_DEG_S)}),
  };
}
const fflt = {
  thumb: mkFingerFilters(),
  index: mkFingerFilters(),
  middle: mkFingerFilters(),
  ring: mkFingerFilters(),
  pinky: mkFingerFilters(),
};

// =============== palm 与手指角解算 ===============
function computePalm(jpos, handed){
  const wrist = jpos["wrist"];
  const idx_m = jpos["index-finger-metacarpal"];
  const pky_m = jpos["pinky-finger-metacarpal"];

  const u = vNorm( vSub(pky_m, idx_m) );                    // 横向 across palm
  let w = vNorm( vCross( vSub(pky_m,wrist), vSub(idx_m,wrist) ) ); // 掌面法线
  if (handed === "left") w = vScale(w, -1);                 // 左右统一法线
  const v = vNorm( vCross(w,u) );
  // 旋转矩阵 [u v w] -> 四元数（用简单从轴到矩阵再转四元数）
  // 这里用矩阵->四元数的安全法：取 trace
  const m00=u.x, m01=v.x, m02=w.x;
  const m10=u.y, m11=v.y, m12=w.y;
  const m20=u.z, m21=v.z, m22=w.z;
  const tr = m00+m11+m22;
  let qw,qx,qy,qz;
  if (tr>0){
    const S = Math.sqrt(tr+1.0)*2; qw=0.25*S;
    qx=(m21-m12)/S; qy=(m02-m20)/S; qz=(m10-m01)/S;
  }else if (m00>m11 && m00>m22){
    const S=Math.sqrt(1.0+m00-m11-m22)*2;
    qw=(m21-m12)/S; qx=0.25*S; qy=(m01+m10)/S; qz=(m02+m20)/S;
  }else if (m11>m22){
    const S=Math.sqrt(1.0+m11-m00-m22)*2;
    qw=(m02-m20)/S; qx=(m01+m10)/S; qy=0.25*S; qz=(m12+m21)/S;
  }else{
    const S=Math.sqrt(1.0+m22-m00-m11)*2;
    qw=(m10-m01)/S; qx=(m02+m20)/S; qy=(m12+m21)/S; qz=0.25*S;
  }
  // 统一到 前/右/上
  const qPalm = qNorm( qMul( qMul(rBasis, {x:qx,y:qy,z:qz,w:qw}), rBasisInv ) );
  const posPalm = remapPositionToFwdRightUp(wrist);
  return { pos:posPalm, quat:qPalm, axes:{u,v,w} };
}

function fingerAngles(jpos, axes, handed){
  const {u,v,w} = axes;
  const wrist = jpos["wrist"];
  const forwardRef = projOnPlane( vSub(jpos["middle-finger-metacarpal"], wrist), w ); // 掌面内“前”
  const out = {};
  for (const name of Object.keys(CHAINS)){
    const chain = CHAINS[name];
    // 骨段
    const P = chain.map(k=>jpos[k]).filter(Boolean);
    if (P.length<3){ continue; }
    const segs = [];
    for (let i=0;i<P.length-1;i++) segs.push( vSub(P[i+1],P[i]) );

    let mcp_flex = Math.PI - angleBetween(segs[0], w);  // 弯曲为正（rad）
    let abd      = signedAngleOnPlane(forwardRef, segs[0], w); // 掌面内带符号角（rad）
    if (handed==='left') abd = -abd;

    let pip=null, dip=null;
    if (name==='thumb'){
      if (segs.length>=2) pip = angleBetween(segs[0],segs[1]);
      if (segs.length>=3) dip = angleBetween(segs[1],segs[2]);
    }else{
      if (segs.length>=3){
        pip = angleBetween(segs[0],segs[1]);
        dip = angleBetween(segs[1],segs[2]);
      }
    }
    out[name] = {
      MCP_flex: mcp_flex, MCP_abd: abd,
      PIP_flex: pip ?? 0, DIP_flex: dip ?? 0
    };
  }
  return out;
}

function clampDegDeg(valDeg, [lo,hi]){
  return clamp(valDeg, lo, hi);
}

// =============== 门控：关键点质量 -> 本指是否更新 ===============
function fingerQuality(jraw, chain){
  // 至少 80% 骨段的两端关键点有效；半径限制；不得 emulated
  let validJ=0, totalJ=0, badList=[];
  for (const j of chain){
    const p = jraw[j];
    totalJ++;
    if (!p){ badList.push(j); continue; }
    const r = p.radius ?? 0.02;
    const emu = p.emulated===true;
    const limit = j.endsWith('tip') ? TIP_RADIUS : MAX_RADIUS;
    if (emu || r>limit){ badList.push(j); continue; }
    validJ++;
  }
  const ratio = validJ/Math.max(totalJ,1);
  return {ok: ratio>=MIN_VALID_RATIO_FINGER, ratio, dropped:badList};
}

// =============== 基线采集（静止标定） ===============
async function calibrate(session, frame, floorSpace){
  // 采集若干帧求平均
  const t0 = performance.now();
  const bufPalm = [];
  const bufAngles = [];

  while (performance.now()-t0 < CALIB_SECS*1000){
    for (const source of session.inputSources){
      if(!source.hand) continue;
      // wrist/index/pinky/middle 均在时才记
      const need = ["wrist","index-finger-metacarpal","pinky-finger-metacarpal","middle-finger-metacarpal"];
      const joints = {};
      let ok=true;
      for (const j of need){
        const js = source.hand.get(j);
        const pose = js && frame.getJointPose(js, floorSpace);
        if (!(pose && pose.transform)){ ok=false; break; }
        joints[j] = remapPositionToFwdRightUp(pose.transform.position);
      }
      if (!ok) continue;
      const { pos, quat, axes } = computePalm(joints, source.handedness);
      bufPalm.push({pos, quat});
      // 用最小集也能算出中指“前”方向
      const jp = {...joints,
        "thumb-metacarpal": joints["wrist"], // 占位避免空
        "thumb-phalanx-proximal": joints["wrist"]
      };
      const f = fingerAngles(jp, axes, source.handedness);
      bufAngles.push(f);
    }
    await new Promise(r=>setTimeout(r, 10));
  }
  if (!bufPalm.length) return false;

  // 平均（四元数简单平均+单位化）
  const avgPos = bufPalm.reduce((a,b)=>({x:a.x+b.pos.x,y:a.y+b.pos.y,z:a.z+b.pos.z}), {x:0,y:0,z:0});
  avgPos.x/=bufPalm.length; avgPos.y/=bufPalm.length; avgPos.z/=bufPalm.length;
  // 简单四元数平均
  let qx=0,qy=0,qz=0,qw=0;
  for (const {quat:q} of bufPalm){ qx+=q.x; qy+=q.y; qz+=q.z; qw+=q.w; }
  const nq = Math.hypot(qx,qy,qz,qw)||1; const quat0={x:qx/nq,y:qy/nq,z:qz/nq,w:qw/nq};

  baseline.palm.pos0 = avgPos;
  baseline.palm.quat0 = quat0;

  // 手指角 baseline（用最后一帧并做裁剪为 0 起点也行，这里取均值）
  const acc = {thumb:{abd:0,mcp:0,ip:0,cmc:0}, index:{mcp:0,pip:0,dip:0},
               middle:{mcp:0,pip:0,dip:0}, ring:{mcp:0,pip:0,dip:0}, pinky:{mcp:0,pip:0,dip:0}};
  for (const ang of bufAngles){
    if (ang.thumb){ acc.thumb.abd += rad2deg(ang.thumb.MCP_abd||0);
      acc.thumb.mcp += rad2deg(ang.thumb.MCP_flex||0);
      acc.thumb.ip  += rad2deg(ang.thumb.DIP_flex||0);
      acc.thumb.cmc += rad2deg(ang.thumb.PIP_flex||0);
    }
    for (const n of ["index","middle","ring","pinky"]){
      if (!ang[n]) continue;
      acc[n].mcp += rad2deg(ang[n].MCP_flex||0);
      acc[n].pip += rad2deg(ang[n].PIP_flex||0);
      acc[n].dip += rad2deg(ang[n].DIP_flex||0);
    }
  }
  const K = bufAngles.length||1;
  for (const n of ["index","middle","ring","pinky"]){
    baseline.fingers[n].mcp = acc[n].mcp/K;
    baseline.fingers[n].pip = acc[n].pip/K;
    baseline.fingers[n].dip = acc[n].dip/K;
  }
  baseline.fingers.thumb.abd = acc.thumb.abd/K;
  baseline.fingers.thumb.mcp = acc.thumb.mcp/K;
  baseline.fingers.thumb.ip  = acc.thumb.ip/K;
  baseline.fingers.thumb.cmc = acc.thumb.cmc/K;

  return true;
}

// =============== 主流程（含渲染，保持你的画面逻辑） ===============
export async function startXR(){
  setupSockets();

  if(!('xr' in navigator)){ console.log("WebXR 不可用"); return; }
  const ok = await navigator.xr.isSessionSupported('immersive-vr');
  if(!ok){ console.log("immersive-vr 不受支持"); return; }

  let session;
  try{
    session = await navigator.xr.requestSession('immersive-vr', {
      requiredFeatures:['local-floor'],
      optionalFeatures:['hand-tracking']
    });
  }catch(e){ console.log("requestSession 失败："+e); return; }

  const canvas = document.createElement('canvas');
  const glCtx = canvas.getContext('webgl', { xrCompatible:true });
  await glCtx.makeXRCompatible();
  session.updateRenderState({ baseLayer: new XRWebGLLayer(session, glCtx) });
  initGLResources(glCtx);

  const viewerSpace = await session.requestReferenceSpace('viewer');
  const floorSpace  = await session.requestReferenceSpace('local-floor');

  let lastT = performance.now();

  session.requestAnimationFrame(function onFrame(time, frame){
    const baseLayer = session.renderState.baseLayer;
    const viewerPose = frame.getViewerPose(viewerSpace);
    if (!viewerPose){ session.requestAnimationFrame(onFrame); return; }

    const now = performance.now();
    const dt  = Math.max((now-lastT)/1000, 1/120); // 保底 dt
    lastT = now;

    // ===== 渲染（不改你的 UI 逻辑）=====
    glCtx.bindFramebuffer(glCtx.FRAMEBUFFER, baseLayer.framebuffer);
    glCtx.clearColor(0.05,0.05,0.05,1);
    glCtx.clear(glCtx.COLOR_BUFFER_BIT);

    for (const view of viewerPose.views) {
      const vp = baseLayer.getViewport(view);
      glCtx.viewport(vp.x, vp.y, vp.width, vp.height);

      const proj = view.projectionMatrix;
      const viewInv = view.transform.inverse.matrix;
      const viewProj = mat4Multiply(proj, viewInv);

      let model = mat4Translate(0,0,-PANEL_DISTANCE);
      if (PANEL_RX_DEG !== 0){
        model = mat4Multiply(mat4RotateX(deg2rad(PANEL_RX_DEG)), model);
      }

      glCtx.useProgram(program3D);
      const uVP = glCtx.getUniformLocation(program3D, "u_viewProj");
      const uM  = glCtx.getUniformLocation(program3D, "u_model");
      glCtx.uniformMatrix4fv(uVP, false, viewProj);
      glCtx.uniformMatrix4fv(uM , false, model);

      if (frameImageBitmap){
        glCtx.activeTexture(glCtx.TEXTURE0);
        glCtx.bindTexture(glCtx.TEXTURE_2D, texture);
        glCtx.pixelStorei(glCtx.UNPACK_FLIP_Y_WEBGL, false);
        glCtx.texImage2D(glCtx.TEXTURE_2D, 0, glCtx.RGBA, glCtx.RGBA, glCtx.UNSIGNED_BYTE, frameImageBitmap);

        const aPos = glCtx.getAttribLocation(program3D, "a_position");
        glCtx.bindBuffer(glCtx.ARRAY_BUFFER, pos3DBuffer);
        glCtx.enableVertexAttribArray(aPos);
        glCtx.vertexAttribPointer(aPos, 3, glCtx.FLOAT, false, 0, 0);

        const aUV  = glCtx.getAttribLocation(program3D, "a_texcoord");
        glCtx.bindBuffer(glCtx.ARRAY_BUFFER, uvBuffer);
        glCtx.enableVertexAttribArray(aUV);
        glCtx.vertexAttribPointer(aUV, 2, glCtx.FLOAT, false, 0, 0);

        glCtx.drawArrays(glCtx.TRIANGLES, 0, 6);
      }
    }

    // ===== 标定（进入后首次若未完成）=====
    if (!baseline.palm.pos0){
      calibrate(session, frame, floorSpace).then(ok=>{
        if(!ok) console.log("[XR] baseline calibration skipped (insufficient data)");
        else    console.log("[XR] baseline calibrated");
      }).catch(()=>{});
    }

    // ===== 采集与发送（world 统一坐标 + 解算角度 + 门控 + 滤波）=====
    for (const source of session.inputSources){
      if(!source.hand) continue;
      const handed = source.handedness;

      // 收集所有关节（位置在 floorSpace）
      const jraw = {};
      let validCount = 0;
      for (const j of JOINTS){
        const js = source.hand.get(j); if(!js) continue;
        const pose = frame.getJointPose(js, floorSpace);
        if (pose && pose.transform){
          const p = pose.transform.position;
          const q = pose.transform.orientation;
          const P1 = remapPositionToFwdRightUp(p);
          // 方向也映射到统一基（虽然不直接用它来算角）
          const q1 = qNorm( qMul( qMul(rBasis, q), rBasisInv ) );
          jraw[j] = { x:P1.x, y:P1.y, z:P1.z, qx:q1.x, qy:q1.y, qz:q1.z, qw:q1.w,
                      radius: pose.radius ?? null, emulated: pose.emulatedPosition===true };
          validCount++;
        }
      }
      if (validCount < MIN_JOINTS ? !gateTrack(handed,false) : !gateTrack(handed,true)) continue;

      // 必要关键点
      const need = ["wrist","index-finger-metacarpal","pinky-finger-metacarpal","middle-finger-metacarpal"];
      if (!need.every(k => !!jraw[k])) continue;

      // palm 解算
      const jpos = {}; for (const k in jraw){ jpos[k] = {x:jraw[k].x, y:jraw[k].y, z:jraw[k].z}; }
      const palm = computePalm(jpos, handed); // pos/quat 已统一坐标
      if (!baseline.palm.pos0){ continue; }   // 还没基线时不发

      // 相对位移/旋转（axis-angle）
      const dp = vSub(palm.pos, baseline.palm.pos0);
      const dpx = filt.dx.step(dp.x, dt);
      const dpy = filt.dy.step(dp.y, dt);
      const dpz = filt.dz.step(dp.z, dt);

      // 相对旋转 R = q0^{-1} * q  -> axis-angle
      // 四元数相对：q_rel = conj(q0) ⊗ q
      const qrel = qMul( quatConj(baseline.palm.quat0), palm.quat );
      const mag = Math.acos(clamp(qrel.w, -1, 1)) * 2; // 角
      const s = Math.sin(mag/2)||1e-6;
      const axis = { x:qrel.x/s, y:qrel.y/s, z:qrel.z/s };
      const rx = filt.rx.step(axis.x*mag, dt);
      const ry = filt.ry.step(axis.y*mag, dt);
      const rz = filt.rz.step(axis.z*mag, dt);

      // 手指角解算 + 门控 + baseline + 裁剪 + 滤波
      const ang = fingerAngles(jpos, palm.axes, handed);
      const fingersOut = {};
      const quality = { visible_ratio: 1.0, radius_mean: 0.0, dropped_joints: [] };
      let accRatio=0, accCnt=0, accRad=0, accRadN=0;

      // 拇指
      {
        const qf = fingerQuality(jraw, CHAINS.thumb);
        accRatio += qf.ratio; accCnt++; if (!qf.ok) quality.dropped_joints.push(...qf.dropped);
        const abd = clampDegDeg(rad2deg(ang.thumb?.MCP_abd || 0) - (baseline.fingers.thumb.abd||0), LIM.thumb_abd);
        const mcp = clampDegDeg(rad2deg(ang.thumb?.MCP_flex||0) - (baseline.fingers.thumb.mcp||0), LIM.thumb_mcp);
        const ip  = clampDegDeg(rad2deg(ang.thumb?.DIP_flex||0) - (baseline.fingers.thumb.ip ||0), LIM.thumb_ip );
        const cmc = clampDegDeg(rad2deg(ang.thumb?.PIP_flex||0) - (baseline.fingers.thumb.cmc||0), LIM.thumb_cmc);
        // 滤波（deg->rad 进入滤波，内部限速单位是 rad/s）
        const abd_r = fflt.thumb.abd.step(deg2rad_(abd), dt);
        const mcp_r = fflt.thumb.mcp.step(deg2rad_(mcp), dt);
        const ip_r  = fflt.thumb.ip .step(deg2rad_(ip ), dt);
        const cmc_r = fflt.thumb.cmc.step(deg2rad_(cmc), dt);
        fingersOut.thumb = { abd:abd_r, mcp:mcp_r, ip:ip_r, cmc:cmc_r, conf: qf.ok?1:0 };
      }
      // 四指
      for (const n of ["index","middle","ring","pinky"]){
        const qf = fingerQuality(jraw, CHAINS[n]);
        accRatio += qf.ratio; accCnt++; if (!qf.ok) quality.dropped_joints.push(...qf.dropped);

        const mcp = clampDegDeg(rad2deg(ang[n]?.MCP_flex||0) - (baseline.fingers[n].mcp||0), LIM.mcp);
        const pip = clampDegDeg(rad2deg(ang[n]?.PIP_flex||0) - (baseline.fingers[n].pip||0), LIM.pip);
        const dip = clampDegDeg(rad2deg(ang[n]?.DIP_flex||0) - (baseline.fingers[n].dip||0), LIM.dip);

        const mcp_r = fflt[n].mcp.step(deg2rad_(mcp), dt);
        const pip_r = fflt[n].pip.step(deg2rad_(pip), dt);
        const dip_r = fflt[n].dip.step(deg2rad_(dip), dt);

        fingersOut[n] = { mcp:mcp_r, pip:pip_r, dip:dip_r, conf: qf.ok?1:0 };
      }
      quality.visible_ratio = accCnt? accRatio/accCnt : 0.0;

      // 半径均值（粗略）
      for (const k in jraw){ const r=jraw[k].radius; if (typeof r==='number'){ accRad += r; accRadN++; } }
      quality.radius_mean = accRadN? accRad/accRadN : 0.0;

      // === 打包 ===
      const payload = {
        t: time,
        space: "world-unified",
        hand: handed,
        palm: {
          origin: [palm.pos.x, palm.pos.y, palm.pos.z],
          quat:   [palm.quat.w, palm.quat.x, palm.quat.y, palm.quat.z],
          origin0: [baseline.palm.pos0.x, baseline.palm.pos0.y, baseline.palm.pos0.z],
          quat0:   [baseline.palm.quat0.w, baseline.palm.quat0.x, baseline.palm.quat0.y, baseline.palm.quat0.z]
        },
        delta: { pos: [dpx,dpy,dpz], rot: [rx,ry,rz] }, // 供下游直接当 Δx,Δy,Δz, Δrx,Δry,Δrz
        fingers: fingersOut,
        quality
      };

      if (wsSend && wsSend.readyState===1){
        wsSend.send(JSON.stringify(payload));
      }
    }

    session.requestAnimationFrame(onFrame);
  });

  console.log("XR session started");
}
