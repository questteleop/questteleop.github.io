import { setupSockets, wsSend, frameImageBitmap } from './ws.js';
import {
  MIN_JOINTS, gateTrack,
  quatMul, quatNorm, qR, qR_inv,     // 仅为兼容导入，未直接使用 qR/qR_inv
  JOINTS, PANEL_DISTANCE, PANEL_RX_DEG
} from './config.js';
import { deg2rad, mat4Multiply, mat4Translate, mat4RotateX } from './math.js';
import { gl, texture, program3D, pos3DBuffer, uvBuffer, initGLResources } from './gl.js';

// ===================== 坐标基与姿态基（FRU：+X前 +Y右 +Z上） =====================
// 位置基变换：x' = -z, y' = x, z' = y
function remapPositionToFwdRightUp(p) { return { x: -p.z, y: p.x, z: p.y }; }
// 姿态基变换：q' = r ⊗ q ⊗ r^{-1}，对应 [[0,0,-1],[1,0,0],[0,1,0]]
const rBasis = { x: 0.5, y: 0.5, z: -0.5, w: 0.5 };
function quatConj(q){ return { x:-q.x, y:-q.y, z:-q.z, w:q.w }; }
const rBasisInv = quatConj(rBasis);

// ================ 去头部 Yaw（把运动固定到“世界”而不是随头转） =================
// 用“viewer 相对 local-floor”的头部方向求出 yaw（绕 Y 轴旋转）
function yawFromHeadOrientation(qHead){
  // 将前向向量 f0=(0,0,-1) 旋转到 f
  const f0 = {x:0,y:0,z:-1};
  const f  = rotateVecByQuat(f0, qHead);
  // 在 XZ 平面上求 yaw：朝 +X 为正，-Z 为前
  // atan2(x, -z) 能把正前（-Z）给出 0，向右（+X）为正角
  const yaw = Math.atan2(f.x, -f.z);
  return yaw;
}
function rotateVecByQuat(v, q){ // v' = q * v * q^{-1}
  const qv = {x:v.x, y:v.y, z:v.z, w:0};
  const t  = quatMul(q, qv);
  const qi = quatConj(q);
  const r  = quatMul(t, qi);
  return {x:r.x, y:r.y, z:r.z};
}
function quatFromYaw(yaw){
  const h = 0.5*yaw;
  return {x:0, y:Math.sin(h), z:0, w:Math.cos(h)};
}
function rotateAroundY(vec, angle){ // 只绕世界 Y 轴旋转
  const c=Math.cos(angle), s=Math.sin(angle);
  const x =  c*vec.x + s*vec.z;
  const z = -s*vec.x + c*vec.z;
  return {x, y:vec.y, z};
}

// ===================== 稳定器（EMA + Deadzone + 轴锁） =====================
const SMOOTH_ALPHA = 0.25;   // 0.1~0.35：越小越稳，越大越跟手
const DEADZONE_M   = 0.01;   // ~1cm 的死区
const LOCK_WINDOW  = 6;      // 轴锁窗口帧数
const LOCK_RATIO   = 0.7;    // 主导轴比例阈值

class JointStabilizer {
  constructor(){
    this.ema = null;      // EMA 位置
    this.last = null;     // 上一稳定输出
    this.velBuf = [];     // 速度窗口（|Δx|,|Δy|,|Δz|）
  }
  _emaUpdate(p){
    if(!this.ema){ this.ema = {...p}; return this.ema; }
    this.ema.x += SMOOTH_ALPHA*(p.x - this.ema.x);
    this.ema.y += SMOOTH_ALPHA*(p.y - this.ema.y);
    this.ema.z += SMOOTH_ALPHA*(p.z - this.ema.z);
    return this.ema;
  }
  _applyDeadzone(p, ref){
    const out = {...p};
    if (Math.abs(p.x - ref.x) < DEADZONE_M) out.x = ref.x;
    if (Math.abs(p.y - ref.y) < DEADZONE_M) out.y = ref.y;
    if (Math.abs(p.z - ref.z) < DEADZONE_M) out.z = ref.z;
    return out;
  }
  _dominantAxis(){
    if(this.velBuf.length===0) return null;
    const sum = this.velBuf.reduce((a,v)=>({x:a.x+v.x,y:a.y+v.y,z:a.z+v.z}),{x:0,y:0,z:0});
    const total = sum.x+sum.y+sum.z + 1e-9;
    const rx=sum.x/total, ry=sum.y/total, rz=sum.z/total;
    const m = Math.max(rx,ry,rz);
    if (m<LOCK_RATIO) return null;
    return (m===rx)?'x':(m===ry)?'y':'z';
  }
  feed(p){
    const smooth = this._emaUpdate(p);
    const ref = this.last || smooth;
    let out = this._applyDeadzone(smooth, ref);

    if (this.last){
      const vx = Math.abs(out.x - this.last.x);
      const vy = Math.abs(out.y - this.last.y);
      const vz = Math.abs(out.z - this.last.z);
      this.velBuf.push({x:vx,y:vy,z:vz});
      if (this.velBuf.length>LOCK_WINDOW) this.velBuf.shift();

      const dom = this._dominantAxis();
      if (dom){
        const keepSmall = (v, lastV)=> (Math.abs(v-lastV)<5*DEADZONE_M)? lastV : v;
        if (dom==='x'){ out.y=keepSmall(out.y,this.last.y); out.z=keepSmall(out.z,this.last.z); }
        else if (dom==='y'){ out.x=keepSmall(out.x,this.last.x); out.z=keepSmall(out.z,this.last.z); }
        else { out.x=keepSmall(out.x,this.last.x); out.y=keepSmall(out.y,this.last.y); }
      }
    }
    this.last = out;
    return out;
  }
}
const STAB = new Map();
function stabilize(handed, joint, p){
  const key = handed+":"+joint;
  let s = STAB.get(key);
  if(!s){ s = new JointStabilizer(); STAB.set(key,s); }
  return s.feed(p);
}

// ===================== 主流程 =====================
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

  console.log("XR session started");

  const viewerSpace = await session.requestReferenceSpace('viewer');       // 用于渲染
  const floorSpace  = await session.requestReferenceSpace('local-floor');  // 用于绝对坐标

  session.requestAnimationFrame(function onFrame(time, frame){
    const baseLayer = session.renderState.baseLayer;

    // 渲染视图（仍然用 viewerSpace）
    const viewerPose = frame.getViewerPose(viewerSpace);
    if (!viewerPose) { session.requestAnimationFrame(onFrame); return; }

    // 头部相对 local-floor 的姿态（用于求 yaw）
    const headPoseFloor = frame.getViewerPose(floorSpace);
    let yaw = 0, qYaw = {x:0,y:0,z:0,w:1}, qYawInv = {x:0,y:0,z:0,w:1};
    if (headPoseFloor){
      const qHead = headPoseFloor.transform.orientation;
      yaw = yawFromHeadOrientation(qHead);
      qYaw = quatFromYaw(yaw);
      qYawInv = quatConj(qYaw);
    }

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

    // ============== 采集手部（pos: local-floor → 去头yaw → FRU；ori: floor → 去yaw → FRU） ==============
    for (const source of session.inputSources){
      if(!source.hand) continue;
      const handed = source.handedness;
      const joints = {}; let validCount = 0;

      for (const j of JOINTS){
        const js = source.hand.get(j); if(!js) continue;

        const pose = frame.getJointPose(js, floorSpace);
        if(!pose || pose.radius==null) continue;

        // 位置：先去头 yaw（把手运动固定在世界），再做 FRU
        const p0 = pose.transform.position;
        const p1 = rotateAroundY(p0, -yaw);           // 去头 yaw
        const Praw = remapPositionToFwdRightUp(p1);   // FRU

        // 若是推测位置，优先用上一帧稳定值，避免突跳
        let P = Praw;
        if (pose.emulatedPosition === true){
          const key = handed+":"+j;
          const stab = STAB.get(key);
          if (stab && stab.last) P = stab.last;
        }
        // 稳定输出
        P = stabilize(handed, j, P);

        // 姿态：去头 yaw（qYawInv ⊗ q），再 FRU（rBasis ⊗ q ⊗ rBasis^{-1}）
        const qF   = pose.transform.orientation;
        const qAligned = quatMul(qYawInv, qF);
        const qTmp = quatMul(rBasis, qAligned);
        const q    = quatNorm(quatMul(qTmp, rBasisInv));

        joints[j] = { x:P.x, y:P.y, z:P.z, qx:q.x, qy:q.y, qz:q.z, qw:q.w, radius:pose.radius };
        validCount++;
      }

      if(validCount>=MIN_JOINTS ? gateTrack(handed,true) : gateTrack(handed,false)){
        const out={ t:time, space:'local-floor(FRU with yaw-removed)', hand:handed, joints };
        if (wsSend && wsSend.readyState===1) wsSend.send(JSON.stringify(out));
      }
    }

    session.requestAnimationFrame(onFrame);
  });
}