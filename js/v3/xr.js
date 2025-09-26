import { setupSockets, wsSend, frameImageBitmap } from './ws.js';
import {
  MIN_JOINTS, gateTrack,
  quatMul, quatNorm, qR, qR_inv, //（保留）
  JOINTS, PANEL_DISTANCE, PANEL_RX_DEG
} from './config.js';
import { deg2rad, mat4Multiply, mat4Translate, mat4RotateX } from './math.js';
import { gl, texture, program3D, pos3DBuffer, uvBuffer, initGLResources } from './gl.js';

// === 统一到 +X 前 / +Y 右 / +Z 上（和你之前一致） ===
function remapPositionToFwdRightUp(p) { return { x: -p.z, y: p.x, z: p.y }; }
// 与上面 remap 匹配的基变换四元数
const rBasis = { x: 0.5, y: 0.5, z: -0.5, w: 0.5 };
function quatConj(q){ return { x:-q.x, y:-q.y, z:-q.z, w:q.w }; }
const rBasisInv = quatConj(rBasis);

// --- 仅提取绕 Y 轴的 yaw（以 local-floor 的 Y 为上）---
function yawFromQuatY(q){ // q = {x,y,z,w}
  // 经典 Tait-Bryan Yaw (Y) 提取，避免万向锁：参考方向为 local-floor
  const {x,y,z,w} = q;
  // yaw about Y:
  const siny_cosp = 2*(w*y + x*z);
  const cosy_cosp = 1 - 2*(y*y + z*z);
  return Math.atan2(siny_cosp, cosy_cosp);
}
function quatFromYawY(yaw){
  const half = yaw * 0.5;
  return { x:0, y:Math.sin(half), z:0, w:Math.cos(half) };
}

// --- 旋转一个位置向量（只绕 Y 轴，用 yaw）---
function rotateVectorAroundY(v, yaw){
  const c = Math.cos(-yaw), s = Math.sin(-yaw); // 用 -yaw 把世界“对齐到头前”
  const x = v.x, y = v.y, z = v.z;
  return { x: c*x + s*z, y, z: -s*x + c*z };
}

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

  const viewerSpace = await session.requestReferenceSpace('viewer');       // 渲染用
  const floorSpace  = await session.requestReferenceSpace('local-floor');  // 绝对坐标

  session.requestAnimationFrame(function onFrame(time, frame){
    const baseLayer = session.renderState.baseLayer;
    const viewerPose = frame.getViewerPose(viewerSpace);
    if (!viewerPose) { session.requestAnimationFrame(onFrame); return; }

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

    // ==== 手部数据（头朝向对齐 + FRU 重映射）====
    // 取“头相对地板”的姿态（只要 yaw）
    const headPose = frame.getPose(viewerSpace, floorSpace);
    const yaw = headPose ? yawFromQuatY(headPose.transform.orientation) : 0.0;
    const qYaw   = quatFromYawY(yaw);
    const qYawInv= quatConj(qYaw);

    for (const source of session.inputSources){
      if(!source.hand) continue;
      const handed = source.handedness;
      const joints = {}; let validCount = 0;

      for (const j of JOINTS){
        const js = source.hand.get(j); if(!js) continue;
        const pose = frame.getJointPose(js, floorSpace); // 同一参考系
        if(!pose || pose.radius==null) continue;

        // 1) 位置：绕 Y 轴旋转 -yaw，把世界对齐到“头前”，再做 FRU 重映射
        const p0 = pose.transform.position;                 // local-floor
        const p1 = rotateVectorAroundY(p0, yaw);            // 头前对齐
        const P  = remapPositionToFwdRightUp(p1);           // FRU

        // 2) 姿态：q' = rBasis ⊗ (qYawInv ⊗ q_floor) ⊗ rBasis^{-1}，最后单位化
        const qF = pose.transform.orientation;              // local-floor
        const qAligned = quatMul(qYawInv, qF);              // 去掉头的水平朝向
        const qTmp = quatMul(rBasis, qAligned);             // 基变换到 FRU
        const q   = quatNorm(quatMul(qTmp, rBasisInv));

        joints[j] = { x:P.x, y:P.y, z:P.z, qx:q.x, qy:q.y, qz:q.z, qw:q.w, radius:pose.radius };
        validCount++;
      }

      if(validCount>=MIN_JOINTS ? gateTrack(handed,true) : gateTrack(handed,false)){
        const out={ t:time, space:'local-floor → head-yaw aligned → FRU', hand:handed, joints };
        if (wsSend && wsSend.readyState===1) wsSend.send(JSON.stringify(out));
      }
    }

    session.requestAnimationFrame(onFrame);
  });
}
