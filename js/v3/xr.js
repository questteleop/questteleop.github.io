import { setupSockets, wsSend, frameImageBitmap } from './ws.js';
import {
  MIN_JOINTS, gateTrack,
  // transformPos,  // ✅ 不再使用原 transformPos
  quatMul, quatNorm, qR, qR_inv, // ✅ qR/qR_inv 也不再用，但保留 import 以减少改动面
  JOINTS, PANEL_DISTANCE, PANEL_RX_DEG
} from './config.js';
import { deg2rad, mat4Multiply, mat4Translate, mat4RotateX } from './math.js';
import { gl, texture, program3D, pos3DBuffer, uvBuffer, initGLResources } from './gl.js';

// ===== 统一坐标系：+X 前进、+Y 右、+Z 上 =====
// WebXR 默认（local/viewer）是 +X 右、+Y 上、+Z 朝向用户（前方是 -Z）。
// 我们要把 XR 坐标转成目标坐标：x' = -z, y' = x, z' = y。
function remapPositionToFwdRightUp(p) {
  return { x: -p.z, y: p.x, z: p.y };
}

// 对应的基变换四元数 r，使向量与姿态从 XR 基变到 (前、右、上)。
// 该 r 满足：v' = R * v，q' = r ⊗ q ⊗ r^{-1}。
// 矩阵为 [[0,0,-1],[1,0,0],[0,1,0]] 的四元数形式为：
const rBasis = { x: 0.5, y: 0.5, z: -0.5, w: 0.5 };
function quatConj(q){ return { x:-q.x, y:-q.y, z:-q.z, w:q.w }; }
const rBasisInv = quatConj(rBasis);

export async function startXR(){
  setupSockets();

  if(!('xr' in navigator)){ console.log("WebXR 不可用"); return; }
  const ok = await navigator.xr.isSessionSupported('immersive-vr');
  if(!ok){ console.log("immersive-vr 不受支持"); return; }

  let session;
  try{
    session = await navigator.xr.requestSession('immersive-vr', {
      requiredFeatures:['local-floor'],   // ✅ 确保可用 local-floor
      optionalFeatures:['hand-tracking']
    });
  }catch(e){ console.log("requestSession 失败："+e); return; }

  const canvas = document.createElement('canvas');
  const glCtx = canvas.getContext('webgl', { xrCompatible:true });
  await glCtx.makeXRCompatible();
  session.updateRenderState({ baseLayer: new XRWebGLLayer(session, glCtx) });
  initGLResources(glCtx);

  console.log("XR session started");

  // ✅ 视图仍用 viewer（不影响你图像渲染）
  const viewerSpace = await session.requestReferenceSpace('viewer');
  // ✅ 手关节改用 local-floor（头动时手不会“相对移动”）
  const handSpace   = await session.requestReferenceSpace('local-floor');

  session.requestAnimationFrame(function onFrame(time, frame){
    const baseLayer = session.renderState.baseLayer;
    const viewerPose = frame.getViewerPose(viewerSpace);
    if (!viewerPose) { session.requestAnimationFrame(onFrame); return; }

    glCtx.bindFramebuffer(glCtx.FRAMEBUFFER, baseLayer.framebuffer);

    // 只清屏一次，避免擦掉先绘制的一只眼
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

    // === 手部数据采集 + 发送（local-floor & 统一坐标系） ===
    for (const source of session.inputSources){
      if(!source.hand) continue;
      const handed = source.handedness;       // 标签不变
      const joints = {}; let validCount = 0;

      for (const j of JOINTS){
        const js = source.hand.get(j); if(!js) continue;
        // ✅ 用 local-floor 取关节姿态
        const pose = frame.getJointPose(js, handSpace);
        if(!pose || pose.radius==null) continue;

        // 位置：XR -> (前、右、上)
        const P1 = remapPositionToFwdRightUp(pose.transform.position);

        // 姿态：q' = rBasis ⊗ qXR ⊗ rBasis^{-1}，再单位化
        const qXR = pose.transform.orientation;
        const qTmp = quatMul(rBasis, qXR);
        const q1 = quatNorm(quatMul(qTmp, rBasisInv));

        joints[j] = { x:P1.x, y:P1.y, z:P1.z, qx:q1.x, qy:q1.y, qz:q1.z, qw:q1.w, radius:pose.radius };
        validCount++;
      }

      if(validCount>=MIN_JOINTS ? gateTrack(handed,true) : gateTrack(handed,false)){
        const out={ t:time, space:'local-floor', hand:handed, joints };
        if (wsSend && wsSend.readyState===1) wsSend.send(JSON.stringify(out));
      }
    }

    session.requestAnimationFrame(onFrame);
  });
}
