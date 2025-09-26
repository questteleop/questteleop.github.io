import { setupSockets, wsSend, frameImageBitmap } from './ws.js';
import {
  MIN_JOINTS, gateTrack,
  quatMul, quatNorm,
  JOINTS, PANEL_DISTANCE, PANEL_RX_DEG
} from './config.js';
import { deg2rad, mat4Multiply, mat4Translate, mat4RotateX } from './math.js';
import { gl, texture, program3D, pos3DBuffer, uvBuffer, initGLResources } from './gl.js';

// ===== 把 XR 坐标统一到：+X 前、+Y 右、+Z 上 =====
function remapPositionToFwdRightUp(p) {
  return { x: -p.z, y: p.x, z: p.y };
}
const rBasis = { x: 0.5, y: 0.5, z: -0.5, w: 0.5 }; // [[0,0,-1],[1,0,0],[0,1,0]]
function quatConj(q){ return { x:-q.x, y:-q.y, z:-q.z, w:q.w }; }
const rBasisInv = quatConj(rBasis);

// ==== 简单平滑缓冲 ====
let lastJoints = {};
const alpha = 0.3;  // 平滑系数 (0~1)

function smoothValue(prev, next) {
  return prev + alpha * (next - prev);
}

export async function startXR(){
  setupSockets();

  if(!('xr' in navigator)){ console.log("WebXR 不可用"); return; }
  const ok = await navigator.xr.isSessionSupported('immersive-vr');
  if(!ok){ console.log("immersive-vr 不受支持"); return; }

  let session;
  try{
    session = await navigator.xr.requestSession('immersive-vr', {
      requiredFeatures:['local-floor'], // 用来渲染的空间仍然需要
      optionalFeatures:['hand-tracking']
    });
  }catch(e){ console.log("requestSession 失败："+e); return; }

  const canvas = document.createElement('canvas');
  const glCtx = canvas.getContext('webgl', { xrCompatible:true });
  await glCtx.makeXRCompatible();
  session.updateRenderState({ baseLayer: new XRWebGLLayer(session, glCtx) });
  initGLResources(glCtx);

  console.log("XR session started");

  const viewerSpace = await session.requestReferenceSpace('viewer');
  const floorSpace  = await session.requestReferenceSpace('local-floor');

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

    // === 手部数据采集 + 平滑 + 发送 ===
    for (const source of session.inputSources){
      if(!source.hand) continue;
      if (source.hand.trackingState && source.hand.trackingState !== "tracked") {
        console.warn("[XR] Hand not tracked, skipping this frame");
        continue;
      }
      const handed = source.handedness;
      const joints = {};
      let validCount = 0;

      for (const j of JOINTS){
        const js = source.hand.get(j); if(!js) continue;
        // 注意：改为用 viewerSpace 得到相对头显位置
        const pose = frame.getJointPose(js, viewerSpace);
        if(!pose) continue;

        const P1 = remapPositionToFwdRightUp(pose.transform.position);
        const qV = pose.transform.orientation;
        const qTmp = quatMul(rBasis, qV);
        const q1   = quatNorm(quatMul(qTmp, rBasisInv));

        const prev = lastJoints[j];
        const smoothed = prev ? {
          x: smoothValue(prev.x, P1.x),
          y: smoothValue(prev.y, P1.y),
          z: smoothValue(prev.z, P1.z),
          qx:q1.x, qy:q1.y, qz:q1.z, qw:q1.w
        } : {x:P1.x, y:P1.y, z:P1.z, qx:q1.x, qy:q1.y, qz:q1.z, qw:q1.w};

        joints[j] = smoothed;
        validCount++;
      }

      lastJoints = joints;

      const needed = ["wrist","index-finger-metacarpal","pinky-finger-metacarpal","middle-finger-metacarpal"];
      const missing = needed.filter(n=>!(n in joints));
      if (missing.length > 0) {
        console.warn(`[XR] Missing key joints: ${missing.join(", ")}`);
      }

      if(validCount>=MIN_JOINTS ? gateTrack(handed,true) : gateTrack(handed,false)){
        const out={ t:time, space:'viewerSpace (relative to HMD)', hand:handed, joints };
        if (wsSend && wsSend.readyState===1) wsSend.send(JSON.stringify(out));
      }
    }

    session.requestAnimationFrame(onFrame);
  });
}
