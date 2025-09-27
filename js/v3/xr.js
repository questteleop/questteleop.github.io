import { setupSockets, wsSend, frameImageBitmap } from './ws.js';
import {
  MIN_JOINTS, gateTrack,
  // transformPos, // 不再使用
  quatMul, quatNorm, qR, qR_inv, // 保留 import 以减少改动面
  JOINTS, PANEL_DISTANCE, PANEL_RX_DEG
} from './config.js';
import { deg2rad, mat4Multiply, mat4Translate, mat4RotateX } from './math.js';
import { gl, texture, program3D, pos3DBuffer, uvBuffer, initGLResources } from './gl.js';

// ===== 把 XR 坐标统一到：+X 前、+Y 右、+Z 上 =====
// 位置基变换：x' = -z, y' = x, z' = y
function remapPositionToFwdRightUp(p) {
  return { x: -p.z, y: p.x, z: p.y };
}
// 姿态基变换：q' = r ⊗ q ⊗ r^{-1}
// 这个 rBasis 对应把 XR 的(大多数设备)轴系变到我们约定的“前/右/上”
function quatConj(q){ return { x:-q.x, y:-q.y, z:-q.z, w:q.w }; }
// 注意：如果你的工程里已有 qR/qR_inv 是这个基变换，也可以直接复用；
// 这里写死一份，避免不同文件的定义不一致。
const rBasis = { x:  0.5, y:  0.5, z: -0.5, w: 0.5 };
const rBasisInv = quatConj(rBasis);

// 关节质量过滤阈值（可按需调整）
const MAX_RADIUS = 0.03;     // m；大于此半径认为不可靠（可设为 null 跳过）
const DROP_EMULATED = true;  // 丢弃 emulatedPosition 的关节

export async function startXR(){
  setupSockets();

  if(!('xr' in navigator)){ console.log("WebXR 不可用"); return; }
  const ok = await navigator.xr.isSessionSupported('immersive-vr');
  if(!ok){ console.log("immersive-vr 不受支持"); return; }

  let session;
  try{
    session = await navigator.xr.requestSession('immersive-vr', {
      requiredFeatures:['local-floor'],     // 需要绝对位置
      optionalFeatures:['hand-tracking']    // 启用手部
    });
  }catch(e){ console.log("requestSession 失败："+e); return; }

  const canvas = document.createElement('canvas');
  const glCtx = canvas.getContext('webgl', { xrCompatible:true });
  await glCtx.makeXRCompatible();
  session.updateRenderState({ baseLayer: new XRWebGLLayer(session, glCtx) });
  initGLResources(glCtx);

  console.log("XR session started");

  // 渲染仍用 viewer（不改你的画面逻辑）
  const viewerSpace = await session.requestReferenceSpace('viewer');
  // 绝对位置与朝向都用 local-floor（统一参考系）
  const floorSpace  = await session.requestReferenceSpace('local-floor');

  session.requestAnimationFrame(function onFrame(time, frame){
    const baseLayer = session.renderState.baseLayer;
    const viewerPose = frame.getViewerPose(viewerSpace);
    if (!viewerPose) { session.requestAnimationFrame(onFrame); return; }

    // ===== 渲染：保持原逻辑 =====
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

    // ===== 手部数据采集 & 上送：pos/ori 都来自 local-floor（绝对），再统一轴系 =====
    for (const source of session.inputSources){
      if(!source.hand) continue;
      const handed = source.handedness;

      const joints = {};
      let validCount = 0;

      for (const j of JOINTS){
        const js = source.hand.get(j); if(!js) continue;

        // 统一用 floorSpace 获取关节姿态
        const pose = frame.getJointPose(js, floorSpace);
        if(!pose || !pose.transform) continue;

        // 质量过滤：半径、是否仿真位置
        const rad = pose.radius ?? null;
        const emu = pose.emulatedPosition === true;
        if ((DROP_EMULATED && emu) || (MAX_RADIUS!=null && rad!=null && rad > MAX_RADIUS)) {
          continue;
        }

        // 位置：floor → 统一轴系（前/右/上）
        const pf = pose.transform.position;
        const P1 = remapPositionToFwdRightUp(pf);

        // 朝向：floor → 统一轴系（q' = r ⊗ q ⊗ r^{-1}）
        const qf = pose.transform.orientation;
        const qTmp = quatMul(rBasis, qf);
        const q1 = quatNorm(quatMul(qTmp, rBasisInv));

        joints[j] = {
          x:P1.x, y:P1.y, z:P1.z,
          qx:q1.x, qy:q1.y, qz:q1.z, qw:q1.w,
          radius: rad, emulated: emu
        };
        validCount++;
      }

      // 追踪门限与状态机（保持你的原逻辑）
      if(validCount>=MIN_JOINTS ? gateTrack(handed,true) : gateTrack(handed,false)){
        const out = {
          t: time,
          space: 'local-floor',  // 统一参考系
          hand: handed,
          joints
        };
        if (wsSend && wsSend.readyState===1) wsSend.send(JSON.stringify(out));
      }
    }

    session.requestAnimationFrame(onFrame);
  });
}
