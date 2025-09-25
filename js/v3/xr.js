import { setupSockets, wsSend, frameImageBitmap } from './ws.js';
import {
  MIN_JOINTS, gateTrack, transformPos, quatMul, quatNorm, qR, qR_inv,
  JOINTS, PANEL_DISTANCE, PANEL_RX_DEG
} from './config.js';
import { deg2rad, mat4Multiply, mat4Translate, mat4RotateX } from './math.js';
import { gl, texture, program3D, pos3DBuffer, uvBuffer, initGLResources } from './gl.js';

// === 新增：是否使用“世界坐标（local-floor）”来渲染和采样手部 ===
const USE_WORLD = true;  // 改回 false 可恢复为原来的 viewer 相对坐标

// === 新增：面板在世界坐标中的固定位置（仅当 USE_WORLD===true 时生效）
const WORLD_POS_X = 0.0;
const WORLD_POS_Y = 1.5;
const WORLD_POS_Z = -2.0;

export async function startXR(){
  setupSockets();

  if(!('xr' in navigator)){ console.log("WebXR 不可用"); return; }
  const ok = await navigator.xr.isSessionSupported('immersive-vr');
  if(!ok){ console.log("immersive-vr 不受支持"); return; }

  let session;
  try{
    session = await navigator.xr.requestSession('immersive-vr', {
      requiredFeatures:['local-floor'],           // 需要世界坐标
      optionalFeatures:['hand-tracking']          // 手部追踪保持不变
    });
  }catch(e){ console.log("requestSession 失败："+e); return; }

  const canvas = document.createElement('canvas');
  const glCtx = canvas.getContext('webgl', { xrCompatible:true });
  await glCtx.makeXRCompatible();
  session.updateRenderState({ baseLayer: new XRWebGLLayer(session, glCtx) });
  initGLResources(glCtx);

  console.log("XR session started");

  // 原有 viewer 空间 + 新增 world 空间；按开关选择
  const viewerSpace = await session.requestReferenceSpace('viewer');
  const worldSpace  = await session.requestReferenceSpace('local-floor');
  const poseSpace   = USE_WORLD ? worldSpace : viewerSpace;

  session.requestAnimationFrame(function onFrame(time, frame){
    const baseLayer = session.renderState.baseLayer;
    const viewerPose = frame.getViewerPose(poseSpace);
    if (!viewerPose) { session.requestAnimationFrame(onFrame); return; }

    glCtx.bindFramebuffer(glCtx.FRAMEBUFFER, baseLayer.framebuffer);

    // 只清屏一次，避免擦掉先绘制的一只眼
    glCtx.clearColor(0.05,0.05,0.05,1);
    glCtx.clear(glCtx.COLOR_BUFFER_BIT);

    for (const view of viewerPose.views) {
      const vp = baseLayer.getViewport(view);
      glCtx.viewport(vp.x, vp.y, vp.width, vp.height);

      const proj = view.projectionMatrix;
      const viewInv = view.transform.inverse.matrix; // 4x4
      const viewProj = mat4Multiply(proj, viewInv);

      // 模型矩阵：
      // - 世界模式：将面板放在固定世界坐标 (WORLD_POS_*)
      // - 头相对模式（原逻辑）：在头前方 -PANEL_DISTANCE
      let model = USE_WORLD
        ? mat4Translate(WORLD_POS_X, WORLD_POS_Y, WORLD_POS_Z)
        : mat4Translate(0, 0, -PANEL_DISTANCE);

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

    // === 手部数据采集 + 发送（保持原逻辑，只是选择参考空间） ===
    for (const source of session.inputSources){
      if(!source.hand) continue;
      const handed = source.handedness;
      const joints = {}; let validCount = 0;
      for (const j of JOINTS){
        const js = source.hand.get(j); if(!js) continue;
        const pose = frame.getJointPose(js, poseSpace); // ★ 使用选定参考系
        if(!pose || pose.radius==null) continue;
        const P = transformPos(pose.transform.position);
        const q = pose.transform.orientation;
        const q2 = quatNorm(quatMul(quatMul(qR,q), qR_inv));
        joints[j] = { x:P.x,y:P.y,z:P.z,qx:q2.x,qy:q2.y,qz:q2.z,qw:q2.w, radius:pose.radius };
        validCount++;
      }
      if(validCount>=MIN_JOINTS ? gateTrack(handed,true) : gateTrack(handed,false)){
        const out={t:time,space: USE_WORLD ? 'local-floor' : 'viewer', hand:handed, joints};
        if (wsSend && wsSend.readyState===1) wsSend.send(JSON.stringify(out));
      }
    }

    session.requestAnimationFrame(onFrame);
  });
}
