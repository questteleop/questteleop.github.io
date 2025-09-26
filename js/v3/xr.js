import { setupSockets, wsSend, frameImageBitmap } from './ws.js';
import {
  MIN_JOINTS, gateTrack,
  // 下面这几个保留 import 以减少你其他文件的改动（本文件不再使用）
  quatMul, quatNorm, qR, qR_inv,
  JOINTS, PANEL_DISTANCE, PANEL_RX_DEG
} from './config.js';
import { deg2rad, mat4Multiply, mat4Translate, mat4RotateX } from './math.js';
import { gl, texture, program3D, pos3DBuffer, uvBuffer, initGLResources } from './gl.js';

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

  // 渲染用 viewer；数据同时采集 floor & viewer 两套
  const viewerSpace = await session.requestReferenceSpace('viewer');
  const floorSpace  = await session.requestReferenceSpace('local-floor');

  session.requestAnimationFrame(function onFrame(time, frame){
    const baseLayer = session.renderState.baseLayer;

    // ===== 渲染（保持原逻辑）=====
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

    // ===== 仅采集 & 上送原始数据：不做变换/去噪/去 yaw =====
    // 额外把头部在 floor & viewer 的姿态也带上（下游更好做配准）
    const headFloorPose  = frame.getViewerPose(floorSpace);
    const headViewerPose = viewerPose;
    const head_raw = {
      floor: headFloorPose ? {
        pos: headFloorPose.views[0]?.transform?.position ?? null,
        quat: headFloorPose.views[0]?.transform?.orientation ?? null,
      } : null,
      viewer: headViewerPose ? {
        pos: headViewerPose.views[0]?.transform?.position ?? null,
        quat: headViewerPose.views[0]?.transform?.orientation ?? null,
      } : null
    };

    // —— 收集两只手的候选，然后只发送“当前被追踪手” ——
    const candidates = []; // {handed, validCountFloor, joints, joints_raw}

    for (const source of session.inputSources){
      if(!source.hand) continue;
      const handed = source.handedness;

      const joints = {};     // 兼容字段：扁平，取 floor 空间
      const joints_raw = {}; // 完整原始：同时带 floor & viewer
      let validCountFloor = 0;

      for (const j of JOINTS){
        const js = source.hand.get(j); if(!js) continue;

        const poseFloor  = frame.getJointPose(js, floorSpace);
        const poseViewer = frame.getJointPose(js, viewerSpace);

        // 扁平（兼容老下游）：floor 有就用 floor
        if (poseFloor && poseFloor.transform){
          const p = poseFloor.transform.position;
          const q = poseFloor.transform.orientation;
          joints[j] = {
            x:p.x, y:p.y, z:p.z,
            qx:q.x, qy:q.y, qz:q.z, qw:q.w,
            radius: poseFloor.radius ?? null,
            emulated: poseFloor.emulatedPosition === true
          };
          validCountFloor++;
        }

        // 完整原始：同时塞 floor/viewer（有就填）
        const entry = {};
        if (poseFloor && poseFloor.transform){
          const pf = poseFloor.transform.position;
          const qf = poseFloor.transform.orientation;
          entry.floor = {
            x:pf.x, y:pf.y, z:pf.z,
            qx:qf.x, qy:qf.y, qz:qf.z, qw:qf.w,
            radius: poseFloor.radius ?? null,
            emulated: poseFloor.emulatedPosition === true
          };
        }
        if (poseViewer && poseViewer.transform){
          const pv = poseViewer.transform.position;
          const qv = poseViewer.transform.orientation;
          entry.viewer = {
            x:pv.x, y:pv.y, z:pv.z,
            qx:qv.x, qy:qv.y, qz:qv.z, qw:qv.w,
            radius: poseViewer.radius ?? null,
            emulated: poseViewer.emulatedPosition === true
          };
        }
        if (entry.floor || entry.viewer){
          joints_raw[j] = entry;
        }
      }

      // 更新状态机；只把“TRACKED”状态的手作为候选
      const isGood = validCountFloor >= MIN_JOINTS;
      if (gateTrack(handed, isGood)) {
        candidates.push({ handed, validCountFloor, joints, joints_raw });
      }
    }

    // 只发送当前“被追踪”的一只：关节数最多者；若并列，偏向已跟踪者（由 gateTrack 保证连续性）
    if (candidates.length > 0) {
      candidates.sort((a,b)=> b.validCountFloor - a.validCountFloor);
      const pick = candidates[0];

      const out = {
        t: time,
        space: 'raw',
        tracking: pick.handed,   // ★ 新增：当前被追踪手 'left' | 'right'
        hand: pick.handed,
        head_raw,
        joints: pick.joints,
        joints_raw: pick.joints_raw
      };
      if (wsSend && wsSend.readyState===1) wsSend.send(JSON.stringify(out));
    }
    // else：两只手都未进入 TRACKED，不发包（维持安静）

    session.requestAnimationFrame(onFrame);
  });
}
