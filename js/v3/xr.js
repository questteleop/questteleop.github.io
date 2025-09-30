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
    return { x: -p.z, y: -p.x, z: p.y };
  }
  // 姿态基变换：q' = r ⊗ q ⊗ r^{-1}
  // 对应矩阵 [[0,0,-1],[1,0,0],[0,1,0]] 的四元数：
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
    // 绝对位置用 local-floor
    const floorSpace  = await session.requestReferenceSpace('local-floor');

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

      // === 手部数据采集 + 发送（手坐标系：以 wrist 为原点/参考姿态 + 统一轴系） ===
      const handsPayload = [];

      // 固定基变换：viewer -> 你定义的全局轴（例如 +X前/+Y右/+Z上）
      const qB = { x: 0.5, y: 0.5, z: -0.5, w: 0.5 }; // 和你前面 rBasis 一致
      const qB_inv = { x: -qB.x, y: -qB.y, z: -qB.z, w: qB.w };

      function quatMul(a,b){
        return {
          x: a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
          y: a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
          z: a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
          w: a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
        };
      }
      function quatConj(q){ return {x:-q.x, y:-q.y, z:-q.z, w:q.w}; }
      function quatNorm(q){ const n=Math.hypot(q.x,q.y,q.z,q.w)||1; return {x:q.x/n,y:q.y/n,z:q.z/n,w:q.w/n}; }
      function rotateVecByQuat(q, v){
        const qv = {x:v.x, y:v.y, z:v.z, w:0};
        const t  = quatMul(q, qv);
        const r  = quatMul(t, quatConj(q));
        return {x:r.x, y:r.y, z:r.z};
      }
      function subVec(a,b){ return {x:a.x-b.x, y:a.y-b.y, z:a.z-b.z}; }

      for (const source of session.inputSources){
        if (!source.hand) continue;
        const handed = source.handedness;   // 'left' | 'right'

        // 1) wrist 作为手系基准（在 viewerSpace 里取）
        const wristSpace = source.hand.get("wrist");
        if (!wristSpace) continue;
        const wristPose = frame.getJointPose(wristSpace, viewerSpace);
        if (!wristPose /*|| wristPose.radius==null*/) continue;  // 不再依赖 radius

        // 先把 wrist 从 viewer 轴旋到“统一轴”
        const Pw_v = wristPose.transform.position;
        const Qw_v = wristPose.transform.orientation;
        const Pw   = rotateVecByQuat(qB, Pw_v);
        const Qw   = quatNorm(quatMul(quatMul(qB, Qw_v), qB_inv));
        const Qw_inv = quatConj(Qw);

        // 2) 逐关节：统一轴 -> 再做手系相对量
        const joints = {};
        let validCount = 0;

        for (const j of JOINTS){
          const js = source.hand.get(j);
          if (!js) continue;
          const pose = frame.getJointPose(js, viewerSpace);
          if (!pose /*|| pose.radius==null*/) continue;

          // 统一轴
          const Pj = rotateVecByQuat(qB, pose.transform.position);
          const Qj = quatNorm(quatMul(quatMul(qB, pose.transform.orientation), qB_inv));

          // 手系：相对 wrist 的位置/姿态
          const dP = subVec(Pj, Pw);
          const p_rel = rotateVecByQuat(Qw_inv, dP);
          const q_rel = quatNorm(quatMul(Qw_inv, Qj));

          joints[j] = {
            x: p_rel.x, y: p_rel.y, z: p_rel.z,
            qx: q_rel.x, qy: q_rel.y, qz: q_rel.z, qw: q_rel.w
            // 可保留 radius: pose.radius 但别依赖它做过滤
          };
          validCount++;
        }

        // 3) 门控：放宽一些，避免整帧被抛（或直接用你的 gateTrack）
        const ok = (validCount >= Math.max(MIN_JOINTS ?? 12, 12))
                  ? gateTrack(handed, true)
                  : gateTrack(handed, false);
        if (!ok) continue;

        // 4) 同时发出“palm/wrist”的全局姿态（统一轴，便于下游做绝对→相对映射）
        //    注意：很多后端喜欢 wxyz 或 xyzw，这里用 wxyz 清楚注明
        const palm = {
          origin: [Pw.x, Pw.y, Pw.z],
          quat:   [Qw.w, Qw.x, Qw.y, Qw.z]  // wxyz
        };

        handsPayload.push({ hand: handed, palm, joints });
      }

      if (handsPayload.length > 0){
        const out = {
          t: time,
          space: 'hand-local(wrist)+world-basis', // 结果既是手系相对量，又统一了轴系
          hands: handsPayload
        };
        if (wsSend && wsSend.readyState === 1){
          wsSend.send(JSON.stringify(out));
        }
      }



      session.requestAnimationFrame(onFrame);
    });
  }