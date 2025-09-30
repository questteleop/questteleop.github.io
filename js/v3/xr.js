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

    // === 手部数据采集 + 发送（位置：local-floor；朝向：viewer；统一到 X前/Y右/Z上） ===
    for (const source of session.inputSources){
      if(!source.hand) continue;
      const handed = source.handedness;   // 'left' | 'right'
      const joints = {}; 
      let validCount = 0;

      for (const j of JOINTS){
        const js = source.hand.get(j); 
        if(!js) continue;

        // 位置用 local-floor（绝对）
        const posePos = frame.getJointPose(js, floorSpace);
        // 朝向用 viewer（方向直觉）
        const poseOri = frame.getJointPose(js, viewerSpace);

        if(!posePos || !poseOri || posePos.radius==null) continue;

        // 位置基变换：XR -> (前/右/上)
        const P1 = remapPositionToFwdRightUp(posePos.transform.position);

        // 朝向基变换：q' = r ⊗ q_viewer ⊗ r^{-1}，并单位化
        const qV = poseOri.transform.orientation;
        const qTmp = quatMul(rBasis, qV);
        const q1 = quatNorm(quatMul(qTmp, rBasisInv)); // {x,y,z,w} (xyzw)

        joints[j] = { 
          x:P1.x, y:P1.y, z:P1.z, 
          qx:q1.x, qy:q1.y, qz:q1.z, qw:q1.w, 
          radius:posePos.radius 
        };
        validCount++;
      }

      // 单手门控：够关节数才发送
      const ok = (validCount >= MIN_JOINTS) ? gateTrack(handed, true)
                                            : gateTrack(handed, false);
      if (!ok) continue;

      // —— 仅在此处“打包与发送”改为单手包，其它逻辑不动 ——

      // palm：用 wrist 近似；若没有 wrist 则不发这只手
      const wr = joints["wrist"];
      if (!wr) continue;

      // palm.origin 为 [x,y,z]；palm.quat 为 **wxyz**（后端会再转 xyzw）
      const palm = {
        origin: [wr.x, wr.y, wr.z],
        quat:   [wr.qw, wr.qx, wr.qy, wr.qz]
      };

      // pinch：thumb-tip 到 index-finger-tip 的距离（米），可选
      let pinch;
      const tt = joints["thumb-tip"];
      const it = joints["index-finger-tip"];
      if (tt && it){
        const dx = tt.x - it.x, dy = tt.y - it.y, dz = tt.z - it.z;
        pinch = { thumb_index_dist: Math.hypot(dx, dy, dz) };
      }

      // 逐手发送“单手包”：hand + palm + pinch(+可选 joints 供调试)
      const pkt = {
        t: time,
        hand: handed,
        palm,
        ...(pinch ? { pinch } : {}),
        joints // 如果不想传 joints，直接删掉这一行即可
      };

      if (wsSend && wsSend.readyState === 1){
        wsSend.send(JSON.stringify(pkt));
      }
    }

    session.requestAnimationFrame(onFrame);
  });
}
