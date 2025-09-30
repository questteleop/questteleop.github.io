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

  const viewerSpace = await session.requestReferenceSpace('viewer');       // 用来取朝向
  const floorSpace  = await session.requestReferenceSpace('local-floor');  // 用来取绝对位置

  // 小工具：安全取某关节的位姿（pos 用 floor，ori 用 viewer）
  function getJointPose(frame, source, joint){
    const js = source.hand.get(joint);
    if(!js) return null;
    const pPos = frame.getJointPose(js, floorSpace);
    const pOri = frame.getJointPose(js, viewerSpace);
    if(!pPos || !pOri || pPos.radius==null) return null;

    // 位置统一基：X前/Y右/Z上
    const P = remapPositionToFwdRightUp(pPos.transform.position);
    // 朝向统一基：q' = r ⊗ q_viewer ⊗ r^{-1}，并单位化
    const qV = pOri.transform.orientation;
    const qTmp = quatMul(rBasis, qV);
    const q1 = quatNorm(quatMul(qTmp, rBasisInv)); // xyzw

    return {
      pos: { x:P.x, y:P.y, z:P.z },
      quat_xyzw: q1
    };
  }

  // 计算拇指尖-食指尖捏合距离（米）；失败返回 null
  function pinchDistance(frame, source){
    const jt = getJointPose(frame, source, "thumb-tip");
    const ji = getJointPose(frame, source, "index-finger-tip");
    if(!jt || !ji) return null;
    const dx = jt.pos.x - ji.pos.x;
    const dy = jt.pos.y - ji.pos.y;
    const dz = jt.pos.z - ji.pos.z;
    return Math.hypot(dx, dy, dz);
  }

  session.requestAnimationFrame(function onFrame(time, frame){
    const baseLayer = session.renderState.baseLayer;
    const viewerPose = frame.getViewerPose(viewerSpace);
    if (!viewerPose) { session.requestAnimationFrame(onFrame); return; }

    // ====== 画面渲染（保持你的逻辑不变） ======
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

    // ====== 手部数据：逐手发送“单手包”，与 Python 端现有解析对齐 ======
    for (const source of session.inputSources){
      if(!source.hand) continue;

      // 用 wrist 作为 palm 近似
      const wrist = getJointPose(frame, source, "wrist");
      if(!wrist) continue; // 该手无效则跳过

      // 组 palm 数据：origin 为 [x,y,z]，quat 为 **wxyz**
      const q = wrist.quat_xyzw; // {x,y,z,w}
      const palm = {
        origin: [ wrist.pos.x, wrist.pos.y, wrist.pos.z ],
        quat:   [ q.w, q.x, q.y, q.z ]  // wxyz，匹配 Python 端的 wxyz_to_xyzw()
      };

      // 捏合距离（可选）
      const dist = pinchDistance(frame, source);
      const pinch = (dist!=null) ? { thumb_index_dist: dist } : undefined;

      // 单手包：hand + palm + pinch（与后端单手 schema 一致）
      const pkt = {
        t: time,
        hand: source.handedness,     // 'left' | 'right'
        palm,                        // {origin:[x,y,z], quat:[w,x,y,z]}
        ...(pinch ? { pinch } : {})  // {thumb_index_dist: m}
      };

      if (wsSend && wsSend.readyState === 1){
        wsSend.send(JSON.stringify(pkt));
      }
    }

    session.requestAnimationFrame(onFrame);
  });
}
