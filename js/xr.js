import { setupSockets, wsSend, frameImageBitmap } from './ws.js';
import {
  MIN_JOINTS, gateTrack, transformPos, quatMul, quatNorm, qR, qR_inv,
  JOINTS, PANEL_DISTANCE, PANEL_RX_DEG
} from './config.js';
import { deg2rad, mat4Multiply, mat4Translate, mat4RotateX } from './math.js';
import { gl, texture, program3D, pos3DBuffer, uvBuffer, initGLResources } from './gl.js';

export async function startXR(){
  setupSockets();

  if(!('xr' in navigator)){ console.log("WebXR 不可用"); return; }
  const ok=await navigator.xr.isSessionSupported('immersive-vr');
  if(!ok){ console.log("immersive-vr 不受支持"); return; }
  let session;
  try{
    session=await navigator.xr.requestSession('immersive-vr',{requiredFeatures:['local-floor'],optionalFeatures:['hand-tracking']});
  }catch(e){ console.log("requestSession 失败："+e); return; }

  const canvas=document.createElement('canvas');
  const glCtx=canvas.getContext('webgl',{xrCompatible:true});
  await glCtx.makeXRCompatible();
  session.updateRenderState({baseLayer:new XRWebGLLayer(session,glCtx)});
  initGLResources(glCtx);

  console.log("XR session started");
  const viewerSpace=await session.requestReferenceSpace('viewer');

  session.requestAnimationFrame(function onFrame(time,frame){
    const baseLayer=session.renderState.baseLayer;
    const viewerPose = frame.getViewerPose(viewerSpace);
    if (!viewerPose) { session.requestAnimationFrame(onFrame); return; }

    glCtx.bindFramebuffer(glCtx.FRAMEBUFFER,baseLayer.framebuffer);

    // ✅ 关键修正：只清屏一次
    glCtx.clearColor(0.05,0.05,0.05,1);
    glCtx.clear(glCtx.COLOR_BUFFER_BIT);

    for (const view of viewerPose.views) {
      const vp = baseLayer.getViewport(view);
      glCtx.viewport(vp.x, vp.y, vp.width, vp.height);

      // 组合矩阵：projection * view^-1
      const proj = view.projectionMatrix;
      const viewInv = view.transform.inverse.matrix; // 4x4
      const viewProj = mat4Multiply(proj, viewInv);

      // 模型矩阵：把面板放到头前方，并保持竖直
      let model = mat4Translate(0,0,-PANEL_DISTANCE);
      if (PANEL_RX_DEG !== 0){
        model = mat4Multiply(mat4RotateX(deg2rad(PANEL_RX_DEG)), model);
      }

      // 上传矩阵
      glCtx.useProgram(program3D);
      const uVP = glCtx.getUniformLocation(program3D,"u_viewProj");
      const uM  = glCtx.getUniformLocation(program3D,"u_model");
      glCtx.uniformMatrix4fv(uVP,false,viewProj);
      glCtx.uniformMatrix4fv(uM,false, model);

      // 如果有帧：更新纹理并绘制
      if(frameImageBitmap){
        glCtx.activeTexture(glCtx.TEXTURE0);
        glCtx.bindTexture(glCtx.TEXTURE_2D,texture);
        glCtx.pixelStorei(glCtx.UNPACK_FLIP_Y_WEBGL, false); // 方向用 UV 控制
        glCtx.texImage2D(glCtx.TEXTURE_2D,0,glCtx.RGBA,glCtx.RGBA,glCtx.UNSIGNED_BYTE,frameImageBitmap);

        const aPos = glCtx.getAttribLocation(program3D,"a_position");
        glCtx.bindBuffer(glCtx.ARRAY_BUFFER,pos3DBuffer);
        glCtx.enableVertexAttribArray(aPos);
        glCtx.vertexAttribPointer(aPos,3,glCtx.FLOAT,false,0,0);

        const aUV  = glCtx.getAttribLocation(program3D,"a_texcoord");
        glCtx.bindBuffer(glCtx.ARRAY_BUFFER,uvBuffer);
        glCtx.enableVertexAttribArray(aUV);
        glCtx.vertexAttribPointer(aUV,2,glCtx.FLOAT,false,0,0);

        glCtx.drawArrays(glCtx.TRIANGLES,0,6);
      }
    }

    // === 手部数据采集 + 发送（原样） ===
    for(const source of session.inputSources){
      if(!source.hand) continue;
      const handed=source.handedness;
      const joints={}; let validCount=0;
      for(const j of JOINTS){
        const js=source.hand.get(j); if(!js) continue;
        const pose=frame.getJointPose(js,viewerSpace);
        if(!pose || pose.radius==null) continue;
        const P=transformPos(pose.transform.position);
        const q=pose.transform.orientation;
        const q2=quatNorm(quatMul(quatMul(qR,q),qR_inv));
        joints[j]={x:P.x,y:P.y,z:P.z,qx:q2.x,qy:q2.y,qz:q2.z,qw:q2.w,radius:pose.radius};
        validCount++;
      }
      if(validCount>=MIN_JOINTS ? gateTrack(handed,true) : gateTrack(handed,false)){
        const out={t:time,space:'viewer',hand:handed,joints};
        if(wsSend&&wsSend.readyState===1) wsSend.send(JSON.stringify(out));
      }
    }

    session.requestAnimationFrame(onFrame);
  });
}
