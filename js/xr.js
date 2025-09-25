import { setupSockets, setOnBitmapCallback, trySendJSON } from './ws.js';
import {
  MIN_JOINTS, gateTrack, transformPos, quatMul, quatNorm, qR, qR_inv,
  JOINTS, PANEL_DISTANCE, PANEL_RX_DEG, uiLog
} from './config.js';
import { deg2rad, mat4Multiply, mat4Translate, mat4RotateX } from './math.js';
import { initGLResources, drawPanel } from './gl_panel.js';

let gl, frameImageBitmap = null;

export async function startXR(){
  setupSockets();
  setOnBitmapCallback((bmp)=>{ frameImageBitmap = bmp; });

  if(!('xr' in navigator)){ uiLog("WebXR 不可用"); return; }
  const ok = await navigator.xr.isSessionSupported('immersive-vr');
  if(!ok){ uiLog("immersive-vr 不受支持"); return; }

  let session;
  try{
    session = await navigator.xr.requestSession('immersive-vr', {
      requiredFeatures:['local-floor'],
      optionalFeatures:['hand-tracking']
    });
  }catch(e){ uiLog("requestSession 失败："+e); return; }

  const canvas = document.createElement('canvas');
  gl = canvas.getContext('webgl',{ xrCompatible:true });
  await gl.makeXRCompatible();
  session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
  initGLResources(gl);

  uiLog("XR session started");
  const viewerSpace = await session.requestReferenceSpace('viewer');

  session.requestAnimationFrame(function onFrame(time, frame){
    const baseLayer = session.renderState.baseLayer;
    const viewerPose = frame.getViewerPose(viewerSpace);
    if (!viewerPose){ session.requestAnimationFrame(onFrame); return; }

    gl.bindFramebuffer(gl.FRAMEBUFFER, baseLayer.framebuffer);

    // ✅ 左右眼只清一次，避免左眼绘制后被右眼清掉
    gl.clearColor(0.05,0.05,0.05,1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    for (const view of viewerPose.views){
      const vp = baseLayer.getViewport(view);
      gl.viewport(vp.x, vp.y, vp.width, vp.height);

      const proj    = view.projectionMatrix;
      const viewInv = view.transform.inverse.matrix;
      const viewProj= mat4Multiply(proj, viewInv);

      let model = mat4Translate(0,0,-PANEL_DISTANCE);
      if (PANEL_RX_DEG !== 0){
        model = mat4Multiply(mat4RotateX(deg2rad(PANEL_RX_DEG)), model);
      }

      drawPanel(viewProj, model, frameImageBitmap);
    }

    // ===== 手部数据发送（原样）=====
    for(const source of session.inputSources){
      if(!source.hand) continue;
      const handed=source.handedness;
      const joints={}; let validCount=0;
      for(const j of JOINTS){
        const js=source.hand.get(j); if(!js) continue;
        const pose=frame.getJointPose(js,viewerSpace); if(!pose || pose.radius==null) continue;
        const P=transformPos(pose.transform.position);
        const q=pose.transform.orientation;
        const q2=quatNorm(quatMul(quatMul(qR,q),qR_inv));
        joints[j]={x:P.x,y:P.y,z:P.z,qx:q2.x,qy:q2.y,qz:q2.z,qw:q2.w,radius:pose.radius};
        validCount++;
      }
      if(!gateTrack(handed,validCount>=MIN_JOINTS)) continue;
      const out={t:time,space:'viewer',hand:handed,joints};
      trySendJSON(out);
    }

    session.requestAnimationFrame(onFrame);
  });
}
