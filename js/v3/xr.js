import { setupSockets, wsSend, frameImageBitmap } from './ws.js';
import {
  quatMul, quatNorm,
  PANEL_DISTANCE, PANEL_RX_DEG
} from './config.js';
import { deg2rad, mat4Multiply, mat4Translate, mat4RotateX } from './math.js';
import { gl, texture, program3D, pos3DBuffer, uvBuffer, initGLResources } from './gl.js';

// ===== 坐标变换：XR → +X前 / +Y右 / +Z上 =====
function remapPositionToFwdRightUp(p) {
  return { x: -p.z, y: p.x, z: p.y };
}
// 基变换四元数（把 XR 默认系旋到 前/右/上）
const rBasis = { x: 0.5, y: 0.5, z: -0.5, w: 0.5 }; // 对应矩阵 [[0,0,-1],[1,0,0],[0,1,0]]
function quatConj(q){ return { x:-q.x, y:-q.y, z:-q.z, w:q.w }; }
const rBasisInv = quatConj(rBasis);

// ==== 控制器简单位置低通（可选） ====
const lastByHand = new Map();       // hand -> {x,y,z}
const alphaPos = 0.25;              // 位置平滑系数（越小越稳，越大越跟手）
function lerp(a, b, t){ return a + (b - a) * t; }

export async function startXR(){
  setupSockets();

  if(!('xr' in navigator)){ console.log("WebXR 不可用"); return; }
  const ok = await navigator.xr.isSessionSupported('immersive-vr');
  if(!ok){ console.log("immersive-vr 不受支持"); return; }

  let session;
  try{
    session = await navigator.xr.requestSession('immersive-vr', {
      requiredFeatures:['local-floor'], // 绝对坐标，头显旋转不引入假位移
      optionalFeatures:[]               // 不需要 hand-tracking 了
    });
  }catch(e){ console.log("requestSession 失败："+e); return; }

  const canvas = document.createElement('canvas');
  const glCtx = canvas.getContext('webgl', { xrCompatible:true });
  await glCtx.makeXRCompatible();
  session.updateRenderState({ baseLayer: new XRWebGLLayer(session, glCtx) });
  initGLResources(glCtx);

  console.log("XR session started");

  const viewerSpace = await session.requestReferenceSpace('viewer');      // 仅用于渲染
  const floorSpace  = await session.requestReferenceSpace('local-floor'); // 控制器位姿用这个

  session.requestAnimationFrame(function onFrame(time, frame){
    const baseLayer = session.renderState.baseLayer;

    // ===== 渲染（保持你原有逻辑）=====
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

    // === 控制器位姿采集 + 发送（不再读 hand joints） ===
    // 取有 gamepad 的 input source 作为控制器
    for (const source of session.inputSources){
      if (!source || !source.gamepad) continue;

      // 优先用 gripSpace（控制器物理握持的位姿）；targetRaySpace 也可作为备选
      const pose = source.gripSpace ? frame.getPose(source.gripSpace, floorSpace)
                                    : (source.targetRaySpace ? frame.getPose(source.targetRaySpace, floorSpace) : null);
      if (!pose) continue;

      // 统一坐标系：位置 & 方向
      const P = remapPositionToFwdRightUp(pose.transform.position);
      const qRaw = pose.transform.orientation;
      const qTmp = quatMul(rBasis, qRaw);
      const q    = quatNorm(quatMul(qTmp, rBasisInv)); // q = (x,y,z,w)

      // 位置低通（可关：把 alphaPos 设为 1.0）
      const key = source.handedness || 'unknown';
      const last = lastByHand.get(key);
      const Px = last ? lerp(last.x, P.x, alphaPos) : P.x;
      const Py = last ? lerp(last.y, P.y, alphaPos) : P.y;
      const Pz = last ? lerp(last.z, P.z, alphaPos) : P.z;
      lastByHand.set(key, {x:Px, y:Py, z:Pz});

      // 触发/握持当作夹爪信号（可按需改按钮索引）
      let gripValue = 1.0;
      try{
        const btns = source.gamepad.buttons || [];
        // 常见：buttons[1] 是 grip，buttons[0] 是 trigger；设备可能不同，自己换
        const bGrip = btns[1] || btns[0];
        if (bGrip) gripValue = (bGrip.value !== undefined) ? bGrip.value : (bGrip.pressed ? 1.0 : 0.0);
      }catch(_){}

      // 为兼容你的 Python 端（读取 pkt['palm']），这里直接构造 palm
      const out = {
        t: time,
        space: 'floorSpace-controller',        // 明确说明来源
        hand: source.handedness || 'unknown',
        palm: {
          origin: [Px, Py, Pz],
          // 你下游常用 (w,x,y,z)，所以这里按 wxyz 发
          quat:   [q.w, q.x, q.y, q.z],
        },
        controller: {
          grip: gripValue
        }
      };

      if (wsSend && wsSend.readyState === 1) {
        wsSend.send(JSON.stringify(out));
      }
    }

    session.requestAnimationFrame(onFrame);
  });
}
