import { setupSockets, wsSend, frameImageBitmap } from './ws.js';
import {
  MIN_JOINTS, gateTrack,
  // 下面这些保留 import 以减少你其他文件的改动（渲染仍使用）
  quatMul, quatNorm, qR, qR_inv,
  JOINTS, PANEL_DISTANCE, PANEL_RX_DEG
} from './config.js';
import { deg2rad, mat4Multiply, mat4Translate, mat4RotateX } from './math.js';
import { gl, texture, program3D, pos3DBuffer, uvBuffer, initGLResources } from './gl.js';

export async function startXR(){
  setupSockets("ws://127.0.0.1:8765/"); // 生产者端连根路径“/”

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

  // 渲染用 viewer；数据采集用 local-floor
  const viewerSpace = await session.requestReferenceSpace('viewer');
  const floorSpace  = await session.requestReferenceSpace('local-floor');

  // ========== 追踪状态 & 基准锚点 ==========
  // 每只手独立维护一个“基准姿态”（首帧或重置帧）
  const ANCHOR = {
    left:  { pos:null, quat:null },  // pos:[x,y,z], quat:[x,y,z,w] (xyzw)
    right: { pos:null, quat:null }
  };

  // --------- 向量/四元数/姿态工具（最小实现） ----------
  const vsub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
  const vdot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  const vcross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const vlen=(a)=>Math.hypot(a[0],a[1],a[2])||1;
  const vnorm=(a)=>{const L=vlen(a);return [a[0]/L,a[1]/L,a[2]/L];};

  // 从 3 个正交轴（列向量）构造四元数（xyzw）
  function quatFromAxes(u,v,w){
    const m00=u[0], m01=v[0], m02=w[0];
    const m10=u[1], m11=v[1], m12=w[1];
    const m20=u[2], m21=v[2], m22=w[2];
    const tr=m00+m11+m22;
    let qx,qy,qz,qw;
    if(tr>0){
      const S=Math.sqrt(tr+1.0)*2;
      qw=0.25*S;
      qx=(m21-m12)/S;
      qy=(m02-m20)/S;
      qz=(m10-m01)/S;
    }else if(m00>m11 && m00>m22){
      const S=Math.sqrt(1.0+m00-m11-m22)*2;
      qw=(m21-m12)/S; qx=0.25*S; qy=(m01+m10)/S; qz=(m02+m20)/S;
    }else if(m11>m22){
      const S=Math.sqrt(1.0+m11-m00-m22)*2;
      qw=(m02-m20)/S; qx=(m01+m10)/S; qy=0.25*S; qz=(m12+m21)/S;
    }else{
      const S=Math.sqrt(1.0+m22-m00-m11)*2;
      qw=(m10-m01)/S; qx=(m02+m20)/S; qy=(m12+m21)/S; qz=0.25*S;
    }
    return [qx,qy,qz,qw]; // xyzw
  }
  const qConj=(q)=>[-q[0],-q[1],-q[2],q[3]];
  function qMul(a,b){ // (xyzw)*(xyzw)
    const ax=a[0],ay=a[1],az=a[2],aw=a[3];
    const bx=b[0],by=b[1],bz=b[2],bw=b[3];
    return [
      aw*bx+ax*bw+ay*bz-az*by,
      aw*by-ax*bz+ay*bw+az*bx,
      aw*bz+ax*by-ay*bx+az*bw,
      aw*bw-ax*bx-ay*by-az*bz
    ];
  }
  const qNorm=(q)=>{const n=Math.hypot(q[0],q[1],q[2],q[3])||1;return[q[0]/n,q[1]/n,q[2]/n,q[3]/n];};

  // 从 WebXR joints（local-floor）估计手掌坐标系 & 四元数（xyzw）
  function computePalmFromJoints(jpos, handed){
    // 需要的关键点：wrist / index-finger-metacarpal / pinky-finger-metacarpal
    const wrist = jpos["wrist"];
    const idx_m = jpos["index-finger-metacarpal"];
    const pky_m = jpos["pinky-finger-metacarpal"];
    if(!wrist||!idx_m||!pky_m) return null;

    const u = vnorm(vsub(pky_m, idx_m));                    // across palm
    let   w = vnorm(vcross(vsub(pky_m, wrist), vsub(idx_m, wrist))); // palm normal
    if(handed==="left") w=[-w[0],-w[1],-w[2]];              // 左手翻法线，统一方向
    const v = vnorm(vcross(w, u));
    // 再正交一次增强数值稳定
    const w2 = vnorm(vcross(u, v));
    const q = quatFromAxes(u, v, w2);                       // xyzw
    return { origin:wrist, quat:q };
  }

  // ========== 主循环 ==========
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

    // ===== 头部（可选，仅用于下游配准/调试）=====
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

    // ===== 采集并发送 =====
    for (const source of session.inputSources){
      if(!source.hand) continue;
      const handed = source.handedness; // "left"|"right"

      const joints = {};
      let valid = 0;
      for (const j of JOINTS){
        const js = source.hand.get(j); if(!js) continue;
        const poseFloor = frame.getJointPose(js, floorSpace);
        if (poseFloor && poseFloor.transform){
          const p = poseFloor.transform.position;
          joints[j] = [p.x, p.y, p.z]; // 仅保留 x/y/z，减小带宽 & 兼容服务端
          valid++;
        }
      }
      const tracked = valid>=MIN_JOINTS ? gateTrack(handed,true) : gateTrack(handed,false);
      if(!tracked) {
        // 丢追踪时可清空锚点（可选）
        // ANCHOR[handed].pos = ANCHOR[handed].quat = null;
        continue;
      }

      // 计算手掌姿态（绝对）
      const jpos = joints; // 同键名，值为 [x,y,z]
      const need = ["wrist","index-finger-metacarpal","pinky-finger-metacarpal"];
      if(!need.every(k=>k in jpos)) continue;

      const palm = computePalmFromJoints(
        { "wrist": jpos["wrist"],
          "index-finger-metacarpal": jpos["index-finger-metacarpal"],
          "pinky-finger-metacarpal":  jpos["pinky-finger-metacarpal"] },
        handed
      );
      if(!palm) continue;

      // 建立或使用锚点（相对位姿）
      const A = ANCHOR[handed];
      if (!A.pos || !A.quat){
        A.pos  = palm.origin.slice();
        A.quat = palm.quat.slice();   // xyzw
      }
      // Δpos = curr - anchor
      const dpos = [ palm.origin[0]-A.pos[0], palm.origin[1]-A.pos[1], palm.origin[2]-A.pos[2] ];
      // Δrot = q_anchor^{-1} * q_curr   （xyzw）
      const dq   = qMul(qConj(A.quat), palm.quat);
      const dq_n = qNorm(dq);

      // 组织输出（兼容 hand_server_hub）
      const out = {
        t: time*0.001,                 // 可选
        space: 'local-floor',          // 明确坐标系
        hand: handed,
        head_raw,                      // 可选
        joints,                        // 兼容字段：每个关节 [x,y,z]
        palm: {
          origin: palm.origin,         // [x,y,z] 绝对
          quat:   palm.quat            // [x,y,z,w] 绝对
        },
        palm_rel: {
          dpos,                        // [dx,dy,dz] 相对锚点
          dquat: dq_n                  // [x,y,z,w] 相对旋转
        }
      };

      if (wsSend && wsSend.readyState===1) {
        wsSend.send(JSON.stringify(out));
      }
    }

    session.requestAnimationFrame(onFrame);
  });
}
