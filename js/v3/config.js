// ===== 手部追踪状态机参数（原样） =====
export const MIN_JOINTS=21, GOOD_FRAMES_TO_LOCK=3, BAD_FRAMES_TO_LOSE=8;
export const TRACK={left:{state:'LOST',good:0,bad:0}, right:{state:'LOST',good:0,bad:0}};
export function gateTrack(h,good){
  const st=TRACK[h]||(TRACK[h]={state:'LOST',good:0,bad:0});
  if(good){st.good++;st.bad=0;if(st.state==='LOST'&&st.good>=GOOD_FRAMES_TO_LOCK)st.state='TRACKED';}
  else {st.bad++;st.good=0;if(st.state==='TRACKED'&&st.bad>=BAD_FRAMES_TO_LOSE)st.state='LOST';}
  return st.state==='TRACKED';
}
export function transformPos(p){return{x:-p.z,y:-p.x,z:p.y};}
export function quatMul(a,b){return{x:a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
                                    y:a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
                                    z:a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w,
                                    w:a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z};}
export function quatConj(q){return{x:-q.x,y:-q.y,z:-q.z,w:q.w};}
export function quatNorm(q){const n=Math.hypot(q.x,q.y,q.z,q.w)||1;return{x:q.x/n,y:q.y/n,z:q.z/n,w:q.w/n};}
export const qR={x:-0.5,y:0.5,z:0.5,w:-0.5}, qR_inv=quatConj(qR);
export const JOINTS=["wrist","thumb-metacarpal","thumb-phalanx-proximal","thumb-phalanx-distal","thumb-tip",
  "index-finger-metacarpal","index-finger-phalanx-proximal","index-finger-phalanx-intermediate","index-finger-phalanx-distal","index-finger-tip",
  "middle-finger-metacarpal","middle-finger-phalanx-proximal","middle-finger-phalanx-intermediate","middle-finger-phalanx-distal","middle-finger-tip",
  "ring-finger-metacarpal","ring-finger-phalanx-proximal","ring-finger-phalanx-intermediate","ring-finger-phalanx-distal","ring-finger-tip",
  "pinky-finger-metacarpal","pinky-finger-phalanx-proximal","pinky-finger-phalanx-intermediate","pinky-finger-phalanx-distal","pinky-finger-tip"];

// ===== 显示屏参数（在你眼前的“看板”）=====
export const PANEL_DISTANCE = 1.8;     // 到头部的距离（米）
export const PANEL_WIDTH    = 1.4;     // 面板宽度（米）
export const PANEL_HEIGHT   = 0.9;     // 面板高度（米）
export const PANEL_RX_DEG   = 0;       // 围绕X轴旋转（度）

// 纹理坐标方向控制
export let ORIENT = 0;     // 0/90/180/270
export let FLIP_X = false;
export let FLIP_Y = true;  // 大多数情况下需要Y翻转

// 与原页面一致的日志
export function log(txt){console.log(txt);const el=document.getElementById('log');if(el)el.textContent=txt;}
