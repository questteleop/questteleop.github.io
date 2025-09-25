import { uiLog } from './config.js';

// WebSocket 发送/接收与帧回调
let wsSend = null, wsRecv = null;

// 外部可设置的“接收帧回调”
let onBitmap = null;
export function setOnBitmapCallback(fn){ onBitmap = fn; }

export function setupSockets(){
  const base = document.getElementById('ws').value.trim();

  wsSend = new WebSocket(base);
  wsSend.onopen = ()=> uiLog("Send WS connected: " + base);
  wsSend.onerror= (e)=> console.warn("Send WS error:", e);
  wsSend.onclose= (e)=> console.warn("Send WS closed:", e.code, e.reason);

  wsRecv = new WebSocket(base + "/sub");
  wsRecv.onopen = ()=> console.log("Recv WS connected: " + base + "/sub");
  wsRecv.onerror= (e)=> console.warn("Recv WS error:", e);
  wsRecv.onclose= (e)=> console.warn("Recv WS closed:", e.code, e.reason);
  wsRecv.onmessage = async ev => {
    try{
      const data = JSON.parse(ev.data);
      if (data.frame){
        // 页面里同步显示一份（便于调试）
        const imgEl = document.getElementById("sim");
        if (imgEl) imgEl.src="data:image/jpeg;base64,"+data.frame;

        // 转为 ImageBitmap，禁用 EXIF 自动旋转
        const blob = await fetch("data:image/jpeg;base64,"+data.frame).then(r=>r.blob());
        const bmp  = await createImageBitmap(blob, { imageOrientation:'none' });

        if (typeof onBitmap === 'function') onBitmap(bmp);
      }
    }catch(e){ console.warn("WS parse error:", e); }
  };
}

export function trySendJSON(obj){
  if (wsSend && wsSend.readyState === 1){
    wsSend.send(JSON.stringify(obj));
  }
}
