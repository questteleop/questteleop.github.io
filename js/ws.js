// ===== WebSocket（原样）=====
export let wsSend = null;
export let wsRecv = null;
export let frameImageBitmap = null;

export function setupSockets(){
  const base = document.getElementById('ws').value;

  wsSend = new WebSocket(base);
  wsSend.onopen = () => {
    const el = document.getElementById('log');
    console.log("Send WS connected: " + base);
    if (el) el.textContent = "Send WS connected: " + base;
  };

  wsRecv = new WebSocket(base + "/sub");
  wsRecv.onopen = () => console.log("Recv WS connected: " + base + "/sub");
  wsRecv.onmessage = async ev => {
    try{
      const data = JSON.parse(ev.data);
      if (data.frame){
        document.getElementById("sim").src = "data:image/jpeg;base64," + data.frame;
        const blob = await fetch("data:image/jpeg;base64," + data.frame).then(r=>r.blob());
        // 禁用 EXIF 自动旋转；方向全部交给 UV 控制
        frameImageBitmap = await createImageBitmap(blob, { imageOrientation: 'none' });
      }
    }catch(e){
      console.warn("WS parse error:", e);
    }
  };
}

// （可选，但有助于某些构建/缓存环境识别导出）
export { wsSend, wsRecv, frameImageBitmap, setupSockets };
