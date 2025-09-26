# hand_server_hub.py
import asyncio, json, http, argparse
import websockets
from websockets.server import serve
import math
import time

# --------- 简单向量/四元数工具 ----------
def vsub(a,b):  return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def vadd(a,b):  return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def vdot(a,b):  return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def vcross(a,b):return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def vlen(a):    return math.sqrt(max(vdot(a,a), 1e-12))
def vnorm(a):   L=vlen(a); return (a[0]/L, a[1]/L, a[2]/L)
def vscale(a,s):return (a[0]*s, a[1]*s, a[2]*s)
def proj_on_plane(v, n):  # 投影到法线 n 的平面
    n = vnorm(n)
    return vsub(v, vscale(n, vdot(v,n)))

def clamp(x, lo=-1.0, hi=1.0): return lo if x<lo else hi if x>hi else x
def angle_between(a,b):  # 无符号角（度）
    a,b = vnorm(a), vnorm(b)
    return math.degrees(math.acos(clamp(vdot(a,b))))

def signed_angle_on_plane(ref, vec, n):  # 带符号角（度），逆时针为正（右手定则）
    ref_p = vnorm(proj_on_plane(ref, n))
    vec_p = vnorm(proj_on_plane(vec, n))
    ang = math.degrees(math.acos(clamp(vdot(ref_p, vec_p))))
    s = vdot(vcross(ref_p, vec_p), vnorm(n))
    return ang if s>=0 else -ang

def quat_from_axes(u, v, w):  # 从3个正交单位轴构造四元数（w,x,y,z）
    # 旋转矩阵列向量 = [u v w]
    m00,m01,m02 = u[0], v[0], w[0]
    m10,m11,m12 = u[1], v[1], w[1]
    m20,m21,m22 = u[2], v[2], w[2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = math.sqrt(tr+1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return (qx,qy,qz,qw)

# finger 关节链（WebXR 25关节命名）
FINGERS = {
    "thumb":  ["thumb-metacarpal","thumb-phalanx-proximal","thumb-phalanx-distal","thumb-tip"],
    "index":  ["index-finger-metacarpal","index-finger-phalanx-proximal","index-finger-phalanx-intermediate","index-finger-phalanx-distal","index-finger-tip"],
    "middle": ["middle-finger-metacarpal","middle-finger-phalanx-proximal","middle-finger-phalanx-intermediate","middle-finger-phalanx-distal","middle-finger-tip"],
    "ring":   ["ring-finger-metacarpal","ring-finger-phalanx-proximal","ring-finger-phalanx-intermediate","ring-finger-phalanx-distal","ring-finger-tip"],
    "pinky":  ["pinky-finger-metacarpal","pinky-finger-phalanx-proximal","pinky-finger-phalanx-intermediate","pinky-finger-phalanx-distal","pinky-finger-tip"],
}

def compute_palm_frame(jpos, handedness):
    """
    以 wrist 为基准，利用 index/pinky metacarpal 估计手掌坐标系：
    - u 轴：从食指掌骨到小指掌骨（横向 across palm）
    - w 轴：手掌法线（大致指向手掌外侧）
    - v 轴：u×w，保证右手系
    """
    wrist = jpos["wrist"]
    idx_m = jpos["index-finger-metacarpal"]
    pky_m = jpos["pinky-finger-metacarpal"]

    u = vnorm(vsub(pky_m, idx_m))
    # palm normal: 用 (pinky - wrist) × (index - wrist)
    w = vnorm(vcross(vsub(pky_m, wrist), vsub(idx_m, wrist)))
    # 让左右手方向一致（可选）：左手时翻转法线
    if handedness == "left":
        w = vscale(w, -1.0)
    v = vnorm(vcross(w, u))
    # 再正交一次，确保数值稳定
    w = vnorm(vcross(u, v))
    origin = wrist  # 也可用四个掌骨平均值
    q = quat_from_axes(u, v, w)
    return origin, (u,v,w), q  # 位置、旋转基、四元数

def finger_angles(jpos, palm_axes, handedness):
    """
    计算每根手指的 MCP/PIP/DIP 屈伸角；MCP 还给一个“外展/内收”角（相对手掌平面）。
    角度单位：度。约定：屈曲为正；外展方向以右手定则给符号（左手会自动取反）。
    """
    u,v,w = palm_axes  # w 约为手掌法线
    wrist = jpos["wrist"]
    mid_ref = jpos["middle-finger-metacarpal"]
    forward_ref = proj_on_plane(vsub(mid_ref, wrist), w)  # 手掌平面内的“前”参考

    results = {}
    for name, chain in FINGERS.items():
        # 取关键点
        P = [jpos[k] for k in chain]
        # 骨向量
        segs = [vsub(P[i+1], P[i]) for i in range(len(P)-1)]
        # MCP 屈伸：用第一节骨与手掌法线的夹角（伸直≈0°，握拳增大）
        mcp_flex = 180.0 - angle_between(segs[0], w)
        # MCP 外展/内收：第一节骨投影到手掌平面，与“前向参考”之间的带符号角
        abd = signed_angle_on_plane(forward_ref, segs[0], w)
        # PIP/DIP：相邻骨夹角
        pip = dip = None
        if name == "thumb":
            # 拇指：近节/远节两段
            pip = angle_between(segs[0], segs[1]) if len(segs)>=2 else None
            dip = angle_between(segs[1], segs[2]) if len(segs)>=3 else None
        else:
            if len(segs) >= 3:
                pip = angle_between(segs[0], segs[1])
                dip = angle_between(segs[1], segs[2])

        # 左手把外展角取反，使“向食指侧”为正
        if handedness == "left" and abd is not None:
            abd = -abd

        results[name] = {
            "MCP_flex": round(mcp_flex, 1),
            "MCP_abd":  round(abd, 1) if abd is not None else None,
            "PIP_flex": round(pip, 1) if pip is not None else None,
            "DIP_flex": round(dip, 1) if dip is not None else None,
        }
    return results

def compute_hand_kinematics(frame):
    """
    frame: 你解出来的单帧 JSON（服务器收到的 out）
    打印：手掌 pose（位置+四元数）与每指关节角
    """
    hand = frame.get("hand")
    joints = frame.get("joints", {})
    # 组装 pos 字典
    jpos = {name: (v["x"], v["y"], v["z"]) for name, v in joints.items()
            if isinstance(v, dict) and all(k in v for k in ("x","y","z"))}
    # 必要关键点是否齐全
    need = {"wrist","index-finger-metacarpal","pinky-finger-metacarpal","middle-finger-metacarpal"}
    if not need.issubset(jpos.keys()):
        print("insufficient joints for kinematics")
        return None

    origin, axes, q = compute_palm_frame(jpos, hand)
    angles = finger_angles(jpos, axes, hand)

    ox,oy,oz = origin
    qx,qy,qz,qw = q
    return hand, ox, oy, oz, qx, qy, qz, qw, angles

# --- 每个订阅者一个队列，慢订阅者丢旧保新 ---
SUB_QUEUES = {}   # ws -> asyncio.Queue
SUB_TASKS = {}    # ws -> sender task

async def process_request(path, request_headers):
    # 兼容非 WS 访问（比如健康检查），返回200而不是报错
    if request_headers.get("Upgrade", "").lower() != "websocket":
        body = b"OK: WebSocket hand hub. POST/WS only.\n"
        return (http.HTTPStatus.OK,
                [("Content-Type","text/plain"),("Content-Length",str(len(body)))],
                body)

async def register_subscriber(ws):
    q = asyncio.Queue(maxsize=256)
    SUB_QUEUES[ws] = q
    async def pump():
        try:
            while True:
                msg = await q.get()
                await ws.send(msg)
        except websockets.ConnectionClosed:
            pass
    t = asyncio.create_task(pump())
    SUB_TASKS[ws] = t
    print("subscriber joined:", ws.remote_address)

def unregister_subscriber(ws):
    t = SUB_TASKS.pop(ws, None)
    if t: t.cancel()
    SUB_QUEUES.pop(ws, None)
    print("subscriber left:", ws.remote_address)

async def broadcast(msg: str):
    dead = []
    for ws, q in SUB_QUEUES.items():
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            # 丢旧保新
            try:
                _ = q.get_nowait()
                q.put_nowait(msg)
            except Exception:
                dead.append(ws)
    for ws in dead:
        unregister_subscriber(ws)

def make_packet(frame: dict):
    """
    输入：网页发来的单帧 dict（含 hand/joints/...）
    输出：发送到另一个进程的精简字典（你也可以直接返回 frame 原样）
    """
    kin = compute_hand_kinematics(frame)
    if kin is None:
        return None

    hand, ox, oy, oz, qx, qy, qz, qw, angles = kin
    out = {
        "t": frame.get("t"),
        "space": frame.get("space", "local-floor"),
        "server_ts": time.time(),
        "hand": hand,
        "palm": {
            "origin": (ox, oy, oz),
            "quat": (qx, qy, qz, qw),   # ← 改成 xyzw（和下游一致）
        },
        "angles": angles
    }
    print(f"[{hand}] PALM pose: pos=({ox:+.3f},{oy:+.3f},{oz:+.3f}) quat=({qx:+.3f},{qy:+.3f},{qz:+.3f},{qw:+.3f})")
    return out

async def handler(ws):
    path = getattr(ws, "path", "/")
    peer = ws.remote_address

    if path.startswith("/sub"):
        await register_subscriber(ws)
        try:
            await ws.wait_closed()
        finally:
            unregister_subscriber(ws)
        return

    print("producer connected:", peer, "path=", path)
    try:
        async for msg in ws:
            try:
                frame = json.loads(msg)
            except json.JSONDecodeError:
                print("[hub] bad JSON, ignored")
                continue

            if "frame" in frame:
                print(f"[hub] got frame message, len={len(frame['frame'])}, broadcasting...")
                await broadcast(json.dumps(frame, separators=(",", ":")))
                continue

            out = make_packet(frame)
            if out:
                print(f"[hub] got hand packet, broadcasting (hand={out['hand']})")
                await broadcast(json.dumps(out, separators=(",", ":")))
            else:
                print("[hub] hand packet ignored (insufficient joints)")
    except websockets.ConnectionClosed:
        print("producer disconnected:", peer)

async def main(host, port):
    async with serve(handler, host, port,
                     process_request=process_request,
                     max_size=2**22, ping_interval=30, ping_timeout=30):
        print(f"[hand-hub] listening ws://{host}:{port}  "
              f"(producers -> / or /ingest, subscribers -> /sub)")
        await asyncio.Future()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    asyncio.run(main(args.host, args.port))
