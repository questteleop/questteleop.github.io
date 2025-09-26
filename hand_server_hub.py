# hand_server_hub.py
import asyncio, json, http, argparse
import websockets
from websockets.server import serve
import math
import time

# =================== 模式选择 ===================
SCHEMA_GRIPPER = "gripper"   # 原 WebXR 关节输入 -> 计算 palm/angles
SCHEMA_DEX     = "dex"       # 灵巧手：dof_names + qpos (+ 可选 pose) 透传

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

def signed_angle_on_plane(ref, vec, n):  # 带符号角（度）
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

# ----------------- WebXR → palm/angles 计算 -----------------
def compute_palm_frame(jpos, handedness):
    wrist = jpos["wrist"]
    idx_m = jpos["index-finger-metacarpal"]
    pky_m = jpos["pinky-finger-metacarpal"]

    u = vnorm(vsub(pky_m, idx_m))                                  # across palm
    w = vnorm(vcross(vsub(pky_m, wrist), vsub(idx_m, wrist)))      # palm normal
    if handedness == "left":
        w = vscale(w, -1.0)  # 左右手统一法线方向
    v = vnorm(vcross(w, u))
    w = vnorm(vcross(u, v))
    origin = wrist
    q = quat_from_axes(u, v, w)
    return origin, (u,v,w), q  # (pos), (axes), quat(x,y,z,w)

def finger_angles(jpos, palm_axes, handedness):
    u,v,w = palm_axes
    wrist = jpos["wrist"]
    mid_ref = jpos["middle-finger-metacarpal"]
    forward_ref = proj_on_plane(vsub(mid_ref, wrist), w)

    results = {}
    for name, chain in FINGERS.items():
        P = [jpos[k] for k in chain]
        segs = [vsub(P[i+1], P[i]) for i in range(len(P)-1)]
        mcp_flex = 180.0 - angle_between(segs[0], w)
        abd = signed_angle_on_plane(forward_ref, segs[0], w)

        pip = dip = None
        if name == "thumb":
            pip = angle_between(segs[0], segs[1]) if len(segs)>=2 else None
            dip = angle_between(segs[1], segs[2]) if len(segs)>=3 else None
        else:
            if len(segs) >= 3:
                pip = angle_between(segs[0], segs[1])
                dip = angle_between(segs[1], segs[2])

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
    WebXR 帧 → (hand, ox,oy,oz, qx,qy,qz,qw, angles)
    """
    hand = frame.get("hand")
    joints = frame.get("joints", {})
    jpos = {name: (v["x"], v["y"], v["z"])
            for name, v in joints.items()
            if isinstance(v, dict) and all(k in v for k in ("x","y","z"))}

    need = {"wrist","index-finger-metacarpal","pinky-finger-metacarpal","middle-finger-metacarpal"}
    if not need.issubset(jpos.keys()):
        return None

    origin, axes, q = compute_palm_frame(jpos, hand)
    angles = finger_angles(jpos, axes, hand)

    ox,oy,oz = origin
    qx,qy,qz,qw = q
    return hand, ox, oy, oz, qx, qy, qz, qw, angles

# ----------------- Dex 输入校验/打包 -----------------
def valid_dex_frame(frame: dict) -> bool:
    """
    Dex 帧的最小条件：
      - hand
      - dof_names(list[str]) 与 qpos(list[float]) 长度一致且 >0
      - pose 可选: {"origin":[x,y,z], "quat":[w,x,y,z]}
    """
    if not isinstance(frame, dict): return False
    if "hand" not in frame: return False
    dn, q = frame.get("dof_names"), frame.get("qpos")
    if not (isinstance(dn, list) and isinstance(q, list) and len(dn)==len(q) and len(dn)>0):
        return False
    return True

def _safe_number_list(xs):
    """### CHANGED: 将 qpos 转成 float 列表并剔除 NaN/Inf（保持长度不变，非法项置 0.0）"""
    out = []
    for v in xs:
        try:
            f = float(v)
            if math.isfinite(f):
                out.append(f)
            else:
                out.append(0.0)
        except Exception:
            out.append(0.0)
    return out

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
            try:
                _ = q.get_nowait()
                q.put_nowait(msg)
            except Exception:
                dead.append(ws)
    for ws in dead:
        unregister_subscriber(ws)

def make_packet_gripper(frame: dict):
    """
    Gripper 模式（WebXR）：计算 palm 与 angles，并输出统一包
    """
    kin = compute_hand_kinematics(frame)
    if kin is None:
        print("[hub] insufficient joints for kinematics (gripper)")
        return None

    hand, ox, oy, oz, qx, qy, qz, qw, angles = kin
    out = {
        "t": frame.get("t"),
        "space": frame.get("space", "local-floor"),
        "server_ts": time.time(),
        "hand": hand,
        "palm": { "origin": (ox, oy, oz), "quat": (qw, qx, qy, qz) },
        "angles": angles
    }
    # ### CHANGED: 打印标签与数值顺序一致（xyzw 标签对应 qx,qy,qz,qw）
    print(f"[{hand}] GRIPPER palm pos=({ox:+.3f},{oy:+.3f},{oz:+.3f}) quat(xyzw)=({qx:+.3f},{qy:+.3f},{qz:+.3f},{qw:+.3f})")
    return out

def make_packet_dex(frame: dict):
    """
    Dex 模式：轻度校验 + 透传（并对 qpos 做最小清洗）
    """
    if not valid_dex_frame(frame):
        print("[hub] invalid dex frame; dropped")
        return None

    hand = frame.get("hand")
    dof_names = list(frame.get("dof_names"))  # 保持原序
    qpos_raw  = frame.get("qpos")
    qpos      = _safe_number_list(qpos_raw)   # ### CHANGED: 转 float 并剔除 NaN/Inf

    palm = None
    pose = frame.get("pose")
    if isinstance(pose, dict):
        origin = pose.get("origin")  # [x,y,z]
        quat   = pose.get("quat")    # [w,x,y,z]
        if isinstance(origin, (list,tuple)) and len(origin)==3 and isinstance(quat, (list,tuple)) and len(quat)==4:
            # 统一为 tuple，避免 numpy 类型
            palm = {"origin": (float(origin[0]), float(origin[1]), float(origin[2])),
                    "quat":   (float(quat[0]),   float(quat[1]),   float(quat[2]),   float(quat[3]))}

    out = {
        "t": frame.get("t"),
        "space": frame.get("space", "dex"),
        "server_ts": time.time(),
        "hand": hand,
        "dex": {
            "dof_names": dof_names,
            "qpos": qpos,
            "dof": len(dof_names),   # ### CHANGED: 附上 DOF 计数，便于下游断言 22/16 等
        }
    }
    if palm is not None:
        out["palm"] = palm

    print(f"[{hand}] DEX dof={len(dof_names)}  pose={'Y' if palm else 'N'}")
    return out

async def handler(ws, forced_schema: str):
    path = getattr(ws, "path", "/")
    peer = ws.remote_address

    if path.startswith("/sub"):
        await register_subscriber(ws)
        try:
            await ws.wait_closed()
        finally:
            unregister_subscriber(ws)
        return

    print("producer connected:", peer, "path=", path, "schema=", forced_schema)
    try:
        async for msg in ws:
            try:
                frame = json.loads(msg)
            except json.JSONDecodeError:
                print("[hub] bad JSON, ignored")
                continue

            # 图片帧透传（与手帧并行）
            if "frame" in frame:
                await broadcast(json.dumps(frame, separators=(",", ":")))
                continue

            out = None
            if forced_schema == SCHEMA_GRIPPER:
                # 期望 WebXR 结构：hand + joints
                if ("hand" in frame) and ("joints" in frame) and isinstance(frame["joints"], dict):
                    out = make_packet_gripper(frame)
                else:
                    print("[hub] non-gripper frame received under gripper mode; dropped")
                    continue
            elif forced_schema == SCHEMA_DEX:
                # 期望 dex 结构：hand + dof_names + qpos (+ pose)
                out = make_packet_dex(frame)
            else:
                print(f"[hub] unknown schema='{forced_schema}'")
                continue

            if out:
                await broadcast(json.dumps(out, separators=(",", ":")))
    except websockets.ConnectionClosed:
        print("producer disconnected:", peer)

async def main(host, port, forced_schema: str):
    if forced_schema not in (SCHEMA_GRIPPER, SCHEMA_DEX):
        raise SystemExit(f"--schema must be one of: {SCHEMA_GRIPPER}, {SCHEMA_DEX}")

    async with serve(lambda ws: handler(ws, forced_schema),
                     host, port,
                     process_request=process_request,
                     max_size=2**22, ping_interval=30, ping_timeout=30):
        print(f"[hand-hub] listening ws://{host}:{port}  "
              f"(producers -> / , subscribers -> /sub)  schema={forced_schema}")
        await asyncio.Future()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--schema", choices=[SCHEMA_GRIPPER, SCHEMA_DEX], required=True,
                    help="输入数据格式：gripper（WebXR 手势→夹爪）| dex（灵巧手 DOF）")
    args = ap.parse_args()
    asyncio.run(main(args.host, args.port, args.schema))
