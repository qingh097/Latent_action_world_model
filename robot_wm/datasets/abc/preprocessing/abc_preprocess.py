"""Preprocess one ABC episode.mcap -> per-camera mp4 (remux, no re-encode) + states.npz.

Handles both ABC rig types: ZED-X (/top-left-camera) and RealSense (/top-camera).
Output cameras: top (ego), left_wrist, right_wrist. States resampled (nearest) onto
the top-camera frame timestamps. Codec (h264/h265) detected per camera.
"""
import os, sys, time
import numpy as np
import av
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

# output name -> candidate topics (first present wins)
CAM_ALIASES = {
    "top": ["/top-left-camera", "/top-camera"],
    "left_wrist": ["/left-wrist-camera"],
    "right_wrist": ["/right-wrist-camera"],
}
TOPIC2OUT = {t: out for out, ts in CAM_ALIASES.items() for t in ts}
ARM = ["/left-arm-state", "/right-arm-state"]
ARM_ACT = ["/left-arm-action", "/right-arm-action"]
EE = ["/left-ee-state", "/right-ee-state"]
EE_ACT = ["/left-ee-action", "/right-ee-action"]


def ts_ns(t):
    return int(t.seconds) * 1_000_000_000 + int(t.nanos)


def remux(packets, out_path, codec="hevc"):
    """Stream-copy ordered H.264/H.265 packets into mp4 (no re-encode)."""
    packets = sorted(packets, key=lambda x: x[0])
    raw = b"".join(data for _, data in packets)
    tmp = out_path + ".raw"
    with open(tmp, "wb") as f:
        f.write(raw)
    inp = av.open(tmp, mode="r", format=codec)
    out = av.open(out_path, mode="w")
    ins = inp.streams.video[0]
    outs = out.add_stream_from_template(ins)
    n = 0
    for pkt in inp.demux(ins):
        if pkt.dts is None:
            pkt.pts = pkt.dts = n
        pkt.stream = outs
        out.mux(pkt)
        n += 1
    out.close()
    inp.close()
    os.remove(tmp)
    return n


def preprocess(mcap_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cam_pkts = {out: [] for out in CAM_ALIASES}
    cam_topic = {}  # output name -> the topic actually used
    cam_fmt = {}
    series = {k: [] for k in ARM + ARM_ACT + EE + EE_ACT}
    instruction = ""
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for sch, ch, msg, dec in reader.iter_decoded_messages():
            t = ch.topic
            if t in TOPIC2OUT:
                out = TOPIC2OUT[t]
                # if both /top-left-camera and /top-camera exist, keep first seen
                if out in cam_topic and cam_topic[out] != t:
                    continue
                cam_topic[out] = t
                cam_pkts[out].append((ts_ns(dec.timestamp), bytes(dec.data)))
                cam_fmt[out] = dec.format
            elif t in ARM or t in ARM_ACT:
                series[t].append((ts_ns(dec.timestamp), list(dec.position)))
            elif t in EE or t in EE_ACT:
                series[t].append((ts_ns(dec.timestamp), float(dec.position[0])))
            elif t == "/instruction":
                instruction = dec.data

    if not cam_pkts["top"]:
        raise RuntimeError("no top camera packets")

    top = sorted(cam_pkts["top"], key=lambda x: x[0])
    frame_ts = np.array([p[0] for p in top], dtype=np.int64)

    def resample(key, dim):
        arr = sorted(series[key], key=lambda x: x[0])
        if not arr:
            return np.zeros((len(frame_ts), dim), dtype=np.float32)
        ts = np.array([a[0] for a in arr], dtype=np.int64)
        vals = np.array([a[1] for a in arr], dtype=np.float32).reshape(len(arr), -1)
        idx = np.clip(np.searchsorted(ts, frame_ts), 0, len(ts) - 1)
        return vals[idx]

    joints = np.concatenate([resample(ARM[0], 6), resample(ARM[1], 6)], axis=1)
    joints_act = np.concatenate([resample(ARM_ACT[0], 6), resample(ARM_ACT[1], 6)], axis=1)
    grip = np.concatenate([resample(EE[0], 1), resample(EE[1], 1)], axis=1)
    grip_act = np.concatenate([resample(EE_ACT[0], 1), resample(EE_ACT[1], 1)], axis=1)

    n = {}
    for out in CAM_ALIASES:
        if not cam_pkts[out]:
            continue
        codec = "hevc" if cam_fmt.get(out) in ("h265", "hevc") else "h264"
        n[out] = remux(cam_pkts[out], os.path.join(out_dir, f"{out}.mp4"), codec=codec)
    np.savez(
        os.path.join(out_dir, "states.npz"),
        joint_states=joints, joint_actions=joints_act,
        gripper_states=grip, gripper_actions=grip_act,
        frame_ts=frame_ts, instruction=instruction,
    )
    return len(frame_ts), n


if __name__ == "__main__":
    mcap_path, out_dir = sys.argv[1], sys.argv[2]
    t = time.time()
    T, n = preprocess(mcap_path, out_dir)
    print(f"preprocessed {T} frames in {time.time()-t:.1f}s | mp4 frames: {n}")
