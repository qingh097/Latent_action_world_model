"""DROID dataset loader that reads directly from RLDS tfrecord files.

Avoids the need for converting tfrecords to H5 format. Presents data in the
same ``episode_data`` dict structure that :class:`DroidTransform` expects.
"""

import io
import json
import logging
import os
import struct
from typing import Any, Optional

import numpy as np

from robot_wm.datasets.base import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Low-level TFRecord helpers
# ---------------------------------------------------------------------------


def _read_tfrecord_at(f, offset: int) -> bytes:
    """Read a single TFRecord entry starting at *offset*."""
    f.seek(offset)
    buf = f.read(8)
    if len(buf) < 8:
        raise EOFError
    length = struct.unpack("<Q", buf)[0]
    f.read(4)  # masked CRC of length
    data = f.read(length)
    f.read(4)  # masked CRC of data
    return data


# ---------------------------------------------------------------------------
# tf.train.Example parser
# ---------------------------------------------------------------------------

def _parse_example(data: bytes) -> dict:
    """Parse a serialised ``tf.train.Example`` into a plain dict.

    Returns ``{key: (kind, values)}`` where *kind* is one of
    ``"bytes"``, ``"float"``, ``"int64"`` and *values* is the raw list.

    Uses a pure-protobuf wire-format parser to avoid descriptor pool issues
    in forked DataLoader workers.
    """
    try:
        from tensorflow.core.example import example_pb2

        ex = example_pb2.Example()
        ex.ParseFromString(data)
        out = {}
        for key, feat in ex.features.feature.items():
            kind = feat.WhichOneof("kind")
            if kind == "bytes_list":
                out[key] = ("bytes", list(feat.bytes_list.value))
            elif kind == "float_list":
                out[key] = ("float", list(feat.float_list.value))
            elif kind == "int64_list":
                out[key] = ("int64", list(feat.int64_list.value))
        return out
    except ImportError:
        pass

    # Fallback: manual protobuf wire-format parser (no descriptor pool needed).
    return _parse_example_raw(data)


def _read_varint(buf: bytes, pos: int) -> tuple:
    """Read a varint from buf starting at pos. Returns (value, new_pos)."""
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return result, pos
        shift += 7


def _parse_example_raw(data: bytes) -> dict:
    """Parse tf.train.Example from raw protobuf wire format."""
    # Example { Features features = 1 }
    # Features { map<string, Feature> feature = 1 }
    # Feature { oneof kind { BytesList=1, FloatList=2, Int64List=3 } }

    def _parse_fields(buf: bytes) -> list:
        """Parse wire-format fields into (field_number, wire_type, value)."""
        fields = []
        pos = 0
        while pos < len(buf):
            tag, pos = _read_varint(buf, pos)
            wire_type = tag & 0x07
            field_number = tag >> 3
            if wire_type == 0:  # varint
                val, pos = _read_varint(buf, pos)
                fields.append((field_number, wire_type, val))
            elif wire_type == 2:  # length-delimited
                length, pos = _read_varint(buf, pos)
                fields.append((field_number, wire_type, buf[pos : pos + length]))
                pos += length
            elif wire_type == 5:  # 32-bit
                fields.append((field_number, wire_type, buf[pos : pos + 4]))
                pos += 4
            elif wire_type == 1:  # 64-bit
                fields.append((field_number, wire_type, buf[pos : pos + 8]))
                pos += 8
            else:
                break
        return fields

    def _parse_feature(buf: bytes):
        """Parse a Feature message, return (kind, values)."""
        fields = _parse_fields(buf)
        for fn, wt, val in fields:
            if fn == 1 and wt == 2:  # bytes_list
                # BytesList { repeated bytes value = 1 }
                inner = _parse_fields(val)
                return ("bytes", [v for ifn, _, v in inner if ifn == 1])
            elif fn == 2 and wt == 2:  # float_list
                # FloatList { repeated float value = 1 } - packed
                inner = _parse_fields(val)
                for ifn, iwt, iv in inner:
                    if ifn == 1 and iwt == 2:  # packed floats
                        count = len(iv) // 4
                        return ("float", list(struct.unpack(f"<{count}f", iv)))
                    elif ifn == 1 and iwt == 5:  # individual float
                        return ("float", [struct.unpack("<f", v)[0]
                                          for ifn2, _, v in inner if ifn2 == 1])
                return ("float", [])
            elif fn == 3 and wt == 2:  # int64_list
                # Int64List { repeated int64 value = 1 } - packed varints
                inner = _parse_fields(val)
                for ifn, iwt, iv in inner:
                    if ifn == 1 and iwt == 2:  # packed varints
                        vals = []
                        p = 0
                        while p < len(iv):
                            v, p = _read_varint(iv, p)
                            vals.append(v)
                        return ("int64", vals)
                    elif ifn == 1 and iwt == 0:  # individual varints
                        return ("int64", [v for ifn2, _, v in inner if ifn2 == 1])
                return ("int64", [])
        return None

    out = {}
    # Example.features is field 1
    example_fields = _parse_fields(data)
    for fn, wt, val in example_fields:
        if fn == 1 and wt == 2:  # Features message
            features_fields = _parse_fields(val)
            for ffn, fwt, fval in features_fields:
                if ffn == 1 and fwt == 2:  # map entry
                    # MapEntry { string key=1, Feature value=2 }
                    entry_fields = _parse_fields(fval)
                    key = None
                    feat_buf = None
                    for efn, ewt, eval_ in entry_fields:
                        if efn == 1 and ewt == 2:
                            key = eval_.decode("utf-8")
                        elif efn == 2 and ewt == 2:
                            feat_buf = eval_
                    if key is not None and feat_buf is not None:
                        parsed = _parse_feature(feat_buf)
                        if parsed is not None:
                            out[key] = parsed
    return out


# ---------------------------------------------------------------------------
# Episode dict builder
# ---------------------------------------------------------------------------


def _parsed_to_episode(parsed: dict) -> dict:
    """Convert the flat parsed feature dict into the nested ``episode_data``
    structure expected by :class:`DroidTransform`."""
    n_steps = len(parsed["steps/discount"][1])

    def _floats(key: str, dim: int) -> np.ndarray:
        vals = np.array(parsed[key][1], dtype=np.float64)
        return vals.reshape(n_steps, dim)

    action = _floats("steps/action", 7)

    observation = {
        "cartesian_position": _floats("steps/observation/cartesian_position", 6),
        "gripper_position": _floats("steps/observation/gripper_position", 1),
        "joint_position": _floats("steps/observation/joint_position", 7),
    }

    for cam in ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"):
        key = f"steps/observation/{cam}"
        if key in parsed:
            observation[cam] = _LazyImageArray(parsed[key][1])

    action_dict = {}
    for sub, dim in [
        ("cartesian_position", 6),
        ("cartesian_velocity", 6),
        ("gripper_position", 1),
        ("gripper_velocity", 1),
        ("joint_position", 7),
        ("joint_velocity", 7),
    ]:
        k = f"steps/action_dict/{sub}"
        if k in parsed:
            action_dict[sub] = _floats(k, dim)

    episode = {
        "episode_data": {
            "action": action,
            "observation": observation,
            "action_dict": action_dict,
        },
        "episode_metadata": {
            "file_path": parsed.get(
                "episode_metadata/file_path", ("bytes", [b""])
            )[1][0].decode("utf-8", errors="replace"),
            "recording_folderpath": parsed.get(
                "episode_metadata/recording_folderpath", ("bytes", [b""])
            )[1][0].decode("utf-8", errors="replace"),
            "trajectory_length": n_steps,
        },
    }
    return episode


class _LazyImageArray:
    """Wraps a list of JPEG byte-strings and decodes on slice/index access.

    Supports ``len()``, integer indexing, and slice indexing — the same
    operations that :class:`DroidTransform` uses on the H5 image datasets.
    """

    def __init__(self, jpeg_list: list[bytes]):
        self._jpegs = jpeg_list
        self._cache: dict[int, np.ndarray] = {}
        self.shape = (len(jpeg_list), 180, 320, 3)  # DROID image dims

    def _decode(self, idx: int) -> np.ndarray:
        if idx not in self._cache:
            from PIL import Image

            img = Image.open(io.BytesIO(self._jpegs[idx]))
            self._cache[idx] = np.asarray(img, dtype=np.uint8)
        return self._cache[idx]

    def __len__(self):
        return len(self._jpegs)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._decode(int(key))
        elif isinstance(key, slice):
            indices = range(*key.indices(len(self._jpegs)))
            return np.stack([self._decode(i) for i in indices], axis=0)
        elif isinstance(key, (np.ndarray, list, tuple)):
            # Sparse history asks for arbitrary, non-contiguous frames.
            return np.stack([self._decode(int(i)) for i in key], axis=0)
        else:
            raise TypeError(f"Unsupported index type: {type(key)}")


# ---------------------------------------------------------------------------
# Shard index with disk caching
# ---------------------------------------------------------------------------

_SHARD_CACHE_DIR = os.environ.get("DROID_SHARD_CACHE", os.path.join(os.environ.get("LACWM_DATA", "./data"), ".droid_shard_offsets"))


def _get_shard_offsets(shard_path: str) -> list[int]:
    """Return byte offsets for each record in a shard, with per-shard disk caching.

    On first call for a given shard, scans the file and saves offsets as a
    small numpy file under ``_SHARD_CACHE_DIR``. Subsequent calls (including
    from other workers/processes) just load from disk — no memory accumulation.
    """
    shard_name = os.path.basename(shard_path)
    cache_path = os.path.join(_SHARD_CACHE_DIR, shard_name + ".offsets.npy")

    if os.path.exists(cache_path):
        return np.load(cache_path).tolist()

    # Scan the shard
    shard_offsets: list[int] = []
    with open(shard_path, "rb") as f:
        while True:
            pos = f.tell()
            buf = f.read(8)
            if len(buf) < 8:
                break
            length = struct.unpack("<Q", buf)[0]
            f.read(4)
            f.seek(length, 1)
            f.read(4)
            shard_offsets.append(pos)

    # Cache to disk
    os.makedirs(_SHARD_CACHE_DIR, exist_ok=True)
    np.save(cache_path, np.array(shard_offsets, dtype=np.int64))
    return shard_offsets


def _build_index_from_dataset_info(data_dir: str) -> tuple[list[str], list[int]]:
    """Use dataset_info.json to build a lightweight (shard_idx, episode_idx)
    index *without* scanning any tfrecord bytes.

    Returns (shard_files, shard_lengths).
    """
    info_path = os.path.join(data_dir, "dataset_info.json")
    with open(info_path) as f:
        info = json.load(f)
    shard_lengths = [int(x) for x in info["splits"][0]["shardLengths"]]
    shard_files = sorted(
        p for p in os.listdir(data_dir) if p.startswith("droid_101-train.tfrecord-")
    )
    assert len(shard_files) == len(shard_lengths), (
        f"Mismatch: {len(shard_files)} shard files vs "
        f"{len(shard_lengths)} entries in dataset_info.json"
    )
    return shard_files, shard_lengths


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------


class DroidTFDSDataset(Dataset):
    """Load DROID episodes directly from local RLDS tfrecord shards.

    Parameters
    ----------
    data_dir : str
        Directory containing ``droid_101-train.tfrecord-*`` files and
        ``dataset_info.json``. Byte offsets within each shard are computed lazily on
        first access and cached under $DROID_SHARD_CACHE.
    """

    def __init__(
        self,
        data_dir: str,
        seed: int = 0,
        infinite: bool = True,
        transform: Optional[Any] = None,
        subsample_traj: Optional[int] = None,
    ):
        super().__init__(seed=seed, infinite=infinite, transform=transform)
        self._data_dir = data_dir

        # Use dataset_info.json for instant startup; byte offsets are
        # computed lazily per-shard on first access.
        self._shard_files, self._shard_lengths = (
            _build_index_from_dataset_info(data_dir)
        )
        # flat index: list of (shard_idx, episode_idx_in_shard)
        self._flat_index: list[tuple[int, int]] = []
        for si, slen in enumerate(self._shard_lengths):
            for ei in range(slen):
                self._flat_index.append((si, ei))

        total = len(self._flat_index)
        logger.info(
            f"Indexed {total:,} episodes across {len(self._shard_files)} shards"
        )

        if subsample_traj is not None and subsample_traj > 0:
            self._flat_index = self._flat_index[:subsample_traj]
            logger.info(
                f"Subsampled trajectories to {subsample_traj}, "
                f"new dataset size: {len(self._flat_index):,}"
            )

        self.ee_action_dim = 10

    @property
    def name(self) -> str:
        return "DroidDataset"  # keep same name for morphology mapping

    def _get_length(self) -> int:
        return len(self._flat_index)

    def __len__(self) -> int:
        return self._get_length()

    def _load_episode(self, shard_idx: int, episode_idx: int) -> dict:
        """Load and parse a single episode from a shard."""
        shard_path = os.path.join(self._data_dir, self._shard_files[shard_idx])
        offsets = _get_shard_offsets(shard_path)
        with open(shard_path, "rb") as f:
            raw = _read_tfrecord_at(f, offsets[episode_idx])
        parsed = _parse_example(raw)
        return _parsed_to_episode(parsed)

    def _get_sample(self, index: int) -> dict[str, Any]:
        shard_idx, loc = self._flat_index[index]
        episode = self._load_episode(shard_idx, loc)
        if self._transform is not None:
            episode = self._transform(episode)
            assert episode["actions"].shape[-1] == self.ee_action_dim + 9, (
                f"Expected action dim {self.ee_action_dim + 9}, "
                f"got {episode['actions'].shape[-1]}"
            )
        return episode

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._get_sample(index)
