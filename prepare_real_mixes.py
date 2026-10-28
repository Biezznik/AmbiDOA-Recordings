#!/usr/bin/env python3
"""prepare_real_mixes.py

Generate 11-channel (or N-channel) WAV files for playback through a discrete
speaker array connected to an RME interface.

Each output file:
  - Has exactly --n-speakers channels (default 11).
  - A random number of speakers are "active" (carry audio).
  - Each active channel contains one unique mono source from the ARCA library
    — no mixing between sources on a single channel.
  - Inactive channels are silence.
  - Each source is first normalised to a common level (RMS by default) so output
    loudness reflects the intentional gain window, not the raw source level.
  - A random per-source gain is applied within a configurable dB window.
  - Shorter sources are looped to the length of the longest active source
    (or to --duration-sec if given).
  - A hard clip guard scales the mix down if any sample would exceed 0.99.

A JSON manifest records per-mix metadata: number of sources, which channels are
active, which ARCA file each carries, and the applied gain.

Usage
-----
    # Uses config.json in the same directory by default:
    python prepare_real_mixes.py

    # Or point to a custom config:
    python prepare_real_mixes.py --config /path/to/my_config.json

    # Batching: generate mixes 800..1599 (filenames mix_00800.wav …):
    python prepare_real_mixes.py --start-index 800 --n-mixes 800

All tunable parameters live in config.json — edit that file, not this script.
`--start-index` / `--n-mixes` override the corresponding config keys so a run can
be split into batches from a shell loop without editing the config each time.

Each mix draws from its own RNG stream seeded from (seed, mix_id), so mix N is
identical no matter which batch produced it — batches can be run in any order,
on separate machines, and concatenated.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from math import gcd
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


# ─────────────────────────────── audio helpers ───────────────────────────────

def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def resample_to(sig: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample a 1-D float32 signal from *sr_in* to *sr_out*.

    Uses SciPy's polyphase (`resample_poly`) filter with integer up/down factors
    reduced by their GCD — high quality, anti-aliased, and dependency-light. This
    lets ARCA sources at any native rate (e.g. 44.1 kHz) be matched to the Zylia
    capture rate set in config.json, rather than being skipped.
    """
    if sr_in == sr_out or len(sig) == 0:
        return sig
    g = gcd(int(sr_in), int(sr_out))
    up = int(sr_out) // g
    down = int(sr_in) // g
    return resample_poly(sig, up, down).astype(np.float32)


def normalize_source(
    sig: np.ndarray, mode: str, target_db: float
) -> Tuple[np.ndarray, float]:
    """Normalize one source to a consistent reference level *before* random gain.

    Equalises the (wildly varying) inherent level of the ARCA source files, so
    output loudness is driven by the intentional random gain window, not by how
    hot each source happened to be recorded.

    mode: "rms"  → scale so the signal's RMS matches target_db (dBFS).
          "peak" → scale so the signal's peak matches target_db (dBFS).
          "none" → leave untouched.

    Returns (normalized_signal, applied_gain_db) so the adjustment is recorded
    in the manifest for exact level reconstruction.
    """
    if mode == "none" or len(sig) == 0:
        return sig, 0.0
    if mode == "rms":
        level = float(np.sqrt(np.mean(sig ** 2)))
    elif mode == "peak":
        level = float(np.max(np.abs(sig)))
    else:
        raise ValueError(f"Unknown source_norm mode: {mode!r} (rms|peak|none).")
    if level <= 0.0:
        return sig, 0.0                      # digital silence — nothing to scale
    scale = db_to_lin(target_db) / level
    return sig * scale, 20.0 * float(np.log10(scale))


def loop_to_length(sig: np.ndarray, target: int) -> np.ndarray:
    """Tile a 1-D float32 signal to exactly *target* samples."""
    n = len(sig)
    if n == 0:
        return np.zeros(target, dtype=np.float32)
    if n >= target:
        return sig[:target]
    reps = int(np.ceil(target / n))
    return np.tile(sig, reps)[:target]


def read_as_mono_float32(
    path: Path, target_sr: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """Read a WAV file, downmix to mono float32, and (optionally) resample.

    If *target_sr* is given and differs from the file's native rate, the signal
    is resampled to *target_sr* so every source lands at the Zylia capture rate.

    Returns (samples_1d, sample_rate).
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        mono = data[:, 0]
    else:
        mono = np.mean(data, axis=1)
    if target_sr is not None and sr != target_sr:
        mono = resample_to(mono, sr, target_sr)
        sr = target_sr
    return mono, sr


# ─────────────────────────── ARCA library scanning ───────────────────────────

def collect_arca_files(
    arca_dir: Path,
    min_base_sec: float,
) -> Tuple[List[Path], List[Path]]:
    """Recursively find usable WAV files under *arca_dir*.

    Sources of any native sample rate are accepted — they are resampled to the
    Zylia capture rate at load time (see :func:`read_as_mono_float32`), so no
    file is dropped for SR mismatch. Duration eligibility is measured in seconds
    from each file's own rate, independent of the output rate.

    Returns
    -------
    all_files  : every usable (readable) file
    long_files : subset with duration >= min_base_sec (eligible as base)

    Unreadable files are logged and skipped.
    """
    all_wavs = sorted(arca_dir.rglob("*.wav")) + sorted(arca_dir.rglob("*.WAV"))
    seen: set = set()
    all_wavs = [p for p in all_wavs if not (p in seen or seen.add(p))]

    if not all_wavs:
        raise FileNotFoundError(f"No WAV files found under: {arca_dir}")

    all_files: List[Path] = []
    long_files: List[Path] = []
    skipped = 0
    for p in all_wavs:
        try:
            info = sf.info(str(p))
            all_files.append(p)
            if info.frames / info.samplerate >= min_base_sec:
                long_files.append(p)
        except Exception as exc:
            print(f"  [SKIP] {p.name}: cannot read metadata ({exc})")
            skipped += 1

    print(
        f"ARCA library: {len(all_files)} usable files "
        f"({skipped} skipped), {len(long_files)} eligible as base (≥ {min_base_sec:.0f} s)."
    )
    if not all_files:
        raise RuntimeError("No usable WAV files remain.")
    if not long_files:
        raise RuntimeError(
            f"No WAV files are long enough (≥ {min_base_sec:.0f} s) to serve as base. "
            "Lower min_base_sec or add longer files."
        )
    return all_files, long_files


# ──────────────────────────── source-count weighting ─────────────────────────

def build_source_count_weights(
    min_sources: int,
    max_sources: int,
    mode: str,
    custom_weights: Optional[List[float]],
) -> Tuple[List[int], List[float]]:
    """Return (values, weights) for sampling the number of active sources.

    Modes
    -----
    linear  — weight(n) = 1 + 0.1 * (n - min_sources); mirrors augment02.py,
              favouring more concurrent sources.
    uniform — every count equally likely.
    custom  — caller supplies explicit per-count weights.
    """
    values = list(range(min_sources, max_sources + 1))
    n = len(values)

    if mode == "uniform":
        weights = [1.0] * n
    elif mode == "custom":
        if custom_weights is None or len(custom_weights) != n:
            raise ValueError(
                f"--custom-weights needs exactly {n} values "
                f"(one per source count in [{min_sources}, {max_sources}])."
            )
        weights = list(custom_weights)
    else:  # linear (default)
        weights = [1.0 + 0.1 * i for i in range(n)]

    return values, weights


def sample_n_sources(values: List[int], weights: List[float], rng: random.Random) -> int:
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for v, w in zip(values, weights):
        acc += w
        if r <= acc:
            return v
    return values[-1]


# ─────────────────────────────── mix builder ─────────────────────────────────

def build_one_mix(
    arca_files: List[Path],
    long_files: List[Path],
    n_speakers: int,
    n_active: int,
    target_sr: int,
    gain_min_db: float,
    gain_max_db: float,
    rng: random.Random,
    source_norm: str = "rms",
    source_norm_target_db: float = -20.0,
    source_peak_cap: float = 0.9,
    peak_limit: Optional[float] = 0.99,
) -> Tuple[np.ndarray, List[Optional[dict]]]:
    """Assemble one N-channel WAV (un-mixed, one source per channel).

    Duration is driven by a randomly chosen base file (must be in long_files).
    Any additional source shorter than the base is looped to match.

    Returns
    -------
    wav           : np.ndarray shape (samples, n_speakers), dtype float32
    channel_meta  : list of length n_speakers; None = silent, dict = active
    """
    # Pick base from the long-file pool; remaining from the full pool
    base_path = rng.choice(long_files)
    extras_pool = [f for f in arca_files if f != base_path]
    extra_paths = rng.sample(extras_pool, n_active - 1) if n_active > 1 else []
    chosen_files = [base_path] + extra_paths

    chosen_channels = sorted(rng.sample(range(n_speakers), n_active))

    # Load base first (resampled to target_sr) to establish output length
    base_sig, _ = read_as_mono_float32(base_path, target_sr)
    out_len = len(base_sig)

    wav = np.zeros((out_len, n_speakers), dtype=np.float32)
    channel_meta: List[Optional[dict]] = [None] * n_speakers

    for ch, src_path in zip(chosen_channels, chosen_files):
        if src_path == base_path:
            sig = base_sig
        else:
            sig, _ = read_as_mono_float32(src_path, target_sr)
        # Match every source to a common level first, then apply the random gain
        # so output loudness reflects the gain window — not the raw source level.
        seg = loop_to_length(sig, out_len)
        seg, norm_gain_db = normalize_source(seg, source_norm, source_norm_target_db)
        # Per-source peak cap: bound a single high-crest-factor / silence-heavy
        # source on its OWN channel, so it can't force the global clip guard to
        # drag the whole mix down (which reintroduces level spread).
        capped_db = 0.0
        if source_peak_cap is not None:
            seg_peak = float(np.max(np.abs(seg)))
            if seg_peak > source_peak_cap:
                cap_scale = source_peak_cap / seg_peak
                seg = seg * cap_scale
                capped_db = 20.0 * float(np.log10(cap_scale))
        gain_db = rng.uniform(gain_min_db, gain_max_db)
        wav[:, ch] = seg * db_to_lin(gain_db)
        channel_meta[ch] = {
            "source_file": str(src_path),
            "gain_db": round(gain_db, 3),
            "norm_mode": source_norm,
            "norm_gain_db": round(norm_gain_db, 3),
            "peak_cap_db": round(capped_db, 3),
            "is_base": src_path == base_path,
        }

    # Clip guard only: scale the whole mix down if it would exceed peak_limit.
    # (Per-source RMS-normalised transients can occasionally peak high.)
    if peak_limit is not None:
        peak = float(np.max(np.abs(wav)))
        if peak > peak_limit:
            wav *= peak_limit / peak

    return wav, channel_meta


# ──────────────────────────────── manifest ───────────────────────────────────

def build_manifest_entry(
    mix_id: int,
    out_filename: str,
    n_active: int,
    channel_meta: List[Optional[dict]],
) -> dict:
    return {
        "mix_id": mix_id,
        "audio_filepath": out_filename,
        "n_sources": n_active,
        "active_channels": [ch for ch, m in enumerate(channel_meta) if m is not None],
        "channels": channel_meta,  # list of n_speakers entries; None = silent
    }


# ──────────────────────────────── config loading ─────────────────────────────

# Defaults applied when a key is absent from config.json
_DEFAULTS: dict = {
    "arca_dir": None,           # required — must be set in config.json
    "out_dir": "./real_mixes",
    "n_mixes": 800,
    "start_index": 0,           # first mix_id — offset for batched runs
    "manifest_name": None,      # None → auto (batch-suffixed when start_index > 0)
    "n_speakers": 11,
    "min_sources": 1,
    "max_sources": None,        # None → n_speakers
    "sample_rate": 44100,
    "min_base_sec": 10.0,
    "bit_depth": "PCM_24",      # PCM_16 | PCM_24 | FLOAT
    # Per-source level equalisation applied BEFORE the random gain window, so
    # output loudness is set by the gain window rather than raw source levels.
    "source_norm": "rms",           # rms | peak | none
    "source_norm_target_db": -20.0,  # target dBFS for the chosen norm mode
    "source_peak_cap": 0.9,          # per-source peak ceiling (linear); null disables
    "gain_min_db": -10.0,
    "gain_max_db": 0.0,
    "weight_mode": "linear",    # linear | uniform | custom
    "custom_weights": None,     # list of floats, only used when weight_mode=custom
    "peak_norm": True,          # True → hard clip guard at 0.99; False → disabled
    "seed": 42,
}


def load_config(config_path: Path) -> dict:
    """Load config.json and merge with defaults.  Unknown keys are ignored."""
    if not config_path.exists():
        sys.exit(f"Error: config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        user = json.load(f)
    cfg = dict(_DEFAULTS)
    cfg.update({k: v for k, v in user.items() if k in _DEFAULTS})
    # Keys starting with "_" are free-form comments in config.json, not settings.
    unknown = [k for k in user if k not in _DEFAULTS and not k.startswith("_")]
    if unknown:
        print(f"[WARN] Unrecognised config keys (ignored): {unknown}")
    return cfg


# ─────────────────────────────────── main ────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate N-channel WAV mixes for a discrete speaker array.",
    )
    p.add_argument(
        "--config", type=Path,
        default=Path(__file__).with_name("config.json"),
        metavar="PATH",
        help="Path to config.json (default: config.json next to this script).",
    )
    p.add_argument(
        "--start-index", type=int, default=None, metavar="N",
        help="First mix_id to generate (overrides config 'start_index'). "
             "Use with --n-mixes to split a run into batches.",
    )
    p.add_argument(
        "--n-mixes", type=int, default=None, metavar="N",
        help="How many mixes to generate in this run (overrides config 'n_mixes').",
    )
    cli = p.parse_args()

    cfg = load_config(cli.config)
    print(f"Loaded config: {cli.config}")

    # CLI overrides — handy for batching without editing config.json
    if cli.start_index is not None:
        cfg["start_index"] = cli.start_index
    if cli.n_mixes is not None:
        cfg["n_mixes"] = cli.n_mixes

    # Resolve paths
    arca_dir = Path(cfg["arca_dir"]) if cfg["arca_dir"] else None
    out_dir  = Path(cfg["out_dir"])
    if arca_dir is None:
        sys.exit("Error: 'arca_dir' must be set in config.json.")

    n_speakers  = int(cfg["n_speakers"])
    min_sources = int(cfg["min_sources"])
    max_sources = int(cfg["max_sources"]) if cfg["max_sources"] is not None else n_speakers
    sr          = int(cfg["sample_rate"])
    n_mixes     = int(cfg["n_mixes"])
    start_index = int(cfg["start_index"])
    min_base_sec  = float(cfg["min_base_sec"])
    gain_min_db   = float(cfg["gain_min_db"])
    gain_max_db   = float(cfg["gain_max_db"])
    source_norm   = str(cfg["source_norm"])
    source_norm_target_db = float(cfg["source_norm_target_db"])
    source_peak_cap = (float(cfg["source_peak_cap"])
                       if cfg["source_peak_cap"] is not None else None)
    weight_mode   = str(cfg["weight_mode"])
    custom_weights = cfg["custom_weights"]
    bit_depth     = str(cfg["bit_depth"])
    # peak_norm now controls only the hard clip guard; None disables it (no more
    # silent-output bug from multiplying by 0.0 when peak_norm is False).
    peak_limit    = 0.99 if cfg["peak_norm"] else None
    seed          = int(cfg["seed"])

    # Validate
    if start_index < 0:
        sys.exit(f"Error: start_index must be >= 0 (got {start_index}).")
    if n_mixes < 1:
        sys.exit(f"Error: n_mixes must be >= 1 (got {n_mixes}).")
    if min_sources < 1:
        sys.exit(f"Error: min_sources must be >= 1 (got {min_sources}).")
    if min_sources > max_sources:
        sys.exit(f"Error: min_sources ({min_sources}) > max_sources ({max_sources}).")
    if max_sources > n_speakers:
        sys.exit(f"Error: max_sources ({max_sources}) > n_speakers ({n_speakers}).")
    if gain_min_db > gain_max_db:
        sys.exit("Error: gain_min_db must be <= gain_max_db.")
    if source_norm not in ("rms", "peak", "none"):
        sys.exit(f"Error: source_norm must be rms, peak, or none (got '{source_norm}').")
    if bit_depth not in ("PCM_16", "PCM_24", "FLOAT"):
        sys.exit(f"Error: bit_depth must be PCM_16, PCM_24, or FLOAT (got '{bit_depth}').")

    np.random.seed(seed)

    # Collect usable ARCA files (any native SR — resampled to `sr` at load time)
    print(f"\nScanning ARCA library: {arca_dir}")
    print(f"Output sample rate (Zylia): {sr} Hz — sources resampled to match.")
    arca_files, long_files = collect_arca_files(
        arca_dir, min_base_sec=min_base_sec
    )

    if len(arca_files) < max_sources:
        sys.exit(
            f"Error: ARCA library has only {len(arca_files)} usable files "
            f"but max_sources={max_sources}. Need at least max_sources files."
        )

    # Build source-count weighting table
    values, weights = build_source_count_weights(
        min_sources, max_sources,
        mode=weight_mode,
        custom_weights=custom_weights,
    )
    print(
        f"\nSource-count distribution ({weight_mode}): "
        + ", ".join(f"{v}→{w:.2f}" for v, w in zip(values, weights))
    )
    print(f"Base file min duration: {min_base_sec:.1f} s  ({len(long_files)} eligible)")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[dict] = []

    stop_index = start_index + n_mixes
    print(
        f"\nGenerating {n_mixes} mixes "
        f"(mix_id {start_index}…{stop_index - 1}) → {out_dir}\n"
    )
    for mix_id in range(start_index, stop_index):
        # Per-mix RNG stream derived from (seed, mix_id): mix N is identical
        # regardless of which batch produced it, so batched runs stitch together
        # exactly like one long run.
        rng = random.Random(f"{seed}:{mix_id}")
        n_active = sample_n_sources(values, weights, rng)

        wav, channel_meta = build_one_mix(
            arca_files=arca_files,
            long_files=long_files,
            n_speakers=n_speakers,
            n_active=n_active,
            target_sr=sr,
            gain_min_db=gain_min_db,
            gain_max_db=gain_max_db,
            rng=rng,
            source_norm=source_norm,
            source_norm_target_db=source_norm_target_db,
            source_peak_cap=source_peak_cap,
            peak_limit=peak_limit,
        )

        out_filename = f"mix_{mix_id:05d}.wav"
        sf.write(str(out_dir / out_filename), wav, sr, subtype=bit_depth)

        entry = build_manifest_entry(mix_id, out_filename, n_active, channel_meta)
        manifest.append(entry)

        print(
            f"[{mix_id - start_index + 1:05d}/{n_mixes}] {out_filename}  "
            f"n_sources={n_active}  channels={entry['active_channels']}  "
            f"len={wav.shape[0]/sr:.2f}s"
        )

    # Batches must not clobber each other's manifest — suffix it with the range
    # unless the config names one explicitly. Merge the per-batch manifests once
    # every batch has run.
    manifest_name = cfg["manifest_name"]
    if not manifest_name:
        manifest_name = (
            "manifest.json" if start_index == 0
            else f"manifest_{start_index:05d}_{stop_index - 1:05d}.json"
        )
    manifest_path = out_dir / manifest_name
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {n_mixes} mixes written (mix_id {start_index}…{stop_index - 1}).")
    print(f"  WAV files : {out_dir}")
    print(f"  Manifest  : {manifest_path}")


if __name__ == "__main__":
    main()
