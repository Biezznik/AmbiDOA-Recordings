#!/usr/bin/env python3
"""playback_record.py

Simultaneously play 11-channel WAV mixes through a Fireface 800 speaker array
and record the acoustic result with a Zylia ZM-1E (19-channel) microphone.

For each mix in the manifest:
  1. Load the WAV from the mixes directory.
  2. Pad with configurable pre/post silence.
  3. Play through the Fireface 800 (configurable output channel mapping).
  4. Record all 19 Zylia channels simultaneously via sd.playrec.
  5. Save the recording as a PCM_24 WAV file.
  6. Append the complete ground truth to one aggregated metadata log
     (recordings_metadata.json) — no per-recording JSON sidecars.

A cumulative recordings_database.json is updated after every successful take
so that interrupted runs can be safely resumed (already-recorded mix IDs are
skipped automatically).

Usage
-----
    # Print available audio devices and their channel counts:
    python playback_record.py --list-devices

    # Validate config and manifests without any audio I/O:
    python playback_record.py --dry-run

    # Full run (reads playback_record_config.json by default):
    python playback_record.py

    # Resume after interruption at manifest index 42:
    python playback_record.py --start-from 42

    # Record specific mix IDs only:
    python playback_record.py --mix-ids 0 5 10 15

    # Verify speaker routing before a session (plays test_wav, records Zylia):
    python playback_record.py --test-routing

    # Custom config path:
    python playback_record.py --config /path/to/config.json

Note on clock synchronisation
------------------------------
The Fireface 800 and Zylia ZM-1E run on independent hardware clocks. Over a
typical 10–60 s recording the resulting drift is < 1 ms (< 50 samples @
44.1 kHz) and negligible for DOA estimation. For longer sessions consider
routing both devices through a macOS Aggregate Audio Device (CoreAudio) or an
ASIO aggregate on Windows so they share a single master clock.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
import yaml


# ─────────────────────────────── defaults ────────────────────────────────────

_DEFAULTS: dict = {
    # Paths
    "mixes_dir":            "./real_mixes",
    "mix_manifest":         "./real_mixes/manifest.json",
    # JSON produced by speaker-array-groundtruth/examples/run_measurement.py
    "calibration_manifest": "./speaker-array-groundtruth/data/speaker_positions.json",
    "recordings_dir":       "./recordings",
    "database_file":        "./recordings/recordings_database.json",
    # Compact augument02-style manifest (one entry per successful take).
    "recording_manifest_file": "./recordings/recording_manifest.json",
    # Aggregated full-metadata log: ONE file holding the complete ground truth
    # for every successful take (replaces the old per-recording <id>.json files).
    "metadata_file":        "./recordings/recordings_metadata.json",
    # Slot budget for the manifest's regions/azimuths/elevations/mask lists.
    # Equals the dataset's maximum concurrent sources (NOT the channel count).
    "manifest_slots": 10,

    # Batch-mode recording: a run processes one slice of `batch_size` mixes.
    # Batch B records mix_ids in [B*batch_size, B*batch_size + batch_size).
    # Set batch_index in config or with --batch on the CLI; null disables.
    "batch_size":  100,
    "batch_index": None,

    # Countdown (seconds) before the first recording starts, giving you time to
    # leave the measurement area so you are not captured in the recordings.
    # 0 disables the wait; override per-run with --startup-delay.
    "startup_delay_sec": 15,

    # Device indices — run --list-devices to find the right numbers.
    # null means unset; the script will exit with an error if left as null.
    "playback_device_index":  None,
    "recording_device_index": None,

    # Channel counts
    "n_playback_channels":  11,
    "n_recording_channels": 19,

    # 0-indexed mix-WAV channel → 0-indexed Fireface output channel.
    # Length must equal n_playback_channels.
    # Default: straight-through (mix ch 0 → Fireface out 0, etc.)
    "output_channel_map": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    # 0-indexed mix-WAV channel → speaker ID used in calibration manifest.
    # Length must equal n_playback_channels.
    # Default: mix ch 0 ↔ speaker 1, mix ch 1 ↔ speaker 2, …
    "channel_to_speaker_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],

    "sample_rate": 44100,

    # Silence (seconds) prepended / appended to every playback+recording
    "pre_silence_sec":   0.5,
    "post_silence_sec":  1.5,

    # Pause between consecutive mixes — lets the room fully decay
    "inter_mix_gap_sec": 3.0,

    # sounddevice latency: "low" | "high" | float (seconds)
    "latency": "low",

    # Output WAV subtype for recordings
    "bit_depth": "PCM_24",

    # Filename prefix for recordings  →  <prefix>_<mix_id:05d>.wav
    "recording_prefix": "rec",

    # 11-channel WAV played by --test-routing to verify speaker routing.
    # Each channel should carry a distinct signal (tone, sweep, noise burst)
    # so every Fireface output → speaker assignment can be confirmed by ear.
    "test_wav": "/Volumes/Bieznik T7/PhD/AmbiDOA_Real_Dataset/TEST_AmbiDOA_REC.wav",

    # Human-readable label per mix channel — populated from config.
    "speaker_layout": {},
}


# ──────────────────────────────── config ─────────────────────────────────────

def load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"Config not found: {path}\n"
                 f"Copy playback_record_config.yaml and fill in your paths.")
    with path.open(encoding="utf-8") as f:
        user = yaml.safe_load(f)
    cfg = dict(_DEFAULTS)
    cfg.update({k: v for k, v in user.items() if k in _DEFAULTS})
    unknown = [k for k in user if k not in _DEFAULTS]
    if unknown:
        print(f"[WARN] Unrecognised config keys (ignored): {unknown}")
    return cfg


# ──────────────────────────── device helpers ─────────────────────────────────

def resolve_device(index: Optional[int], key: str) -> int:
    """Validate a device index from config and return it.

    Exits with a helpful message if the index is None (unset) or out of range.
    """
    if index is None:
        sys.exit(
            f"'{key}' is not set in config.\n"
            f"Run --list-devices to see available devices and their indices,\n"
            f"then set '{key}' to the correct integer."
        )
    n_devices = len(sd.query_devices())
    if not (0 <= index < n_devices):
        sys.exit(
            f"'{key}' = {index} is out of range (0–{n_devices - 1}).\n"
            f"Run --list-devices to see available devices."
        )
    return index


# ──────────────────────────── manifest I/O ───────────────────────────────────

def load_mix_manifest(path: Path) -> List[dict]:
    if not path.exists():
        sys.exit(f"Mix manifest not found: {path}\n"
                 f"Run prepare_real_mixes.py first.")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_calibration_manifest(path: Path) -> Optional[dict]:
    """Load speaker_positions.json from run_measurement.py.

    Returns the inner 'speakers' dict:
        { "1": {"azimuth_deg": …, "elevation_deg": …}, … }
    Returns None if the file does not exist (recordings still proceed, but
    azimuth/elevation fields in ground truth will be null).
    """
    if not path.exists():
        print(f"[WARN] Calibration manifest not found: {path}")
        print("       Azimuth/elevation ground truth will be null.")
        print("       Run speaker-array-groundtruth/examples/run_measurement.py first.")
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    speakers = data.get("speakers", {})
    if not speakers:
        print(f"[WARN] 'speakers' key missing or empty in: {path}")
        return None
    return speakers


# ────────────────────────── ground truth assembly ────────────────────────────

def build_ground_truth(
    mix_entry: dict,
    calib_speakers: Optional[dict],
    channel_to_speaker_id: List[int],
    n_playback_channels: int,
) -> dict:
    """Merge mix-manifest channel metadata with calibration speaker positions.

    Returns:
        {
          "n_sources": int,
          "active_speakers": [ {mix_channel_index, speaker_id, azimuth_deg,
                                 elevation_deg, gain_db, source_file, is_base} ],
          "silent_channels": [int, …]
        }
    """
    active_speakers = []
    silent_channels = []

    for ch in range(n_playback_channels):
        ch_meta = mix_entry["channels"][ch]
        if ch_meta is None:
            silent_channels.append(ch)
            continue

        spk_id = channel_to_speaker_id[ch]
        spk_pos: Optional[dict] = None
        if calib_speakers is not None:
            spk_pos = (calib_speakers.get(str(spk_id))
                       or calib_speakers.get(spk_id))
        if spk_pos is None and calib_speakers is not None:
            print(f"  [WARN] No calibration position for speaker_id={spk_id} "
                  f"(mix channel {ch})")

        active_speakers.append({
            "mix_channel_index": ch,
            "speaker_id":        spk_id,
            "azimuth_deg":       spk_pos["azimuth_deg"]   if spk_pos else None,
            "elevation_deg":     spk_pos["elevation_deg"] if spk_pos else None,
            "gain_db":           ch_meta["gain_db"],
            "source_file":       ch_meta["source_file"],
            "is_base":           ch_meta["is_base"],
        })

    return {
        "n_sources":       mix_entry["n_sources"],
        "active_speakers": active_speakers,
        "silent_channels": silent_channels,
    }


def build_recording_manifest_entry(
    mix_entry: dict,
    calib_speakers: Optional[dict],
    channel_to_speaker_id: List[int],
    n_playback_channels: int,
    recording_filename: str,
    manifest_slots: int,
) -> dict:
    """Build an augument02-style manifest entry for one recording.

    Active sources are packed into the first n_sources slots in playback-channel
    order; silent channels are skipped. Remaining slots are padded (mask=0).
    """
    regions:    List[str] = ["" for _ in range(manifest_slots)]
    azimuths:   List[Optional[float]] = [None] * manifest_slots
    elevations: List[Optional[float]] = [None] * manifest_slots
    mask:       List[int] = [0] * manifest_slots

    slot = 0
    for ch in range(n_playback_channels):
        ch_meta = mix_entry["channels"][ch]
        if ch_meta is None:
            continue
        if slot >= manifest_slots:
            raise ValueError(
                f"mix_id={mix_entry.get('mix_id')} has more active sources "
                f"than manifest_slots ({manifest_slots})."
            )
        spk_id = channel_to_speaker_id[ch]
        spk_pos: Optional[dict] = None
        if calib_speakers is not None:
            spk_pos = (calib_speakers.get(str(spk_id))
                       or calib_speakers.get(spk_id))
        regions[slot]    = ch_meta["source_file"]
        azimuths[slot]   = spk_pos["azimuth_deg"]   if spk_pos else None
        elevations[slot] = spk_pos["elevation_deg"] if spk_pos else None
        mask[slot] = 1
        slot += 1

    return {
        "audio_filepath": recording_filename,
        "n_sources":      mix_entry["n_sources"],
        "regions":        regions,
        "azimuths":       azimuths,
        "elevations":     elevations,
        "mask":           mask,
    }


# ──────────────────────────────── audio I/O ──────────────────────────────────

def load_mix_wav(path: Path, expected_channels: int) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[1] != expected_channels:
        raise ValueError(
            f"{path.name}: expected {expected_channels} channels, got {data.shape[1]}"
        )
    return data, sr


def pad_with_silence(wav: np.ndarray, pre: int, post: int) -> np.ndarray:
    n_ch = wav.shape[1]
    return np.concatenate([
        np.zeros((pre,  n_ch), dtype=np.float32),
        wav,
        np.zeros((post, n_ch), dtype=np.float32),
    ], axis=0)


def record_mix(
    playback_data: np.ndarray,
    sample_rate: int,
    rec_device_idx: int,
    play_device_idx: int,
    input_mapping: List[int],
    output_mapping: List[int],
    latency,
) -> np.ndarray:
    """Play *playback_data* through the output device while recording from the
    input device.

    *input_mapping* and *output_mapping* are 1-indexed device channel lists.
    Returns recorded array of shape (n_frames, len(input_mapping)), float32.
    """
    return sd.playrec(
        playback_data,
        samplerate=sample_rate,
        dtype="float32",
        device=(rec_device_idx, play_device_idx),  # (input, output)
        input_mapping=input_mapping,
        output_mapping=output_mapping,
        latency=latency,
        blocking=True,
    )


# ───────────────────────────── metadata helpers ──────────────────────────────

def build_metadata(
    recording_id: str,
    mix_entry: dict,
    ground_truth: dict,
    calib_speakers: Optional[dict],
    calib_manifest_path: str,
    wav_info: dict,
    devices: dict,
) -> dict:
    return {
        "recording_id":   recording_id,
        "mix_id":         mix_entry["mix_id"],
        "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
        "files": {
            "recording_wav": f"{recording_id}.wav",
            "playback_wav":  mix_entry["audio_filepath"],
        },
        "audio":        wav_info,
        "devices":      devices,
        "ground_truth": ground_truth,
        "mix_manifest_entry": mix_entry,
        "calibration": {
            "source_file": calib_manifest_path,
            "speakers":    calib_speakers,
        },
    }


# ──────────────────────────── database helpers ───────────────────────────────

def load_or_init_database(path: Path, config: dict) -> dict:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return {
        "created_utc":      datetime.now(timezone.utc).isoformat(),
        "last_updated_utc": None,
        "config":           config,
        "recordings":       [],
    }


def save_database(db: dict, path: Path) -> None:
    db["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


def get_recorded_mix_ids(db: dict) -> set:
    return {r["mix_id"] for r in db["recordings"] if r.get("status") == "ok"}


def load_or_init_recording_manifest(path: Path) -> List[dict]:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return []


def save_recording_manifest(manifest: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_or_init_metadata_log(path: Path) -> List[dict]:
    """Aggregated full-metadata log: a cumulative list of per-take metadata."""
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return []


def save_metadata_log(metadata_log: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata_log, f, indent=2, ensure_ascii=False)


# ─────────────────────────── batch-log helpers ───────────────────────────────

def spherical_to_cartesian_unit(
    azimuth_deg: float, elevation_deg: float,
) -> Tuple[float, float, float]:
    """AmbiX unit vector from (az, el) in degrees.

    Convention (matches speaker-array-groundtruth):
      +x forward, +y left, +z up; az CCW from +x in xy-plane; el up from xy.
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return (
        float(np.cos(el) * np.cos(az)),
        float(np.cos(el) * np.sin(az)),
        float(np.sin(el)),
    )


def write_batch_log(
    log_path: Path,
    rec_prefix: str,
    batch_index: int,
    batch_size: int,
    mix_id_lo: int,
    mix_id_hi: int,
    mix_manifest_path: Path,
    recording_manifest_path: Path,
    calib_manifest_path: Path,
    cfg: dict,
    n_play_ch: int,
    out_ch_map: List[int],
    output_mapping: List[int],
    ch_to_spk_id: List[int],
    calib_speakers: Optional[dict],
    processed: List[dict],
) -> None:
    """Write a human-readable .txt log for one batch run.

    Carries everything that doesn't live in the compact recording manifest:
    speaker routing, speaker numbers, positions (spherical + cartesian unit
    vector), manifest filenames, and the per-mix outcome of this run.
    """
    speaker_layout = cfg.get("speaker_layout", {}) or {}
    timestamp = datetime.now(timezone.utc).isoformat()

    lines: List[str] = []
    lines.append(f"{rec_prefix} recording log")
    lines.append("=" * 60)
    lines.append(f"Timestamp UTC          : {timestamp}")
    # Prefix identifies the rig pose and names every WAV; batch index only picks
    # the mix slice. Both are reported because they are free to differ.
    lines.append(f"Recording prefix       : {rec_prefix}")
    lines.append(f"Batch index (mix slice): {batch_index}")
    lines.append(f"Batch size             : {batch_size}")
    lines.append(f"Mix ID range           : [{mix_id_lo}, {mix_id_hi})")
    lines.append(f"Input mix manifest     : {mix_manifest_path}")
    lines.append(f"Output recording manif.: {recording_manifest_path}")
    lines.append(f"Calibration manifest   : {calib_manifest_path}")
    lines.append("")

    # ---- speaker routing -----------------------------------------------------
    lines.append("Speaker routing")
    lines.append("-" * 60)
    lines.append(
        f"{'mix_ch':>6}  {'ff_out_0idx':>11}  {'ff_out_1idx':>11}  "
        f"{'speaker_id':>10}  label"
    )
    for ch in range(n_play_ch):
        label = speaker_layout.get(f"mix_ch_{ch}", "")
        lines.append(
            f"{ch:>6d}  {out_ch_map[ch]:>11d}  {output_mapping[ch]:>11d}  "
            f"{ch_to_spk_id[ch]:>10d}  {label}"
        )
    lines.append("")

    # ---- speaker positions ---------------------------------------------------
    lines.append("Speaker positions")
    lines.append("-" * 60)
    lines.append(
        f"{'spk_id':>6}  {'az_deg':>9}  {'el_deg':>9}  "
        f"{'x':>8}  {'y':>8}  {'z':>8}"
    )
    if calib_speakers is None:
        lines.append("(calibration manifest unavailable — positions unknown)")
    else:
        for spk_id in ch_to_spk_id:
            pos = (calib_speakers.get(str(spk_id))
                   or calib_speakers.get(spk_id))
            if pos is None:
                lines.append(f"{spk_id:>6d}  (no calibration entry)")
                continue
            az = pos["azimuth_deg"]
            el = pos["elevation_deg"]
            x, y, z = spherical_to_cartesian_unit(az, el)
            lines.append(
                f"{spk_id:>6d}  {az:>9.3f}  {el:>9.3f}  "
                f"{x:>8.4f}  {y:>8.4f}  {z:>8.4f}"
            )
    lines.append("")

    # ---- per-mix outcomes ----------------------------------------------------
    lines.append("Mixes processed in this run")
    lines.append("-" * 60)
    lines.append(
        f"{'mix_id':>6}  {'recording_wav':<28}  "
        f"{'n_src':>5}  active_channels  status"
    )
    if not processed:
        lines.append("(none)")
    else:
        for r in processed:
            lines.append(
                f"{r['mix_id']:>6d}  {r['recording_wav']:<28}  "
                f"{r['n_sources']:>5d}  {r['active_channels']}  {r['status']}"
            )
    lines.append("")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ────────────────────────────── test routing ─────────────────────────────────

def run_test_routing(
    cfg: dict,
    recordings_dir: Path,
    sr: int,
    n_play_ch: int,
    n_rec_ch: int,
    output_mapping: List[int],
    input_mapping: List[int],
    play_dev_idx: int,
    rec_dev_idx: int,
    latency,
    bit_depth: str,
    devices_meta: dict,
) -> None:
    """Play test_wav through the Fireface and record the Zylia.

    Saved as test_routing_<timestamp>.wav + .json in recordings_dir.
    Does not touch the recordings database.

    The speaker layout baked into the metadata mirrors the output_channel_map
    so you can cross-reference the recording against the expected routing.
    """
    test_wav_path = Path(cfg["test_wav"])
    if not test_wav_path.exists():
        sys.exit(
            f"test_wav not found: {test_wav_path}\n"
            f"Update 'test_wav' in your config to point to TEST.wav."
        )

    print(f"\nTest-routing mode")
    print(f"  Source   : {test_wav_path}")

    # Load — accept any channel count; warn if it differs from n_play_ch
    wav, file_sr = sf.read(str(test_wav_path), dtype="float32", always_2d=True)
    n_ch_actual = wav.shape[1]
    if n_ch_actual != n_play_ch:
        print(
            f"  [WARN] test_wav has {n_ch_actual} channels but n_playback_channels={n_play_ch}.\n"
            f"         Truncating/padding to {n_play_ch} channels."
        )
        if n_ch_actual < n_play_ch:
            pad = np.zeros((wav.shape[0], n_play_ch - n_ch_actual), dtype=np.float32)
            wav = np.concatenate([wav, pad], axis=1)
        else:
            wav = wav[:, :n_play_ch]

    sr_actual = file_sr
    if file_sr != sr:
        print(f"  [WARN] test_wav SR={file_sr} Hz differs from config SR={sr} Hz. "
              f"Using file SR.")

    duration = wav.shape[0] / sr_actual
    print(f"  Duration : {duration:.2f}s  ({wav.shape[0]} frames × {n_play_ch} ch)")
    print(f"  Output ch: {output_mapping}  (1-indexed Fireface channels)")
    print(f"  Input  ch: {input_mapping[0]}–{input_mapping[-1]}  (Zylia channels)")

    recordings_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    stem = f"test_routing_{timestamp}"

    print(f"\nPlaying + recording…")
    t0 = time.perf_counter()
    recorded = record_mix(
        playback_data   = wav,
        sample_rate     = sr_actual,
        rec_device_idx  = rec_dev_idx,
        play_device_idx = play_dev_idx,
        input_mapping   = input_mapping,
        output_mapping  = output_mapping,
        latency         = latency,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Recorded {recorded.shape[0]} frames × {recorded.shape[1]} ch "
          f"in {elapsed:.2f}s")

    # Save WAV
    rec_wav_path = recordings_dir / f"{stem}.wav"
    sf.write(str(rec_wav_path), recorded, sr_actual, subtype=bit_depth)

    # Build speaker layout annotation from config
    out_ch_map       = list(cfg["output_channel_map"])
    ch_to_spk_id     = list(cfg["channel_to_speaker_id"])
    speaker_layout   = cfg.get("speaker_layout", {})
    channel_details  = []
    for ch in range(n_play_ch):
        channel_details.append({
            "mix_channel_index":    ch,
            "fireface_output_ch":   out_ch_map[ch],      # 0-indexed
            "fireface_output_1idx": output_mapping[ch],  # 1-indexed (used by sounddevice)
            "speaker_id":           ch_to_spk_id[ch],
            "label":                speaker_layout.get(f"mix_ch_{ch}", ""),
        })

    metadata = {
        "test_routing_id": stem,
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
        "test_wav":        str(test_wav_path),
        "files": {
            "recording_wav": f"{stem}.wav",
            "metadata_json": f"{stem}.json",
        },
        "audio": {
            "sample_rate":             sr_actual,
            "n_recording_channels":    n_rec_ch,
            "n_playback_channels":     n_play_ch,
            "total_recording_samples": recorded.shape[0],
            "duration_sec":            round(recorded.shape[0] / sr_actual, 6),
        },
        "devices":          devices_meta,
        "channel_routing":  channel_details,
    }
    meta_path = recordings_dir / f"{stem}.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nSaved:")
    print(f"  {rec_wav_path}")
    print(f"  {meta_path}")
    print(f"\nRouting used:")
    for ch in channel_details:
        print(f"  mix ch {ch['mix_channel_index']:2d}  →  "
              f"Fireface out {ch['fireface_output_1idx']:2d}  "
              f"(speaker {ch['speaker_id']:2d})  {ch['label']}")


# ──────────────────────────── time estimation ────────────────────────────────

def fmt_duration(sec: float) -> str:
    """Human-readable duration: '1h04m09s' | '4m09s' | '9s'."""
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def estimate_mix_seconds(
    to_process: List[dict],
    mixes_dir: Path,
    pre_silence_sec: float,
    post_silence_sec: float,
) -> List[float]:
    """Per-mix wall-clock seconds spent playing+recording each mix.

    Each mix's playback/record time is its own duration (read cheaply from the
    WAV header, not the samples) plus the pre/post silence pads. The inter-mix
    gap is added separately by the caller since it falls *between* mixes. A file
    whose header can't be read contributes only the silence pads (its duration
    is unknown until load time).
    """
    per_mix: List[float] = []
    for e in to_process:
        wav_path = mixes_dir / e["audio_filepath"]
        try:
            info = sf.info(str(wav_path))
            mix_sec = info.frames / info.samplerate
        except Exception:
            mix_sec = 0.0
        per_mix.append(pre_silence_sec + mix_sec + post_silence_sec)
    return per_mix


# ──────────────────────────────── main ───────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play multichannel mixes (Fireface 800) and record (Zylia ZM-1E).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).with_name("playback_record_config.yaml"),
        metavar="PATH",
        help="Config JSON (default: playback_record_config.json next to this script).",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="Print all available audio devices with channel counts and exit.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and manifests without performing any audio I/O.",
    )
    parser.add_argument(
        "--start-from", type=int, default=0, metavar="IDX",
        help="Start from this 0-based manifest index. Useful for resuming.",
    )
    parser.add_argument(
        "--mix-ids", type=int, nargs="+", metavar="ID",
        help="Record only these specific mix IDs (overrides --start-from).",
    )
    parser.add_argument(
        "--batch", type=int, default=None, metavar="IDX",
        help=(
            "Record only batch IDX (0-based): mix_ids in "
            "[IDX*batch_size, IDX*batch_size + batch_size). Each run writes its "
            "own timestamped database/manifest/metadata JSONs and a "
            "batch_<IDX>_<timestamp>_log.txt into recordings_dir, so a batch's "
            "metadata is never mixed with another's. Overrides --start-from; "
            "ignored if --mix-ids is given."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help=(
            "Allow recording over existing <recording_prefix>_<mix_id>.wav "
            "files. Without this the run aborts before touching anything, so a "
            "stale recording_prefix cannot silently destroy a previous batch."
        ),
    )
    parser.add_argument(
        "--startup-delay", type=float, default=None, metavar="SEC",
        help=(
            "Seconds to wait before the first recording, to leave the "
            "measurement area (default from config, 15s). 0 disables."
        ),
    )
    parser.add_argument(
        "--test-routing", action="store_true",
        help=(
            "Play test_wav through the Fireface and record the Zylia to verify "
            "speaker routing. Saved as test_routing_<timestamp>.wav/.json in "
            "recordings_dir. Does not affect the recordings database."
        ),
    )
    args = parser.parse_args()

    # ── list-devices shortcut ───────────────────────────────────────────────
    if args.list_devices:
        print(sd.query_devices())
        return

    # ── load config ─────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    print(f"Config loaded: {args.config}")

    mixes_dir         = Path(cfg["mixes_dir"])
    mix_manifest_path = Path(cfg["mix_manifest"])
    calib_path        = Path(cfg["calibration_manifest"])
    recordings_dir    = Path(cfg["recordings_dir"])
    database_path     = Path(cfg["database_file"])
    recording_manifest_path = Path(cfg["recording_manifest_file"])
    metadata_path           = Path(cfg["metadata_file"])
    manifest_slots          = int(cfg["manifest_slots"])
    batch_size              = int(cfg["batch_size"])
    batch_index_cfg         = cfg.get("batch_index")
    batch_index = (args.batch if args.batch is not None
                   else (int(batch_index_cfg) if batch_index_cfg is not None else None))
    startup_delay_sec = (args.startup_delay if args.startup_delay is not None
                         else float(cfg.get("startup_delay_sec", 0) or 0))
    rec_prefix       = cfg["recording_prefix"]

    # ── per-run, timestamped output files ────────────────────────────────────
    # In batch mode, each RUN writes its own database/manifest/metadata (and
    # batch log) tagged with the RECORDING PREFIX and a run timestamp, e.g.
    #   recordings_database_Batch_6_20260808T165736.json
    # so a run's metadata is never mixed with another's, and re-running produces
    # a fresh, clearly-named set instead of appending to a shared cumulative
    # file. Because the run starts from an empty (new) database, the whole batch
    # is (re)recorded — for a crash mid-batch, resume with --start-from IDX.
    # Non-batch runs keep the original shared filenames.
    #
    # The tag is the PREFIX, not batch_index, because those two mean different
    # things: recording_prefix identifies the rig pose (and already names every
    # WAV), while batch_index only selects which mix slice to record. They are
    # free to differ — pose 6 recording mix slice 5, say — and tagging by index
    # then produces two different runs claiming the same name, silently
    # overwriting or interleaving each other's metadata. Prefix-tagged run files
    # always pair 1:1 with the WAVs they describe.
    run_stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    batch_log_path: Optional[Path] = None
    if batch_index is not None:
        tag = f"{rec_prefix}_{run_stamp}"

        def _stamped(p: Path) -> Path:
            return p.with_name(f"{p.stem}_{tag}{p.suffix}")

        database_path           = _stamped(database_path)
        recording_manifest_path = _stamped(recording_manifest_path)
        metadata_path           = _stamped(metadata_path)
        batch_log_path          = recordings_dir / f"{tag}_log.txt"
        print(f"Batch mode: per-run output files tagged '{tag}'  "
              f"(prefix '{rec_prefix}', mix slice {batch_index})")
        print(f"  database : {database_path.name}")
        print(f"  manifest : {recording_manifest_path.name}")
        print(f"  metadata : {metadata_path.name}")
        print(f"  log      : {batch_log_path.name}")

    sr               = int(cfg["sample_rate"])
    n_play_ch        = int(cfg["n_playback_channels"])
    n_rec_ch         = int(cfg["n_recording_channels"])
    out_ch_map       = list(cfg["output_channel_map"])    # 0-indexed
    ch_to_spk_id     = list(cfg["channel_to_speaker_id"])
    pre_sil          = int(round(float(cfg["pre_silence_sec"])  * sr))
    post_sil         = int(round(float(cfg["post_silence_sec"]) * sr))
    inter_gap        = float(cfg["inter_mix_gap_sec"])
    latency          = cfg["latency"]
    bit_depth        = cfg["bit_depth"]

    # sounddevice uses 1-indexed channel lists
    output_mapping = [c + 1 for c in out_ch_map]
    input_mapping  = list(range(1, n_rec_ch + 1))

    # ── validate config ──────────────────────────────────────────────────────
    if len(out_ch_map) != n_play_ch:
        sys.exit(f"output_channel_map length ({len(out_ch_map)}) "
                 f"!= n_playback_channels ({n_play_ch})")
    if len(ch_to_spk_id) != n_play_ch:
        sys.exit(f"channel_to_speaker_id length ({len(ch_to_spk_id)}) "
                 f"!= n_playback_channels ({n_play_ch})")

    # ── load manifests (skipped for --test-routing) ──────────────────────────
    if not args.test_routing:
        print(f"\nLoading mix manifest : {mix_manifest_path}")
        mix_manifest = load_mix_manifest(mix_manifest_path)
        print(f"  {len(mix_manifest)} mixes.")

        print(f"Loading calibration  : {calib_path}")
        calib_speakers = load_calibration_manifest(calib_path)
        if calib_speakers:
            print(f"  {len(calib_speakers)} speakers: {sorted(calib_speakers.keys())}")

    # ── find audio devices ───────────────────────────────────────────────────
    if args.test_routing and args.dry_run:
        sys.exit("--test-routing performs real audio I/O and cannot be combined with --dry-run.")

    if not args.dry_run:
        print("\nResolving audio devices...")
        play_dev_idx = resolve_device(cfg["playback_device_index"],  "playback_device_index")
        rec_dev_idx  = resolve_device(cfg["recording_device_index"], "recording_device_index")

        play_info = sd.query_devices(play_dev_idx)
        rec_info  = sd.query_devices(rec_dev_idx)
        print(f"  Playback : [{play_dev_idx}] {play_info['name']}  "
              f"(max out: {play_info['max_output_channels']} ch)")
        print(f"  Recording: [{rec_dev_idx}]  {rec_info['name']}  "
              f"(max in:  {rec_info['max_input_channels']} ch)")

        if play_info["max_output_channels"] < max(out_ch_map) + 1:
            sys.exit(f"Playback device has only {play_info['max_output_channels']} output "
                     f"channels but output_channel_map requests channel "
                     f"{max(out_ch_map) + 1} (0-indexed).")
        if rec_info["max_input_channels"] < n_rec_ch:
            sys.exit(f"Recording device has only {rec_info['max_input_channels']} input "
                     f"channels but n_recording_channels={n_rec_ch}.")

        devices_meta = {
            "playback":  {"name": play_info["name"], "device_index": play_dev_idx},
            "recording": {"name": rec_info["name"],  "device_index": rec_dev_idx},
        }
    else:
        play_dev_idx = rec_dev_idx = -1
        devices_meta = {
            "playback":  {"name": f"index {cfg['playback_device_index']}",  "device_index": cfg["playback_device_index"]},
            "recording": {"name": f"index {cfg['recording_device_index']}", "device_index": cfg["recording_device_index"]},
        }

    # ── test-routing shortcut ────────────────────────────────────────────────
    if args.test_routing:
        run_test_routing(
            cfg           = cfg,
            recordings_dir= recordings_dir,
            sr            = sr,
            n_play_ch     = n_play_ch,
            n_rec_ch      = n_rec_ch,
            output_mapping= output_mapping,
            input_mapping = input_mapping,
            play_dev_idx  = play_dev_idx,
            rec_dev_idx   = rec_dev_idx,
            latency       = latency,
            bit_depth     = bit_depth,
            devices_meta  = devices_meta,
        )
        return

    # ── prepare output dir and database ─────────────────────────────────────
    recordings_dir.mkdir(parents=True, exist_ok=True)
    db = load_or_init_database(database_path, cfg)
    already_done = get_recorded_mix_ids(db)
    if already_done:
        print(f"\n{len(already_done)} mixes already in database — will skip.")

    recording_manifest = load_or_init_recording_manifest(recording_manifest_path)
    metadata_log = load_or_init_metadata_log(metadata_path)
    if recording_manifest:
        print(f"  Existing recording manifest has {len(recording_manifest)} entries.")

    # ── filter mix list ──────────────────────────────────────────────────────
    batch_lo = batch_hi = None
    if args.mix_ids:
        target_ids = set(args.mix_ids)
        to_process = [e for e in mix_manifest if e["mix_id"] in target_ids]
        print(f"\nTargeting {len(to_process)} specified mix IDs.")
    elif batch_index is not None:
        if batch_index < 0:
            sys.exit(f"--batch must be >= 0 (got {batch_index}).")
        batch_lo = batch_index * batch_size
        batch_hi = batch_lo + batch_size
        to_process = [e for e in mix_manifest if batch_lo <= e["mix_id"] < batch_hi]
        print(f"\nBatch {batch_index}: mix_ids [{batch_lo}, {batch_hi})  "
              f"→ {len(to_process)} mixes.")
    else:
        to_process = mix_manifest[args.start_from:]
        print(f"\nProcessing {len(to_process)} mixes from manifest index {args.start_from}.")

    to_process = [e for e in to_process if e["mix_id"] not in already_done]
    print(f"  → {len(to_process)} mixes to record after skipping already-done.")

    if not to_process:
        print("Nothing to do.")
        return

    # ── overwrite guard ──────────────────────────────────────────────────────
    # Recording filenames are <recording_prefix>_<mix_id>.wav, and the prefix is
    # a free-form config string that is NOT derived from batch_index. A run
    # started with a stale prefix (or a re-run of the same batch) will therefore
    # write straight over a previous batch's audio, which is unrecoverable —
    # the per-run database/manifest/log are timestamped, but the WAVs are not.
    # Refuse to start if any destination already exists unless --overwrite.
    collisions = [
        (e["mix_id"], recordings_dir / f"{rec_prefix}_{e['mix_id']:05d}.wav")
        for e in to_process
        if (recordings_dir / f"{rec_prefix}_{e['mix_id']:05d}.wav").exists()
    ]
    if collisions and not args.overwrite:
        print(f"\nERROR: {len(collisions)} of {len(to_process)} recordings would "
              f"overwrite existing files in {recordings_dir}:")
        for mix_id, path in collisions[:10]:
            print(f"  ✗ mix_id {mix_id:>5}  {path.name}")
        if len(collisions) > 10:
            print(f"  … and {len(collisions) - 10} more")
        sys.exit(
            f"\nRefusing to start — recording would destroy this audio.\n"
            f"  recording_prefix = {rec_prefix!r}"
            + (f", batch_index = {batch_index}" if batch_index is not None else "")
            + "\n"
            f"Check that recording_prefix / batch_index / calibration_manifest in "
            f"the config all refer to the batch you mean to record.\n"
            f"To record anyway, re-run with --overwrite."
        )
    if collisions:
        print(f"\n[--overwrite] {len(collisions)} existing recording(s) will be "
              f"replaced.")

    # ── batch time estimate ──────────────────────────────────────────────────
    # Schedule-based: each mix's own duration + pre/post silence pads, plus one
    # inter-mix gap between consecutive mixes. Startup delay is added on top.
    pre_sil_sec  = float(cfg["pre_silence_sec"])
    post_sil_sec = float(cfg["post_silence_sec"])
    mix_seconds  = estimate_mix_seconds(to_process, mixes_dir, pre_sil_sec, post_sil_sec)
    n_gaps       = max(0, len(to_process) - 1)
    total_est_sec = sum(mix_seconds) + inter_gap * n_gaps + startup_delay_sec
    finish_at = datetime.now() + timedelta(seconds=total_est_sec)
    print(
        f"\nEstimated batch time: ~{fmt_duration(total_est_sec)}  "
        f"→ finish ~{finish_at.strftime('%H:%M:%S')}\n"
        f"  {len(to_process)} mixes  +  {pre_sil_sec:g}s/{post_sil_sec:g}s silence pads  "
        f"+  {n_gaps}×{inter_gap:g}s gaps  +  {startup_delay_sec:g}s startup"
    )

    if args.dry_run:
        print(f"\n[DRY RUN] Config and manifests OK.  "
              f"Would record {len(to_process)} mixes → {recordings_dir}")
        return

    # ── startup delay ────────────────────────────────────────────────────────
    # Give the operator time to leave the measurement area before playback so
    # they are not captured in the recordings. Skipped for --dry-run (above).
    if startup_delay_sec and startup_delay_sec > 0:
        print(f"\nStarting in {startup_delay_sec:.0f}s — leave the measurement "
              f"area now (Ctrl-C to abort)…")
        remaining = int(round(startup_delay_sec))
        for remaining in range(remaining, 0, -1):
            print(f"  {remaining:>3d}s", end="\r", flush=True)
            time.sleep(1)
        print("  0s — recording now.        ")

    # ── main recording loop ──────────────────────────────────────────────────
    total = len(to_process)
    errors = 0
    processed_log: List[dict] = []  # per-mix outcomes for the batch log

    for loop_idx, mix_entry in enumerate(to_process):
        mix_id       = mix_entry["mix_id"]
        recording_id = f"{rec_prefix}_{mix_id:05d}"
        wav_path     = mixes_dir / mix_entry["audio_filepath"]

        print(f"\n[{loop_idx + 1:04d}/{total}] mix_id={mix_id}  "
              f"file={mix_entry['audio_filepath']}  "
              f"n_sources={mix_entry['n_sources']}  "
              f"active_ch={mix_entry['active_channels']}")

        # remaining batch time = this mix + all later mixes + the gaps between
        # them (schedule-based, from the header durations computed up front)
        remaining_sec = sum(mix_seconds[loop_idx:]) + inter_gap * (total - loop_idx - 1)
        eta_finish = datetime.now() + timedelta(seconds=remaining_sec)
        print(f"  Batch ETA: ~{fmt_duration(remaining_sec)} left  "
              f"→ finish ~{eta_finish.strftime('%H:%M:%S')}")

        # load WAV
        try:
            wav, file_sr = load_mix_wav(wav_path, n_play_ch)
        except Exception as exc:
            print(f"  [ERROR] Could not load WAV: {exc}")
            db["recordings"].append({
                "recording_id": recording_id,
                "mix_id":       mix_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "error":  str(exc),
            })
            save_database(db, database_path)
            processed_log.append({
                "mix_id":          mix_id,
                "recording_wav":   f"{recording_id}.wav",
                "n_sources":       mix_entry["n_sources"],
                "active_channels": mix_entry["active_channels"],
                "status":          f"error: load_wav ({exc})",
            })
            errors += 1
            continue

        sr_actual = file_sr
        if file_sr != sr:
            print(f"  [WARN] WAV SR={file_sr} Hz differs from config SR={sr} Hz. "
                  f"Using file SR.")

        padded      = pad_with_silence(wav, pre_sil, post_sil)
        mix_samples = wav.shape[0]
        total_samp  = padded.shape[0]

        print(f"  Duration: {mix_samples / sr_actual:.2f}s mix + "
              f"{pre_sil / sr_actual:.2f}s pre + {post_sil / sr_actual:.2f}s post "
              f"= {total_samp / sr_actual:.2f}s total")

        # ground truth (does not depend on audio I/O)
        gt = build_ground_truth(mix_entry, calib_speakers, ch_to_spk_id, n_play_ch)

        # play + record
        try:
            t0 = time.perf_counter()
            recorded = record_mix(
                playback_data  = padded,
                sample_rate    = sr_actual,
                rec_device_idx = rec_dev_idx,
                play_device_idx= play_dev_idx,
                input_mapping  = input_mapping,
                output_mapping = output_mapping,
                latency        = latency,
            )
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            print(f"  [ERROR] Audio I/O failed: {exc}")
            db["recordings"].append({
                "recording_id": recording_id,
                "mix_id":       mix_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "error":  str(exc),
            })
            save_database(db, database_path)
            processed_log.append({
                "mix_id":          mix_id,
                "recording_wav":   f"{recording_id}.wav",
                "n_sources":       mix_entry["n_sources"],
                "active_channels": mix_entry["active_channels"],
                "status":          f"error: playrec ({exc})",
            })
            errors += 1
            continue

        print(f"  Recorded {recorded.shape[0]} frames × {recorded.shape[1]} ch "
              f"in {elapsed:.2f}s")

        # save recording WAV
        rec_wav_path = recordings_dir / f"{recording_id}.wav"
        sf.write(str(rec_wav_path), recorded, sr_actual, subtype=bit_depth)

        # save per-recording metadata JSON
        wav_info = {
            "sample_rate":             sr_actual,
            "n_recording_channels":    n_rec_ch,
            "n_playback_channels":     n_play_ch,
            "pre_silence_samples":     pre_sil,
            "post_silence_samples":    post_sil,
            "mix_samples":             mix_samples,
            "total_recording_samples": total_samp,
            "duration_sec":            round(total_samp / sr_actual, 6),
        }
        metadata = build_metadata(
            recording_id       = recording_id,
            mix_entry          = mix_entry,
            ground_truth       = gt,
            calib_speakers     = calib_speakers,
            calib_manifest_path= str(calib_path),
            wav_info           = wav_info,
            devices            = devices_meta,
        )
        # append full metadata to the aggregated log (one file for all takes,
        # written after every take for crash safety — no per-mix JSON sidecars)
        metadata_log.append(metadata)
        save_metadata_log(metadata_log, metadata_path)

        # update database (written after every take for crash safety)
        db["recordings"].append({
            "recording_id":   recording_id,
            "mix_id":         mix_id,
            "timestamp_utc":  metadata["timestamp_utc"],
            "recording_wav":  f"{recording_id}.wav",
            "n_sources":      mix_entry["n_sources"],
            "active_channels":mix_entry["active_channels"],
            "status":         "ok",
        })
        save_database(db, database_path)

        # update augument02-style recording manifest (one entry per take)
        recording_manifest.append(build_recording_manifest_entry(
            mix_entry             = mix_entry,
            calib_speakers        = calib_speakers,
            channel_to_speaker_id = ch_to_spk_id,
            n_playback_channels   = n_play_ch,
            recording_filename    = f"{recording_id}.wav",
            manifest_slots        = manifest_slots,
        ))
        save_recording_manifest(recording_manifest, recording_manifest_path)

        processed_log.append({
            "mix_id":          mix_id,
            "recording_wav":   f"{recording_id}.wav",
            "n_sources":       mix_entry["n_sources"],
            "active_channels": mix_entry["active_channels"],
            "status":          "ok",
        })

        print(f"  Saved {rec_wav_path.name}  (metadata → {metadata_path.name})")

        # inter-mix gap
        if loop_idx < total - 1 and inter_gap > 0:
            print(f"  Pausing {inter_gap:.1f}s for room decay…")
            time.sleep(inter_gap)

    # ── summary ─────────────────────────────────────────────────────────────
    ok = total - errors
    print(f"\n{'─' * 50}")
    print(f"Done.  {ok}/{total} mixes recorded successfully.")
    if errors:
        print(f"  {errors} error(s) — see database for details.")
    print(f"  Recordings : {recordings_dir}")
    print(f"  Database   : {database_path}")
    print(f"  Manifest   : {recording_manifest_path}")
    print(f"  Metadata   : {metadata_path}")

    # ── batch log ────────────────────────────────────────────────────────────
    if batch_index is not None:
        log_path = batch_log_path
        write_batch_log(
            log_path                = log_path,
            rec_prefix              = rec_prefix,
            batch_index             = batch_index,
            batch_size              = batch_size,
            mix_id_lo               = batch_lo,
            mix_id_hi               = batch_hi,
            mix_manifest_path       = mix_manifest_path,
            recording_manifest_path = recording_manifest_path,
            calib_manifest_path     = calib_path,
            cfg                     = cfg,
            n_play_ch               = n_play_ch,
            out_ch_map              = out_ch_map,
            output_mapping          = output_mapping,
            ch_to_spk_id            = ch_to_spk_id,
            calib_speakers          = calib_speakers,
            processed               = processed_log,
        )
        print(f"  Batch log  : {log_path}")


if __name__ == "__main__":
    main()
