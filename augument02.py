#!/usr/bin/env python3
"""audio_augmentation_mixer.py

Augments a multi-channel audio dataset according to mixing rules specified by
Professor Małecki. For every *base* entry in the input manifest, the script can
create *k* new mixtures, each of which always contains the base entry plus a
random number (1–10 total sources; i.e., 0–9 additional entries). Additional
entries must be at least five indices away from the base entry **and** from
each other, and pass angular separation checks. The resulting mixtures:
    • are extended to the **longest** constituent length by *looping* shorter signals,
    • are peak-normalised to 0.99 (float32),
    • are written to <out_dir>/mix_<base>_<rep>.wav,
    • are described in a new manifest that mirrors the input structure.

Set *k* in the “MANUAL CONFIGURATION” section.
"""
from __future__ import annotations

import json, glob
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

# ----------------------------- utility functions ----------------------------- #

POLE_EPS = 1e-6

def load_manifest(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(data: List[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def first_active_region(entry: dict) -> Tuple[str, float | None, float | None]:
    # Works with any list length (e.g., 10 or 12 slots) assuming lists are aligned
    for idx, active in enumerate(entry["mask"]):
        if active:
            return (
                entry["regions"][idx],
                entry["azimuths"][idx],
                entry["elevations"][idx],
            )
    return "", None, None  # should never happen if data are valid


def first_active_azel(entry: dict) -> Tuple[float | None, float | None]:
    """Return azimuth/elevation of the first active source in entry."""
    for idx, active in enumerate(entry["mask"]):
        if active:
            az = entry["azimuths"][idx]
            el = entry["elevations"][idx]
            if az is None or el is None:
                return None, None
            return float(az), float(el)
    return None, None


def circular_azimuth_diff_deg(a: float, b: float) -> float:
    """Shortest azimuth distance on a circle (0..180)."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def same_pole_point(el1: float, el2: float) -> bool:
    """At +/-90 deg elevation azimuth is undefined: all az map to one pole."""
    at_north_1 = abs(el1 - 90.0) <= POLE_EPS
    at_north_2 = abs(el2 - 90.0) <= POLE_EPS
    at_south_1 = abs(el1 + 90.0) <= POLE_EPS
    at_south_2 = abs(el2 + 90.0) <= POLE_EPS
    return (at_north_1 and at_north_2) or (at_south_1 and at_south_2)


def violates_angular_separation(
    cand_az: float,
    cand_el: float,
    existing: List[Tuple[float, float]],
    min_angle_sep_deg: float,
) -> bool:
    """Reject if candidate is too close to any existing source direction.

    Rule requested by user:
      keep candidate only when it is farther than threshold in azimuth OR elevation.
      Equivalent reject condition: (d_az <= thr) AND (d_el <= thr).

    Extra physical guard:
      reject same-pole points (el=+/-90) even if azimuth differs.
    """
    for az, el in existing:
        if same_pole_point(cand_el, el):
            return True
        d_az = circular_azimuth_diff_deg(cand_az, az)
        d_el = abs(cand_el - el)
        if d_az <= min_angle_sep_deg and d_el <= min_angle_sep_deg:
            return True
    return False

def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2) + 1e-12))

def scale_to_target_rms(x: np.ndarray, target_rms: float) -> np.ndarray:
    cur = rms(x)
    if cur == 0.0:
        return x
    return x * (target_rms / cur)

def read_random_wav(path_dir: Path) -> Tuple[np.ndarray, int]:
    """Wczytuje losowy plik WAV z katalogu. Zwraca [frames, channels], sr."""
    paths = sorted([p for p in (path_dir).glob("**/*.wav")])
    if not paths:
        raise FileNotFoundError(f"Brak plików WAV w katalogu: {path_dir}")
    p = random.choice(paths)
    data, sr = sf.read(p, dtype="float32", always_2d=True)
    return data, sr, p

def _ensure_16ch_from_mono_or_16(x: np.ndarray) -> np.ndarray:
    """Akceptuje 1-kanał lub 16-kanałów. Mono replikujemy do 16. Inaczej błąd."""
    if x.shape[1] == 16:
        return x
    if x.shape[1] == 1:
        return np.repeat(x, 16, axis=1)
    raise ValueError(f"Plik ma {x.shape[1]} kanałów; dozwolone: 1 albo 16.")

def _ensure_mono(x: np.ndarray) -> np.ndarray:
    """Redukuje do mono, jeżeli wielokanałowe (średnia kanałów)."""
    if x.shape[1] == 1:
        return x
    return np.mean(x, axis=1, keepdims=True)

# ----------------------------- mixing functions ------------------------------ #

def select_extra_indices(
    base_idx: int,
    total: int,
    num_extra: int,
    manifest: List[dict],
    min_distance: int = 5,
    min_angle_sep_deg: float = 4.0,
    rng: random.Random | None = None,
) -> List[int]:
    """Select *num_extra* indices, satisfying distance constraints."""
    rng = rng or random
    candidates = [j for j in range(total) if abs(j - base_idx) > min_distance]
    rng.shuffle(candidates)

    base_az, base_el = first_active_azel(manifest[base_idx])
    existing_dirs: List[Tuple[float, float]] = []
    if base_az is not None and base_el is not None:
        existing_dirs.append((base_az, base_el))

    selected: List[int] = []
    for idx in candidates:
        if len(selected) == num_extra:
            break
        if not all(abs(idx - s) > min_distance for s in selected):
            continue

        cand_az, cand_el = first_active_azel(manifest[idx])
        if cand_az is None or cand_el is None:
            continue

        if violates_angular_separation(cand_az, cand_el, existing_dirs, min_angle_sep_deg):
            continue

        selected.append(idx)
        existing_dirs.append((cand_az, cand_el))

    if len(selected) < num_extra:
        raise RuntimeError(
            f"Cannot select {num_extra} extra indices for base {base_idx}. "
            "Dataset might be too small for the given index/angular constraints."
        )
    return selected


def read_audio(audio_path: Path) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    return data, sr


def _loop_to_length(sig: np.ndarray, target_len: int) -> np.ndarray:
    """Loop (tile) a [frames, channels] signal to reach *target_len* frames."""
    frames, ch = sig.shape
    if frames == target_len:
        return sig
    reps = int(np.ceil(target_len / frames))
    tiled = np.tile(sig, (reps, 1))
    return tiled[:target_len, :]


def mix_audio_signals(signals: List[np.ndarray]) -> np.ndarray:
    """Sum-mix signals po wcześniejszym zapętlaniu do najdłuższego.
    Każde źródło skaluje losowy gain ∈ [-6, +6] dB (wokół poziomów plików).
    Na końcu: peak clamp do 0.99 (jeśli trzeba).
    """
    max_len = max(s.shape[0] for s in signals)
    stretched = [_loop_to_length(s, max_len) for s in signals]

    # Losowe wzmocnienia per-źródło
    gains_db = np.random.uniform(-10.0, 10.0, size=len(stretched))
    gains = np.array([db_to_lin(g) for g in gains_db], dtype=np.float32)

    # Zastosuj gainy
    stretched = [s * g for s, g in zip(stretched, gains)]

    mix = np.sum(stretched, axis=0)

    max_abs = np.max(np.abs(mix))
    if max_abs > 0.99:
        mix *= 0.99 / max_abs
    return mix


# ----------------------------- manifest helpers ----------------------------- #

def build_output_entry(
    base_entry: dict,
    extra_entries: List[dict],
    out_relative_path: str,
) -> dict:
    all_entries = [base_entry] + extra_entries
    n_sources = len(all_entries)

    # Dynamically match the slot count of the base entry (supports 10 or 12, etc.)
    slots = len(base_entry.get("regions", [])) or 10

    regions_out = ["" for _ in range(slots)]
    azimuths_out: List[float | None] = [None for _ in range(slots)]
    elevations_out: List[float | None] = [None for _ in range(slots)]
    mask_out = [0 for _ in range(slots)]

    # Fill first n_sources positions; any remaining (e.g., last two when slots==12)
    # stay zero/empty by construction.
    for idx, src in enumerate(all_entries[:slots]):
        region, az, el = first_active_region(src)
        regions_out[idx] = region
        azimuths_out[idx] = az
        elevations_out[idx] = el
        mask_out[idx] = 1

    return {
        "audio_filepath": out_relative_path,
        "n_sources": n_sources,
        "regions": regions_out,
        "azimuths": azimuths_out,
        "elevations": elevations_out,
        "mask": mask_out,
    }


# ------------------------------ sampling helpers ---------------------------- #

def sample_n_total(rng: random.Random) -> int:
    """Sample total number of sources (1..10) with linearly increasing weights.

    Weight for n is 1 + 0.1*(n-1), i.e., n=2 is 10% more likely than n=1,
    n=3 is 20% more likely than n=1, ..., n=10 is 90% more likely than n=1.
    """
    values = list(range(1, 11))
    weights = [1.0 + 0.1 * (n - 1) for n in values]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for v, w in zip(values, weights):
        acc += w
        if r <= acc:
            return v
    return values[-1]


# ------------------------------ main pipeline ------------------------------- #

def process_dataset(
    manifest: List[dict],
    audio_dir: Path,
    out_dir: Path,
    k: int,
    rng: random.Random,
    min_angle_sep_deg: float = 4.0,
) -> List[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest: List[dict] = []

    sr_ref: int | None = None  # for consistency check

    for base_idx, base_entry in enumerate(manifest):
        for rep in range(k):
            # Weighted sampling of total sources
            n_total = sample_n_total(rng)
            n_extra = n_total - 1

            # Try to honour constraints; if impossible, step down n_total
            while True:
                try:
                    extra_indices = select_extra_indices(
                        base_idx,
                        total=len(manifest),
                        num_extra=n_extra,
                        manifest=manifest,
                        min_angle_sep_deg=min_angle_sep_deg,
                        rng=rng,
                    )
                    break
                except RuntimeError:
                    if n_total > 1:
                        n_total -= 1
                        n_extra = n_total - 1
                    else:
                        # Fall back to only the base entry
                        extra_indices = []
                        break

            extra_entries = [manifest[j] for j in extra_indices]

            # ---- audio loading ---- #
            audio_paths = [audio_dir / e["audio_filepath"] for e in [base_entry] + extra_entries]
            signals = []
            for p in audio_paths:
                sig, sr = read_audio(p)
                if sr_ref is None:
                    sr_ref = sr
                elif sr != sr_ref:
                    raise ValueError(f"Sample-rate mismatch: {p} has {sr} Hz, expected {sr_ref} Hz.")
                if sig.shape[1] != 16:
                    raise ValueError(f"{p} has {sig.shape[1]} channels, expected 16.")
                signals.append(sig)

            mix = mix_audio_signals(signals)


            # --- DODATKI CO TRZECIA PRÓBKA ---
            # Identyfikator globalny miksu (naprzemienne przypadki)
            mix_idx = base_idx * k + rep

            # 1) co trzecia (case 0): dodaj losowy SZUM z audio_dir/noise/
            if mix_idx % 3 == 0:
                noise_dir = audio_dir / "noise"
                try:
                    noise_sig, noise_sr, noise_path = read_random_wav(noise_dir)
                except FileNotFoundError:
                    noise_sig = None
                    print(f"[WARN] Brak plików szumu w {noise_dir}; pomijam dodatek szumu.")

                if noise_sig is not None:
                    if sr_ref is not None and noise_sr != sr_ref:
                        raise ValueError(f"SR mismatch dla szumu {noise_path}: {noise_sr} vs {sr_ref}")
                    # 1- lub 16-kanał → 16 kanałów
                    noise_sig = _ensure_16ch_from_mono_or_16(noise_sig)
                    # dopasuj długość
                    noise_sig = _loop_to_length(noise_sig, mix.shape[0])
                    # losowy target relacji: [-30, -10] dB względem AKTUALNEGO miksu
                    rel_db = random.uniform(-10.0, 6.0)
                    target_rms = rms(mix) * db_to_lin(rel_db)
                    # skalowanie RMS
                    # (jeśli mix jest „prawie 0”, nic nie dodajemy istotnego)
                    if target_rms > 0.0:
                        # wspólny cel RMS dla wszystkich 16 kanałów szumu
                        # (zachowujemy wewnętrzne proporcje pliku szumu)
                        cur_rms = rms(noise_sig)
                        if cur_rms > 0.0:
                            noise_sig = noise_sig * (target_rms / cur_rms)
                            mix = mix + noise_sig

            # 2) inna co trzecia (case 1): dodaj losowe MONO z audio_dir/audios/ tylko do W
            elif mix_idx % 3 == 1:
                audios_dir = audio_dir / "audios"
                try:
                    mono_sig, mono_sr, mono_path = read_random_wav(audios_dir)
                except FileNotFoundError:
                    mono_sig = None
                    print(f"[WARN] Brak plików audio w {audios_dir}; pomijam dodatek mono.")

                if mono_sig is not None:
                    if sr_ref is not None and mono_sr != sr_ref:
                        raise ValueError(f"SR mismatch dla mono {mono_path}: {mono_sr} vs {sr_ref}")
                    # zapewnij mono
                    mono_sig = _ensure_mono(mono_sig)  # [frames,1]
                    # dopasuj długość
                    mono_sig = _loop_to_length(mono_sig, mix.shape[0])
                    # relacja [-30, -10] dB względem aktualnego miksu
                    rel_db = random.uniform(-10.0, 6.0)
                    target_rms = rms(mix) * db_to_lin(rel_db)
                    if target_rms > 0.0:
                        mono_sig = scale_to_target_rms(mono_sig, target_rms)
                        # dodaj TYLKO do kanału W (index 0)
                        mix[:, 0] = mix[:, 0] + mono_sig[:, 0]

            # Po dodatkach: bezpieczeństwo peak ≤ 0.99
            max_abs = np.max(np.abs(mix))
            if max_abs > 0.99:
                mix *= 0.99 / max_abs




            # ---- output ---- #
            out_filename = f"mix_{base_idx:05d}_{rep}.wav"
            out_path = out_dir / out_filename
            sf.write(out_path, mix, sr_ref, subtype="PCM_24")

            entry = build_output_entry(base_entry, extra_entries, out_relative_path=out_filename)
            out_manifest.append(entry)

            print(f"[OK] base {base_idx:05d}, rep {rep} | n_sources={n_total} -> {out_filename}")

    return out_manifest


# ------------------------------ configuration ------------------------------- #

def main() -> None:
    # ---- MANUAL CONFIGURATION ---- #
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "New_Database"
    manifest_path = (
        dataset_root / "samples2locator_5000_regions_update_manifest_10_samples.json"
    )
    audio_dir = dataset_root
    k_reps = 2  # number of mixtures per base entry

    out_dir = audio_dir / "out"
    out_manifest_path = manifest_path.with_name(manifest_path.stem + "_aug2_hnoise.json")

    rng = random.Random()

    manifest = load_manifest(manifest_path)
    out_manifest = process_dataset(manifest, audio_dir, out_dir, k=k_reps, rng=rng)
    save_manifest(out_manifest, out_manifest_path)

    print("\nDone!  Augmented dataset written to:")
    print(f"  audio files : {out_dir}")
    print(f"  manifest    : {out_manifest_path}")


if __name__ == "__main__":
    main()
