# Speaker Array Ground-Truth Measurement

Measure spatial directions (azimuth, elevation) of speakers in a static array
using a single 360° photo from an Insta360 camera co-mounted with an
ambisonic microphone.

## What this does

Given:
- A rigid rig: Insta360 + ambisonic mic mounted together
- AprilTag markers stuck on each speaker at the acoustic-center point
- A few calibration markers placed around the room

Produces:
- `(azimuth, elevation)` for each speaker in the mic's coordinate frame
- Typical accuracy: ~0.2–0.5° per direction

## Why this approach

Because we only need direction (not range), a single 360° photo from the
mic position contains all the geometric information needed. An equirectangular
image is a direct pixel-to-(az, el) map by construction; the only unknown is
the rotational offset between the camera and the mic, which is solved once
per rig assembly using a handful of known reference points.

This avoids photogrammetry (no walking around with the rig), avoids LiDAR
(2D scanners can't see drivers, 3D scanners are expensive), and gives a
single-shot workflow that takes seconds per measurement after setup.

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Print AprilTags (see examples/generate_apriltags.py)
python examples/generate_apriltags.py

# 3. Set up calibration scene, measure marker positions, edit
#    calibration/calibration_input.yaml

# 4. Run one-time calibration
python examples/run_calibration.py

# 5. For each photo, run measurement
python examples/run_measurement.py --image data/scene.jpg
```

## Documentation

- [Pipeline overview](docs/01-pipeline-overview.md)
- [Marker sizing for distance](docs/02-marker-sizing.md)
- [Calibration procedure](docs/03-calibration-procedure.md)
- [Coordinate conventions](docs/04-coordinate-conventions.md)
- [Troubleshooting](docs/05-troubleshooting.md)

## Hardware notes

- **Camera:** Insta360 X3/X4 or similar, 5.7K stills minimum (11K preferred)
- **Mic:** Any ambisonic mic, rigidly mounted to the camera
- **Markers:** Print AprilTag DICT_APRILTAG_16h5 on matte paper. Size depends
  on distance — see [docs/02-marker-sizing.md](docs/02-marker-sizing.md).
- **Calipers** for measuring camera-to-mic offset (one-time, ~mm precision)
- **Tape measure or laser distance meter** for calibration marker positions

## Coordinate convention

AmbiX / SN3D:
- `+x` forward, `+y` left, `+z` up
- Azimuth: CCW from +x, in the xy-plane
- Elevation: angle above the xy-plane

## License

Internal research code. Adapt freely for your own measurement work.
