# VR180 Stereo → Point Cloud

Usage:

1. Install dependencies:

```bash
python -m pip install -r stereo_vr/requirements.txt
```

2. Run the converter:

```bash
python stereo_vr/stereo_vr.py LEFT_EQR.jpg RIGHT_EQR.jpg --out output.ply --fov 90 --out_w 1024 --out_h 768 --baseline 0.06
```

Notes:
- The script projects equirectangular images to a rectilinear perspective view for a given yaw/pitch and computes disparity with OpenCV SGBM.
- `--baseline` should be provided in meters to obtain metric depths.
- For higher-quality results, tune `--num_disp` and `--block_size`.
