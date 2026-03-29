# Running the stereo_vr converter (step-by-step)

This document explains how to set up a Python virtual environment on Windows (PowerShell), install dependencies, run the converter, and troubleshoot common issues.

## 1. Prerequisites
- Python 3.9 or later installed and on your PATH.
- PowerShell (default on Windows).
- A pair of VR180 equirectangular images: a left-eye and a right-eye image.
- Workspace root containing the `stereo_vr` folder.

## 2. Open PowerShell in the project root
1. Open File Explorer to your project folder (for example: `E:\Documents\Projects\VRToDepth`).
2. Right-click inside the folder and choose "Open in Terminal" (or open PowerShell and `cd` to the project root).

## 3. Create and activate a virtual environment
Run these commands in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by execution policy, run (one-time, current session):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\.venv\Scripts\Activate.ps1
```

After activation your prompt will show the virtual environment name (for example `(.venv) PS C:\...`).

## 4. Upgrade pip and install dependencies
From the project root:

```powershell
python -m pip install --upgrade pip
python -m pip install -r stereo_vr/requirements.txt
```

This installs `numpy` and `opencv-python`. If you prefer a different OpenCV build (e.g., `opencv-contrib-python`) you can edit `stereo_vr/requirements.txt` and re-run the install command.

## 5. Run the converter (single perspective)
Basic usage (replace filenames):

```powershell
python stereo_vr/stereo_vr.py stereo_vr/LEFT_EQR.jpg stereo_vr/RIGHT_EQR.jpg --out output.ply
```

Common useful options:
- `--fov 90` : Field of view of the perspective projection (degrees).
- `--yaw 0 --pitch 0` : Orientation of the perspective crop on the equirectangular image (degrees).
- `--out_w 1024 --out_h 768` : Output perspective image resolution.
- `--baseline 0.06` : Stereo baseline in meters (use a realistic value to get metric depths).
- `--num_disp 256 --block_size 7` : Disparity tuning (higher `num_disp` and odd `block_size` can improve results; `num_disp` must be multiple of 16 internally).

Example with options:

```powershell
python stereo_vr/stereo_vr.py stereo_vr/left.jpg stereo_vr/right.jpg --out out.ply --fov 90 --yaw 10 --pitch 0 --out_w 1280 --out_h 720 --baseline 0.065 --num_disp 256 --block_size 7
```

## 6. Inspect the output
- The script writes a PLY file (ASCII) at the path given by `--out` (default `cloud.ply`).
- Open the PLY with MeshLab (meshlab.net) or CloudCompare (cloudcompare.org) to view the point cloud.

## 7. Tips for better results
- Increase `--num_disp` (multiple of 16) and tune `--block_size` for texture/noise tradeoffs.
- Try multiple `--yaw` angles (e.g., -90..+90) and merge the resulting PLYs if you need wider coverage. You can later align/merge them in MeshLab/CloudCompare.
- If disparity looks noisy, lower `--block_size` or increase `--speckleWindowSize` and `--speckleRange` inside the code's StereoSGBM parameters (advanced).
- Ensure left/right images are correctly ordered and correspond to the same scene instant.

## 8. Deactivate the virtual environment
When finished:

```powershell
deactivate
```

## 9. Troubleshooting
- "Module not found" after activation: ensure the venv is activated and you installed the requirements in that same environment.
- Images not found: use correct relative paths from the project root (e.g., `stereo_vr/left.jpg`).
- Very sparse point cloud: increase `--num_disp` and `--out_w`/`--out_h` to provide more pixel detail.

## 10. Next steps (optional)
- Create a small script to iterate yaw angles and produce/merge several PLYs for full 180 coverage.
- Replace SGBM parameters with tuned values or use external stereo networks for higher quality depth (requires additional dependencies).

---
Files:
- `stereo_vr/stereo_vr.py`
- `stereo_vr/requirements.txt`
- `stereo_vr/README.md`
- `stereo_vr/RUNNING.md` (this file)

If you want, I can: create a wrapper to sample multiple yaw angles and merge PLYs, or run a quick test if you upload a small left/right sample pair.
