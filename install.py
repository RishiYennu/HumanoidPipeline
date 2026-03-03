import subprocess
import os
import sys

HOME = os.path.expanduser("~")
GVHMR_DIR = os.path.join(HOME, "GVHMR")
GMR_DIR = os.path.join(HOME, "GMR")


def run(cmd, cwd=None, env=None):
    """Run a shell command and stream output."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
    if result.returncode != 0:
        print(f"WARNING: Command exited with code {result.returncode}")
    return result.returncode

def install_gvhmr():
    if os.path.isdir(GVHMR_DIR):
        print(f"[GVHMR] Already exists at {GVHMR_DIR}, skipping install.")
        return

    print("=== Installing GVHMR ===")

    run("git clone https://github.com/zju3dv/GVHMR", cwd=HOME)

    run("conda create -y -n gvhmr python=3.10", cwd=GVHMR_DIR)
    conda_run = "conda run -n gvhmr --no-capture-output"

    run(f"{conda_run} pip install -r requirements.txt", cwd=GVHMR_DIR)
    run(f"{conda_run} pip install -e .", cwd=GVHMR_DIR)

    ckpt_dir = os.path.join(GVHMR_DIR, "inputs", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("\n[GVHMR] Code installed!")
    print("  You still need to manually download:")
    print("  1. SMPL models:  https://smpl.is.tue.mpg.de/")
    print("  2. SMPLX models: https://smpl-x.is.tue.mpg.de/")
    print(f"     Place in: {ckpt_dir}/body_models/smpl/ and smplx/")
    print("  3. Pretrained checkpoints from Google Drive:")
    print("     https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD")
    print(f"     Place in: {ckpt_dir}/")
    print("     Expected: dpvo/dpvo.pth, gvhmr/gvhmr_siga24_release.ckpt,")
    print("               hmr2/epoch=10-step=25000.ckpt, vitpose/vitpose-h-multi-coco.pth, yolo/yolov8x.pt")


def install_gmr():
    if os.path.isdir(GMR_DIR):
        print(f"[GMR] Already exists at {GMR_DIR}, skipping install.")
        return

    print("\n=== Installing GMR ===")

    run("git clone https://github.com/YanjieZe/GMR", cwd=HOME)

    run("conda create -y -n gmr python=3.10", cwd=GMR_DIR)
    conda_run = "conda run -n gmr --no-capture-output"

    run(f"{conda_run} pip install -e .", cwd=GMR_DIR)

    run(f"{conda_run} conda install -c conda-forge libstdcxx-ng -y", cwd=GMR_DIR)
    body_model_dir = os.path.join(GMR_DIR, "assets", "body_models", "smplx")
    os.makedirs(body_model_dir, exist_ok=True)

    print("\n[GMR] Code installed!")
    print("  You still need to manually download:")
    print("  1. SMPLX body models from: https://smpl-x.is.tue.mpg.de/")
    print(f"     Place SMPLX_NEUTRAL.pkl, SMPLX_FEMALE.pkl, SMPLX_MALE.pkl in:")
    print(f"     {body_model_dir}/")
    print("  2. (Optional) AMASS motion data from: https://amass.is.tue.mpg.de/")
    print("  3. (Optional) LAFAN1 motion data from: https://github.com/ubisoft/ubisoft-laforge-animation-dataset")
    print("\n  IMPORTANT: After installing SMPLX, change 'ext' in smplx/body_models.py")
    print("  from 'npz' to 'pkl' if you are using SMPL-X pkl files.")


if __name__ == "__main__":
    print(f"Home directory: {HOME}")
    print(f"Checking for GVHMR at: {GVHMR_DIR}")
    print(f"Checking for GMR at:   {GMR_DIR}")
    print("=" * 50)

    install_gvhmr()
    install_gmr()

    print("\n" + "=" * 50)
    print("All done!")
    print(f"  GVHMR env: conda activate gvhmr")
    print(f"  GMR env:   conda activate gmr")
    print("=" * 50)