"""
Patch for snapshot-??.pt rename crash while training
---------------------------------------------------
Fixes DeepLabCut 3.x FileExistsError:
  [WinError 183] Cannot create a file when that file already exists
Triggered when 'snapshot-best-XXX.pt' already exists during rename.

Usage:
    1. Adjust the path to match your DeepLabCut installation.
    2. Run this script ONCE after installing or updating DeepLabCut.

Author: Thomas (custom patch)
License: MIT
"""

from pathlib import Path
import shutil
import sys

# === 🔧 USER CONFIGURATION ===
# Change this path to match your DLC environment folder
# Example for conda env 'dlc': r"C:\Users\Thomas\conda_envs\dlc\lib\site-packages\deeplabcut\pose_estimation_pytorch\runners\snapshots.py"
dlc_path = Path(
    r"C:\Users\Thomas\conda_envs\dlc\lib\site-packages\deeplabcut\pose_estimation_pytorch\runners\snapshots.py"
)

# === Script logic starts here ===
if not dlc_path.exists():
    sys.exit(f"❌ snapshots.py not found at:\n{dlc_path}\nPlease check your DLC installation path.")

backup_path = dlc_path.with_suffix(".bak")

# --- Patch code text ---
patch_marker = "Skipping rename from"
replacement_snippet = (
    "if not new_name.exists():\n"
    "            current_best.path.rename(new_name)\n"
    "        else:\n"
    "            warnings.warn("
    "f\"[DLC Warning] Snapshot {new_name.name} already exists. "
    "Skipping rename from {current_best.path.name}.\")"
)

# --- Apply patch if not present ---
code = dlc_path.read_text(encoding="utf-8")

if patch_marker not in code:
    print("📦 Applying patch: Fix for snapshot-??.pt rename crash while training...")
    shutil.copy(dlc_path, backup_path)
    patched = code.replace("current_best.path.rename(new_name)", replacement_snippet)
    dlc_path.write_text(patched, encoding="utf-8")
    print(f"✅ Patch applied and backup created → {backup_path.name}")
else:
    print("✅ Patch already present — no action needed.")

# --- Verification step ---
verify_text = dlc_path.read_text(encoding="utf-8")
if patch_marker in verify_text:
    print("🔍 Verification successful — patch confirmed active in snapshots.py")
else:
    print("⚠️ Verification failed — patch code not detected.")
