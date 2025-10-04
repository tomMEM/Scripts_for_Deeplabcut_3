from pathlib import Path
import deeplabcut
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as DQ
import cv2
import shutil



# -----------------------------
# USER INPUT
# -----------------------------
project_name = "Feb"
experimenter = "Thomas"
working_dir  = Path(r"C:\Users\thomas\users\2P_Feb_Social")

videos = [
    r"C:\Users\thomas\users\2P_Feb_Social\Feb2025_behav\1stHab-1-2025.2.25\MiceVideo2\MiceVideo\CHB_2025-02-25_18-26-40-0000.avi"
]

bodyparts = ["Nose", "Head", "Body", "Tail"]


from pathlib import Path
import os

def sanitize_avi_filenames(root_dir):
    """
    Recursively scan for .avi files under root_dir and rename them
    by replacing spaces ' ' with underscores '_'.
    Prints each change and warns the user.
    """
    root = Path(root_dir)
    print(f"🔎 Scanning for AVI files in: {root}")

    renamed = []
    for avi in root.rglob("*.avi"):
        if " " in avi.name:
            new_name = avi.name.replace(" ", "_")
            new_path = avi.with_name(new_name)
            print(f"⚠️  Renaming: {avi.name} -> {new_name}")
            avi.rename(new_path)
            renamed.append((avi, new_path))

    if renamed:
        print(f"\n✅ Done. {len(renamed)} files were renamed.")
        print("⚠️ WARNING: Spaces in AVI filenames were replaced with '_' to avoid DeepLabCut/Windows issues.")
    else:
        print("✅ No AVI files with spaces found.")

    return renamed




root_dir = working_dir# r"C:\Users\thomas\users\2P_Feb_Social"
sanitize_avi_filenames(root_dir)

# -----------------------------
# CREATE PROJECT
# -----------------------------
config_path = deeplabcut.create_new_project(
    project=project_name,
    experimenter=experimenter,
    videos=videos,
    working_directory=str(working_dir),
    copy_videos=True
)
print("✅ Project created:", config_path)

config_file = Path(config_path)  # ensure Path object
proj_dir    = config_file.parent
vid_dir     = proj_dir / "videos"

# -----------------------------
# STEP 1: FIX BROKEN VIDEO PATHS AS TEXT
# -----------------------------
print("🔧 Repairing config.yaml (text stage)...")

# Backup original
backup_path = config_file.with_suffix(".bak")
shutil.copy(config_file, backup_path)
print(f"💾 Backup saved at: {backup_path}")

# Read raw text
with open(config_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Repair lines: join broken video path lines
fixed_lines = []
skip_next = False
for i, line in enumerate(lines):
    if skip_next:
        skip_next = False
        continue
    if line.strip().endswith("CHB"):  # broken line
        # join with next line
        joined = line.rstrip("\n") + lines[i+1]
        fixed_lines.append(joined)
        skip_next = True
    else:
        fixed_lines.append(line)

# Write fixed text back
with open(config_file, "w", encoding="utf-8") as f:
    f.writelines(fixed_lines)

print("📝 Video path lines repaired (as text).")

# -----------------------------
# STEP 2: LOAD WITH YAML & REWRITE SAFELY
# -----------------------------
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)
yaml.width = 10**6

with open(config_file, "r", encoding="utf-8") as f:
    cfg = yaml.load(f)

# Rebuild clean video_sets
video_sets = {}
for vf in vid_dir.iterdir():
    if vf.suffix.lower() not in [".avi", ".mp4", ".mov", ".mkv"]:
        continue

    cap = cv2.VideoCapture(str(vf))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        w = h = 0
    else:
        h, w = frame.shape[:2]

    # Always quote video path
    video_sets[DQ(str(vf))] = {"crop": [0, int(w), 0, int(h)]}

cfg["video_sets"] = video_sets
cfg["bodyparts"]  = bodyparts
cfg["scorer"]     = experimenter

with open(config_file, "w", encoding="utf-8") as f:
    yaml.dump(cfg, f)

print("✅ Config repaired and updated with bodyparts at:", config_file)

# -----------------------------
# STEP 3: TEST RELOAD
# -----------------------------
try:
    with open(config_file, "r", encoding="utf-8") as f:
        _ = yaml.load(f)
    print("✅ Config YAML reload test: OK")
except Exception as e:
    print("❌ Still invalid YAML:", e)


from ruamel.yaml import YAML
from pathlib import Path

config_file = config_path #Path(r"C:\Users\thomas\users\2P_Feb_Social\Feb-Thomas-2025-10-03\config.yaml")
yaml = YAML()

with open(config_file, "r", encoding="utf-8") as f:
    cfg = yaml.load(f)

print("Video sets:", list(cfg["video_sets"].keys()))


# -----------------------------
# VERIFY THAT VIDEO CAN BE OPENED
# -----------------------------
import cv2

print("\n🔎 Verifying video accessibility from config.yaml...")

video_paths = list(cfg["video_sets"].keys())
for vp in video_paths:
    print(f"Checking: {vp}")
    vp_path = Path(vp)

    if not vp_path.exists():
        print(f"  ❌ File does not exist: {vp_path}")
        continue

    cap = cv2.VideoCapture(str(vp_path))
    if not cap.isOpened():
        print(f"  ❌ OpenCV could not open: {vp_path}")
    else:
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"  ✅ OpenCV read first frame successfully ({w}x{h})")
        else:
            print(f"  ⚠️ OpenCV opened but could not read a frame: {vp_path}")
    cap.release()
