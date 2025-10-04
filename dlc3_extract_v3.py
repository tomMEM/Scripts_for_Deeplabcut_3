import deeplabcut
from pathlib import Path
import ruamel.yaml
import cv2
import shutil

# -----------------------------
# CONFIG PATH
# -----------------------------
config_path = Path(r"C:\Users\thomas\users\2P_Feb_Social\Feb-Thomas-2025-10-03\config.yaml")
backup_path = config_path.with_suffix(".yaml.bak")

# -----------------------------
# USER INPUT: VIDEOS
# -----------------------------
print("Paste full paths to videos (AVI/MP4).")
print("Enter blank line when finished:\n")

videos_to_add = []
while True:
    v = input("Video path: ").strip().strip('"').strip("'")
    if not v:
        break
    p = Path(v)
    if p.exists() and p.suffix.lower() in [".avi", ".mp4", ".mov", ".mkv"]:
        videos_to_add.append(str(p.resolve()))
    else:
        print(f"❌ Skipping (not found or not a supported video): {v}")

if not videos_to_add:
    print("⚠️ No valid videos provided, exiting.")
    exit(0)

print(f"\n✅ Videos to add:\n" + "\n".join(videos_to_add))

# -----------------------------
# LOAD CONFIG
# -----------------------------
yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.load(f)

# Backup old config
shutil.copy(config_path, backup_path)
print(f"💾 Backup saved at {backup_path}")

# -----------------------------
# REPLACE video_sets with ONLY new videos
# -----------------------------
cfg["video_sets"] = {}

for v in videos_to_add:
    cap = cv2.VideoCapture(v)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print(f"⚠️ Warning: Could not read frames from {v}")
        continue
    h, w = frame.shape[:2]
    cfg["video_sets"][v] = {"crop": f"0,{w},0,{h}"}

# Save the modified config.yaml
with open(config_path, "w", encoding="utf-8") as f:
    yaml.dump(cfg, f)

print(f"🔧 Updated config.yaml with {len(videos_to_add)} new videos ONLY.")

# -----------------------------
# RUN EXTRACTION
# -----------------------------
deeplabcut.extract_frames(
    str(config_path),
    mode="automatic",
    algo="kmeans",   # or "uniform"
    crop=False,
    userfeedback=False,
    cluster_step=10,          # speed up extraction (downsample frames)
    cluster_resizewidth=150   # smaller frames for kmeans
)

print("\n✅ Done! You can now label the frames in the GUI.")
print(f"⚠️ Remember: your old config.yaml is backed up at {backup_path}")
