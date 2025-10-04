import deeplabcut
from pathlib import Path
import shutil
from ruamel.yaml import YAML

# ===================================================================
#                      INTERACTIVE DLC3 PIPELINE
# ===================================================================

print("🐭 DeepLabCut 3.x Interactive Trainer\n")

# --- STEP 0: BASIC USER INPUT ---
config_path = input("Enter full path to your DLC config.yaml: ").strip().strip('"')
if not Path(config_path).exists():
    raise FileNotFoundError(f"❌ Config file not found: {config_path}")

video_root_directory = input("\nEnter the top-level directory where your videos are stored:\n").strip().strip('"')
video_root_directory = Path(video_root_directory)
if not video_root_directory.exists():
    raise FileNotFoundError(f"❌ Video root directory not found: {video_root_directory}")

# Get iteration + shuffle interactively
try:
    TARGET_ITERATION = int(input("\nEnter target iteration number (e.g. 0): ").strip() or "0")
    TARGET_SHUFFLE = int(input("Enter shuffle number (e.g. 1): ").strip() or "1")
except ValueError:
    raise ValueError("❌ Invalid input for iteration or shuffle. Please enter integers.")

project_path = Path(config_path).parent

# ===================================================================
# STEP 1: FIND ALL VIDEO FILES THAT MATCH LABELED FOLDERS
# ===================================================================

print(f"\nSTEP 1/5: 🔍 Scanning labeled-data folders and matching videos...")

labeled_data_dir = Path(config_path).parent / "labeled-data"
if not labeled_data_dir.exists():
    raise FileNotFoundError(f"❌ 'labeled-data' folder not found at {labeled_data_dir}")

# Get the base folder names under labeled-data (each corresponds to one video)
labeled_folders = [p.name for p in labeled_data_dir.iterdir() if p.is_dir()]
print(f"🗂 Found {len(labeled_folders)} labeled folders.")

# Search for matching video files anywhere under your video root
all_video_paths = []
for label_folder in labeled_folders:
    # find matching video (avi/mp4) whose filename contains the labeled folder name
    for ext in ['.avi', '.mp4']:
        for p in video_root_directory.rglob(f"*{label_folder}*{ext}"):
            all_video_paths.append(str(p.resolve()))

# Deduplicate and sort for consistency
all_video_paths = sorted(set(all_video_paths))

if not all_video_paths:
    print("❌ No matching video files found for your labeled folders.")
    exit(1)

print(f"✅ Found {len(all_video_paths)} video(s) corresponding to labeled folders:")
for v in all_video_paths:
    print("   -", v)

# ===================================================================
# STEP 2: DIRECTLY UPDATE CONFIG (no copy, no symlink)
# ===================================================================
from ruamel.yaml import YAML
import cv2

yaml = YAML()
yaml.preserve_quotes = True

print("\nSTEP 2/5: 🧩 Directly updating config.yaml with labeled videos only (no copy/symlink)...")

# --- Ask user which folder or path part to exclude ---
exclude_pattern = input("Enter part of path to EXCLUDE (e.g., 'MiceVideo1', or leave blank for none): ").strip()

# Load config.yaml
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.load(f)
print("TrainingFraction:", cfg["TrainingFraction"])
cfg["TrainingFraction"] = [0.8]   # was [0.95]

# Reset the list of videos
cfg["video_sets"] = {}

# Find all labeled-data folders to match videos
labeled_dir = Path(project_path) / "labeled-data"
labeled_folders = [f.name for f in labeled_dir.iterdir() if f.is_dir()]
print(f"   Found {len(labeled_folders)} labeled folders.")

# Add only videos whose filename matches a labeled folder
added = 0
for v in all_video_paths:
    # Skip any that match the exclusion rule
    if exclude_pattern and exclude_pattern.lower() in str(v).lower():
        print(f"🚫 Excluded (matched '{exclude_pattern}'): {v}")
        continue

    video_name = Path(v).stem
    if any(video_name in lf for lf in labeled_folders):
        cap = cv2.VideoCapture(v)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"⚠️ Skipping unreadable video: {v}")
            continue
        h, w = frame.shape[:2]
        cfg["video_sets"][v] = {"crop": f"0,{w},0,{h}"}
        print(f"   + Added labeled video: {v}")
        added += 1




# Save cleaned config
with open(config_path, "w", encoding="utf-8") as f:
    yaml.dump(cfg, f)

print(f"\n✅ Config.yaml updated with {added} labeled videos (excluding any matching '{exclude_pattern}').")



# ===================================================================
# STEP 3: RESET ITERATION
# ===================================================================

print(f"\nSTEP 3/5: 🔁 Resetting to iteration {TARGET_ITERATION}...")
training_dataset_path = project_path / "training-datasets" / f"iteration-{TARGET_ITERATION}"
if training_dataset_path.exists():
    print(f"   - Removing old training dataset at {training_dataset_path}")
    shutil.rmtree(training_dataset_path)

yaml = YAML()
with open(config_path, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f)
cfg['iteration'] = TARGET_ITERATION
with open(config_path, 'w', encoding='utf-8') as f:
    yaml.dump(cfg, f)
print(f"   - Set 'iteration: {TARGET_ITERATION}' in config.yaml")

# ===================================================================
# STEP 4: CREATE TRAINING DATASET
# ===================================================================

print(f"\nSTEP 4/5: 🧱 Creating new training dataset (iteration-{TARGET_ITERATION})...")
deeplabcut.create_training_dataset(config_path)
print("✅ Created new training dataset successfully.")

# ===================================================================
# STEP 5: TRAIN NETWORK
# ===================================================================

#print(f"\nSTEP 5/5: 🧠 Starting training (iteration-{TARGET_ITERATION}, shuffle-{TARGET_SHUFFLE})...")
#deeplabcut.train_network(config_path, shuffle=TARGET_SHUFFLE)
#print("\n🎉 Training complete! 🎉")

# ===================================================================
# OPTIONAL POST-STEP
# ===================================================================
#print("\n💡 Tip: You can now run `deeplabcut.evaluate_network` or open the GUI to inspect results.")
