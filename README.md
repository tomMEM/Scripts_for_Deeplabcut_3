## 🧠 About These Scripts

These scripts are built to automate and streamline workflows using **[DeepLabCut 3.x](https://deeplabcut.github.io/DeepLabCut/)** — an open-source toolbox for markerless pose estimation of animals and humans.


> These scripts were developed for **research and teaching purposes**, to enhance and automate workflows in **DeepLabCut 3.x**.
> They rely on DeepLabCut’s **official backend functions, APIs, and configuration structures**.
>
> ⚖️ **Disclaimer:** All DeepLabCut intellectual property, algorithms, and core functionality remain owned and maintained by the official [DeepLabCut Development Team](https://github.com/DeepLabCut) at the Mathis Group (Harvard/EPFL).

---

### 📦 Official Repository

👉 **GitHub:** [https://github.com/DeepLabCut/DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)

# 🐭 DeepLabCut 3 Interactive Command-Line Suite

This toolkit re-creates the most important DeepLabCut 3 functions without using the GUI.
Each script corresponds to one step of the standard DLC workflow — from project creation, through data preparation, to training.

> 💡 Designed for **DLC 3.x (PyTorch engine)** on Windows systems
> Compatible with **non-symlink filesystems** (like exFAT or NTFS without admin privileges)

---

## 📁 Folder Structure

Each project typically has this structure:

```
YourProject/
│
├─ config.yaml
├─ videos/
├─ labeled-data/
├─ training-datasets/
├─ dlc-models-pytorch/
└─ scripts/
   ├─ dlc3_create_v1.py
   ├─ dlc3_crop_settings.py
   ├─ dlc3_extract_v3.py
   ├─ dlc3_syncvideos_createdataset.py
   └─ dlc3_v2.py
```

---

## 1️⃣ **Project Creation**

### **Script:** `dlc3_create_v1.py`

Creates a new DeepLabCut project via command line.
Equivalent to `deeplabcut.create_new_project()` in the GUI.

**🧠 User inputs:**

* Project name (e.g. `"FebOct3"`)
* Your name (e.g. `"Thomas"`)
* Path to a folder containing your videos
* Whether to copy or link videos (it auto-handles exFAT by copying)
* Whether to create a multi-animal or single-animal project

**💾 Output:**

* `config.yaml` file
* `/videos/` folder populated
* `/labeled-data/` initialized

---

## 2️⃣ **Adjust Crop Parameters**

### **Script:** `dlc3_crop_settings.py`

Opens and edits cropping values inside `config.yaml` for all listed videos.

**🧠 User inputs:**

* Full path to the project’s `config.yaml`
* Crop coordinates or a scaling factor to resize all videos
* Optionally auto-detect crop size using OpenCV

**💡 Notes:**
Useful for ensuring that all videos share consistent resolution or removing frame borders before extraction.

---

## 3️⃣ **Frame Extraction (for .AVI or all videos)**

### **Script:** `dlc3_extract_v3.py`

Replacement for the DLC GUI’s **“Extract frames”** function — but working even for `.AVI` files.

**🧠 User inputs:**

* Path to `config.yaml`
* Frame extraction method (e.g., `kmeans`, `uniform`)
* Number of frames per video (e.g., `40`)
* Resize factor or crop region (optional)
* Whether to overwrite old extracted frames

**💡 Notes:**
This step populates `/labeled-data/VideoName/` folders with extracted frames ready for manual labeling.

---

## 4️⃣ **Sync Labeled Videos and Create Training Dataset**

### **Script:** `dlc3_syncvideos_createdataset.py`

Finds all labeled frames (`CollectedData_*.h5`) across subfolders
and automatically updates `config.yaml` to include **all labeled videos**.
Then rebuilds the **training dataset**.

**🧠 User inputs:**

* Path to `config.yaml`
* Root directory where videos are stored
* Target iteration (e.g., `0`)
* Shuffle number (usually `1`)
* Optional: pattern to **exclude** (e.g., `"MiceVideo1"`)

**💡 Notes:**

* Automatically rebuilds the dataset with `deeplabcut.create_training_dataset()`
* Sets `TrainingFraction: [0.8]` (train/test split)
* Skips unreadable or excluded videos
* Cleans up iteration folders before rebuilding

**✅ Output:**

* Updated `config.yaml`
* New training dataset under `/training-datasets/iteration-0/`
* Ready for training

---

## 5️⃣ **Train (or Resume) Network**

### **Script:** `dlc3_v2.py`

Interactive training manager for DeepLabCut 3 (PyTorch).
Allows you to start new training or resume from existing snapshots.

**🧠 User inputs:**

* Path to your DLC project folder
* Choice of training run (lists all `train` subfolders found in `/dlc-models-pytorch/`)
* If resuming: select snapshot epoch (latest or specific)
* Number of **epochs** to train (e.g., `500`)
* Save interval (e.g., every `50` epochs)

**🧩 Extra features:**

* Automatically updates `pytorch_config.yaml` (real training file)
* Adds safe defaults:

  * `batch_size = 8`
  * `display_iters = 100`
  * `lr = 0.0003` (for small datasets)
* Supports **offline training** (`HF_HUB_OFFLINE=1`)
* Merges new logs into existing `learning_stats.csv`
* Auto-adjusts epoch numbering after resume

**💾 Output:**

* Training snapshots (`snapshot-*.pt`)
* Updated training logs (`learning_stats.csv`)
* Fine-tuned model weights in `/dlc-models-pytorch/.../train/`

---

## 🧠 Typical Workflow Summary

| Step | Script                             | Purpose                                | Input                        | Output               |
| ---- | ---------------------------------- | -------------------------------------- | ---------------------------- | -------------------- |
| 1️⃣  | `dlc3_create_v1.py`                | Create project                         | project name, videos         | config.yaml          |
| 2️⃣  | `dlc3_crop_settings.py`            | Adjust crop size                       | config.yaml                  | updated config       |
| 3️⃣  | `dlc3_extract_v3.py`               | Extract frames                         | config.yaml                  | labeled-data/ frames |
| 4️⃣  | `dlc3_syncvideos_createdataset.py` | Combine labeled data & rebuild dataset | config.yaml, exclude pattern | new training dataset |
| 5️⃣  | `dlc3_v2.py`                       | Train or resume                        | project path, epochs         | trained model        |

---

## ⚙️ Example Use Case

```bash
# 1. Create a project
python dlc3_create_v1.py

# 2. Adjust crop region if needed
python dlc3_crop_settings.py

# 3. Extract 40 frames per video
python dlc3_extract_v3.py

# 4. Sync labeled data, exclude MiceVideo1
python dlc3_syncvideos_createdataset.py

# 5. Train for 500 epochs, saving every 50
python dlc3_v2.py
```

---

## 🧩 Pro Tips

* Use **TrainingFraction = [0.8]** for balanced validation.
* If you move videos, always re-sync with `dlc3_syncvideos_createdataset.py`.
* For small datasets, 300–500 epochs often suffice.
* You can inspect training progress live via the console or `learning_stats.csv`.
* When switching datasets (e.g. 80 % vs 95 %), remove old training folders to avoid confusion.
