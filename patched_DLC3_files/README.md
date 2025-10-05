# 🧩 DeepLabCut Snapshot Rename Safety Patch

---

<details>
<summary><b>📘 Purpose — Fix for Snapshot Rename Crash</b></summary>

This patch fixes a known issue in **DeepLabCut 3.x** that causes a crash during training:

```

FileExistsError: [WinError 183] Cannot create a file when that file already exists:
'snapshot-best-050.pt' -> 'snapshot-050.pt'

````

This happens when DeepLabCut tries to rename an existing `snapshot-best-XXX.pt` file.  
The patch adds a safety check to prevent overwriting existing snapshots.

</details>

---

<details>
<summary><b>⚙️ How It Works</b></summary>

- Automatically **backs up** the original `snapshots.py` file.  
- Adds a safety guard before renaming:
  ```python
  if not new_name.exists():
      current_best.path.rename(new_name)
  else:
      warnings.warn(f"[DLC Warning] Snapshot {new_name.name} already exists. Skipping rename.")
    ````

* Verifies the patch was successfully applied by scanning the modified file.

📂 **Patched file location:**

```
deeplabcut/pose_estimation_pytorch/runners/snapshots.py
(Line ~115 in DLC 3.x)
```



---


<summary><b>🪄 Usage Instructions</b></summary>

1. **Locate your DeepLabCut installation path**

   Example for conda:

   ```
   C:\Users\<username>\conda_envs\dlc\lib\site-packages\deeplabcut\pose_estimation_pytorch\runners\snapshots.py
   ```

2. **Edit the path in the patch script**

   ```python
   dlc_path = Path(r"YOUR_PATH_HERE")
   ```

3. **Run the patch once after each DeepLabCut update**

   ```bash
   python patch_dlc_snapshots.py
   ```

4. ✅ You should see:

   ```
   📦 Applying patch: Fix for snapshot rename crash...
   ✅ Backup created: snapshots.bak
   🔍 Verification successful — patch active
   ```



---


<summary><b>💡 Notes & Tips</b></summary>

* Safe to run multiple times (the patch won’t duplicate itself).
* Creates a `.bak` backup of your original file.
* Works on **Windows, Linux, and macOS** (Python ≥3.9).
* If you update or reinstall DeepLabCut, simply **re-run this patch**.

💡 **Optional**: Create a quick batch file to reapply it:

```batch
@echo off
echo Running DeepLabCut snapshot patch...
python patch_dlc_snapshots.py
pause
```






<summary><b>📦 Example Folder Layout</b></summary>

```
deeplabcut-snapshot-patch/
│
├── patch_dlc_snapshots.py
└── README.md
```



---


<summary><b>🧑‍💻 Author & License</b></summary>

Developed by **Thomas**,
for research and teaching support using **DeepLabCut 3.x**.

**License:** MIT — free to use, modify, and distribute.

</details>




