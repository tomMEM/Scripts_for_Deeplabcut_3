## 🧠 About This Toolkit

These scripts are built to automate and streamline workflows using **[DeepLabCut 3.x](https://deeplabcut.github.io/DeepLabCut/)** — an open-source toolbox for markerless pose estimation of animals and humans.

> These scripts were developed for **research and teaching purposes**, to enhance and automate workflows in **DeepLabCut 3.x**.
> They rely on DeepLabCut’s **official backend functions, APIs, and configuration structures**.
>
> ⚖️ **Disclaimer:** All DeepLabCut intellectual property, algorithms, and core functionality remain owned and maintained by the official [DeepLabCut Development Team](https://github.com/DeepLabCut) at the Mathis Group (Harvard/EPFL).

---

### 📦 Official Repository

👉 **GitHub:** [https://github.com/DeepLabCut/DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)

# DeepLabCut v3 Installation and Training Guide

This guide provides a comprehensive walkthrough for installing DeepLabCut v3, handling common issues, manually caching models for offline use, and correctly evaluating model performance to avoid overfitting.

## Table of Contents

1.  [**Installation**](#installation)
    *   [Recommended Pip-Based Install (Windows)](#recommended-pip-based-install-windows)
    *   [Legacy TF 2.10 Install](#legacy-tf-210-install)
    *   [Troubleshooting](#troubleshooting)
2.  [**Offline Model Caching**](#offline-model-caching)
    *   [The Problem: Network Errors](#the-problem-network-errors)
    *   [Step 1: Gather Information](#step-1-gather-information)
    *   [Step 2: Download and Hash the Model](#step-2-download-and-hash-the-model)
    *   [Step 3: Locate Cache Directory](#step-3-locate-cache-directory)
    *   [Step 4: Build Cache Structure](#step-4-build-cache-structure)
    *   [Step 5: Test in Offline Mode](#step-5-test-in-offline-mode)
3.  [**Model Evaluation and Best Practices**](#model-evaluation-and-best-practices)
    *   [Understanding the Key Metrics](#understanding-the-key-metrics)
    *   [Diagnosing Overfitting with `learning_stats.csv`](#diagnosing-overfitting-with-learning_statscsv)
    *   [Analyzing Final Performance with `evaluation-results.csv`](#analyzing-final-performance-with-evaluation-resultscsv)
    *   [The Solution: Selecting the Best Snapshot](#the-solution-selecting-the-best-snapshot)

---

## Installation

### Recommended Pip-Based Install (Windows)

> This process targets **Python 3.10+** and the default **PyTorch engine** for DeepLabCut ≥3.0. It includes critical fixes for NumPy and Pandas compatibility.

**a) Create & activate a fresh conda environment**
```bash
conda create -n dlc python=3.10 -y
conda activate dlc
```

**b) Install PyTorch (pick the CUDA build for your GPU)**
Example for CUDA **12.4** (RTX 30xx/40xx series):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
*Note: If you need another CUDA version (e.g., 11.8), get the exact command from the [PyTorch website](https://pytorch.org/get-started/locally/).*

**c) Install PyTables using conda**
This is recommended on Windows to avoid ABI mismatches.
```bash
conda install -c conda-forge pytables==3.8.0 -y
```

**d) Fix NumPy / Pandas Compatibility (Critical!)**
DLC is not yet compatible with NumPy 2.x. Pinning versions is required.
```bash
pip install "numpy<2.0" "pandas>=2.2,<3.0"
```
Also fix potential numba/llvmlite conflicts:```bash
pip install "numba<0.60" "llvmlite<0.42" numexpr


**e) Install DeepLabCut with GUI support**
```bash
pip install "deeplabcut[gui]" : Gives not always DLC3

Safer is to install over:
pip install --upgrade --force-reinstall "deeplabcut[gui] @ git+https://github.com/DeepLabCut/DeepLabCut.git"
OR @3.0.0rc13, Note: Avoid @3.0.0rc12
pip install --force-reinstall "deeplabcut[gui] @ git+https://github.com/DeepLabCut/DeepLabCut.git@3.0.0rc13"

```
If you also need TensorFlow for legacy models, use:
```bash
pip install "deeplabcut[gui,tf]"
```

**f) Verify the Installation**
```python
import numpy, pandas, torch, deeplabcut
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("DeepLabCut import OK")
```

**g) Launch the GUI**
```bash
python -m deeplabcut
```
---

### Legacy TF 2.10 Install
If you specifically need the older TensorFlow 2.10 environment.

```bash
# 1. Create a conda environment with Python 3.9
conda create -n deeplabcut_env python=3.9 -y
conda activate deeplabcut_env

# 2. Install CUDA and cuDNN via conda-forge
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y

# 3. Install TensorFlow 2.10
pip install tensorflow==2.10.0

# 4. Install PyTorch (optional but recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 5. Install DeepLabCut with TF and GUI support
pip install "deeplabcut[gui,tf]"
```

### Troubleshooting

*   **DLL load failed (`_multiarray_umath`)**: Caused by NumPy ≥2.0.
    *   **Fix**: `pip install --force-reinstall "numpy<2.0" "pandas<3.0"`

*   **`llvmlite` / `numba` DLL errors**: Version mismatch.
    *   **Fix**: See step `d)` under the recommended install to pin compatible versions.

*   **Corrupted DLC install**:
    *   **Fix**:
        ```bash
        pip uninstall deeplabcut -y
        pip cache purge
        pip install "deeplabcut[gui]"
        ```
*   **GPU not used**: Driver or PyTorch/CUDA mismatch.
    *   **Test with**: `import torch; print(torch.cuda.is_available())`

[🔼 Back to top](#table-of-contents)

---

## Offline Model Caching

### The Problem: Network Errors
When firewalls, proxies, or network issues block access to the Hugging Face Hub, DeepLabCut fails to download pretrained models, causing `ConnectTimeout` or `LocalEntryNotFoundError` errors. This guide shows how to create a local cache to solve this. This example uses `timm/resnet50_gn.a1h_in1k`.

### Step 1: Gather Information
1.  **Model Name**: `timm/resnet50_gn.a1h_in1k`
2.  **Download URL**: `https://huggingface.co/timm/resnet50_gn.a1h_in1k/resolve/main/model.safetensors`
3.  **Full Commit Hash**: Find the latest commit on the model's Hugging Face "History" tab and copy the full hash (e.g., `b5b1dcf729e34c1143273175402a5102518f8e81`).
4.  **Model File SHA256 Hash**: To be calculated in the next step.

### Step 2: Download and Hash the Model
1.  Download `model.safetensors` from the URL above.
2.  Calculate its SHA256 hash.
    *   **Windows (PowerShell)**: `Get-FileHash C:\path\to\your\model.safetensors`
    *   **Linux/macOS**: `sha256sum /path\to\your/model.safetensors`

### Step 3: Locate Cache Directory
Find or create the Hugging Face cache directory. All subsequent work happens here.
*   **Windows**: `C:\Users\<Your-Username>\.cache\huggingface\hub\`
*   **Linux/macOS**: `~/.cache/huggingface/hub/`

### Step 4: Build Cache Structure
Build the precise folder structure the library expects.

1.  **Create the main model directory**: Convert `/` in the model name to `--`.
    *   `models--timm--resnet50_gn.a1h_in1k`
2.  **Store the model data in `blobs`**:
    *   Inside the model directory, create a `blobs` folder.
    *   Move `model.safetensors` into `blobs` and rename it to its **SHA256 hash** (with no file extension).
3.  **Create the `refs` pointer**:
    *   Create a `refs` folder. Inside it, create a text file named `main`.
    *   Paste the **Full Commit Hash** into this file.
4.  **Create the `snapshots` directory**:
    *   Create a `snapshots` folder. Inside it, create a new folder named with the **Full Commit Hash**.
5.  **Populate the snapshot directory**:
    *   Go into the commit-named folder (`snapshots/b5b1dcf.../`).
    *   **Copy** the hash-named file from `blobs` and paste it here.
    *   **Rename this copied file** back to `model.safetensors`.
6.  **Create final pointer files**:
    *   Inside the commit-named folder, create a `.huggingface` folder.
    *   Inside `.huggingface`, create a `blobs` folder.
    *   Inside this `blobs` folder, create a text file named `model.safetensors`.
    *   Paste the model's **SHA256 Hash** into this text file.

**Final Directory Structure Summary**
```text
.
└── models--timm--resnet50_gn.a1h_in1k/
    +-- blobs/
    |   └── <YOUR_FILE_SHA256_HASH>
    +-- refs/
    |   └── main
    └── snapshots/
        └── b5b1dcf729e34c1143273175402a5102518f8e81/
            +-- .huggingface/
            |   └── blobs/
            |       └── model.safetensors   (Text file with the SHA256 hash)
            └── model.safetensors           (The actual model weights file)
```

### Step 5: Test in Offline Mode
Force the library to use the local cache to verify your setup.
1.  Open your terminal.
2.  Set the environment variable:
    *   **Windows (cmd)**: `set HF_HUB_OFFLINE=1`
    *   **Linux/macOS**: `export HF_HUB_OFFLINE=1`
3.  In the same terminal, run your DeepLabCut command. Success is when training starts without network errors.

[🔼 Back to top](#table-of-contents)

---

## Model Evaluation and Best Practices

### Understanding the Key Metrics
Evaluating your model correctly is critical. Focus on metrics from the **test/evaluation set**, as they measure performance on unseen data.

| Column Name                         | What it Measures                           | What to Look For                  | Primary Use                                     |
| ----------------------------------- | ------------------------------------------ | --------------------------------- | ----------------------------------------------- |
| **`metrics/test.mAP`**              | Overall keypoint accuracy                  | **As HIGH as possible**           | **Finding the single best model snapshot**      |
| **`metrics/test.rmse`**             | Average pixel error on the test set        | As LOW as possible                | Understanding model precision in pixels         |
| **`losses/eval.total_loss`**        | Model's error on unseen data               | **Decrease then PLATEAU**         | **Identifying the point of peak learning**      |
| `losses/train.total_loss`           | Model's error on training data             | Consistent decrease               | Confirming the model is learning anything       |

### Diagnosing Overfitting with `learning_stats.csv`
After training, open `dlc-models/.../train/learning_stats.csv`. Plot `losses/train.total_loss` vs. `losses/eval.total_loss` to find the overfitting point.
*   **Good Training**: Both losses decrease together.
*   **Overfitting**: `train.total_loss` continues to decrease while `eval.total_loss` flattens or rises.

### Analyzing Final Performance with `evaluation-results.csv`
Run the "Evaluate Network" step to generate `evaluation-results.csv`. This provides a final report card. Compare the `train` and `test` metrics to diagnose overfitting.

> #### **Example Analysis**
>
> 1.  **Performance on the Training Set ("Memorized" Data)**
>     *   **`train rmse`**: 1.04 pixels
>     *   **`train mAP`**: 45.08
>
>     *Interpretation: Phenomenal performance. The model learned the training images with an average error of just **1 pixel**, proving it has sufficient capacity.*
>
> 2.  **Performance on the Test Set (New Data)**
>     *   **`test rmse`**: 5.98 pixels
>     *   **`test mAR`**: 0
>
>     *Interpretation: The model's real-world performance. The error balloons to nearly **6 pixels**, which is too high to be considered correct by the strict mAP metric. This large gap between train and test performance is undeniable proof of overfitting.*

### The Solution: Selecting the Best Snapshot
The final snapshot is likely overfit. You must manually select an earlier, better-performing snapshot.

1.  **Identify the Best Epoch**: Review `learning_stats.csv` and your loss plot. Find the epoch where `losses/eval.total_loss` or `metrics/test.rmse` was at its lowest point (e.g., epoch 90).
2.  **Manually Specify the Snapshot**: Open your project's `config.yaml` file. Find `snapshotindex` and change its value from `all` to your chosen epoch number.
    ```yaml
    # In config.yaml, change 'all' to the epoch number you chose
    # For example, to use the model from epoch 90:
    snapshotindex: 90
    ```
3.  **Re-evaluate**: Save the `config.yaml` and run "Evaluate Network" again. The new results should show a much smaller gap between train and test metrics, confirming you have a better, more generalized model.

[🔼 Back to top](#table-of-contents)
