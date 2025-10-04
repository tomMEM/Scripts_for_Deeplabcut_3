# FILE: interactive_dlc3_training_FINAL_V2.py
# Purpose: Resume or start DeepLabCut 3.x PyTorch training interactively, safely, and efficiently.

import deeplabcut
import os
import yaml
import glob
import re
from pathlib import Path
from deeplabcut.utils import auxiliaryfunctions
import pandas as pd

def find_training_folders(project_path):
    pytorch_models_dir = project_path / "dlc-models-pytorch"
    if not pytorch_models_dir.is_dir():
        return []
    return list(pytorch_models_dir.glob("**/train"))

def get_training_folder_choice(folders):
    print("\nMultiple training runs found. Please choose which one to resume from:")
    for i, folder in enumerate(folders):
        rel_path = os.path.relpath(folder.parent, folder.parents[2])
        print(f"  [{i+1}] {rel_path}")
    while True:
        try:
            choice = int(input(f"Enter your choice (1-{len(folders)}): "))
            if 1 <= choice <= len(folders):
                return folders[choice-1]
        except ValueError:
            pass
        print("❌ Invalid input, try again.")

def find_snapshots(train_folder_path):
    print(f"🔍 Searching for snapshots in: {train_folder_path}")
    snapshot_pattern = os.path.join(train_folder_path, 'snapshot-*.pt')
    found_files = glob.glob(snapshot_pattern)
    snapshots = {}
    for f in found_files:
        match = re.search(r'snapshot-(\d+)', Path(f).name)
        if match:
            snapshots[int(match.group(1))] = f
    return dict(sorted(snapshots.items(), key=lambda item: item[0], reverse=True))

def get_snapshot_choice(snapshots):
    latest_epoch = max(snapshots.keys())
    while True:
        choice = input(f"\nLatest snapshot found: epoch {latest_epoch}. Resume from this one? (Y/n/list): ").lower().strip()
        if choice in ('y', 'yes', ''):
            return latest_epoch
        elif choice == 'list':
            print("\n--- Available Snapshots ---")
            for e in snapshots.keys(): print(f"  Epoch {e}")
            try:
                ep = int(input("Enter epoch to resume from: "))
                if ep in snapshots: return ep
            except ValueError:
                pass
            print("❌ Invalid choice.")
        elif choice in ('n', 'no', 'exit', 'quit'):
            return None

def run_interactive_training():
    # --- Project selection ---
    while True:
        project_input = input("Enter full path to your DLC project folder: ").strip().strip('"')
        project_path = Path(project_input).resolve()
        config_path = project_path / "config.yaml"
        if config_path.exists():
            print(f"✅ Found project: {config_path}")
            break
        else:
            print("❌ 'config.yaml' not found, try again.")

    # --- Locate training folders ---
    train_folders = find_training_folders(project_path)
    if not train_folders:
        print("⚠️ No previous training found — new training will be started.")
        train_folder_path = None
    elif len(train_folders) == 1:
        train_folder_path = train_folders[0]
        print(f"✅ Using training folder: {train_folder_path}")
    else:
        train_folder_path = get_training_folder_choice(train_folders)

    available_snapshots = find_snapshots(str(train_folder_path)) if train_folder_path else {}

    # ===============================================================
    # FRESH TRAINING
    # ===============================================================
    if not available_snapshots:
        print("\n🚀 Starting *fresh* training run.")
        total_epochs = int(input("How many epochs to train? (e.g., 500): "))
        save_interval = int(input("Save a snapshot every X epochs (e.g., 50): "))
        deeplabcut.train_network(str(config_path), maxiters=total_epochs, saveiters=save_interval)
        print("✅ Training started successfully.")
        return

    # ===============================================================
    # RESUME TRAINING
    # ===============================================================
    resume_epoch = get_snapshot_choice(available_snapshots)
    if resume_epoch is None:
        print("Exiting.")
        return

    chosen_snapshot_path = Path(available_snapshots[resume_epoch]).resolve()
    print(f"▶️ Resuming from: {chosen_snapshot_path}")

    additional_epochs = int(input("How many *additional* epochs? (e.g., 200): "))
    save_interval = int(input("Save a snapshot every X epochs (e.g., 50): "))
    
    pytorch_config_path = train_folder_path / "pytorch_config.yaml"
    if pytorch_config_path.exists():
        with open(pytorch_config_path, "r") as f:
            pt_cfg = yaml.safe_load(f)
    
        # ✅ Train only for the *new* epochs (not total cumulative)
        pt_cfg["train_settings"]["epochs"] = additional_epochs

        pt_cfg["runner"]["snapshots"]["save_epochs"] = save_interval
        pt_cfg["runner"]["snapshots"]["max_snapshots"] = 50
        pt_cfg["runner"]["snapshots"]["save_optimizer_state"] = True
        pt_cfg["runner"]["snapshots"]["save_best"] = False  # ensure regular incremental saves
        #pt_cfg["runner"]["snapshots"]["save_best"] = True
        pt_cfg["runner"]["snapshots"]["keep_best"] = False

        
    
        # ✅ Resume from the previous snapshot
        snapshot_base = Path(chosen_snapshot_path).with_suffix("")
        pt_cfg["runner"]["snapshots"]["resume_from"] = str(snapshot_base)
        pt_cfg["init_weights"] = str(chosen_snapshot_path)
    
        # ✅ Clean training defaults for Windows
        pt_cfg["train_settings"]["batch_size"] = 8
        pt_cfg["train_settings"]["display_iters"] = 1000
        pt_cfg["train_settings"]["dataloader_workers"] = 0  # <— keep only 0, prevents DLC reloading spam
    
        # ✅ Auto-tune learning rate for small datasets
        if "optimizer" in pt_cfg.get("runner", {}):
            lr = pt_cfg["runner"]["optimizer"]["params"].get("lr", 0.0005)
            if lr > 0.0003:
                pt_cfg["runner"]["optimizer"]["params"]["lr"] = 0.0003
                print("⚙️ Lowered learning rate to 0.0003 for small dataset stability.")
    
        # ✅ Write updated config
        with open(pytorch_config_path, "w") as f:
            yaml.dump(pt_cfg, f)
    
        print(f"✅ Updated pytorch_config.yaml → training for {additional_epochs} new epochs "
              f"(resuming from {resume_epoch}), saving every {save_interval} epochs.")
    else:
        print("⚠️ pytorch_config.yaml not found, proceeding with defaults.")
    
    print(f"⚙️ Resuming training from snapshot file: {snapshot_base}")


    # --- Update main config.yaml ---
    cfg = auxiliaryfunctions.read_config(str(config_path))
    cfg["init_weights"] = str(chosen_snapshot_path)
    auxiliaryfunctions.write_config(str(config_path), cfg)
    print("✅ Updated main config with 'init_weights'.")

    # --- Optional: record learning stats offset ---
    learning_stats = train_folder_path / "learning_stats.csv"
    if learning_stats.exists():
        try:
            df = pd.read_csv(learning_stats)
            if not df.empty and 'epoch' in df.columns:
                last_epoch = int(df['epoch'].max())
                print(f"📈 Found learning stats up to epoch {last_epoch}.")
        except Exception:
            pass

    # --- Environment setup for deterministic GPU behavior ---
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["DLC_SEED"] = "42"
    os.environ["HF_HUB_OFFLINE"] = "1"

    print("\n🚀 Launching resumed training...")
    #deeplabcut.train_network(str(config_path))
    deeplabcut.train_network(str(config_path), maxiters=additional_epochs, saveiters=save_interval)

    print("\n✅ Training process finished successfully.")

if __name__ == "__main__":
    run_interactive_training()
