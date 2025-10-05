# FILE: dlc3_v5.py
# Purpose: Resume or start DeepLabCut 3.x PyTorch training interactively, safely, and efficiently.
# Version: v5, GUI-compatible, and resilient against snapshot rename crashes.

import deeplabcut
import os
import yaml
import glob
import re
from pathlib import Path
from deeplabcut.utils import auxiliaryfunctions
import pandas as pd

def find_training_folders(project_path):
    """Finds all 'train' directories within the dlc-models-pytorch structure."""
    pytorch_models_dir = project_path / "dlc-models-pytorch"
    if not pytorch_models_dir.is_dir():
        return []
    return list(pytorch_models_dir.glob("**/train"))

def get_training_folder_choice(folders):
    """Prompts the user to select a training folder if multiple exist."""
    print("\nMultiple training runs found. Please choose which one to resume from:")
    for i, folder in enumerate(folders):
        rel_path = os.path.relpath(folder.parent, folder.parents[2])
        print(f"  [{i+1}] {rel_path}")
    while True:
        try:
            choice = int(input(f"Enter your choice (1-{len(folders)}): "))
            if 1 <= choice <= len(folders):
                return folders[choice - 1]
        except ValueError:
            pass
        print("❌ Invalid input, try again.")

def find_snapshots(train_folder_path):
    """Finds and sorts all snapshots in a given training folder."""
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
    """Asks the user to select a snapshot to resume from."""
    if not snapshots:
        return None
    latest_epoch = max(snapshots.keys())
    while True:
        choice = input(f"\nLatest snapshot found: epoch {latest_epoch}. Resume from this one? (Y/n/list): ").lower().strip()
        if choice in ('y', 'yes', ''):
            return latest_epoch
        elif choice == 'list':
            print("\n--- Available Snapshots ---")
            for e in sorted(snapshots.keys()):
                print(f"  Epoch {e}")
            try:
                ep = int(input("Enter epoch to resume from: "))
                if ep in snapshots:
                    return ep
            except ValueError:
                pass
            print("❌ Invalid choice.")
        elif choice in ('n', 'no', 'exit', 'quit'):
            return None

def run_interactive_training():
    """Main function to run the interactive training script."""
    # --- Project selection ---
    while True:
        project_input = input("Enter the full path to your DLC project folder: ").strip().strip('"')
        project_path = Path(project_input).resolve()
        config_path = project_path / "config.yaml"
        if config_path.exists():
            print(f"✅ Found project: {config_path}")
            break
        else:
            print("❌ 'config.yaml' not found. Please try again.")

    # --- Locate training folders ---
    train_folders = find_training_folders(project_path)
    train_folder_path = None
    if not train_folders:
        print("⚠️ No previous training found — a new training will be started.")
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
        print("\n🚀 Starting a *fresh* training run.")
        total_epochs = int(input("How many epochs to train? (e.g., 500): "))
        save_interval = int(input("Save a snapshot every X epochs (e.g., 50): "))

        deeplabcut.train_network(
            config=str(config_path),
            shuffle=1,
            maxiters=total_epochs,
            saveiters=save_interval,
            displayiters=1000,
        )
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

    additional_epochs = int(input("How many *additional* epochs do you want to train? (e.g., 200): "))
    save_interval = int(input("Save a new snapshot every X epochs (e.g., 50): "))

    # --- Lower learning rate for continued training if necessary ---
    pytorch_config_path = train_folder_path / "pytorch_config.yaml"
    if pytorch_config_path.exists():
        with open(pytorch_config_path, "r") as f:
            pt_cfg = yaml.safe_load(f)
        
        if "optimizer" in pt_cfg.get("runner", {}):
            current_lr = pt_cfg["runner"]["optimizer"]["params"].get("lr", 0.0005)
            if current_lr > 0.0003:
                pt_cfg["runner"]["optimizer"]["params"]["lr"] = 0.0003
                print("⚙️ Lowered learning rate to 0.0003 for continued training stability.")
        
        with open(pytorch_config_path, "w") as f:
            yaml.dump(pt_cfg, f)
            print("✅ Updated pytorch_config.yaml with a potentially lower learning rate.")

    total_epochs = resume_epoch + additional_epochs

    print(f"\n⚙️ Training for {additional_epochs} new epochs (up to a total of {total_epochs}). Saving every {save_interval} epochs.")
    print("\n🚀 Launching resumed training...")

    # Pass snapshot_path directly to train_network to resume training
    deeplabcut.train_network(
        config=str(config_path),
        shuffle=1,
        maxiters=total_epochs,
        saveiters=save_interval,
        displayiters=1000,
        snapshot_path=str(chosen_snapshot_path),
    )

    print("\n✅ Training process finished successfully.")

if __name__ == "__main__":
    run_interactive_training()