# FILE: interactive_dlc3_training.py

import deeplabcut
import os
import sys
import yaml
import glob
import re
import subprocess
from pathlib import Path

def find_training_folders(project_path):
    pytorch_models_dir = project_path / "dlc-models-pytorch"
    if not pytorch_models_dir.is_dir():
        return []
    return list(pytorch_models_dir.glob("**/train"))

def get_training_folder_choice(folders):
    print("\nMultiple training runs found. Please choose which one to resume from:")
    for i, folder in enumerate(folders):
        relative_path = os.path.relpath(folder.parent, folder.parents[2])
        print(f"  [{i+1}] {relative_path}")
    while True:
        try:
            choice = int(input(f"Enter your choice (1-{len(folders)}): "))
            if 1 <= choice <= len(folders):
                return folders[choice-1]
            else:
                print(f"❌ Error: Please enter a number between 1 and {len(folders)}.")
        except ValueError:
            print("❌ Error: Please enter a valid number.")

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
        prompt = (f"\nLatest snapshot found is at epoch {latest_epoch}."
                  f"\nUse this one? (Y/n) or type 'list' to see all options: ")
        choice = input(prompt).lower().strip()
        if choice in ('y', 'yes', ''):
            return latest_epoch
        elif choice == 'list':
            print("\n--- Available Snapshots ---")
            for epoch in snapshots.keys():
                print(f"  Epoch: {epoch}")
            print("---------------------------")
            while True:
                try:
                    num_choice = int(input("Please type the epoch number you want to resume from: "))
                    if num_choice in snapshots:
                        return num_choice
                    else:
                        print(f"❌ Error: Snapshot for epoch {num_choice} not found.")
                except ValueError:
                    print("❌ Error: Please enter a valid number.")
        else:
            print("Exiting.")
            return None

def run_interactive_training():
    # --- Get DLC project path ---
    while True:
        project_path_input = input("Please enter the full path to your DLC project folder: ").strip()
        # Normalize path — remove quotes, smart quotes, and trailing slashes
        project_path_input = (
            project_path_input.strip()
            .strip('"')
            .strip("'")
            .replace("“", "")
            .replace("”", "")
            .rstrip("\\/")
        )
        project_path = Path(project_path_input).resolve()
        config_path = project_path / "config.yaml"
        if config_path.exists():
            print(f"✅ Project found: {config_path}")
            break
        else:
            print(f"❌ 'config.yaml' not found in the directory: {project_path}")


    # --- Find training folders ---
    all_train_folders = find_training_folders(project_path)
    if not all_train_folders:
        print("❌ Could not find any training folders in 'dlc-models-pytorch'.")
        return
    elif len(all_train_folders) == 1:
        train_folder_path = all_train_folders[0]
        relative_path = os.path.relpath(train_folder_path.parent, train_folder_path.parents[2])
        print(f"✅ Automatically selected the only training folder found: {relative_path}")
    else:
        train_folder_path = get_training_folder_choice(all_train_folders)

    # --- Find available snapshots ---
    available_snapshots = find_snapshots(str(train_folder_path))
    if not available_snapshots:
        print("⚠️ No snapshots found — this looks like a NEW training set.")
        print("➡️ Starting fresh training from scratch.")
    
        # Ask for total epochs and snapshot interval
        while True:
            try:
                total_epochs = int(input("How many total epochs would you like to train for? (e.g., 500): "))
                if total_epochs > 0:
                    break
            except ValueError:
                print("❌ Please enter a valid number.")
    
        while True:
            try:
                save_interval_epochs = int(input("Save a snapshot every X epochs (e.g., 10, 50): "))
                if save_interval_epochs > 0:
                    break
            except ValueError:
                print("❌ Please enter a valid number.")
    
        print("\n🚀 Launching *fresh* training run...")
        python_command = f"import deeplabcut; deeplabcut.train_network(r'{config_path}', maxiters={total_epochs}, saveiters={save_interval_epochs})"
        # Ensure deterministic behavior across runs
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        os.environ["PYTHONHASHSEED"] = "0"
        os.environ["DLC_SEED"] = "42"

        subprocess.run([sys.executable, "-c", python_command], check=True)
        print("\n✅ Training started successfully.")
        return

    resume_epoch_num = get_snapshot_choice(available_snapshots)
    if resume_epoch_num is None:
        return
        
    chosen_snapshot_path = available_snapshots[resume_epoch_num]
    print(f"▶️ Resuming from snapshot: {chosen_snapshot_path}")

    # --- Get training parameters ---
    while True:
        try:
            additional_epochs = int(input("How many *additional* epochs would you like to train for? "))
            if additional_epochs > 0:
                break
        except ValueError:
            print("❌ Please enter a valid number.")
            
    while True:
        try:
            save_interval_epochs = int(input("Save a snapshot every X epochs (e.g., 10, 50): "))
            if save_interval_epochs > 0:
                break
        except ValueError:
            print("❌ Please enter a valid number.")

    # --- Force update pytorch_config.yaml (the *real* training file) ---
    pytorch_config_path = train_folder_path / "pytorch_config.yaml"
    if pytorch_config_path.exists():
        with open(pytorch_config_path, "r") as f:
            pt_cfg = yaml.safe_load(f)
    
        # ✅ Compute total epochs (resume + additional)
        total_epochs = int(resume_epoch_num) + int(additional_epochs)
        pt_cfg.setdefault("train_settings", {})
        pt_cfg["train_settings"]["epochs"] = total_epochs
        pt_cfg.setdefault("runner", {}).setdefault("snapshots", {})
        pt_cfg["runner"]["snapshots"]["save_epochs"] = int(save_interval_epochs)
    
        # ✅ Ensure good training defaults
        pt_cfg["train_settings"]["batch_size"] = 8              # Fits most GPUs
        pt_cfg["train_settings"]["dataloader_workers"] = 2      # Faster loading
        pt_cfg["train_settings"]["display_iters"] = 100          # More feedback
    
        # ✅ Auto-tune learning rate for small datasets
        if "optimizer" in pt_cfg.get("runner", {}):
            lr = pt_cfg["runner"]["optimizer"]["params"].get("lr", 0.0005)
            if lr > 0.0003:
                pt_cfg["runner"]["optimizer"]["params"]["lr"] = 0.0003
                print("⚙️ Reduced learning rate to 0.0003 for small dataset stability.")
    
        # ✅ Write back changes
        with open(pytorch_config_path, "w") as f:
            yaml.dump(pt_cfg, f)
    
        print(f"✅ Updated {pytorch_config_path} → total epochs={total_epochs} "
              f"(resume {resume_epoch_num} + {additional_epochs}), "
              f"save every {save_interval_epochs} epochs")
    else:
        print(f"⚠️ Could not find {pytorch_config_path} (training may use default 200 epochs)")



   
    import pandas as pd
    
    # --- Adjust learning_stats.csv so logs append instead of restarting ---
    learning_stats_path = train_folder_path / "learning_stats.csv"
    if learning_stats_path.is_file():
        try:
            df = pd.read_csv(learning_stats_path)
            if not df.empty and 'epoch' in df.columns:
                last_logged_epoch = int(df['epoch'].max())
                print(f"📈 Found existing learning stats up to epoch {last_logged_epoch}.")
                
                # Save marker for offset
                offset_marker = train_folder_path / "epoch_offset.txt"
                with open(offset_marker, "w") as f:
                    f.write(str(last_logged_epoch))
                print(f"  - Set epoch offset marker to {last_logged_epoch}.")
            else:
                print("ℹ️ Learning stats file found but empty or missing 'epoch' column.")
        except Exception as e:
            print(f"⚠️ Could not read existing learning stats: {e}")
    
    
    # --- Update DLC config.yaml ---
    from deeplabcut.utils import auxiliaryfunctions
    cfg = auxiliaryfunctions.read_config(str(config_path))
    cfg['init_weights'] = chosen_snapshot_path
    auxiliaryfunctions.write_config(str(config_path), cfg)
    print(f"  - Main config updated with 'init_weights'.")
    
    print("\n🚀 Launching training process in OFFLINE mode...")
    my_env = os.environ.copy()
    my_env["HF_HUB_OFFLINE"] = "1"
    python_command = f"import deeplabcut; deeplabcut.train_network(r'{config_path}')"
    
    try:
        # Ensure deterministic behavior across runs
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        os.environ["PYTHONHASHSEED"] = "0"
        os.environ["DLC_SEED"] = "42"

        
        subprocess.run([sys.executable, "-c", python_command], check=True, env=my_env)
        print("\n✅ Training process finished successfully.")
    
        # --- Post-process learning_stats.csv to fix epoch numbers ---
        try:
            if learning_stats_path.is_file() and (train_folder_path / "epoch_offset.txt").is_file():
                with open(train_folder_path / "epoch_offset.txt", "r") as f:
                    offset = int(f.read().strip())
    
                df = pd.read_csv(learning_stats_path)
                if 'epoch' in df.columns:
                    df['epoch'] = df['epoch'] + offset
                    df.to_csv(learning_stats_path, index=False)
                    print(f"📈 Adjusted learning_stats.csv with offset {offset}.")
        except Exception as e:
            print(f"⚠️ Could not adjust learning_stats.csv: {e}")
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Training process exited with an error (Code: {e.returncode}).")
    except KeyboardInterrupt:
        print("\n⏹️ Training stopped by user.")


if __name__ == '__main__':
    run_interactive_training()
