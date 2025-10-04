from ruamel.yaml import YAML
from pathlib import Path

config_file = Path(r"C:\Users\thomas\users\2P_Feb_Social\Feb-Thomas-2025-10-03\config.yaml")

yaml = YAML()
with open(config_file, "r", encoding="utf-8") as f:
    cfg = yaml.load(f)

# Fix crop values
for v in cfg["video_sets"].values():
    if isinstance(v["crop"], list):  # convert list → string
        v["crop"] = ",".join(str(x) for x in v["crop"])

with open(config_file, "w", encoding="utf-8") as f:
    yaml.dump(cfg, f)

print("🔧 Fixed crop fields back to strings.")