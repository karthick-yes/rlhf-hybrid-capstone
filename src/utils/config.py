import yaml
import argparse
import os


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/base_config.yaml", help="Path to the base config file")
    parser.add_argument("--ablation", type=str, default=None, help= "Path to ablation overrise config")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    with open(args.config, "r") as f:
        config= yaml.safe_load(f)
    
    if args.ablation:
        if not os.path.exists(args.ablation):
            raise FileNotFoundError(f"Ablation file not found: {args.ablation}")
        with open(args.ablation, 'r') as f:
            override = yaml.safe_load(f)
            config.update(override)
            print(f"!!! ABLATION APPLIED: {args.ablation} !!!")
    
    return config