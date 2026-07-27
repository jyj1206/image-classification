import argparse
from pathlib import Path

from utils import (
    load_train_config,
    prepare_checkpoint_dir,
    prepare_result_dir,
    save_result_config,
)
from utils.utils_dist import (
    cleanup_distributed,
    init_distributed,
    is_main_process,
)

from builds import build_dataloader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,  required=True)
    parser.add_argument('--resume', default=None, description='path to latest checkpoint or result directory')
    return parser.parse_args()

def main():
    args = parse_args()
    config_path = Path(args.config)
    resume = Path(args.resume) if args.resume else None

    config, result_dir = load_train_config(config_path, resume)
    dist_info = init_distributed(config)

    try:
        result_dir = prepare_result_dir(config, result_dir)
        checkpoint_dir = prepare_checkpoint_dir(result_dir, config)
        save_result_config(config, result_dir)

        if is_main_process():
            print(f"Result directory: {result_dir}")
            print(f"Checkpoint directory: {checkpoint_dir}")
            print(f"Device: {dist_info['device']}")
            
        train_loader, val_loader, test_loader = build_dataloader(config)
        
        
        
        

        
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
