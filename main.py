"""
Brain Tumor Detection - entry point.

Usage:
python main.py --data_dir ./data --epochs 5 --img_size 224 --batch_size 16 --lr 1e-3
"""
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Brain Tumor Detection")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()

def main():
    args = parse_args()
    print("[INFO] Starting pipeline with:", vars(args))
    print("[TODO] Load data from", args.data_dir)
    print("[TODO] Build model")
    print("[TODO] Train/Evaluate model")
    print("[DONE] Placeholder run complete.")

if __name__ == "__main__":
    main()
