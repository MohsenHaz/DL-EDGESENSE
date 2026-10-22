"""Generic edge deployment script."""
import argparse, torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["rpi4", "jetson", "custom"])
    parser.add_argument("--checkpoint"); parser.add_argument("--mode", default="standard")
    args = parser.parse_args()
    print(f"Deploying: platform={args.platform}, mode={args.mode}")

if __name__ == "__main__":
    main()
