"""Deploy DL-EgSense to edge devices."""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--platform", choices=["rpi4", "jetson_nano"], required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--mode", default="standard")
args = parser.parse_args() if __name__ == "__main__" else None

if args:
    print(f"Deploying to {args.platform} in {args.mode} mode from {args.checkpoint}")
