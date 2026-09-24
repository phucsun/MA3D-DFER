import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="original checkpoint")
    parser.add_argument("--dst", type=str, required=True, help="output slim checkpoint")
    return parser.parse_args()


def main():
    args = get_args()

    ckpt = torch.load(args.src, map_location="cpu")

    # handle both old formats safely
    if isinstance(ckpt, dict) and "model" in ckpt:
        model_state = ckpt["model"]
    else:
        model_state = ckpt  # already a raw state_dict

    # keep same structure as training checkpoint
    slim_ckpt = {
        "model": model_state
    }

    torch.save(slim_ckpt, args.dst)

    print(f"Original keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'state_dict only'}")
    print(f"Saved slim checkpoint (with 'model' key) to: {args.dst}")


if __name__ == "__main__":
    main()