import torch
import argparse

from models.MA3D import MA3D
from build_dataloader import get_dataloaders
from engine import validate


def get_args():
    parser = argparse.ArgumentParser("SFER Evaluation")

    parser.add_argument("--data_type", default="RAF-DB",
                        choices=["RAF-DB", "VKIST", "Cheo", "FerPlus", "Caers", "CheoFaMo"])
    parser.add_argument("--num_classes", type=int, default=7)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--ckpt_path", type=str, default="checkpoints/last.pth")

    return parser.parse_args()


def load_model(ckpt_path, device, num_classes):
    model = MA3D(num_classes=num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    loss, acc = validate(model, loader, criterion, device, 0, 1)
    return loss, acc


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, val_loader = get_dataloaders(args)

    model = load_model(args.ckpt_path, device, args.num_classes)

    loss, acc = evaluate(model, val_loader, device)

    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Loss: {loss:.4f}")
    print(f"Acc : {acc * 100:.2f}%")


if __name__ == "__main__":
    main()