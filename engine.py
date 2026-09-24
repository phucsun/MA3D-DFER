import torch
from tqdm import tqdm


def get_loss(logits, labels, CE_criterion, lsce_criterion, MA_criterion, epoch):
    CE_loss = CE_criterion(logits, labels)
    lsce_loss = lsce_criterion(logits, labels)
    MA_loss = MA_criterion(logits, labels)

    loss = 2 * lsce_loss + CE_loss
    if epoch < 10:
        loss = MA_loss

    return loss

def prepare_batch(batch, device):
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)

    npy = batch["npy"]
    exp = npy["exp"].to(device, non_blocking=True)
    detail = npy["detail"].to(device, non_blocking=True)
    pose = npy["pose"].to(device, non_blocking=True)
    shape = npy["shape"].to(device, non_blocking=True)
    tex = npy["tex"].to(device, non_blocking=True)

    x_3d = torch.cat([exp, detail, pose, shape, tex], dim=1)

    return images, labels, x_3d

def train_one_epoch(model, loader, CE_criterion, lsce_criterion, MA_criterion,
                    optimizer, device, epoch, epochs):

    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, batch in enumerate(
        tqdm(loader, desc=f"Training [{epoch + 1}/{epochs}]", leave=False)
    ):
        images, labels, x_3d = prepare_batch(batch, device)

        logits, features, attn = model(images, x_3d)
        loss = get_loss(logits, labels, CE_criterion, lsce_criterion, MA_criterion, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.first_step(zero_grad=True)


        logits_2, features_2, attn = model(images, x_3d)
        loss_2 = get_loss(logits_2, labels, CE_criterion, lsce_criterion, MA_criterion, epoch)
        loss_2.backward()
        optimizer.second_step(zero_grad=True)


        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, epochs):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc=f"Validation [{epoch + 1}/{epochs}]", leave=False):
        images, labels, x_3d = prepare_batch(batch, device)

        logits, features, attn = model(images, x_3d)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc