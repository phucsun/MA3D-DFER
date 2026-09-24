import torch
from tqdm import tqdm


def get_loss(logits, labels, CE_criterion, lsce_criterion, MA_criterion, epoch):
    CE_loss = CE_criterion(logits, labels)
    lsce_loss = lsce_criterion(logits, labels)

    return 2 * lsce_loss + CE_loss


def prepare_video_batch(batch, device, use_3dmm: bool):
    """
    Chuyển batch từ VideoDataset lên device.

    Returns:
        video:    (B, T, 3, H, W)
        labels:   (B,)
        lengths:  (B,)
        video_3d: (B, T, 334) hoặc None
    """
    video = batch["video"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    lengths = batch["lengths"].to(device, non_blocking=True)

    video_3d = None
    if use_3dmm and "video_3d" in batch:
        video_3d = batch["video_3d"].to(device, non_blocking=True)

    return video, labels, lengths, video_3d


def train_one_epoch_video(
    model,
    loader,
    CE_criterion,
    lsce_criterion,
    MA_criterion,
    optimizer,
    device,
    epoch,
    epochs,
    use_3dmm: bool,
    use_amp: bool = False,
):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    is_cuda = device != "cpu" and torch.cuda.is_available()

    pbar = tqdm(
        loader,
        desc=f"  Train {epoch + 1:>3}/{epochs}",
        leave=False,
        unit="batch",
        dynamic_ncols=True,
    )

    for batch in pbar:
        video, labels, lengths, video_3d = prepare_video_batch(batch, device, use_3dmm)

        # SAM first step
        with torch.amp.autocast("cuda", enabled=use_amp and is_cuda):
            logits, _ = model(video, video_3d, seq_lengths=lengths)
            loss = get_loss(logits, labels, CE_criterion, lsce_criterion, MA_criterion, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # SAM second step (tại vị trí đã perturb)
        with torch.amp.autocast("cuda", enabled=use_amp and is_cuda):
            logits_2, _ = model(video, video_3d, seq_lengths=lengths)
            loss_2 = get_loss(logits_2, labels, CE_criterion, lsce_criterion, MA_criterion, epoch)

        loss_2.backward()
        optimizer.second_step(zero_grad=True)

        B = labels.size(0)
        running_loss += loss.item() * B
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += B

        postfix = {
            "loss": f"{running_loss / total:.4f}",
            "acc":  f"{correct / total * 100:.1f}%",
        }
        if is_cuda:
            postfix["vram"] = f"{torch.cuda.memory_allocated() / 1e9:.1f}G"
        pbar.set_postfix(postfix)

    pbar.close()
    return running_loss / total, correct / total


@torch.no_grad()
def validate_video(model, loader, criterion, device, epoch, epochs, use_3dmm: bool,
                   use_amp: bool = False):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    is_cuda = device != "cpu" and torch.cuda.is_available()

    pbar = tqdm(
        loader,
        desc=f"    Val {epoch + 1:>3}/{epochs}",
        leave=False,
        unit="batch",
        dynamic_ncols=True,
    )

    for batch in pbar:
        video, labels, lengths, video_3d = prepare_video_batch(batch, device, use_3dmm)

        with torch.amp.autocast("cuda", enabled=use_amp and is_cuda):
            logits, _ = model(video, video_3d, seq_lengths=lengths)
            loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{running_loss / total:.4f}",
            "acc":  f"{correct / total * 100:.1f}%",
        })

    pbar.close()
    return running_loss / total, correct / total
