import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet34_Weights, resnet34

from .data import IGNORE


STRUCTURES = {
    "fundus": {"od": (1, 2), "oc": (2,)},
    "polyp": {"polyp": (1,)},
}
NUM_CLASSES = {"fundus": 3, "polyp": 2}


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.convs = conv_block(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.convs(torch.cat((x, skip), dim=1))


class ResUNet34(nn.Module):
    def __init__(self, imagenet_weights=False, *, num_classes=3):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if imagenet_weights else None
        encoder = resnet34(weights=weights)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1, self.layer2 = encoder.layer1, encoder.layer2
        self.layer3, self.layer4 = encoder.layer3, encoder.layer4
        self.decode3 = DecoderBlock(512, 256, 256)
        self.decode2 = DecoderBlock(256, 128, 128)
        self.decode1 = DecoderBlock(128, 64, 64)
        self.decode0 = DecoderBlock(64, 64, 32)
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, image):
        x0 = self.stem(image)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.decode3(x4, x3)
        x = self.decode2(x, x2)
        x = self.decode1(x, x1)
        x = self.decode0(x, x0)
        return F.interpolate(self.head(x), size=image.shape[-2:], mode="bilinear", align_corners=False)


def image_tensor(rgb, normalization="imagenet"):
    image = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)
    if normalization == "minmax":
        low, high = torch.aminmax(image)
        return (image - low) / (high - low).clamp_min(1e-6)
    if normalization != "imagenet":
        raise ValueError(f"Unknown image normalization: {normalization}")
    mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    return (image - mean) / std


def segmentation_loss(logits, target):
    probs = logits.softmax(dim=1)
    valid = target != IGNORE
    if logits.shape[1] == 2:
        pred = probs[:, 1:2]
        actual = target[:, None] == 1
    else:
        pred = torch.stack((probs[:, 1] + probs[:, 2], probs[:, 2]), dim=1)
        actual = torch.stack(((target == 1) | (target == 2), target == 2), dim=1)
    pred = pred * valid[:, None]
    intersection = (pred * actual).sum(dim=(-2, -1))
    total = pred.sum(dim=(-2, -1)) + actual.sum(dim=(-2, -1))
    dice = (2 * intersection + 1e-6) / (total + 1e-6)
    return F.cross_entropy(logits, target, ignore_index=IGNORE) + 1 - dice.mean()


def structure_masks(prediction, target, groups):
    valid = target != IGNORE
    for labels in groups:
        pred = torch.zeros_like(valid)
        actual = torch.zeros_like(valid)
        for label in labels:
            pred |= prediction == label
            actual |= target == label
        yield pred & valid, actual


def binary_overlap(prediction, target):
    return (prediction & target).sum().item(), (prediction | target).sum().item()


def overlap_counts(prediction, target, groups=tuple(STRUCTURES["fundus"].values())):
    return [binary_overlap(pred, actual) for pred, actual in structure_masks(prediction, target, groups)]


def dice_score(intersection, union):
    return 2 * intersection / (intersection + union) if union else 1.0


def iou_score(intersection, union):
    return intersection / union if union else 1.0


def image_dice(prediction, target, groups=tuple(STRUCTURES["fundus"].values())):
    return [dice_score(intersection, union) for intersection, union in overlap_counts(prediction, target, groups)]
