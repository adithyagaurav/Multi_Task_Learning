import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from utils import Normalise, RandomCrop, ToTensor, RandomMirror
from dataset import HydranetDataset
from torch.utils.data import DataLoader
from Hydranet import Hydranet
from utils import InvHuberLoss
from utils import AverageMeter
from utils import MeanIoU, RMSE
from tqdm import tqdm

num_classes = (1, 40)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
crop_size=400
img_scale = 1.0 / 255
depth_scale = 5000.0

img_mean = np.array([0.485, 0.456, 0.406])
img_std = np.array([0.229, 0.224, 0.225])
transform_train = transforms.Compose([RandomMirror(),
                                      RandomCrop(crop_size=crop_size),
                                      Normalise(scale=img_scale, mean=img_mean.reshape((1,1,3)), std=img_std.reshape(((1,1,3))), depth_scale=depth_scale),
                                      ToTensor()])
transform_valid = transforms.Compose([Normalise(scale=img_scale, mean=img_mean.reshape((1,1,3)), std=img_std.reshape(((1,1,3))), depth_scale=depth_scale),
                                      ToTensor()])

train_batch_size = 4
valid_batch_size = 4
train_file = "train_list_depth.txt"
valid_file = "val_list_depth.txt"
print("[INFO]: Loading data")
trainloader = DataLoader(HydranetDataset(data_file=train_file, transform=transform_train),
                         batch_size=train_batch_size,
                         shuffle=True, num_workers=4,
                         drop_last=True)
valloader = DataLoader(HydranetDataset(data_file=valid_file, transform=transform_valid),
                       batch_size=valid_batch_size,
                       shuffle=False, num_workers=4,
                       drop_last=False)
print("[INFO]: Loading model")
hydranet = Hydranet(2,40)
ckpt = torch.load("mobilenetv2-e6e8dd43.pth", map_location='cpu')
hydranet.enc.load_state_dict(ckpt)
print("[INFO]: Model has {} parameters".format(sum([p.numel() for p in hydranet.parameters()])))
print("[INFO]: Model and weights loaded successfully")
for param in hydranet.enc.parameters():
    param.requires_grad=False

ignore_index = 255
ignore_depth = 0

crit_segm = nn.CrossEntropyLoss(ignore_index=ignore_index).to(device)
crit_depth = InvHuberLoss(ignore_index=ignore_depth).to(device)

lr_encoder = 1e-2
lr_decoder = 1e-3
momentum_encoder = 0.9
momentum_decoder = 0.9
weight_decay_encoder = 1e-5
weight_decay_decoder = 1e-5
n_epochs = 1000

optims = [torch.optim.SGD(hydranet.enc.parameters(), lr=lr_encoder, momentum=momentum_encoder, weight_decay=weight_decay_encoder),
          torch.optim.SGD(hydranet.dec.parameters(), lr=lr_decoder, momentum=momentum_decoder, weight_decay=weight_decay_decoder)]

opt_scheds = []
for opt in optims:
    opt_scheds.append(torch.optim.lr_scheduler.MultiStepLR(opt, np.arange(0, n_epochs, 100), gamma=0.1))


def train(model, opts, crits, dataloader, loss_coeffs=(1.0,), grad_norm=0.0):
    model.train()
    loss_meter = AverageMeter()
    pbar = tqdm(dataloader)

    for sample in pbar:
        loss = 0.0
        image = sample["image"].float().to(device)
        targets = [sample[k].to(device) for k in dataloader.dataset.mask_names]
        output = model(image)

        for out, target, crit, loss_coeff in zip(output, targets, crits, loss_coeffs):
            loss += loss_coeff * crit(F.interpolate(out, target.size()[1:], mode="bilinear", align_corners=False).squeeze(dim=1),
                                      target.squeeze(dim=1))

        for opt in opts:
            opt.zero_grad()
        loss.backward()
        if grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        for opt in opts:
            opt.step()
        loss_meter.update(loss.item())
        pbar.set_description(
            "Loss {:.3f} | Avg. Loss {:.3f}".format(loss.item(), loss_meter.avg)
        )


def validate(model, metrics, dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model.eval()
    for metric in metrics:
        metric.reset()

    pbar = tqdm(dataloader)

    def get_val(metrics):
        results = [(m.name, m.val()) for m in metrics]
        names, vals = list(zip(*results))
        out = ["{} : {:4f}".format(name, val) for name, val in results]
        return vals, " | ".join(out)

    with torch.no_grad():
        for sample in pbar:
            # Get the Data
            input = sample["image"].float().to(device)
            targets = [sample[k].to(device) for k in dataloader.dataset.masks_names]

            #input, targets = get_input_and_targets(sample=sample, dataloader=dataloader, device=device)
            targets = [target.squeeze(dim=1).cpu().numpy() for target in targets]

            # Forward
            outputs = model(input)
            #outputs = make_list(outputs)

            # Backward
            for out, target, metric in zip(outputs, targets, metrics):
                metric.update(
                    F.interpolate(out, size=target.shape[1:], mode="bilinear", align_corners=False)
                    .squeeze(dim=1)
                    .cpu()
                    .numpy(),
                    target,
                )
            pbar.set_description(get_val(metrics)[1])
    vals, _ = get_val(metrics)
    print("----" * 5)
    return vals


val_every = 5
loss_coeffs = (0.5, 0.5)
print("[INFO]: Start Training")
for i in range(0, n_epochs):
    for sched in opt_scheds:
        sched.step(i)

    print("Epoch {:d}".format(i))
    train(hydranet, optims, [crit_depth, crit_segm], trainloader, loss_coeffs)

    if i % val_every == 0:
        metrics = [MeanIoU(num_classes[1]), RMSE(ignore_val=ignore_depth)]
        with torch.no_grad():
            vals = validate(hydranet, metrics, valloader)

    print("Save Checkpoint")
    torch.save(hydranet.state_dict(), "chekcpoint.pth")