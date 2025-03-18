from collections import OrderedDict
import os
import time
from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)

from monai.utils import set_determinism
import torch
from configs import get_swinunetr, image_configs, frft_configs
from swinunetr import SwinUNETR
from unetr import UNETR
from utils import load_pretrain, plot_metrics
from transform import train_transform, val_transform
import argparse

parser = argparse.ArgumentParser(description="Test different SwinUNETR models with FRFT configurations.")
parser.add_argument("--model", type=str, choices=frft_configs.keys(), required=True, help="Specify the model type to test.")

args = parser.parse_args()
model_name = args.model


root_dir = "/data/datasets/MSD"

set_determinism(seed=0)

train_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=train_transform,
    section="training",
    download=False,
    cache_rate=0.0,
    num_workers=4,
)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

val_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=val_transform,
    section="validation",
    download=False,
    cache_rate=0.0,
    num_workers=4,
)

val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

max_epochs = 50
val_interval = 1
TRAIN_AMP = False
VAL_AMP = False


device = torch.device("cuda:1")
model = get_swinunetr(model_name).to(device)
# model = UNETR().to(device)
caption = model_name


load_pretrain(model)

loss_function = DiceCELoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-6, T_max=max_epochs, verbose=True)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])


# define inference method
def inference(input):
    with torch.cuda.amp.autocast(enabled=VAL_AMP):
        return sliding_window_inference(
            inputs=input,
            roi_size=image_configs["img_size"],
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

scaler = torch.cuda.amp.GradScaler(enabled=TRAIN_AMP)
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

best_metric = -1
best_metric_epoch = -1
best_metrics = OrderedDict({
    "metric": [],
    "epoch": [],
    "time": [],
})
epoch_loss_values = []
metric_records = {
    "mean_dice": [],
    "tumor_core": [],
    "whole_tumor": [],
    "enhancing_tumor": [],
}

total_start = time.time()
for epoch in range(max_epochs):
    epoch_start = time.time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=TRAIN_AMP):
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}"
            f", train_loss: {loss.item():.4f}"
            f", step time: {(time.time() - step_start):.4f}"
        )
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_outputs = inference(val_inputs)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)


            metric_batch = dice_metric_batch.aggregate()
            metric = dice_metric.aggregate().item()
            
            metric_records["mean_dice"].append(metric)
            metric_records["tumor_core"].append(metric_batch[0].item())
            metric_records["whole_tumor"].append(metric_batch[1].item())
            metric_records["enhancing_tumor"].append(metric_batch[2].item())
            
            dice_metric.reset()
            dice_metric_batch.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics["metric"].append(best_metric)
                best_metrics["epoch"].append(best_metric_epoch)
                best_metrics["time"].append(time.time() - total_start)
                torch.save(
                    model.state_dict(),
                    os.path.join(root_dir, "best_metric_model.pth"),
                )
                print("saved new best metric model")
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    plot_metrics(val_interval, epoch_loss_values, 
                 metric_records["mean_dice"], metric_records["tumor_core"], metric_records["whole_tumor"], metric_records["enhancing_tumor"], 
                 caption=caption)
total_time = time.time() - total_start

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")




