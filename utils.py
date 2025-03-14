from collections import OrderedDict
import numpy as np
import torch
from matplotlib import pyplot as plt


def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    locked_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = total_params - locked_params
    trainable_param_names = [name for name, param in model.named_parameters() if param.requires_grad]
    
    print("Total Parameters: ", total_params)
    print("Locked Parameters (not trainable): ", locked_params)
    print("Trainable Parameters: ", trainable_params)
    print("Trainable Parameter Names:")
    for name in trainable_param_names:
        print(name)
        
def load_pretrain(model, path = "ssl_pretrained_weights.pth"):
    ssl_dict = torch.load(path)
    ssl_weights = ssl_dict["state_dict"]
    missing_keys, unexpected_keys = model.load_state_dict(ssl_weights, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    print_model_parameters(model)


def plot_metrics(val_interval, epoch_loss_values, 
                 metric_values, metric_values_tc, metric_values_wt, metric_values_et, 
                 caption="residual"):
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="red")
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="green")
    plt.savefig(f"results/{caption}_Loss.png")
    plt.close()

    plt.figure("train", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Val Mean Dice TC")
    x = [val_interval * (i + 1) for i in range(len(metric_values_tc))]
    y = metric_values_tc
    plt.xlabel("epoch")
    plt.plot(x, y, color="blue")
    plt.subplot(1, 3, 2)
    plt.title("Val Mean Dice WT")
    x = [val_interval * (i + 1) for i in range(len(metric_values_wt))]
    y = metric_values_wt
    plt.xlabel("epoch")
    plt.plot(x, y, color="brown")
    plt.subplot(1, 3, 3)
    plt.title("Val Mean Dice ET")
    x = [val_interval * (i + 1) for i in range(len(metric_values_et))]
    y = metric_values_et
    plt.xlabel("epoch")
    plt.plot(x, y, color="purple")
    plt.savefig(f"results/{caption}_Dice.png")
    plt.close()