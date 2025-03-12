from collections import OrderedDict
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
    monai_loadable_state_dict = OrderedDict()
    model_prior_dict = model.state_dict()
    model_update_dict = model_prior_dict

    for key, value in ssl_weights.items():
        if key[:8] == "encoder.":
            if key[8:19] == "patch_embed":
                new_key = "swinViT." + key[8:]
            else:
                new_key = "swinViT." + key[8:18] + key[20:]
            monai_loadable_state_dict[new_key] = value
        else:
            monai_loadable_state_dict[key] = value

    model_update_dict.update(monai_loadable_state_dict)
    model.load_state_dict(model_update_dict, strict=False)
    model_final_loaded_dict = model.state_dict()

    # Safeguard test to ensure that weights got loaded successfully
    layer_counter = 0
    for k, _v in model_final_loaded_dict.items():
        if k in model_prior_dict:
            layer_counter = layer_counter + 1

            old_wts = model_prior_dict[k]
            new_wts = model_final_loaded_dict[k]

            old_wts = old_wts.to("cpu").numpy()
            new_wts = new_wts.to("cpu").numpy()
            diff = np.mean(np.abs(old_wts, new_wts))
            print("Layer {}, the update difference is: {}".format(k, diff))
            if diff == 0.0:
                print("Warning: No difference found for layer {}".format(k))
                
            model_final_loaded_dict[k].requires_grad = False
    print("Total updated layers {} / {}".format(layer_counter, len(model_prior_dict)))
    print("Pretrained Weights Succesfully Loaded !")
    
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
    plt.savefig(f"{caption}_Loss.png")
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
    plt.savefig(f"{caption}_Dice.png")
    plt.close()