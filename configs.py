from swinunetr import SwinUNETR


image_configs = {
        "img_size": (96, 96, 96),
        "in_channels": 4,
        "out_channels": 3,
        "feature_size": 12,  # Corrected key for feature_size
    }

# Define FRFT transformation configurations
frft_configs = {
    "normal": {"dims": None},
    "frft_1_trainable": {"orders": [0.5], "dims": [-4], "trainable": True},
    "frft_1_fixed": {"orders": [0.5], "dims": [-4], "trainable": False},
    "frft_5_trainable": {"orders": [0.1, 0.3, 0.5, 0.7, 0.9], "dims": [-4], "trainable": True},
    "frft_5_fixed": {"orders": [0.1, 0.3, 0.5, 0.7, 0.9], "dims": [-4], "trainable": False},
    
    "spatial_frft_1_fixed": {"orders": [0.5], "dims": [-3, -2, -1], "trainable": False},
    "spatial_frft_1_trainable": {"orders": [0.5], "dims": [-3, -2, -1], "trainable": True},
    "spatial_frft_5_trainable": {"orders": [0.1, 0.3, 0.5, 0.7, 0.9], "dims": [-3, -2, -1], "trainable": True},
    "spatial_frft_5_fixed": {"orders": [0.1, 0.3, 0.5, 0.7, 0.9], "dims": [-3, -2, -1], "trainable": False},
    
    "mixed_frft_1_fixed": {"orders": [0.5], "dims": [-4, -3, -2, -1], "trainable": False},
    "mixed_frft_1_trainable": {"orders": [0.5], "dims": [-4, -3, -2, -1], "trainable": True},
    "mixed_frft_5_trainable": {"orders": [0.1, 0.3, 0.5, 0.7, 0.9], "dims": [-4, -3, -2, -1], "trainable": True},
    "mixed_frft_5_fixed": {"orders": [0.1, 0.3, 0.5, 0.7, 0.9], "dims": [-4, -3, -2, -1], "trainable": False},
}

# Function to initialize a SwinUNETR model
def get_swinunetr(model_type):
    if model_type in frft_configs:
        return SwinUNETR(**image_configs, frft_config=frft_configs[model_type])
    else:
        raise ValueError(f"Undefined model type: {model_type}")
