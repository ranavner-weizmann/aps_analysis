import torch
import numpy as np

def avail_mask(t):
    # for each feature for each station, we will check if the whole feature is missing or just runtime error.
    is_nan_tensor = torch.isnan(t)
    all_nan_features_mask = torch.all(is_nan_tensor, dim=(0, 1, 3))
    # nan_feature_indices = torch.nonzero(all_nan_features_mask, as_tuple=True)[0]

    return all_nan_features_mask