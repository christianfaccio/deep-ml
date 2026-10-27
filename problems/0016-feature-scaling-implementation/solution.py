import torch

def feature_scaling(data) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standardize and Min-Max normalize input data using PyTorch.
    Input: Tensor or convertible of shape (m,n).
    Returns (standardized_data, normalized_data), both rounded to 4 decimals.
    """
    data_t = torch.as_tensor(data, dtype=torch.float32)

    # Standardization (using population std: unbiased=False)
    mean = torch.mean(data_t, dim=0)
    std = torch.std(data_t, dim=0, unbiased=False)
    standardized_data = (data_t - mean) / std

    # Min-Max Normalization
    min_val = torch.min(data_t, dim=0).values
    max_val = torch.max(data_t, dim=0).values
    normalized_data = (data_t - min_val) / (max_val - min_val)

    return (torch.round(standardized_data, decimals=4), torch.round(normalized_data, decimals=4))