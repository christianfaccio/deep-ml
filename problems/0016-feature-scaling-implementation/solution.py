import torch

def feature_scaling(data) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standardize and Min-Max normalize input data using PyTorch.
    Input: Tensor or convertible of shape (m,n).
    Returns (standardized_data, normalized_data), both rounded to 4 decimals.
    """
    data_t = torch.as_tensor(data, dtype=torch.float)
    standardized_data = (data_t - torch.mean(data_t, dim=0)) / torch.std(data_t, dim=0, unbiased=False)
    normalized_data = (data_t - torch.min(data_t, dim=0).values) / (torch.max(data_t, dim=0).values - torch.min(data_t, dim=0).values)

    return (torch.round(standardized_data, decimals=4), torch.round(normalized_data, decimals=4))
