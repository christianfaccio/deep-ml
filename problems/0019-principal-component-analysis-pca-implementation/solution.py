import torch

def pca(data, k) -> torch.Tensor:
    """
    Perform PCA on `data`, returning the top `k` principal components as a tensor.
    Input: Tensor or convertible of shape (n_samples, n_features).
    Returns: a torch.Tensor of shape (n_features, k), with floats rounded to 4 decimals.
    Note: If an eigenvector's first non-zero value is negative, flip its sign.
    """
    data = torch.as_tensor(data, dtype=torch.float32)

    means = torch.mean(data, dim=0)
    stds = torch.std(data, dim=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = (data[i][j] - means[j]) / stds[j]

    cov_mat = torch.cov(data.T)     # transpose since 
                                    # cov takes variables
                                    # as rows
    vals, vecs = torch.linalg.eigh(cov_mat)

    for i in range(vecs.shape[1]):
        for elem in vecs[:,i]:
            if torch.abs(elem) > 1e-10 and elem < 0:
                vecs[:,i] *= -1
                break 

    idxs = []
    for i in range(1,k+1):
        idxs.append(vals.shape[0]-i)
    
    return torch.index_select(vecs, 1, torch.as_tensor(idxs))