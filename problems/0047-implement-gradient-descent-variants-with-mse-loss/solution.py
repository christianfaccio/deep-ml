import torch

def gradient_descent(X: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, 
                    learning_rate: float, n_epochs: int, 
                    batch_size: int = 1, method: str = 'batch') -> torch.Tensor:
    """
    Implements three variants of gradient descent: Batch, Stochastic, and Mini-Batch.
    Uses Mean Squared Error (MSE) as the loss function.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights as a tensor
    """
    match method:
        case 'batch':
            for i in range(n_epochs):
                weights = weights - learning_rate * (2/X.shape[0]) * ((X @ weights - y) @ X)
        
        case 'stochastic':
            for j in range(n_epochs):            
                for i in range(X.shape[0]):
                    weights = weights - learning_rate * 2 * ((X[i,:] @ weights - y[i]) * X[i,:])

        case 'mini_batch':
            idxs = []
            for i in range(batch_size):
                idxs.append(i)
            for i in range(n_epochs):
                # skipping m=b check for simplicity
                for j in range(int(X.shape[0] / batch_size)):
                    actual_idxs = torch.as_tensor(idxs) + j*batch_size
                    X_b = torch.index_select(X,0,actual_idxs)
                    y_b = torch.index_select(y,0,actual_idxs)
                    weights = weights - learning_rate * (2/batch_size) * (( X_b @ weights - y_b) @ X_b) 
            
    return weights