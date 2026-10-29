import torch

def precision_recall_curve(y_true: torch.Tensor, y_scores: torch.Tensor) -> tuple:
    """
    Compute precision-recall pairs for different probability thresholds.
    
    Args:
        y_true: Tensor of true binary labels (0 or 1)
        y_scores: Tensor of predicted probabilities or confidence scores
    
    Returns:
        Tuple of (precisions, recalls, thresholds) where each is a torch.Tensor
    """
    precisions = []
    recalls = []
    thresholds = torch.clone(y_scores)
    count_actual_pos = 0

    for i in range(y_scores.shape[0]):
        count_actual_pos += y_scores[i]
    
    for i in range(y_scores.shape[0]):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for j in range(y_true.shape[0]):
            if y_scores[j] >= y_scores[i]:
                if (y_true[j]):
                    TP += 1
                else:
                    FP += 1
            else:
                if (y_true[j]):
                    FN += 1
                else:
                    TN += 1
        
        if (TP == 0 and FP == 0):
            precisions.append(1.0)
        else:
            if (TP+FP) == 0:
                recalls.append(1.0)
            else:
                precisions.append(TP/(TP+FP))

        if count_actual_pos == 0:
            recalls.append(0.0)
        else:
            if (TP+FN) == 0:
                recalls.append(0.0)
            else:
                recalls.append(TP/(TP+FN))

    return (torch.as_tensor(precisions), torch.as_tensor(recalls), thresholds)