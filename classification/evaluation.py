import torch
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC, BinarySpecificity
import matplotlib.pyplot as plt
import datetime
import os
import enlighten
from tqdm import tqdm
from networks.ssm_nnMamba import nnMambaEncoder

def evaluate_model(device_in, uuid, ld_helper):
    device = device_in
    manager = enlighten.get_manager()
    
    log_path = os.path.join("..", "logs", f"{uuid}.txt")
    os.makedirs(os.path.join("..", "logs"), exist_ok=True)

    with open(log_path, 'a' if os.path.exists(log_path) else 'w') as filein:
        filein.write(
            f"\n==========================\n"
            f"===== Log for camull =====\n"
            f"==========================\n"
            f"----- Date: {datetime.datetime.now():%Y-%m-%d_%H:%M:%S} -----\n\n\n"
        )

        path = os.path.join("..", "weights", ld_helper.get_task_string(), uuid, "best_weight.pth")
        model = nnMambaEncoder().to(device)  # Move to device first
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()  # Set model to evaluation mode
        test_dl = ld_helper.get_test_dl(0)

        graph_path = os.path.join("..", "graphs", uuid)
        os.makedirs(graph_path, exist_ok=True)

        try:
            # get_roc_auc returns 9 items in total
            metrics = get_roc_auc(model, test_dl, True, graph_path, 1)
            # Unpack the 9 metrics
            accuracy, sensitivity, specificity, precision, f1, auroc, youdens_max, threshold, avg_auroc = metrics

            filein.write(
                f"=====   Fold 1  =====\n"
                f"Threshold {threshold:.4f}\n"
                f"--- Accuracy     : {accuracy:.4f}\n"
                f"--- Sensitivity  : {sensitivity:.4f}\n"
                f"--- Specificity  : {specificity:.4f}\n"
                f"--- Precision    : {precision:.4f}\n"
                f"--- F1           : {f1:.4f}\n"
                f"--- Youdens stat : {youdens_max:.4f}\n\n"
                f"(Variable Threshold)\n"
                f"--- ROC AUC      : {avg_auroc:.4f}\n\n"
            )
        except Exception as e:
            filein.write(f"Error during evaluation: {str(e)}\n")
            raise

def get_roc_auc(model_in, test_dl, figure=False, path=None, fold=1):
    model_in.eval()  # Ensure model is in evaluation mode
    thresholds = torch.linspace(0, 1, 11)  # Increased number of threshold points for better accuracy
    metrics_list = []
    
    try:
        for thresh in tqdm(thresholds):
            metrics = get_metrics(model_in, test_dl, thresh.item())
            metrics_list.append(metrics)
    
        tpr = [m[1] for m in metrics_list]  # sensitivity
        fpr = [1 - m[2] for m in metrics_list]  # 1 - specificity
        
        youdens = [sens + spec - 1 for _, sens, spec, _, _, _ in metrics_list]
        best_idx = max(range(len(youdens)), key=lambda i: youdens[i])
        
        if figure:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, 'darkorange', lw=2, 
                     label=f'ROC curve (area = {metrics_list[best_idx][5]:.4f})')
            plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - Fold {fold}')
            plt.legend(loc="lower right")
            
            if path:
                plt.savefig(os.path.join(path, f"auc-fold{fold}-{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.png"))
                plt.close()
            else:
                plt.savefig(os.path.join("..", "graphs", f"auc-{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.png"))
                plt.close()

        # Return 9 items total:
        # (best accuracy, best recall=best sensitivity, best specificity, best precision, best f1,
        #  best auroc, best youdens, best threshold, average auroc)
        return (
            *metrics_list[best_idx],
            youdens[best_idx],
            thresholds[best_idx].item(),
            sum(m[5] for m in metrics_list) / len(metrics_list)
        )
    
    except Exception as e:
        print(f"Error in get_roc_auc: {str(e)}")
        raise

def get_metrics(model_in, test_dl, thresh=0.5):
    device = next(model_in.parameters()).device
    
    metrics = {
        'accuracy': BinaryAccuracy(threshold=thresh).to(device),
        'recall': BinaryRecall(threshold=thresh).to(device),
        'specificity': BinarySpecificity(threshold=thresh).to(device),
        'precision': BinaryPrecision(threshold=thresh).to(device),
        'f1': BinaryF1Score(threshold=thresh).to(device),
        'auroc': BinaryAUROC().to(device)
    }
    
    try:
        with torch.no_grad():
            for batch in test_dl:
                X = batch['mri'].to(device)
                y = batch['label'].to(device)
                preds = model_in(X).sigmoid()
                
                for metric in metrics.values():
                    metric.update(preds, y)
        
        return (
            metrics['accuracy'].compute().item(),
            metrics['recall'].compute().item(),
            metrics['specificity'].compute().item(),
            metrics['precision'].compute().item(),
            metrics['f1'].compute().item(),
            metrics['auroc'].compute().item()
        )
    
    except Exception as e:
        print(f"Error in get_metrics: {str(e)}")
        raise