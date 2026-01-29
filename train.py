from numpy import True_
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================
# Triplet Loss
# =========================
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, feats):
        bs = feats.shape[0] // 2
        anchor, positive = feats[:bs], feats[bs:]
        pos_dist = torch.norm(anchor - positive, dim=-1)
        neg_dist = torch.cdist(anchor, positive).min(dim=-1)[0]  # hardest negative
        loss = F.relu(pos_dist - neg_dist + self.margin).mean()
        return loss


# =========================
# Combined Loss: BCE + Triplet
# =========================
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.02):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.triplet = TripletLoss()

    def forward(self, logits, feats, targets):
        loss_ce = self.bce(logits, targets)
        loss_triplet = self.triplet(feats)
        return loss_ce, self.alpha * loss_triplet


# =========================
# Training Function
# =========================
def train(loader, model, optimizer, scheduler=None, device=device, epoch=0, alpha=0.02):
    model.train()
    criterion = CombinedLoss(alpha=alpha)
    all_preds = []
    all_labels = []
    total_loss = 0

    for step, (ninput, nlabel, ainput, alabel) in tqdm(enumerate(loader), total=len(loader)):
        optimizer.zero_grad()

        # Concatenate normal + abnormal inputs
        inputs = torch.cat([ninput, ainput], 0).to(device)
        labels = torch.cat([nlabel, alabel], 0).float().to(device)

        # Forward pass
        logits, feats = model(inputs)
        feats = F.normalize(feats, dim=-1)  # Normalize for triplet loss

        # Compute loss
        loss_ce, loss_triplet = criterion(logits.squeeze(), feats, labels)
        loss = loss_ce + loss_triplet

        # Backprop
        loss.backward()
        optimizer.step()

        # Optional: scheduler step per batch if it's a step-based scheduler
        if scheduler is not None and hasattr(scheduler, "_step_count"):  # OneCycleLR etc.
            scheduler.step()

        total_loss += loss.item()

        # Save predictions for AUC
        all_preds.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    # Step scheduler per epoch if it requires epoch
    if scheduler is not None and not hasattr(scheduler, "_step_count"):
        scheduler.step(epoch)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    pr_auc = auc(recall, precision)

    print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f} | ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f}")
    return total_loss / len(loader)


# =========================
# Validation Function
# =========================
def validate(loader, model, device=device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for ninput, nlabel, ainput, alabel in loader:
            inputs = torch.cat([ninput, ainput], 0).to(device)
            labels = torch.cat([nlabel, alabel], 0).float().to(device)
            logits, _ = model(inputs)
            all_preds.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    pr_auc = auc(recall, precision)
    print(f"Validation | ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f}")
    return roc_auc, pr_auc
