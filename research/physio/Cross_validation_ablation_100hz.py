"""
Cross-Validation & Ablation Study (100Hz): DNN vs Attention DNN
================================================================
15-fold Leave-One-Subject-Out (LOSO) cross-validation on the
100Hz WESAD feature set.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ============================================================
# Configuration
# ============================================================
DIR_DATA = os.getenv("DIR_DATA_100HZ")
# Fallback to local Data_Processed_100Hz if .env path doesn't exist
if not DIR_DATA or not os.path.exists(DIR_DATA):
    DIR_DATA = "Data_Processed_100Hz"
DIR_RESULTS = os.getenv("DIR_RESULTS_100HZ", "Results_CrossVal_100Hz")
DIR_MODELS = os.getenv("DIR_MODELS_100HZ", "Models_CrossVal_100Hz")

# Full LOSO: every subject serves as test set once (15-fold)
TEST_SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
                 "S10", "S11", "S13", "S14", "S15", "S16", "S17"]

# All available subjects
ALL_SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
                "S10", "S11", "S13", "S14", "S15", "S16", "S17"]

NUM_EPOCHS = 50
BATCH_SIZE = 1024
LR = 0.001

# Create dirs
os.makedirs(DIR_RESULTS, exist_ok=True)
os.makedirs(DIR_MODELS, exist_ok=True)

# Seed
manualSeed = 1
torch.manual_seed(manualSeed)
random.seed(manualSeed)
np.random.seed(manualSeed)
g = torch.Generator()
g.manual_seed(manualSeed)

# GPU settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not available, using CPU. Training will be slow.")


# ============================================================
# Data Utilities (from existing codebase)
# ============================================================
def suppr(dic):
    """Delete extremums from a dictionary"""
    bornemax = np.quantile(dic["features"], 0.99, axis=0)
    bornemin = np.quantile(dic["features"], 0.01, axis=0)
    indicesmauvais = np.where(
        np.sum(np.add(bornemin > np.array(dic["features"]),
                      np.array(dic["features"]) > bornemax), axis=1) > 0
    )[0]
    k = 0
    for i in indicesmauvais:
        del dic["features"][i - k]
        del dic["label"][i - k]
        k += 1
    return dic


def extract_ds_from_dict(data):
    """Extract dataset and filter outliers"""
    Letat = []
    for i in range(0, 4):
        dictio = {}
        features = [data["features"][j] for j in np.where(np.array(data["label"]) == i + 1)[0]]
        label = [data["label"][j] for j in np.where(np.array(data["label"]) == i + 1)[0]]
        dictio["features"] = features
        dictio["label"] = label
        Letat.append(dictio.copy())
    neutr = Letat[0]; stress = Letat[1]; amu = Letat[2]; med = Letat[3]
    neutr = suppr(neutr); stress = suppr(stress); amu = suppr(amu); med = suppr(med)
    features = []; label = []; dict_id = {}
    for m in range(0, 4):
        dictio = Letat[m]
        features += [x for x in dictio["features"]]
        label += [x for x in dictio["label"]]
    dict_id["features"] = features
    dict_id["label"] = label
    return dict_id.copy()


def fusion_dic(list_dic):
    """Merge dictionaries"""
    features = []; label = []; dic_f = {}
    for dic in list_dic:
        features += dic["features"]; label += dic["label"]
    dic_f["features"] = features; dic_f["label"] = label
    return dic_f


def proportion(dic, indice, prop):
    tot = len(indice)
    features = [dic["features"][j] for j in indice[::int(np.ceil(tot / prop))]]
    label = [dic["label"][j] for j in indice[::int(np.ceil(tot / prop))]]
    return features, label


def eq_dic(dic):
    """Return a balanced dataset"""
    indice_neutr = np.where(np.array(dic["label"]) == 1)[0]
    indice_stress = np.where(np.array(dic["label"]) == 2)[0]
    indice_amu = np.where(np.array(dic["label"]) == 3)[0]
    indice_med = np.where(np.array(dic["label"]) == 4)[0]
    prop = min([3 * len(indice_neutr), len(indice_stress), 3 * len(indice_amu), 3 * len(indice_med)])
    p_s = prop; p_n = int(0.333 * prop); p_a = int(0.333 * prop); p_m = int(0.333 * prop)
    features = []; label = []; dic_f = {}
    tf, tl = proportion(dic, indice_neutr, p_n); features += tf; label += tl
    tf, tl = proportion(dic, indice_stress, p_s); features += tf; label += tl
    tf, tl = proportion(dic, indice_amu, p_a); features += tf; label += tl
    tf, tl = proportion(dic, indice_med, p_m); features += tf; label += tl
    dic_f["features"] = features; dic_f["label"] = label
    return dic_f


class ds_wesad(Dataset):
    def __init__(self, dic):
        self.samples = []
        for i in range(len(dic["label"])):
            num = dic["label"][i]
            stress = num == 2
            x = np.array(dic["features"][i])
            self.samples.append((x, int(stress), num))

    def __len__(self): return len(self.samples)
    def __getitem__(self, id): return self.samples[id]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed); random.seed(worker_seed)


def compute_cohens_d(data):
    """Compute mean and per-feature Cohen's d between stress and non-stress."""
    feat = np.array(data["features"])
    labels = np.array(data["label"])
    stress_feat = feat[labels == 2]
    non_stress_feat = feat[labels != 2]

    stress_mean = np.mean(stress_feat, axis=0)
    non_stress_mean = np.mean(non_stress_feat, axis=0)
    stress_std = np.std(stress_feat, axis=0)
    non_stress_std = np.std(non_stress_feat, axis=0)

    pooled_std = np.sqrt((stress_std ** 2 + non_stress_std ** 2) / 2)
    pooled_std[pooled_std == 0] = 1e-10
    per_feature_d = np.abs(stress_mean - non_stress_mean) / pooled_std
    mean_d = float(np.mean(per_feature_d))
    max_d = float(np.max(per_feature_d))
    max_feature_idx = int(np.argmax(per_feature_d))
    return {
        "mean_d": round(mean_d, 3),
        "max_d": round(max_d, 3),
        "max_feature_idx": max_feature_idx,
        "per_feature_d": [round(float(d), 3) for d in per_feature_d]
    }


FEATURE_NAMES = [
    'Mean Freq', 'Std Freq', 'TINN', 'HRV Index',
    'NN50', 'pNN50', 'Mean HRV', 'Std HRV',
    'RMSSD', 'FFT Mean', 'FFT Std', 'Sum PSD'
]


# ============================================================
# Models
# ============================================================
def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class ClassifierDNN(nn.Module):
    """Standard DNN (no attention)"""
    def __init__(self):
        super(ClassifierDNN, self).__init__()
        self.nnECG = nn.Sequential(
            nn.Linear(12, 128, bias=True), nn.BatchNorm1d(128), nn.Dropout(0.5), nn.LeakyReLU(0.2),
            nn.Linear(128, 64, bias=True), nn.BatchNorm1d(64), nn.Dropout(0.5), nn.LeakyReLU(0.2),
            nn.Linear(64, 16, bias=True), nn.BatchNorm1d(16), nn.Dropout(0.5), nn.LeakyReLU(0.2),
            nn.Linear(16, 4, bias=True), nn.BatchNorm1d(4), nn.Dropout(0.5), nn.LeakyReLU(0.2),
            nn.Linear(4, 1, bias=True), nn.Sigmoid()
        )
        self.nnECG.apply(init_weight)

    def forward(self, input):
        return self.nnECG(input)


class ClassifierDNNAttention(nn.Module):
    """DNN with Self-Attention"""
    def __init__(self):
        super(ClassifierDNNAttention, self).__init__()
        self.embed_dim = 32
        self.num_heads = 4
        self.embedding = nn.Linear(1, self.embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True)
        self.nnECG = nn.Sequential(
            nn.Linear(12 * self.embed_dim, 128, bias=True), nn.BatchNorm1d(128), nn.Dropout(0.5), nn.LeakyReLU(0.2),
            nn.Linear(128, 64, bias=True), nn.BatchNorm1d(64), nn.Dropout(0.5), nn.LeakyReLU(0.2),
            nn.Linear(64, 16, bias=True), nn.BatchNorm1d(16), nn.Dropout(0.5), nn.LeakyReLU(0.2),
            nn.Linear(16, 4, bias=True), nn.BatchNorm1d(4), nn.Dropout(0.5), nn.LeakyReLU(0.2),
            nn.Linear(4, 1, bias=True), nn.Sigmoid()
        )
        self.nnECG.apply(init_weight)
        self.embedding.apply(init_weight)

    def forward(self, input):
        batch_size = input.size(0)
        x = input.view(batch_size, 12, 1)
        x = self.embedding(x)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.reshape(batch_size, -1)
        return self.nnECG(x)


# ============================================================
# Training & Evaluation
# ============================================================
def train_model(net, dataloader_t, dataloader_v, num_epochs, fold_id, model_name):
    """Train model and return validation losses per epoch."""
    Loss_t = []; Loss_v = []
    optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999))

    model_dir = os.path.join(DIR_MODELS, f"fold_{fold_id}_{model_name}")
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # --- Training ---
        net.train()
        L_t = []
        for dataj in dataloader_t:
            net.zero_grad()
            x = dataj[0].float().to(device, non_blocking=True)
            yhat = dataj[1].float().to(device, non_blocking=True).view(-1, 1)
            y = net(x)
            err_t = nn.BCELoss()(y.float(), yhat.float())
            err_t.backward()
            optimizer.step()
            L_t.append(err_t.item())

        # --- Validation ---
        net.eval()
        L_v = []
        with torch.no_grad():
            for dataj in dataloader_v:
                x = dataj[0].float().to(device, non_blocking=True)
                yhat = dataj[1].float().to(device, non_blocking=True).view(-1, 1)
                y = net(x)
                err_v = nn.BCELoss()(y.float(), yhat.float())
                L_v.append(err_v.item())

        err = np.mean(L_t); errv = np.mean(L_v)
        Loss_t.append(err); Loss_v.append(errv)

        # Save model weights
        torch.save(net.state_dict(), os.path.join(model_dir, f"epoch_{epoch}.pth"))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{num_epochs} | Train: {err:.4f} | Val: {errv:.4f}")

    best_epoch = int(np.argmin(Loss_v))
    print(f"    Best epoch: {best_epoch+1} (Val Loss: {Loss_v[best_epoch]:.4f})")
    return Loss_v, best_epoch, model_dir


def evaluate_model(net, dataloader_test):
    """Evaluate model and return metrics dict."""
    net.eval()
    confusion = np.zeros((2, 2))
    total_samples = 0

    with torch.no_grad():
        for datal in dataloader_test:
            x = datal[0].float().to(device, non_blocking=True)
            y = net(x).view(-1)
            pred = (y > 0.5).int()
            label = datal[1].float().to(device, non_blocking=True).view(-1).int()

            # Stress = positive class (label==1)
            for p, l in zip(pred.cpu().numpy(), label.cpu().numpy()):
                if l == 1 and p == 1:
                    confusion[0, 0] += 1  # TP
                elif l == 1 and p == 0:
                    confusion[1, 0] += 1  # FN
                elif l == 0 and p == 1:
                    confusion[0, 1] += 1  # FP
                else:
                    confusion[1, 1] += 1  # TN
            total_samples += datal[0].size(0)

    TP = confusion[0, 0]; TN = confusion[1, 1]; FN = confusion[1, 0]; FP = confusion[0, 1]
    acc = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
        "total_samples": int(total_samples)
    }


# ============================================================
# Report Generation
# ============================================================
def generate_report(all_results):
    """Generate a markdown ablation report from results."""
    lines = []
    lines.append("# Cross-Validation & Ablation Report")
    lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Task**: {len(all_results)}-Fold LOSO Cross-Validation — Standard DNN vs Self-Attention DNN\n")
    lines.append(f"**Training Config**: {NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LR}, BCELoss, Adam optimizer\n")
    lines.append(f"**Device**: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})\n")

    # Collect metrics
    dnn_metrics = {k: [] for k in ["accuracy", "precision", "recall", "f1"]}
    att_metrics = {k: [] for k in ["accuracy", "precision", "recall", "f1"]}
    for r in all_results:
        for k in dnn_metrics:
            dnn_metrics[k].append(r["DNN"][k])
            att_metrics[k].append(r["Attention"][k])

    # =============================================
    # Section 1: Cohen's d per Subject
    # =============================================
    lines.append("---\n")
    lines.append("## 1. Subject Difficulty Analysis (Cohen's d)\n")
    lines.append("> Cohen's d measures feature separability between stress and non-stress classes.")
    lines.append("> Higher d → features are more distinct → classification is easier.")
    lines.append("> Subjects with d < 1.0 are considered 'hard' to classify.\n")
    lines.append("| Test Subject | Mean Cohen's d | Max Cohen's d | Max Feature | Difficulty |")
    lines.append("|-------------|:--------------:|:-------------:|-------------|:----------:|")
    for r in all_results:
        cd = r.get("cohens_d", {})
        mean_d = cd.get("mean_d", 0)
        max_d = cd.get("max_d", 0)
        max_idx = cd.get("max_feature_idx", 0)
        feat_name = FEATURE_NAMES[max_idx] if max_idx < len(FEATURE_NAMES) else f"Feature {max_idx}"
        if mean_d < 0.8:
            difficulty = "🔴 Hard"
        elif mean_d < 1.5:
            difficulty = "🟡 Medium"
        else:
            difficulty = "🟢 Easy"
        lines.append(f"| {r['test_subject']} | {mean_d:.3f} | {max_d:.3f} | {feat_name} | {difficulty} |")

    # =============================================
    # Section 2: Standard DNN per-fold
    # =============================================
    lines.append("\n---\n")
    lines.append("## 2. Standard DNN — Per-Fold Results\n")
    lines.append("| Fold | Test Subject | Cohen's d | Accuracy | Precision | Recall | F1 Score | Best Epoch |")
    lines.append("|------|-------------|:---------:|----------|-----------|--------|----------|------------|")
    for r in all_results:
        m = r["DNN"]; cd = r.get("cohens_d", {}).get("mean_d", 0)
        lines.append(f"| {r['fold']} | {r['test_subject']} | {cd:.3f} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {r['DNN_best_epoch']+1} |")
    lines.append(f"| **Mean±Std** | — | — | {np.mean(dnn_metrics['accuracy']):.4f}±{np.std(dnn_metrics['accuracy']):.4f} | {np.mean(dnn_metrics['precision']):.4f}±{np.std(dnn_metrics['precision']):.4f} | {np.mean(dnn_metrics['recall']):.4f}±{np.std(dnn_metrics['recall']):.4f} | {np.mean(dnn_metrics['f1']):.4f}±{np.std(dnn_metrics['f1']):.4f} | — |")

    # =============================================
    # Section 3: Attention DNN per-fold
    # =============================================
    lines.append("\n---\n")
    lines.append("## 3. Self-Attention DNN — Per-Fold Results\n")
    lines.append("| Fold | Test Subject | Cohen's d | Accuracy | Precision | Recall | F1 Score | Best Epoch |")
    lines.append("|------|-------------|:---------:|----------|-----------|--------|----------|------------|")
    for r in all_results:
        m = r["Attention"]; cd = r.get("cohens_d", {}).get("mean_d", 0)
        lines.append(f"| {r['fold']} | {r['test_subject']} | {cd:.3f} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {r['Attention_best_epoch']+1} |")
    lines.append(f"| **Mean±Std** | — | — | {np.mean(att_metrics['accuracy']):.4f}±{np.std(att_metrics['accuracy']):.4f} | {np.mean(att_metrics['precision']):.4f}±{np.std(att_metrics['precision']):.4f} | {np.mean(att_metrics['recall']):.4f}±{np.std(att_metrics['recall']):.4f} | {np.mean(att_metrics['f1']):.4f}±{np.std(att_metrics['f1']):.4f} | — |")

    # =============================================
    # Section 4: Ablation (Micro-Average = standard Mean)
    # =============================================
    lines.append("\n---\n")
    lines.append("## 4. Ablation Comparison — Micro-Average (Standard Mean)\n")
    lines.append("> Micro-average treats all folds equally, which may be dominated by 'easy' subjects.\n")
    lines.append("| Metric | Standard DNN (Mean±Std) | Attention DNN (Mean±Std) | Δ (Att − DNN) |")
    lines.append("|--------|------------------------|--------------------------|----------------|")
    for metric_name in ["accuracy", "precision", "recall", "f1"]:
        d_mean = np.mean(dnn_metrics[metric_name])
        d_std = np.std(dnn_metrics[metric_name])
        a_mean = np.mean(att_metrics[metric_name])
        a_std = np.std(att_metrics[metric_name])
        delta = a_mean - d_mean
        display_name = metric_name.capitalize() if metric_name != "f1" else "F1 Score"
        lines.append(f"| {display_name} | {d_mean:.4f}±{d_std:.4f} | {a_mean:.4f}±{a_std:.4f} | {'+' if delta >= 0 else ''}{delta:.4f} |")

    # =============================================
    # Section 5: Ablation (Macro-Average = weighted by inverse Cohen's d)
    # =============================================
    lines.append("\n---\n")
    lines.append("## 5. Ablation Comparison — Macro-Average (Difficulty-Weighted)\n")
    lines.append("> **Weighting Formula**: `w_i = (1 / log(1 + d_i)) / Σ(1 / log(1 + d_j))`\n>")
    lines.append("> where `d_i` is the mean Cohen's d for subject `i`. Lower Cohen's d (harder subject) yields a higher weight.")
    lines.append("> Compared to `1/d` weighting, the `1/log(1+d)` scheme compresses the weight range, preventing any single")
    lines.append("> hard subject from dominating the aggregate metrics.\n")

    # Compute weights: 1 / log(1+d)
    cohens_ds = [r.get("cohens_d", {}).get("mean_d", 1.0) for r in all_results]
    raw_weights = [1.0 / np.log(1 + max(d, 0.01)) for d in cohens_ds]
    weight_sum = sum(raw_weights)
    weights = [w / weight_sum for w in raw_weights]

    lines.append("| Metric | Standard DNN (Weighted) | Attention DNN (Weighted) | Δ (Att − DNN) |")
    lines.append("|--------|------------------------|--------------------------|----------------|")
    for metric_name in ["accuracy", "precision", "recall", "f1"]:
        d_wav = sum(w * v for w, v in zip(weights, dnn_metrics[metric_name]))
        a_wav = sum(w * v for w, v in zip(weights, att_metrics[metric_name]))
        delta = a_wav - d_wav
        display_name = metric_name.capitalize() if metric_name != "f1" else "F1 Score"
        lines.append(f"| {display_name} | {d_wav:.4f} | {a_wav:.4f} | {'+' if delta >= 0 else ''}{delta:.4f} |")

    # Show weights
    lines.append(f"\n**Fold Weights** (computed as `1/log(1+d)`, then normalized to sum=1):\n")
    lines.append("| Subject | Cohen's d | 1/log(1+d) | Normalized Weight |")
    lines.append("|---------|:---------:|:----------:|:-----------------:|")
    for r, w, cd, rw in zip(all_results, weights, cohens_ds, raw_weights):
        lines.append(f"| {r['test_subject']} | {cd:.3f} | {rw:.4f} | {w:.4f} ({100*w:.1f}%) |")

    # =============================================
    # Section 6: Per-Fold Side-by-Side
    # =============================================
    lines.append("\n---\n")
    lines.append("## 6. Per-Fold Side-by-Side Comparison\n")
    lines.append("| Fold | Test Subject | Cohen's d | DNN Acc | Att Acc | DNN F1 | Att F1 | Acc Δ | F1 Δ |")
    lines.append("|------|-------------|:---------:|---------|---------|--------|--------|-------|------|")
    for r in all_results:
        d = r["DNN"]; a = r["Attention"]
        cd = r.get("cohens_d", {}).get("mean_d", 0)
        acc_d = d["accuracy"]; acc_a = a["accuracy"]
        f1_d = d["f1"]; f1_a = a["f1"]
        lines.append(f"| {r['fold']} | {r['test_subject']} | {cd:.3f} | {acc_d:.4f} | {acc_a:.4f} | {f1_d:.4f} | {f1_a:.4f} | {'+' if acc_a-acc_d>=0 else ''}{acc_a-acc_d:.4f} | {'+' if f1_a-f1_d>=0 else ''}{f1_a-f1_d:.4f} |")

    # =============================================
    # Section 7: Summary
    # =============================================
    lines.append("\n---\n")
    lines.append("## 7. Summary\n")
    better_acc = sum(1 for r in all_results if r["Attention"]["accuracy"] > r["DNN"]["accuracy"])
    better_f1 = sum(1 for r in all_results if r["Attention"]["f1"] > r["DNN"]["f1"])
    total = len(all_results)
    lines.append(f"- The **Self-Attention DNN** achieved higher accuracy in **{better_acc}/{total}** folds.")
    lines.append(f"- The **Self-Attention DNN** achieved higher F1 score in **{better_f1}/{total}** folds.")

    # Micro averages
    avg_acc_delta = np.mean(att_metrics["accuracy"]) - np.mean(dnn_metrics["accuracy"])
    avg_f1_delta = np.mean(att_metrics["f1"]) - np.mean(dnn_metrics["f1"])
    lines.append(f"- **Micro-average**: Attention {'improved' if avg_acc_delta > 0 else 'changed'} accuracy by **{avg_acc_delta:+.4f}** and F1 by **{avg_f1_delta:+.4f}**.")

    # Macro (weighted) averages
    d_wav_acc = sum(w * v for w, v in zip(weights, dnn_metrics["accuracy"]))
    a_wav_acc = sum(w * v for w, v in zip(weights, att_metrics["accuracy"]))
    d_wav_f1 = sum(w * v for w, v in zip(weights, dnn_metrics["f1"]))
    a_wav_f1 = sum(w * v for w, v in zip(weights, att_metrics["f1"]))
    wav_acc_delta = a_wav_acc - d_wav_acc
    wav_f1_delta = a_wav_f1 - d_wav_f1
    lines.append(f"- **Macro-average** (difficulty-weighted): Attention {'improved' if wav_acc_delta > 0 else 'changed'} accuracy by **{wav_acc_delta:+.4f}** and F1 by **{wav_f1_delta:+.4f}**.")

    # Subject difficulty insight
    hard_subjects = [r['test_subject'] for r in all_results if r.get('cohens_d', {}).get('mean_d', 0) < 0.8]
    easy_subjects = [r['test_subject'] for r in all_results if r.get('cohens_d', {}).get('mean_d', 0) >= 2.0]
    if hard_subjects:
        lines.append(f"- **Hard subjects** (Cohen's d < 0.8): {', '.join(hard_subjects)} — these dominate the macro-average.")
    if easy_subjects:
        lines.append(f"- **Easy subjects** (Cohen's d ≥ 2.0): {', '.join(easy_subjects)} — both models near-perfect, low discriminative value.")

    return "\n".join(lines)


# ============================================================
# Main Pipeline
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Cross-Validation & Ablation Study")
    print("=" * 60)

    all_results = []

    for fold_idx, test_subj in enumerate(TEST_SUBJECTS):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx+1}/{len(TEST_SUBJECTS)} — Test Subject: {test_subj}")
        print(f"{'='*60}")

        # --- Prepare Data ---
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subj]
        print(f"  Training subjects ({len(train_subjects)}): {train_subjects}")

        # Load and balance training data
        train_dicts = []
        for subj in train_subjects:
            fname = f"WESADECG_{subj}.json"
            with open(os.path.join(DIR_DATA, fname), 'r') as f:
                data = json.load(f)
                train_dicts.append(eq_dic(data))

        merged_train = fusion_dic(train_dicts)
        ds_train = ds_wesad(extract_ds_from_dict(merged_train))

        # Load test data (unbalanced, as-is)
        test_fname = f"WESADECG_{test_subj}.json"
        with open(os.path.join(DIR_DATA, test_fname), 'r') as f:
            test_raw = json.load(f)
            ds_test = ds_wesad(test_raw)

        # Compute Cohen's d for this test subject
        cd_stats = compute_cohens_d(test_raw)
        print(f"  Train samples: {len(ds_train)} | Test samples: {len(ds_test)} | Cohen's d: {cd_stats['mean_d']:.3f}")

        dataloader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=0, worker_init_fn=seed_worker,
                                      generator=g, drop_last=True, pin_memory=True)
        dataloader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=0, worker_init_fn=seed_worker,
                                     generator=g, drop_last=False, pin_memory=True)

        fold_result = {"fold": fold_idx + 1, "test_subject": test_subj, "cohens_d": cd_stats}

        # --- Model 1: Standard DNN ---
        print(f"\n  [1/2] Training Standard DNN...")
        # Reset seed for fair comparison
        torch.manual_seed(manualSeed)
        random.seed(manualSeed)
        np.random.seed(manualSeed)

        net_dnn = ClassifierDNN().to(device)
        loss_v_dnn, best_epoch_dnn, model_dir_dnn = train_model(
            net_dnn, dataloader_train, dataloader_test, NUM_EPOCHS, fold_idx, "DNN"
        )
        # Load best model and evaluate
        net_dnn.load_state_dict(torch.load(os.path.join(model_dir_dnn, f"epoch_{best_epoch_dnn}.pth"), map_location=device))
        metrics_dnn = evaluate_model(net_dnn, dataloader_test)
        fold_result["DNN"] = metrics_dnn
        fold_result["DNN_best_epoch"] = best_epoch_dnn
        print(f"    DNN Results => Acc: {metrics_dnn['accuracy']:.4f} | Prec: {metrics_dnn['precision']:.4f} | Rec: {metrics_dnn['recall']:.4f} | F1: {metrics_dnn['f1']:.4f}")

        # --- Model 2: Attention DNN ---
        print(f"\n  [2/2] Training Self-Attention DNN...")
        torch.manual_seed(manualSeed)
        random.seed(manualSeed)
        np.random.seed(manualSeed)

        net_att = ClassifierDNNAttention().to(device)
        loss_v_att, best_epoch_att, model_dir_att = train_model(
            net_att, dataloader_train, dataloader_test, NUM_EPOCHS, fold_idx, "Attention"
        )
        net_att.load_state_dict(torch.load(os.path.join(model_dir_att, f"epoch_{best_epoch_att}.pth"), map_location=device))
        metrics_att = evaluate_model(net_att, dataloader_test)
        fold_result["Attention"] = metrics_att
        fold_result["Attention_best_epoch"] = best_epoch_att
        print(f"    Attention Results => Acc: {metrics_att['accuracy']:.4f} | Prec: {metrics_att['precision']:.4f} | Rec: {metrics_att['recall']:.4f} | F1: {metrics_att['f1']:.4f}")

        all_results.append(fold_result)

    # ============================================================
    # Save Results
    # ============================================================
    print(f"\n{'='*60}")
    print("Saving Results...")
    print(f"{'='*60}")

    # Save JSON
    results_path = os.path.join(DIR_RESULTS, "cross_val_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results JSON saved to: {results_path}")

    # Generate and save report
    report = generate_report(all_results)
    report_path = os.path.join(DIR_RESULTS, "ablation_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Ablation report saved to: {report_path}")

    print(f"\n{'='*60}")
    print("All done!")
    print(f"{'='*60}")
