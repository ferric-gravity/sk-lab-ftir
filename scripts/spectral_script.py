# If not installed
!pip install fastai

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score

from fastai.torch_core import *
from fastai.learner import Learner
from fastai.data.core import DataLoaders
import torch
import torch.nn as nn

from fastai.optimizer import OptimWrapper
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def adagrad_opt(params, lr=0.01):
    return OptimWrapper(params, optim.Adagrad, lr=lr)


patients = pd.read_csv("patients.csv")
control  = pd.read_csv("control.csv")

wn_pat = patients.iloc[:,0].values
wn_ctrl = control.iloc[:,0].values

assert np.allclose(wn_pat, wn_ctrl), "Wavenumber axes are NOT identical!"
wavenumbers = wn_pat

# spectra: transpose so each row = subject spectrum
X_pat = patients.iloc[:,1:].T.values
X_ctrl = control.iloc[:,1:].T.values

# labels
y_pat = np.ones(X_pat.shape[0])
y_ctrl = np.zeros(X_ctrl.shape[0])

# combine
X = np.vstack([X_pat, X_ctrl]).astype(float)
y = np.concatenate([y_pat, y_ctrl]).astype(int)

print("Subjects (patients):", X_pat.shape[0])
print("Subjects (control):", X_ctrl.shape[0])
print("Features per spectrum:", X.shape[1])

def sample_zscore(X):
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

Xn = sample_zscore(X)



class FNN4(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.SELU(),
            nn.Dropout(0.1),

            nn.Linear(400, 400),
            nn.SELU(),
            nn.Dropout(0.1),

            nn.Linear(400, 400),
            nn.SELU(),
            nn.Dropout(0.1),

            nn.Linear(400, 400),
            nn.SELU(),

            nn.Linear(400, 2)
        )
    def forward(self, x):
        return self.model(x)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using:", device)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=21)

accs, aucs, recalls, ppvs = [], [], [], []

for train_idx, test_idx in skf.split(Xn, y):
    
    Xtr = torch.tensor(Xn[train_idx], dtype=torch.float32)
    ytr = torch.tensor(y[train_idx], dtype=torch.long)

    Xts = torch.tensor(Xn[test_idx], dtype=torch.float32)
    yts = y[test_idx]

    dls = DataLoaders.from_dsets(
        TensorDataset(Xtr, ytr),
        TensorDataset(Xts, torch.tensor(yts)),
        bs=16, shuffle=True
    )

    model = FNN4(Xn.shape[1]).to(device)
    # learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(), opt_func=torch.optim.Adagrad)
    learn = Learner(
    dls,
    model,
    loss_func=nn.CrossEntropyLoss(),
    opt_func=adagrad_opt)
    learn.fit(40)

    preds = learn.get_preds(ds_idx=1)[0].softmax(dim=1).cpu().numpy()
    y_pred = np.argmax(preds, axis=1)

    accs.append(accuracy_score(yts, y_pred))
    aucs.append(roc_auc_score(yts, preds[:,1]))
    recalls.append(recall_score(yts, y_pred))
    ppvs.append(precision_score(yts, y_pred))


print("\n--- Neural Network Performance (Table-3 Equivalent) ---")
print("Accuracy:", np.mean(accs)*100)
print("AUC:", np.mean(aucs)*100)
print("Recall:", np.mean(recalls)*100)
print("PPV:", np.mean(ppvs)*100)


Xtorch = torch.tensor(Xn, dtype=torch.float32)
ytorch = torch.tensor(y, dtype=torch.long)

dls_full = DataLoaders.from_dsets(
    TensorDataset(Xtorch, ytorch),
    TensorDataset(Xtorch, ytorch),
    bs=16, shuffle=True
)

final_model = FNN4(Xn.shape[1]).to(device)
# learn_final = Learner(dls_full, final_model, loss_func=nn.CrossEntropyLoss(), opt_func=torch.optim.Adagrad)
learn_final = Learner(
    dls_full,
    final_model,
    loss_func=nn.CrossEntropyLoss(),
    opt_func=adagrad_opt
)
learn_final.fit(80)


perturb = np.linspace(-0.5, 0.5, 21)
ideal = torch.eye(2)[ytorch].to(device)
mse_resp = []

with torch.no_grad():
    base_out = learn_final.model(Xtorch.to(device)).softmax(dim=1)

for j in range(Xn.shape[1]):
    mvals = []
    for p in perturb:
        Xp = Xtorch.clone()
        Xp[:,j] += Xp[:,j] * p
        out = learn_final.model(Xp.to(device)).softmax(dim=1)
        mse = ((out - ideal)**2).mean().item()
        mvals.append(mse)
    mse_resp.append(np.mean(mvals))

mse_resp = np.array(mse_resp)


plt.figure(figsize=(10,4))
plt.plot(wavenumbers, mse_resp)
plt.gca().invert_xaxis()
plt.title("Neural Network Sensitivity — Figure-5 Equivalent")
plt.xlabel("Wavenumber (cm⁻1)")
plt.ylabel("Sensitivity (MSE Response)")
plt.grid()
plt.savefig('senstivity-plot.png')