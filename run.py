import torch
import torch.nn as nn
import torch.optim as optim
from model import WIP
from dataset import DataHandler
from evaluator import Evaluator
from train_val import fit


# DATASET_PATH = "dataset/delicious.pkl"
DATASET_PATH = "dataset/reddit.pkl"

# Config
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

# Experiment
dataset = DataHandler(DATASET_PATH, batch_size=64, n_friends=5, limit_friend_sesscnt=5, limit_hist_sesscnt=5, device=device)
PADDING_ITEM = dataset.padding_item
NUM_ITEMS = dataset.num_items + 1

model = WIP(dim_model=32, num_users=dataset.num_users, num_items=NUM_ITEMS, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dropout=0.2, padding_idx=PADDING_ITEM, device=device)
opt = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-6)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
loss_fn = nn.CrossEntropyLoss(ignore_index=PADDING_ITEM)
evaluator = Evaluator(num_items=NUM_ITEMS, k_list=[5, 20], padding_idx=PADDING_ITEM)

# Execute
fit(model, opt, loss_fn, evaluator, dataset, epochs=50, checkpoints=True, checkpoint_name="WIP_v2_REDDIT", scheduler=scheduler)