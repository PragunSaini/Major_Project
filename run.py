import torch
import torch.nn as nn
import torch.optim as optim
from model import WIP
from dataset import DataHandler
from evaluator import Evaluator
from train_val import fit


DATASET_PATH = "dataset/delicious.pkl"
# DATASET_PATH = "dataset/reddit.pkl"

# Config
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

# Experiment
dataset = DataHandler(DATASET_PATH, batch_size=4, device=device)
PADDING_ITEM = dataset.padding_item
START_TOKEN = dataset.start_token
NUM_ITEMS = dataset.num_items + 2

model = WIP(hidden_size=32, num_users=dataset.num_users, num_items=NUM_ITEMS, dropout=0.2, padding_idx=PADDING_ITEM, device=device)
opt = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-6)
loss_fn = nn.CrossEntropyLoss(ignore_index=PADDING_ITEM)
evaluator = Evaluator(num_items=NUM_ITEMS, k_list=[5, 20], padding_idx=PADDING_ITEM)

# Execute
# fit(model, opt, loss_fn, evaluator, dataset, epochs=100, checkpoints=True, checkpoint_name="WIP_reddit")

dataset.reset_train_batch()
while True:
    batch_users, X, y_input, y_expected, src_key_mask, target_key_mask, target_mask, cur_sess_len, hist_sess, hist_sess_key_mask, hist_sizes, frds_sess = dataset.get_next_train_batch()
    pred = model(X, y_input, target_mask=target_mask, src_pad_mask=src_key_mask, target_pad_mask=target_key_mask, hist_sess=hist_sess, hist_sess_pad_mask=hist_sess_key_mask, hist_sizes=hist_sizes)
    print(pred)
