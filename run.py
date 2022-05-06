import sys
import torch
import torch.nn as nn
import torch.optim as optim
from model import Transformer4Rec, load_model_checkpoint
from dataset import DataHandler
from evaluator import Evaluator
from train_val import fit, predict
import torch
import torch.nn as nn
import torch.optim as optim


DATASET_PATH = "dataset/reddit_processed_split_mapped.pickle"
# DATASET_PATH = "dataset/delicious_processed_split_mapped.pickle"

# Config
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

# Experiment
dataset = DataHandler(DATASET_PATH, batch_size=256, cloze_proba=0.6, device=device)

# Training and testing
model = Transformer4Rec(num_users=dataset.num_users, num_items=dataset.items_embedding_size, dim_model=32, nhead=2, num_encoder_layers=2, layer_norm_eps=1e-05, dropout=0.0, padding_idx=dataset.padding_item, maxseqlen=20, device=device)
opt = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.padding_item)
evaluator = Evaluator(num_items=dataset.items_embedding_size, k_list=[5, 20], masking_token=dataset.masking_item)
fit(model, opt, loss_fn, evaluator, dataset, epochs=50, checkpoints=True, checkpoint_name="Transformer4Rec_Reddit", resume_from_checkpoint="models/Transformer4Rec_Reddit", scheduler=scheduler, testonly=True)


# Inference
model = Transformer4Rec(num_users=dataset.num_users, num_items=dataset.items_embedding_size, dim_model=32, nhead=2, num_encoder_layers=2, layer_norm_eps=1e-05, dropout=0.0, padding_idx=dataset.padding_item, maxseqlen=20, device=device)
predict([7, 8, 143, 185, 1, 50], model, dataset, model_checkpoint="models/Transformer4Rec_Reddit")
# ['news', 'worldnews', 'politics', 'science', 'AskReddit', 'pics'] 
