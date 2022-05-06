import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import progressbar
from model import load_model_checkpoint


def calc_cloze_loss(inputs, targets, predictions, loss_fn, mask_token):
  predictions = predictions.reshape(predictions.size(0)*predictions.size(1), -1) # [SEQ LEN * BATCH, NUM ITEMS]
  targets = targets.reshape(-1) # [BATCH * SEQ LEN]
  mask = inputs.reshape(-1) == mask_token
  return loss_fn(predictions[mask], targets[mask])


def train_loop(model, opt, loss_fn, dataset):
    model.train()
    losses = []

    dataset.reset_train_batch()
    num_batches = dataset.get_num_remain_batches()
    batch_idx = 0
    bar = progressbar.ProgressBar(max_value=num_batches)

    while True:
        batch_users, cur_sess, cur_sess_targets, cur_sess_len, key_mask = dataset.get_next_train_batch()
        c_batch_size = len(batch_users)
        if c_batch_size == 0:
            break
        
        pred = model(cur_sess, cur_sess_len, key_mask) # [SEQ LEN, BATCH, NUM ITEMS]
        loss = calc_cloze_loss(cur_sess, cur_sess_targets, pred, loss_fn, dataset.masking_item)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # Gradient Clipping
        opt.step()
        losses.append(loss.detach().item())

        bar.update(batch_idx)
        batch_idx += 1

    return np.mean(losses)


def validation_loop(model, loss_fn, evaluator, dataset):
    model.eval()
    losses = []
    evaluator.initialize()

    dataset.reset_val_batch()
    num_batches = dataset.get_num_remain_batches()
    batch_idx = 0
    bar = progressbar.ProgressBar(max_value=num_batches)
    
    with torch.no_grad():
        while True:
            batch_users, cur_sess, cur_sess_targets, cur_sess_len, key_mask = dataset.get_next_val_batch()
            c_batch_size = len(batch_users)
            if c_batch_size == 0:
                break

            pred = model(cur_sess, cur_sess_len, key_mask) # [SEQ LEN, BATCH, NUM ITEMS]
            loss = calc_cloze_loss(cur_sess, cur_sess_targets, pred, loss_fn, dataset.masking_item)
            losses.append(loss.detach().item())
            evaluator.evaluate_batch(cur_sess, cur_sess_targets, pred.permute(1, 0, 2))

            bar.update(batch_idx)
            batch_idx += 1

    return np.mean(losses), evaluator.get_stats()


def test_loop(model, loss_fn, evaluator, dataset):
    model.eval()
    losses = []
    evaluator.initialize()

    dataset.reset_test_batch()
    num_batches = dataset.get_num_remain_batches()
    batch_idx = 0
    bar = progressbar.ProgressBar(max_value=num_batches)
    
    with torch.no_grad():
        while True:
            batch_users, cur_sess, cur_sess_targets, cur_sess_len, key_mask = dataset.get_next_test_batch()
            c_batch_size = len(batch_users)
            if c_batch_size == 0:
                break

            pred = model(cur_sess, cur_sess_len, key_mask) # [SEQ LEN, BATCH, NUM ITEMS]
            loss = calc_cloze_loss(cur_sess, cur_sess_targets, pred, loss_fn, dataset.masking_item)
            losses.append(loss.detach().item())
            evaluator.evaluate_batch(cur_sess, cur_sess_targets, pred.permute(1, 0, 2))

            bar.update(batch_idx)
            batch_idx += 1

    return np.mean(losses), evaluator.get_stats()


def test(model, loss_fn, evaluator, dataset):
  print("Testing Model")
  test_loss, test_results = test_loop(model, loss_fn, evaluator, dataset)
  print(test_results)
  print(f"Test loss: {test_loss:.4f}")


def fit(model, opt, loss_fn, evaluator, dataset, epochs=5, checkpoints=True, checkpoint_name="", resume_from_checkpoint=None, scheduler=None, testonly=False):
    resume_from_epoch = 0
    if resume_from_checkpoint != None:
        checkpoint = torch.load(resume_from_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        resume_from_epoch = checkpoint['epoch']

    if not testonly:
        print("Training model and Validating Model")

        for epoch in range(resume_from_epoch, epochs):
            print("-"*25, f"Epoch {epoch}","-"*25)
        
            train_loss = train_loop(model, opt, loss_fn, dataset)
            print(f"\nTraining loss: {train_loss:.4f}\n")

            eval_loss, eval_results = validation_loop(model, loss_fn, evaluator, dataset)
            print(eval_results)
            print(f"Validation loss: {eval_loss:.4f}")

            if checkpoints:
                checkpoint_path = f"checkpoints/{checkpoint_name}_{time.strftime('%Y-%m-%d-%H:%M:%S')}"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'train_loss': train_loss,
                    'eval_loss': eval_loss,
                    'eval_results': eval_results
                }, checkpoint_path)
                print(f"\nModel at Epoch {epoch} saved at {checkpoint_path}\n\n")
            
            if scheduler is not None:
                scheduler.step(eval_loss)

    test(model, loss_fn, evaluator, dataset)


def predict(items, model, dataset, model_checkpoint, k=10):
    model = load_model_checkpoint(model, model_checkpoint)
    items.append(dataset.masking_item)
    sess, sesslens, mask = dataset.generate_prediction_data(items)
    target_mask = sess == dataset.masking_item
    with torch.no_grad():
        pred = model(sess, sesslens, mask)
        pred = pred.permute(1, 0, 2)
        pred = pred[target_mask]
    result = pred[0]
    result = torch.topk(result, k+10)[1]
    dup_mask = torch.vstack([result != i for i in items]).prod(0).bool()
    result = result[dup_mask][:k]
    result = dataset.item_to_name(result)
    print(result)