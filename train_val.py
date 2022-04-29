import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import progressbar


def train_loop(model, opt, loss_fn, dataset):
    model.train()
    losses = []

    dataset.reset_train_batch()
    num_batches = dataset.get_num_remain_batches()
    batch_idx = 0
    bar = progressbar.ProgressBar(max_value=num_batches)

    while True:
        batch_users, X, y, target_key_mask, target_mask, cur_sess_len, hist_sess, hist_sess_key_mask, hist_sizes, friend_sess, friend_sess_key_mask, friend_sizes\
            = dataset.get_next_train_batch()
        c_batch_size = len(batch_users)
        if c_batch_size == 0:
            break
        
        pred = model(X, y, target_key_mask, target_mask, cur_sess_len, hist_sess, hist_sess_key_mask, hist_sizes, friend_sess, friend_sess_key_mask, friend_sizes)
        pred = pred.permute(1, 2, 0)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.detach().item())

        bar.update(batch_idx)
        batch_idx += 1

    return np.mean(losses)


def validation_loop(model, loss_fn, evaluator, dataset):
    model.eval()
    losses = []
    evaluator.initialize()

    dataset.reset_test_batch()
    num_batches = dataset.get_num_remain_batches()
    batch_idx = 0
    bar = progressbar.ProgressBar(max_value=num_batches)
    
    with torch.no_grad():
        while True:
            batch_users, X, y, target_key_mask, target_mask, cur_sess_len, hist_sess, hist_sess_key_mask, hist_sizes, friend_sess, friend_sess_key_mask, friend_sizes\
                = dataset.get_next_test_batch()
            c_batch_size = len(batch_users)
            if c_batch_size == 0:
                break

            pred = model(X, y, target_key_mask, target_mask, cur_sess_len, hist_sess, hist_sess_key_mask, hist_sizes, friend_sess, friend_sess_key_mask, friend_sizes)
            pred = pred.permute(1, 2, 0)

            loss = loss_fn(pred, y)
            losses.append(loss.detach().item())

            pred = pred.permute(0, 2, 1)
            evaluator.evaluate_batch(pred, y)

            bar.update(batch_idx)
            batch_idx += 1

    return np.mean(losses), evaluator.get_stats()


def fit(model, opt, loss_fn, evaluator, dataset, epochs=5, checkpoints=True, checkpoint_name="", resume_from_checkpoint=None, scheduler=None):
    resume_from_epoch = 0
    if resume_from_checkpoint != None:
        checkpoint = torch.load(resume_from_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        resume_from_epoch = checkpoint['epoch']

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
