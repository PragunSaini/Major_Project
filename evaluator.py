import math
import numpy as np
import torch


class Evaluator(object):
    def __init__(self, num_items, k_list=[5, 20], masking_token=-1):
        self.num_items = num_items
        self.k_list = k_list
        self.masking_token = masking_token
        self.initialize()

    def initialize(self):
        self.recall = []
        self.mrr = []

    def evaluate_batch(self, inputs, targets, predictions):
        """
        inputs : [BATCH, SEQLEN]
        targets : [BATCH, SEQLEN]
        predictions : [BATCH, SEQLEN, NUMITEMS (probabilities)]
        """
        predictions = predictions.reshape(-1, self.num_items)
        targets = targets.reshape(-1)
        # mask = targets != self.padding_idx
        mask = inputs.reshape(-1) == self.masking_token
        self.recall.append(self.get_recall(predictions[mask], targets[mask]))
        self.mrr.append(self.get_mrr(predictions[mask], targets[mask]))

    def get_recall(self, predictions, targets):
        recalls = []
        for k in self.k_list:
            _, topk = torch.topk(predictions, k=k, dim=1)
            targets_exp = targets.view(-1, 1).expand_as(topk)
            hits = (targets_exp == topk).nonzero()
            if len(hits) == 0:
                recalls.append(0.0)
                continue
            n_hits = hits[:, :-1].size(0)
            recall = float(n_hits) / targets_exp.size(0)
            recalls.append(recall)
        return recalls

    def get_mrr(self, predictions, targets):
        mrrs = []
        for k in self.k_list:
            _, topk = torch.topk(predictions, k=k, dim=1)
            targets_exp = targets.view(-1, 1).expand_as(topk)
            hits = (targets_exp == topk).nonzero()
            ranks = hits[:, -1] + 1
            ranks = ranks.float()
            reciprocal_ranks = torch.reciprocal(ranks)
            n_hits = hits[:, :-1].size(0)
            mrr = torch.sum(reciprocal_ranks).item() / targets_exp.size(0)
            mrrs.append(mrr)
        return mrrs

    def get_stats(self):
        eval_result = "\nEvaluation Results"
        recalls = torch.tensor(self.recall)
        mrrs = torch.tensor(self.mrr)

        for i in range(len(self.k_list)):
            recall_at_k = recalls[:, i].mean().item()
            mrr_at_k = mrrs[:, i].mean().item()
            eval_result += f"\nRecall@{self.k_list[i]}\t{recall_at_k:.4f}"
            eval_result += f"\nMRR@{self.k_list[i]}\t\t{mrr_at_k:.4f}"
        
        eval_result += "\n"
        return eval_result