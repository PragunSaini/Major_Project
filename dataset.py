import pickle
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
from utils import load_pickle


class DataHandler():
    def __init__(self, data_path, batch_size=64, cloze_proba=0.4, device='cpu'):
        print(f'Loading dataset from {data_path}')
        stime = time.time()
        self.device = device
        self.batch_size = batch_size
        self.cloze_proba = cloze_proba

        dataset = load_pickle(data_path)
        trainset, train_session_lengths = dataset['train'], dataset['train_lens']
        valset, val_session_lengths = dataset['val'], dataset['val_lens']
        testset, test_session_lengths = dataset['test'], dataset['test_lens']

        self.item_name_map = dataset['item_name_mapping']
        self.item_index_map = dataset['item_index_mapping']
        self.item_index_map = dict((v,k) for k,v in self.item_index_map.items())
        self.item_name_map = dict((v,k) for k,v in self.item_name_map.items())
        if type(self.item_index_map) == dict:
            self.item_name_map = pd.Series(self.item_name_map)

        user_set = set(trainset.keys())
        self.num_users = len(trainset)

        assert min(user_set) == 0
        assert (max(user_set) + 1) == len(user_set)
        for user in testset.keys():
            assert user in user_set
        for user in valset.keys():
            assert user in user_set

        padding_item = -1
        self.num_session_train = 0
        self.num_session_test = 0
        self.num_session_val = 0
        self.user_item = defaultdict(set)

        self.train_data = {}
        self.train_session_lengths = {}

        for user, session_list in trainset.items():
            assert len(session_list) >= 1
            sessions = np.array(session_list)
            sess = torch.from_numpy(sessions).to(self.device)
            self.train_data[user] = sess
            self.train_session_lengths[user] = torch.tensor(train_session_lengths[user])
            padding_item = max(padding_item, sessions.max())
            self.num_session_train += len(session_list)

        self.val_data = {}
        self.val_session_lengths = {}

        for user, session_list in valset.items():
            assert len(session_list) >= 1
            sessions = np.array(session_list)
            sess = torch.from_numpy(sessions).to(self.device)
            self.val_data[user] = sess
            self.val_session_lengths[user] = torch.tensor(val_session_lengths[user])
            padding_item = max(padding_item, sessions.max())
            self.num_session_val += len(session_list)

        self.test_data = {}
        self.test_session_lengths = {}

        for user, session_list in testset.items():
            assert len(session_list) >= 1
            sessions = np.array(session_list)
            sess = torch.from_numpy(sessions).to(self.device)
            self.test_data[user] = sess
            self.test_session_lengths[user] = torch.tensor(test_session_lengths[user])
            padding_item = max(padding_item, sessions.max())
            self.num_session_test += len(session_list)

        # max index of items is the padding item
        self.padding_item = int(padding_item)
        self.num_items = int(padding_item)
        self.masking_item = self.num_items + 1
        self.items_embedding_size = self.masking_item + 1

        print(f'Dataset loaded in {(time.time() - stime):.1f}s')


    def get_similar_users(self):
        row = []
        col = []
        for usr, itms in self.user_item.items():
            col.extend(list(itms))
            row.extend([usr]*len(itms))
        row = np.array(row)
        col = np.array(col)
        idxs = col != self.padding_item
        col = col[idxs]
        row = row[idxs]  # ! user id start from 0 to N-1
        feature_mtx = coo_matrix(([1]*len(row), (row, col)), shape=(self.num_users, self.num_items))
        similarity = cosine_similarity(feature_mtx)
        return similarity.argsort()[:, -(self.n_friends+1):]


    def reset_batch(self, dataset, start_index=1):
        self.num_remain_sessions = np.zeros(self.num_users, int)
        self.index_cur_session = np.ones(self.num_users, int) * start_index
        for user, session_list in dataset.items():
            self.num_remain_sessions[user] = len(session_list) - start_index
        assert self.num_remain_sessions.min() >= 0

    def reset_train_batch(self):
        self.reset_batch(self.train_data, start_index=0) # ***** this is 1 when using user history, but ensure >= 2 sessions per user *****

    def reset_test_batch(self):
        self.reset_batch(self.test_data, start_index=0)
      
    def reset_val_batch(self):
        self.reset_batch(self.val_data, start_index=0)


    def get_next_batch(self, dataset, dataset_session_lengths, training=True):
        # select users for the batch
        if (self.num_remain_sessions > 0).sum() >= self.batch_size:
            batch_users = np.argsort(self.num_remain_sessions)[-self.batch_size:]
        else:
            batch_users = np.where(self.num_remain_sessions > 0)[0]

        if len(batch_users) == 0:
            # end of the epoch
            return batch_users, None, None, None, None

        cur_sess = []  # current sessions
        cur_sess_targets = []
        cur_sess_len = []

        for user in batch_users:
            selected_sess = dataset[user][self.index_cur_session[user], :].clone().detach()
            selected_sess_len = dataset_session_lengths[user][self.index_cur_session[user]]
            if training: # Generate cloze samples
              mask = torch.rand(selected_sess_len) < self.cloze_proba
              selected_sess[:selected_sess_len][mask] = self.masking_item
              cur_sess.append(selected_sess.unsqueeze(0))
              cur_sess_targets.append(dataset[user][self.index_cur_session[user], :])
              cur_sess_len.append(selected_sess_len)
            selected_sess_last_masked = dataset[user][self.index_cur_session[user], :].clone().detach() # generate next-item recommendation sample
            selected_sess_last_masked[selected_sess_len - 1] = self.masking_item
            cur_sess.append(selected_sess_last_masked.unsqueeze(0))
            cur_sess_targets.append(dataset[user][self.index_cur_session[user], :])
            cur_sess_len.append(selected_sess_len)

            self.index_cur_session[user] += 1
            self.num_remain_sessions[user] -= 1


        # Current Session
        # Create masked sessions with cloze_proba, and also sessions with last item masked for sequential recommendation
        cur_sess = torch.cat(cur_sess, dim=0)
        cur_sess_targets = torch.cat(cur_sess_targets, dim=0)
        cur_sess_len = torch.tensor(cur_sess_len)
        key_mask = self.create_pad_mask_by_len(cur_sess, cur_sess_len).to(self.device)

        return batch_users, cur_sess, cur_sess_targets, cur_sess_len, key_mask

    def get_next_train_batch(self):
        return self.get_next_batch(self.train_data, self.train_session_lengths, training=True)

    def get_next_test_batch(self):
        return self.get_next_batch(self.test_data, self.test_session_lengths, training=False)

    def get_next_val_batch(self):
        return self.get_next_batch(self.val_data, self.val_session_lengths, training=False)

    def get_num_remain_batches(self):
        return math.ceil(self.num_remain_sessions.sum()/self.batch_size)

    def get_target_mask(self, size):
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, value=float('-inf'))
        mask = mask.masked_fill(mask == 1, value=float(0.0))
        return mask

    def create_pad_mask(self, matrix, pad_token):
        return (matrix == pad_token)

    def create_pad_mask_by_len(self, matrix, seq_len):
        return torch.arange(matrix.size(1)).repeat(matrix.size(0), 1) >= seq_len.unsqueeze(1)
        
    def generate_prediction_data(self, items):
        sess = torch.tensor(items).unsqueeze(0)
        sesslens = torch.tensor([len(items)])
        mask =  self.create_pad_mask_by_len(sess, sesslens)
        return sess.to(self.device), sesslens.to(self.device), mask.to(self.device)

    def item_to_name(self, items):
        items = [self.item_index_map[x.item()] for x in items]
        return self.item_name_map.loc[items]