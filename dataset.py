import pickle
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
from utils import load_pickle


class DataHandler():
    def __init__(self, data_path, batch_size=64, n_friends=10, device='cpu'):
        print(f'Loading dataset from {data_path}')
        stime = time.time()
        self.device = device
        self.batch_size = batch_size

        dataset = load_pickle(data_path)
        trainset = dataset['trainset']
        testset = dataset['testset']
        train_session_lengths = dataset['train_session_lengths']
        test_session_lengths = dataset['test_session_lengths']

        user_set = set(trainset.keys())
        self.num_users = len(trainset)
        self.n_friends = min(n_friends, self.num_users)

        assert min(user_set) == 0
        assert (max(user_set) + 1) == len(user_set)
        for user in testset.keys():
            assert user in user_set

        padding_item = -1
        self.num_session_train = 0
        self.num_session_test = 0
        self.user_item = defaultdict(set)

        self.train_data = {}
        self.train_session_start_time = defaultdict(list)
        self.train_session_lengths = {}

        for user, session_list in trainset.items():
            assert len(session_list) >= 2
            for s in session_list:
                self.user_item[user].update(set([i[1] for i in s]))
            sessions = np.array(session_list)
            ordered_index = np.argsort(sessions[:, 0, 0])
            sess = torch.from_numpy(sessions[:, :, 1].astype(np.int64)[ordered_index]).to(self.device)
            self.train_data[user] = sess
            self.train_session_lengths[user] = np.array(train_session_lengths[user])[ordered_index]
            padding_item = max(padding_item, sessions[:, :, 1].max())
            self.num_session_train += len(session_list)

        self.test_data = {}
        self.test_session_lengths = {}

        for user, session_list in testset.items():
            assert len(session_list) >= 1
            for s in session_list:
                self.user_item[user].update(set([i[1] for i in s]))
            sessions = np.array(session_list)
            ordered_index = np.argsort(sessions[:, 0, 0])
            sess = torch.from_numpy(sessions[:, :, 1].astype(np.int64)[ordered_index]).to(self.device)
            self.test_data[user] = sess
            self.test_session_lengths[user] = np.array(test_session_lengths[user])[ordered_index]
            self.num_session_test += len(session_list)

        # max index of items is the padding item
        self.padding_item = int(padding_item)
        self.start_token = self.padding_item + 1
        self.num_items = int(padding_item)

        if n_friends > 0:
            self.user_similarity = self.get_similar_users()

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
        self.reset_batch(self.train_data, start_index=1)


    def reset_test_batch(self):
        self.reset_batch(self.test_data, start_index=0)


    # @timeit
    def get_next_batch(self, dataset, dataset_session_lengths, training=True):
        # select users for the batch
        if (self.num_remain_sessions > 0).sum() >= self.batch_size:
            batch_users = np.argsort(self.num_remain_sessions)[-self.batch_size:]
        else:
            batch_users = np.where(self.num_remain_sessions > 0)[0]

        if len(batch_users) == 0:
            # end of the epoch
            return batch_users, None, None, None, None, None, None, None, None, None, None, None

        cur_sess = []  # current sessions
        cur_sess_len = []
        hist_sess = []  # history sessions for each user
        friend_sess = []  # friends' sessions for each user

        for user in batch_users:
            cur_sess.append(dataset[user][self.index_cur_session[user], :])
            cur_sess_len.append(dataset_session_lengths[user][self.index_cur_session[user]])

            if training:
                prev_sess = self.train_data[user][:self.index_cur_session[user], :]
            else:
                if self.index_cur_session[user] > 0:
                    prev_sess = torch.cat([self.train_data[user], self.test_data[user][:self.index_cur_session[user], :]], dim=0)
                else:
                    prev_sess = self.train_data[user]
            hist_sess.append(prev_sess)

            # friend sessions
            if self.n_friends > 0:
                friend_sess_item_list = []
                for frd in self.user_similarity[user]:
                    if frd != user:
                        friend_sess_item_list.append(self.train_data[frd])
                sess_itm_data = torch.cat((friend_sess_item_list), dim=0)
                friend_sess.append(sess_itm_data)

            self.index_cur_session[user] += 1
            self.num_remain_sessions[user] -= 1


        # Current Session
        cur_btch_size = len(batch_users)
        cur_sess = torch.cat(cur_sess).view(len(batch_users), -1)
        cur_sess_len = np.array(cur_sess_len)
        X, y = cur_sess[:, :-1], cur_sess[:, 1:]
        X = X.to(self.device)
        y_input = torch.column_stack((torch.full((cur_btch_size,), self.start_token).to(self.device), y[:, :-1])).to(self.device)
        y_expected = y.to(self.device)
        target_mask = self.get_target_mask(y_input.size(1)).to(self.device)
        target_key_mask = self.create_pad_mask_by_len(y_input, cur_sess_len).to(self.device)
        src_key_mask = self.create_pad_mask_by_len(X, cur_sess_len).to(self.device)

        # Hist Session
        hist_sizes = torch.tensor([sess.size(0) for sess in hist_sess]).to(self.device)
        hist_sess = torch.cat(hist_sess, dim=0)
        hist_sess_key_mask = self.create_pad_mask(hist_sess, self.padding_item)

        return batch_users, X, y_input, y_expected, src_key_mask, target_key_mask, target_mask, np.array(cur_sess_len), hist_sess, hist_sess_key_mask, hist_sizes, friend_sess


    def get_next_train_batch(self):
        return self.get_next_batch(self.train_data, self.train_session_lengths, training=True)


    def get_next_test_batch(self):
        return self.get_next_batch(self.test_data, self.test_session_lengths, training=False)


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
        return torch.arange(matrix.size(1)).repeat(matrix.size(0), 1) >= torch.tensor(seq_len).unsqueeze(1)
