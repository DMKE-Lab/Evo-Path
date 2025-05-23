# coding=UTF-8
from tqdm import tqdm
import json
import numpy as np
from collections import defaultdict
import csv
import random
import os
import datetime


class RelationEntityBatcher():
    def __init__(self, input_dir, batch_size, entity_vocab, relation_vocab, time_vocab, mode="train"):
        self.input_dir = input_dir
        self.input_file = input_dir+'/{0}.txt'.format(mode)
        self.batch_size = batch_size
        print('Reading vocab...')
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.time_vocab = time_vocab
        self.mode = mode
        self.create_triple_store(self.input_file)
        print(mode + " batcher loaded")

    def get_next_batch(self):
        if self.mode == 'train':
            yield self.yield_next_batch_train()
        else:
            yield self.yield_next_batch_test()

    def create_triple_store(self, input_file):
        self.store_all_correct = defaultdict(set)
        self.store = []
        if self.mode == 'train':
            with open(input_file) as raw_input_file:
                csv_file = csv.reader(raw_input_file, delimiter='\t' )
                for line in csv_file:
                    e1 = self.entity_vocab[line[0]]
                    r = self.relation_vocab[line[1]]
                    e2 = self.entity_vocab[line[2]]
                    t = self.time_vocab[line[3]]
                    self.store.append([e1, r, e2, t])
                    self.store_all_correct[(e1, r)].add(e2)
            self.store = np.array(self.store)
        else:
            with open(input_file) as raw_input_file:
                csv_file = csv.reader(raw_input_file, delimiter='\t' )
                for line in csv_file:
                    e1 = line[0]
                    r = line[1]
                    e2 = line[2]
                    t = line[3]
                    if e1 in self.entity_vocab and e2 in self.entity_vocab:
                        e1 = self.entity_vocab[e1]
                        r = self.relation_vocab[r]
                        e2 = self.entity_vocab[e2]
                        t = self.time_vocab[t]
                        self.store.append([e1, r, e2, t])
            self.store = np.array(self.store)

            for f in ['graph.txt']:
                with open(self.input_dir+'/'+f) as raw_input_file:
                    csv_file = csv.reader(raw_input_file, delimiter='\t')
                    for line in csv_file:
                        e1 = line[0]
                        r = line[1]
                        e2 = line[2]
                        t = line[3]
                        if e1 in self.entity_vocab and e2 in self.entity_vocab and t in self.time_vocab:
                            e1 = self.entity_vocab[e1]
                            r = self.relation_vocab[r]
                            e2 = self.entity_vocab[e2]
                            t = self.time_vocab[t]
                            self.store_all_correct[(e1, r)].add(e2)

    def yield_next_batch_train(self):
        while True:
            random_t = np.random.randint(1, len(self.time_vocab))
            t_index = np.where(self.store[:, 3] == random_t)
            if t_index[0].shape[0] == 0:
                continue
            store_t = self.store[t_index[0], :]
            batch_idx = np.random.randint(0, store_t.shape[0], size=self.batch_size)
            batch = store_t[batch_idx, :]
            e1 = batch[:, 0]
            r = batch[:, 1]
            e2 = batch[:, 2]
            t = batch[:, 3]
            all_e2s = []
            for i in range(e1.shape[0]):
                all_e2s.append(self.store_all_correct[(e1[i], r[i])])
            assert e1.shape[0] == e2.shape[0] == r.shape[0] == t.shape[0] == len(all_e2s)
            yield e1, r, e2, t, all_e2s

    def yield_next_batch_test(self):
        remaining_triples = self.store.shape[0]
        current_idx = 0
        while True:
            if remaining_triples == 0:
                return
            if remaining_triples - self.batch_size > 0:
                batch_idx = np.arange(current_idx, current_idx+self.batch_size)
                current_idx += self.batch_size
                remaining_triples -= self.batch_size
            else:
                batch_idx = np.arange(current_idx, self.store.shape[0])
                current_idx += batch_idx.shape[0]
                remaining_triples = 0
            batch = self.store[batch_idx, :]

            t = batch[:, 3]
            t_list = t.tolist()
            t_set = set(t_list)
            if len(t_set) == 1:
                e1 = batch[:, 0]
                r = batch[:, 1]
                e2 = batch[:, 2]
                t = batch[:, 3]
            else:
                count = 0
                t0 = t[0]
                for i in range(batch.shape[0]):
                    if t[i] == t0:
                        count += 1
                    else:
                        break
                e1 = batch[0:count, 0]
                r = batch[0:count, 1]
                e2 = batch[0:count, 2]
                t = batch[0:count, 3]
                current_idx -= batch.shape[0] - count
                remaining_triples += batch.shape[0] - count

            all_e2s = []
            for i in range(e1.shape[0]):
                all_e2s.append(self.store_all_correct[(e1[i], r[i])])
            assert e1.shape[0] == e2.shape[0] == r.shape[0] == t.shape[0] == len(all_e2s)
            yield e1, r, e2, t, all_e2s
