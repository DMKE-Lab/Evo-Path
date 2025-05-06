# coding=UTF-8
from __future__ import absolute_import
from __future__ import division
import numpy as np
from code.data.feed_data import RelationEntityBatcher
from code.data.grapher import RelationEntityGrapher
import logging

logger = logging.getLogger()


class Episode(object):
    def __init__(self, graph, data, params):
        self.grapher = graph
        self.batch_size, self.path_len, num_rollouts, test_rollouts, positive_reward, negative_reward, mode, batcher = params
        self.mode = mode
        if self.mode == 'train':
            self.num_rollouts = num_rollouts
        else:
            self.num_rollouts = test_rollouts
        self.current_hop = 0
        start_entities, query_relation, end_entities, query_time, all_answers = data
        self.no_examples = start_entities.shape[0]
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        start_entities = np.repeat(start_entities, self.num_rollouts)
        batch_query_relation = np.repeat(query_relation, self.num_rollouts)
        end_entities = np.repeat(end_entities, self.num_rollouts)
        query_time = np.repeat(query_time, self.num_rollouts)
        self.start_entities = start_entities
        self.end_entities = end_entities
        self.current_entities = np.array(start_entities)
        self.query_relation = batch_query_relation
        self.query_time = query_time
        self.all_answers = all_answers
        self.pre_action_time = query_time

        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.query_time, self.pre_action_time,
                                                        self.end_entities, self.all_answers,
                                                        self.current_hop == self.path_len - 1, self.num_rollouts)
        self.state = {}
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['next_times'] = next_actions[:, :, 2]
        self.state['current_entities'] = self.current_entities

    def get_state(self):
        return self.state

    def get_query_relation(self):
        return self.query_relation

    def get_query_time(self):
        return self.query_time

    def get_reward(self):
        reward = (self.current_entities == self.end_entities)
        # set the True and False values to the values of positive and negative rewards.
        condlist = [reward == True, reward == False]
        choicelist = [self.positive_reward, self.negative_reward]
        reward = np.select(condlist, choicelist)
        return reward

    def get_gat_reward(self):
        out = []
        for i in self.all_answers:
            i = list(i)
            out.append(np.sum(np.eye(len(self.grapher.entity_vocab))[i], axis=0))
        out = np.array(out)
        out = np.repeat(out, self.num_rollouts, axis=0)
        return out

    def get_gat_reward1(self):
        return self.end_entities

    def get_mask_mt(self):
        nb_node = len(self.grapher.entity_vocab)
        ce_mt = np.eye(nb_node)[self.current_entities]
        mt_list = []
        for i in range(0, ce_mt.shape[0], self.num_rollouts):
            mt_list.append(np.sum(ce_mt[i:i + self.num_rollouts, :], axis=0, keepdims=True))
        mt = np.concatenate(mt_list, axis=0)
        mask = mt == 0
        mt[mask] = 1
        mt[~mask] = 0
        mt += np.eye(nb_node)[0]
        mt += np.eye(nb_node)[1]
        # mt += se_mt_new
        mask = mt != 0
        mt[mask] = 1
        mt *= -1e9
        mt = np.repeat(mt, self.num_rollouts, axis=0)
        return mt

    def __call__(self, action):
        self.current_hop += 1
        self.current_entities = self.state['next_entities'][np.arange(self.no_examples * self.num_rollouts), action]
        self.pre_action_time = self.state['next_times'][np.arange(self.no_examples * self.num_rollouts), action]

        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.query_time, self.pre_action_time,
                                                        self.end_entities, self.all_answers,
                                                        self.current_hop == self.path_len - 1, self.num_rollouts )

        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['next_times'] = next_actions[:, :, 2]
        self.state['current_entities'] = self.current_entities
        return self.state


class env(object):
    def __init__(self, params, mode='train'):
        self.batch_size = params['batch_size']
        self.num_rollouts = params['num_rollouts']
        self.positive_reward = params['positive_reward']
        self.negative_reward = params['negative_reward']
        self.mode = mode
        self.path_len = params['path_length']
        self.test_rollouts = params['test_rollouts']
        input_dir = params['data_input_dir']
        if mode == 'train':
            self.batcher = RelationEntityBatcher(input_dir=input_dir,
                                                 batch_size=params['batch_size'],
                                                 entity_vocab=params['entity_vocab'],
                                                 relation_vocab=params['relation_vocab'],
                                                 time_vocab=params['time_vocab'])
        else:
            self.batcher = RelationEntityBatcher(input_dir=input_dir,
                                                 mode=mode,
                                                 batch_size=params['batch_size'],
                                                 entity_vocab=params['entity_vocab'],
                                                 relation_vocab=params['relation_vocab'],
                                                 time_vocab=params['time_vocab'])

            self.total_no_examples = self.batcher.store.shape[0]
        print("construct " + mode + " KG")
        self.grapher = RelationEntityGrapher(triple_store=params['data_input_dir'] + '/' + 'graph.txt',
                                             max_num_actions=params['max_num_actions'],
                                             entity_vocab=params['entity_vocab'],
                                             relation_vocab=params['relation_vocab'],
                                             time_vocab=params['time_vocab'])

    def get_episodes(self):
        params = self.batch_size, self.path_len, self.num_rollouts, self.test_rollouts, self.positive_reward, \
                 self.negative_reward, self.mode, self.batcher
        if self.mode == 'train':
            for data in self.batcher.yield_next_batch_train():
                yield Episode(self.grapher, data, params)
        else:
            for data in self.batcher.yield_next_batch_test():
                if data == None:
                    return
                yield Episode(self.grapher, data, params)
