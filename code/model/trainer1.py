# coding=UTF-8
from __future__ import absolute_import
from __future__ import division
from tqdm import tqdm
import json
import os
import logging
import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

from code.model.agent import Agent
from code.options import read_options
from code.model.environment import env
import codecs
from collections import defaultdict
import gc
import resource
import sys
from code.model.baseline import ReactiveBaseline
from code.model.nell_eval import nell_eval
from scipy.misc import logsumexp as lse


logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Trainer(object):
    def __init__(self, params):
        for key, val in params.items(): setattr(self, key, val);

        self.agent = Agent(params)
        self.save_path = None
        self.train_environment = env(params, 'train')
        self.dev_test_environment = env(params, 'dev')
        self.test_test_environment = env(params, 'test')
        self.test_environment = self.dev_test_environment
        self.rev_relation_vocab = self.train_environment.grapher.rev_relation_vocab
        self.rev_entity_vocab = self.train_environment.grapher.rev_entity_vocab
        self.rev_time_vocab = self.train_environment.grapher.rev_time_vocab
        self.max_hits_at_10 = 0
        self.ePAD = self.entity_vocab['PAD']
        self.rPAD = self.relation_vocab['PAD']
        # optimize
        self.baseline = ReactiveBaseline(l=self.Lambda)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

    def calc_reinforce_loss(self):
        loss = tf.stack(self.per_example_loss, axis=1)

        self.tf_baseline = self.baseline.get_baseline_value()

        final_reward = self.cum_discounted_reward - self.tf_baseline
        reward_mean, reward_var = tf.nn.moments(final_reward, axes=[0, 1])
        reward_std = tf.sqrt(reward_var) + 1e-6
        final_reward = tf.div(final_reward - reward_mean, reward_std)

        loss = tf.multiply(loss, final_reward)
        self.loss_before_reg = loss
        total_loss = tf.reduce_mean(loss) - self.decaying_beta * self.entropy_reg_loss(
            self.per_example_logits)

        return total_loss

    def entropy_reg_loss(self, all_logits):
        all_logits = tf.stack(all_logits, axis=2)
        entropy_policy = - tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.exp(all_logits), all_logits), axis=1))
        return entropy_policy

    def initialize(self, restore=None, sess=None):
        logger.info("Creating TF graph...")
        self.candidate_relation_sequence = []
        self.candidate_entity_sequence = []
        self.candidate_time_sequence = []
        self.input_path = []
        self.first_state_of_test = tf.placeholder(tf.bool, name="is_first_state_of_test")
        self.query_relation = tf.placeholder(tf.int32, [None], name="query_relation")
        self.query_time = tf.placeholder(tf.int32, [None], name="query_time")
        self.range_arr = tf.placeholder(tf.int32, shape=[None, ])
        self.global_step = tf.Variable(0, trainable=False)
        self.decaying_beta = tf.train.exponential_decay(self.beta, self.global_step,
                                                        200, 0.90, staircase=False)
        self.entity_sequence = []

        self.cum_discounted_reward = tf.placeholder(tf.float32, [None, self.path_length],
                                                    name="cumulative_discounted_reward")

        for t in range(self.path_length):
            next_possible_relations = tf.placeholder(tf.int32, [None, self.max_num_actions],
                                                     name="next_relations_{}".format(t))
            next_possible_entities = tf.placeholder(tf.int32, [None, self.max_num_actions],
                                                    name="next_entities_{}".format(t))
            next_possible_times = tf.placeholder(tf.int32, [None, self.max_num_actions],
                                                 name="next_times_{}".format(t))
            input_label_relation = tf.placeholder(tf.int32, [None], name="input_label_relation_{}".format(t))
            start_entities = tf.placeholder(tf.int32, [None, ])
            self.input_path.append(input_label_relation)
            self.candidate_relation_sequence.append(next_possible_relations)
            self.candidate_entity_sequence.append(next_possible_entities)
            self.candidate_time_sequence.append(next_possible_times)
            self.entity_sequence.append(start_entities)

        self.loss_before_reg = tf.constant(0.0)

        self.per_example_loss, self.per_example_logits, self.action_idx = self.agent(
            self.candidate_relation_sequence, self.candidate_entity_sequence, self.candidate_time_sequence,
            self.entity_sequence, self.input_path, self.query_relation, self.query_time,
            self.range_arr, self.first_state_of_test, self.path_length)

        self.loss_op = self.calc_reinforce_loss()

        self.train_op = self.bp(self.loss_op)

        # Building the test graph
        self.prev_state = tf.placeholder(tf.float32, self.agent.get_mem_shape(), name="memory_of_agent")
        self.prev_relation = tf.placeholder(tf.int32, [None, ], name="previous_relation")
        self.prev_time = tf.placeholder(tf.int32, [None, ], name="previous_time")
        self.query_embedding = tf.nn.embedding_lookup(self.agent.relation_lookup_table, self.query_relation)
        self.query_time_t = tf.placeholder(tf.int32, [None], name="query_time")
        layer_state = tf.unstack(self.prev_state, self.LSTM_layers)
        formated_state = [tf.unstack(s, 2) for s in layer_state]
        self.next_relations = tf.placeholder(tf.int32, shape=[None, self.max_num_actions])
        self.next_entities = tf.placeholder(tf.int32, shape=[None, self.max_num_actions])
        self.next_times = tf.placeholder(tf.int32, shape=[None, self.max_num_actions])

        self.current_entities = tf.placeholder(tf.int32, shape=[None, ])

        with tf.variable_scope("policy_steps_unroll") as scope:
            scope.reuse_variables()
            self.test_loss, test_state, self.test_logits, self.test_action_idx, self.chosen_relation, self.chosen_time = self.agent.step(
                self.next_relations, self.next_entities, self.next_times, formated_state, self.prev_relation,
                self.prev_time, self.query_embedding, self.query_time_t, self.current_entities, self.input_path[0],
                self.range_arr, self.first_state_of_test)
            self.test_state = tf.stack(test_state)

        logger.info('TF Graph creation done..')
        self.model_saver = tf.train.Saver(max_to_keep=2)

        # return the variable initializer Op.
        if not restore:
            return tf.global_variables_initializer()
        else:
            return self.model_saver.restore(sess, restore)

    def initialize_pretrained_embeddings(self, sess):
        if self.pretrained_embeddings_action != '':
            embeddings = np.loadtxt(open(self.pretrained_embeddings_action))
            _ = sess.run((self.agent.relation_embedding_init),
                         feed_dict={self.agent.action_embedding_placeholder: embeddings})
        if self.pretrained_embeddings_entity != '':
            embeddings = np.loadtxt(open(self.pretrained_embeddings_entity))
            _ = sess.run((self.agent.entity_embedding_init),
                         feed_dict={self.agent.entity_embedding_placeholder: embeddings})

    def bp(self, cost):
        self.baseline.update(tf.reduce_mean(self.cum_discounted_reward))
        tvars = tf.trainable_variables()
        grads = tf.gradients(cost, tvars)
        grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)
        train_op = self.optimizer.apply_gradients(zip(grads, tvars))
        with tf.control_dependencies([train_op]):  # see https://github.com/tensorflow/tensorflow/issues/1899
            self.dummy = tf.constant(0)
        return train_op

    def calc_cum_discounted_reward(self, rewards):
        """
        calculates the cumulative discounted reward.
        :param rewards:
        :param T:
        :param gamma:
        :return:
        """
        running_add = np.zeros([rewards.shape[0]])
        cum_disc_reward = np.zeros([rewards.shape[0], self.path_length])
        cum_disc_reward[:, self.path_length - 1] = rewards  # set the last time step to the reward received at the last state
        for t in reversed(range(self.path_length)):
            running_add = self.gamma * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add
        return cum_disc_reward

    def gpu_io_setup(self):
        fetches_train = self.per_example_loss + self.action_idx + [self.loss_op] + self.per_example_logits + [
            self.dummy]
        fetches = self.per_example_loss + self.action_idx + [self.loss_op] + self.per_example_logits
        feeds = [self.first_state_of_test] + self.candidate_relation_sequence + self.candidate_entity_sequence + \
                self.input_path + [self.query_relation] + [self.cum_discounted_reward] + [self.range_arr] + \
                self.entity_sequence + self.candidate_time_sequence + [self.query_time]

        feed_dict = [{} for _ in range(self.path_length)]

        feed_dict[0][self.first_state_of_test] = False
        feed_dict[0][self.query_relation] = None
        feed_dict[0][self.query_time] = None
        feed_dict[0][self.range_arr] = np.arange(self.batch_size * self.num_rollouts)
        for i in range(self.path_length):
            feed_dict[i][self.input_path[i]] = np.zeros(self.batch_size * self.num_rollouts)  # placebo
            feed_dict[i][self.candidate_relation_sequence[i]] = None
            feed_dict[i][self.candidate_entity_sequence[i]] = None
            feed_dict[i][self.candidate_time_sequence[i]] = None
            feed_dict[i][self.entity_sequence[i]] = None

        return fetches_train, fetches, feeds, feed_dict

    def train(self, sess):
        fetches_train, fetches, feeds, feed_dict = self.gpu_io_setup()

        train_loss = 0.0
        self.batch_counter = 0
        for episode in self.train_environment.get_episodes():
            self.batch_counter += 1

            h_train = sess.partial_run_setup(fetches=fetches_train, feeds=feeds)

            feed_dict[0][self.query_relation] = episode.get_query_relation()
            feed_dict[0][self.query_time] = episode.get_query_time()
            # get initial state
            state = episode.get_state()
            # for each time step
            loss_before_regularization = []
            logits = []
            trace = []
            trace_relation = []
            trace_time = []
            for i in range(self.path_length):
                feed_dict[i][self.candidate_relation_sequence[i]] = state['next_relations']
                feed_dict[i][self.candidate_entity_sequence[i]] = state['next_entities']
                feed_dict[i][self.candidate_time_sequence[i]] = state['next_times']
                feed_dict[i][self.entity_sequence[i]] = state['current_entities']
                trace.append(state['current_entities'])
                per_example_loss, per_example_logits, idx = sess.partial_run(h_train, [self.per_example_loss[i],
                                                                                           self.per_example_logits[i],
                                                                                           self.action_idx[i]],
                                                                                 feed_dict=feed_dict[i])
                loss_before_regularization.append(per_example_loss)
                logits.append(per_example_logits)

                relation = []
                for j in range(state['next_relations'].shape[0]):
                    relation.append(state['next_relations'][j, idx[j]])
                trace_relation.append(relation)
                time = []
                for j in range(state['next_times'].shape[0]):
                    time.append(state['next_times'][j, idx[j]])
                trace_time.append(time)

                state = episode(idx)
            trace.append(state['current_entities'])
            loss_before_regularization = np.stack(loss_before_regularization, axis=1)

            # get the final reward from the environment
            rewards = episode.get_reward()

            # computed cumulative discounted reward
            cum_discounted_reward = self.calc_cum_discounted_reward(rewards)

            # backprop
            batch_total_loss, _ = sess.partial_run(h_train, [self.loss_op, self.dummy],
                                                       feed_dict={self.cum_discounted_reward: cum_discounted_reward})

            # print statistics
            train_loss = 0.98 * train_loss + 0.02 * batch_total_loss
            avg_reward = np.mean(rewards)
            # now reshape the reward to [orig_batch_size, num_rollouts], I want to calculate for how many of the
            # entity pair, at least one of the path get to the right answer
            reward_reshape = np.reshape(rewards, (self.batch_size, self.num_rollouts))  # [orig_batch, num_rollouts]
            reward_reshape = np.sum(reward_reshape, axis=1)  # [orig_batch]
            reward_reshape = (reward_reshape > 0)
            num_ep_correct = np.sum(reward_reshape)
            if np.isnan(train_loss):
                raise ArithmeticError("Error in computing loss")

            logger.info("batch_counter: {0:4d},      num_hits: {1:7.4f}, avg_reward_per_batch {2:7.4f},\n"
                        "num_ep_correct {3:4d}, avg_ep_correct {4:7.4f},           train loss {5:7.4f}".
                        format(self.batch_counter, np.sum(rewards), avg_reward,
                               num_ep_correct, (num_ep_correct / self.batch_size), train_loss))

            if self.batch_counter % self.eval_every == 0:
                with open(self.output_dir + '/scores.txt', 'a') as score_file:
                    score_file.write("Score for iteration " + str(self.batch_counter) + "\n")
                os.mkdir(self.path_logger_file + "/" + str(self.batch_counter))
                self.path_logger_file_ = self.path_logger_file + "/" + str(self.batch_counter) + "/paths"

                self.test(sess, beam=True, print_paths=False)

            logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            gc.collect()
            if self.batch_counter >= self.total_iterations:
                break

    def test(self, sess, beam=False, print_paths=False, save_model=True, auc=False):
        batch_counter = 0
        example_counter = 0
        paths = defaultdict(list)
        answers = []
        feed_dict = {}
        all_final_reward_1 = 0
        all_final_reward_3 = 0
        all_final_reward_5 = 0
        all_final_reward_10 = 0
        all_final_reward_20 = 0
        auc = 0

        total_examples = self.test_environment.total_no_examples
        for episode in tqdm(self.test_environment.get_episodes()):
            batch_counter += 1
            temp_batch_size = episode.no_examples
            example_counter += temp_batch_size
            logger.info('test total examples: ' + str(total_examples))
            logger.info('test batch number: ' + str(batch_counter))
            logger.info('test batch size: ' + str(temp_batch_size))
            logger.info('test example count: ' + str(example_counter))

            self.qr = episode.get_query_relation()
            self.qt = episode.get_query_time()
            feed_dict[self.query_relation] = self.qr
            feed_dict[self.query_time_t] = episode.get_query_time()
            # set initial beam probs
            beam_probs = np.zeros((temp_batch_size * self.test_rollouts, 1))
            # get initial state
            state = episode.get_state()
            mem = self.agent.get_mem_shape()
            agent_mem = np.zeros((mem[0], mem[1], temp_batch_size * self.test_rollouts, mem[3])).astype('float32')
            previous_relation = np.ones((temp_batch_size * self.test_rollouts,), dtype='int64') * self.relation_vocab[
                'DUMMY_START_RELATION']
            previous_time = np.zeros((temp_batch_size * self.test_rollouts,), dtype='float32')
            feed_dict[self.range_arr] = np.arange(temp_batch_size * self.test_rollouts)
            feed_dict[self.input_path[0]] = np.zeros(temp_batch_size * self.test_rollouts)

            ####logger code####
            if print_paths:
                self.entity_trajectory = []
                self.relation_trajectory = []
                self.time_trajectory = []
            ####################

            self.log_probs = np.zeros((temp_batch_size * self.test_rollouts,)) * 1.0
            trace = []
            trace_time = []
            trace_relation = []
            # for each time step
            for i in range(self.path_length):
                if i == 0:
                    feed_dict[self.first_state_of_test] = True
                trace.append(state['current_entities'])
                feed_dict[self.next_relations] = state['next_relations']
                feed_dict[self.next_entities] = state['next_entities']
                feed_dict[self.next_times] = state['next_times']
                feed_dict[self.current_entities] = state['current_entities']
                feed_dict[self.prev_state] = agent_mem
                feed_dict[self.prev_relation] = previous_relation
                feed_dict[self.prev_time] = previous_time

                loss, agent_mem, test_scores, test_action_idx, chosen_relation, chosen_time = sess.run(
                    [self.test_loss, self.test_state, self.test_logits, self.test_action_idx, self.chosen_relation,
                     self.chosen_time],
                    feed_dict=feed_dict)

                if beam:
                    k = self.test_rollouts
                    new_scores = test_scores + beam_probs
                    if i == 0:
                        idx = np.argsort(new_scores)
                        idx = idx[:, -k:]
                        ranged_idx = np.tile([b for b in range(k)], temp_batch_size)
                        idx = idx[np.arange(k * temp_batch_size), ranged_idx]
                    else:
                        idx = self.top_k(new_scores, k)

                    y = idx // self.max_num_actions
                    x = idx % self.max_num_actions

                    y += np.repeat([b * k for b in range(temp_batch_size)], k)

                    state['current_entities'] = state['current_entities'][y]
                    state['next_relations'] = state['next_relations'][y, :]
                    state['next_entities'] = state['next_entities'][y, :]
                    state['next_times'] = state['next_times'][y, :]
                    agent_mem = agent_mem[:, :, y, :]
                    test_action_idx = x
                    chosen_relation = state['next_relations'][np.arange(temp_batch_size * k), x]
                    chosen_time = state['next_times'][np.arange(temp_batch_size * k), x]
                    beam_probs = new_scores[y, x]
                    beam_probs = beam_probs.reshape((-1, 1))
                    if print_paths:
                        for j in range(i):
                            self.entity_trajectory[j] = self.entity_trajectory[j][y]
                            self.relation_trajectory[j] = self.relation_trajectory[j][y]
                            self.time_trajectory[j] = self.time_trajectory[j][y]
                previous_relation = chosen_relation
                previous_time = chosen_time

                trace_relation.append(chosen_relation)
                trace_time.append(chosen_time)

                ####logger code####
                if print_paths:
                    self.entity_trajectory.append(state['current_entities'])
                    self.relation_trajectory.append(chosen_relation)
                    self.time_trajectory.append(chosen_time)
                ####################
                state = episode(test_action_idx)
                self.log_probs += test_scores[np.arange(self.log_probs.shape[0]), test_action_idx]

            trace.append(state['current_entities'])

            if beam:
                self.log_probs = beam_probs

            ####Logger code####

            if print_paths:
                self.entity_trajectory.append(state['current_entities'])

            # ask environment for final reward
            rewards = episode.get_reward()  # [trB]
            reward_reshape = np.reshape(rewards, (temp_batch_size, self.test_rollouts))
            self.log_probs = np.reshape(self.log_probs, (temp_batch_size, self.test_rollouts))
            sorted_indx = np.argsort(-self.log_probs)
            final_reward_1 = 0
            final_reward_3 = 0
            final_reward_5 = 0
            final_reward_10 = 0
            final_reward_20 = 0
            AP = 0
            # current entity
            ce = episode.state['current_entities'].reshape((temp_batch_size, self.test_rollouts))
            # start entity
            se = episode.start_entities.reshape((temp_batch_size, self.test_rollouts))

            for b in range(temp_batch_size):
                answer_pos = None
                seen = set()
                pos = 0
                if self.pool == 'max':
                    for r in sorted_indx[b]:
                        if reward_reshape[b, r] == self.positive_reward:
                            answer_pos = pos
                            break
                        if ce[b, r] not in seen:
                            seen.add(ce[b, r])
                            pos += 1
                if self.pool == 'sum':
                    scores = defaultdict(list)
                    answer = ''
                    for r in sorted_indx[b]:
                        scores[ce[b, r]].append(self.log_probs[b, r])
                        if reward_reshape[b, r] == self.positive_reward:
                            answer = ce[b, r]
                    final_scores = defaultdict(float)
                    for e in scores:
                        final_scores[e] = lse(scores[e])
                    sorted_answers = sorted(final_scores, key=final_scores.get, reverse=True)
                    if answer in sorted_answers:
                        answer_pos = sorted_answers.index(answer)
                    else:
                        answer_pos = None

                if answer_pos != None:
                    if answer_pos < 20:
                        final_reward_20 += 1
                        if answer_pos < 10:
                            final_reward_10 += 1
                            if answer_pos < 5:
                                final_reward_5 += 1
                                if answer_pos < 3:
                                    final_reward_3 += 1
                                    if answer_pos < 1:
                                        final_reward_1 += 1
                if answer_pos == None:
                    AP += 0
                else:
                    AP += 1.0 / (answer_pos + 1)

                if print_paths:
                    qr = self.train_environment.grapher.rev_relation_vocab[
                        self.qr[b * self.test_rollouts]]  # test relation
                    qt = self.train_environment.grapher.rev_time_vocab[self.qt[b * self.test_rollouts]]
                    start_e = self.rev_entity_vocab[episode.start_entities[b * self.test_rollouts]]
                    end_e = self.rev_entity_vocab[episode.end_entities[b * self.test_rollouts]]
                    paths[str(qr)].append(
                        'Query: ' + str(start_e) + "\t" + str(qr) + '\t' + str(end_e) + '\t' + str(qt) + "\n")
                    paths[str(qr)].append(
                        "Reward(pos<10): " + str(1 if answer_pos != None and answer_pos < 10 else 0) + "\n")
                    paths[str(qr)].append('**********************' + '\n')
                    tr = 0
                    for r in sorted_indx[b]:
                        tr += 1
                        indx = b * self.test_rollouts + r
                        if rewards[indx] == self.positive_reward:
                            rev = 1
                        else:
                            rev = -1
                        answers.append(
                            self.rev_entity_vocab[se[b, r]] + '\t' + self.rev_entity_vocab[ce[b, r]] + '\t' + str(
                                self.log_probs[b, r]) + '\n')
                        paths[str(qr)].append(
                            'Test rollout number: ' + str(tr) + '\n' +
                            '\t'.join([str(self.rev_entity_vocab[e[indx]]) for e in self.entity_trajectory]) + '\n' +
                            '\t'.join(
                                [str(self.rev_relation_vocab[re[indx]]) for re in self.relation_trajectory]) + '\n' +
                            '\t'.join([str(self.rev_time_vocab[t[indx]]) for t in self.time_trajectory]) + '\n' +
                            'Reward: ' + str(rev) + '\n' +
                            'Score: ' + str(self.log_probs[b, r]) + '\n' +
                            '_______________________' + '\n')
                    paths[str(qr)].append("#############################\n")

            all_final_reward_1 += final_reward_1
            all_final_reward_3 += final_reward_3
            all_final_reward_5 += final_reward_5
            all_final_reward_10 += final_reward_10
            all_final_reward_20 += final_reward_20
            auc += AP

        all_final_reward_1 /= total_examples
        all_final_reward_3 /= total_examples
        all_final_reward_5 /= total_examples
        all_final_reward_10 /= total_examples
        all_final_reward_20 /= total_examples
        auc /= total_examples

        if save_model:
            if all_final_reward_10 >= self.max_hits_at_10:
                self.max_hits_at_10 = all_final_reward_10
                self.save_path = self.model_saver.save(sess, self.model_dir + "model" + '.ckpt')

        if print_paths:
            logger.info("[ printing paths at {} ]".format(self.output_dir + '/test_beam/'))
            for q in paths:
                j = q.replace('/', '-')
                with codecs.open(self.path_logger_file_ + '_' + j, 'a', 'utf-8') as pos_file:
                    for p in paths[q]:
                        pos_file.write(p)
            with open(self.path_logger_file_ + 'answers', 'w') as answer_file:
                for a in answers:
                    answer_file.write(a)

        with open(self.output_dir + '/scores.txt', 'a') as score_file:
            score_file.write("Hits@1: {0:7.4f}".format(all_final_reward_1))
            score_file.write("\n")
            score_file.write("Hits@3: {0:7.4f}".format(all_final_reward_3))
            score_file.write("\n")
            score_file.write("Hits@10: {0:7.4f}".format(all_final_reward_10))
            score_file.write("\n")
            score_file.write("MRR: {0:7.4f}".format(auc))
            score_file.write("\n")
            score_file.write("\n")

        logger.info("Hits@1: {0:7.4f}".format(all_final_reward_1))
        logger.info("Hits@3: {0:7.4f}".format(all_final_reward_3))
        logger.info("Hits@10: {0:7.4f}".format(all_final_reward_10))
        logger.info("MRR: {0:7.4f}".format(auc))

    def top_k(self, scores, k):
        scores = scores.reshape(-1, k * self.max_num_actions)
        idx = np.argsort(scores, axis=1)
        idx = idx[:, -k:]
        return idx.reshape((-1))


if __name__ == '__main__':
    # read command line options
    options = read_options()
    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    logfile = logging.FileHandler(options['log_file_name'], 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    # read the vocab files, it will be used by many classes hence global scope
    logger.info('reading vocab files...')
    options['relation_vocab'] = json.load(open(options['vocab_dir'] + '/relation_vocab.json'))
    options['entity_vocab'] = json.load(open(options['vocab_dir'] + '/entity_vocab.json'))
    options['time_vocab'] = json.load(open(options['vocab_dir'] + '/time_vocab.json'))

    logger.info('Reading mid to name map')
    mid_to_word = {}
    logger.info('Done..')

    logger.info('Total number of entities {}'.format(len(options['entity_vocab'])))
    logger.info('Total number of relations {}'.format(len(options['relation_vocab'])))
    logger.info('Total number of times {}'.format(len(options['time_vocab'])))
    save_path = ''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True

    # Training
    if not options['load_model']:
        trainer = Trainer(options)
        with tf.Session(config=config) as sess:
            sess.run(trainer.initialize())
            trainer.initialize_pretrained_embeddings(sess=sess)

            trainer.train(sess)
            save_path = trainer.save_path
            path_logger_file = trainer.path_logger_file
            output_dir = trainer.output_dir

        tf.reset_default_graph()
    else:
        logger.info("Skipping training")
        logger.info("Loading model from {}".format(options["model_load_dir"]))

    trainer = Trainer(options)
    if options['load_model']:
        save_path = options['model_load_dir']
        path_logger_file = trainer.path_logger_file
        output_dir = trainer.output_dir

    # Testing on test with best model
    with tf.Session(config=config) as sess:
        trainer.initialize(restore=save_path, sess=sess)

        trainer.test_rollouts = 100

        os.mkdir(path_logger_file + "/" + "test_beam")
        trainer.path_logger_file_ = path_logger_file + "/" + "test_beam" + "/paths"
        with open(output_dir + '/scores.txt', 'a') as score_file:
            score_file.write("Test (beam) scores with best model from " + save_path + "\n")
        trainer.test_environment = trainer.test_test_environment
        trainer.test_environment.test_rollouts = 100  # 测试rollouts = 100

        trainer.test(sess, beam=True, print_paths=True, save_model=False)

        print(options['nell_evaluation'])
        if options['nell_evaluation'] == 1:
            nell_eval(path_logger_file + "/" + "test_beam/" + "pathsanswers",
                      trainer.data_input_dir + '/sort_test.pairs')

