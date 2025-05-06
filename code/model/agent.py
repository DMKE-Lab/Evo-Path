# coding=UTF-8
import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


class Agent(object):
    def __init__(self, params):
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.time_vocab_size = len(params['time_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.ePAD = tf.constant(params['entity_vocab']['PAD'], dtype=tf.int32)
        self.rPAD = tf.constant(params['relation_vocab']['PAD'], dtype=tf.int32)
        if params['use_entity_embeddings']:
            self.entity_initializer = tf.keras.initializers.glorot_normal()
        else:
            self.entity_initializer = tf.zeros_initializer()
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']

        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.LSTM_Layers = params['LSTM_layers']
        self.batch_size = params['batch_size'] * params['num_rollouts']
        self.dummy_start_label = tf.constant(
            np.ones(self.batch_size, dtype='int64') * params['relation_vocab']['DUMMY_START_RELATION'])

        self.entity_embedding_size = self.embedding_size
        self.use_entity_embeddings = params['use_entity_embeddings']

        if self.use_entity_embeddings:
            self.m = 4 + 2
        else:
            self.m = 2 + 2

        with tf.variable_scope("action_lookup_table"):
            self.action_embedding_placeholder = tf.placeholder(tf.float32,
                                                               [self.action_vocab_size, 2 * self.embedding_size])

            self.relation_lookup_table = tf.get_variable("relation_lookup_table",
                                                         shape=[self.action_vocab_size, 2 * self.embedding_size],
                                                         dtype=tf.float32,
                                                         initializer=tf.keras.initializers.glorot_normal(),
                                                         trainable=self.train_relations)
            self.relation_embedding_init = self.relation_lookup_table.assign(self.action_embedding_placeholder)

        with tf.variable_scope("entity_lookup_table"):
            self.entity_embedding_placeholder = tf.placeholder(tf.float32,
                                                               [self.entity_vocab_size, 2 * self.embedding_size])
            self.entity_lookup_table = tf.get_variable("entity_lookup_table",
                                                       shape=[self.entity_vocab_size, 2 * self.entity_embedding_size],
                                                       dtype=tf.float32,
                                                       initializer=self.entity_initializer,
                                                       trainable=self.train_entities)
            self.entity_embedding_init = self.entity_lookup_table.assign(self.entity_embedding_placeholder)

        with tf.variable_scope("time_lookup_table"):
            self.time_embedding_placeholder = tf.placeholder(tf.float32,
                                                               [self.time_vocab_size, 2 * self.embedding_size])
            self.time_lookup_table = tf.get_variable("time_lookup_table",
                                                       shape=[self.time_vocab_size, 2 * self.entity_embedding_size],
                                                       dtype=tf.float32,
                                                       initializer=self.entity_initializer,
                                                       trainable=self.train_entities)
            self.time_embedding_init = self.time_lookup_table.assign(self.time_embedding_placeholder)

        with tf.variable_scope("policy_step"):
            cells = []
            for _ in range(self.LSTM_Layers):
                cells.append(tf.nn.rnn_cell.LSTMCell(self.m * self.hidden_size, use_peepholes=True, state_is_tuple=True))
            self.policy_step = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

    def get_mem_shape(self):
        return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)

    def policy_MLP(self, state):
        with tf.variable_scope("MLP_for_policy"):
            output = tf.layers.dense(state, self.m * self.embedding_size, activation=tf.nn.relu)
        return output

    def action_encoder(self, next_relations, next_entities, next_times):
        with tf.variable_scope("lookup_table_edge_encoder"):
            relation_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, next_relations)
            entity_embedding = tf.nn.embedding_lookup(self.entity_lookup_table, next_entities)
            time_embedding = tf.nn.embedding_lookup(self.time_lookup_table, next_times)

            if self.use_entity_embeddings:
                action_embedding = tf.concat([relation_embedding, entity_embedding], axis=-1)
            else:
                action_embedding = relation_embedding
            action_embedding = tf.concat([action_embedding, time_embedding], axis=-1)
        return action_embedding

    def step(self, next_relations, next_entities, next_times, prev_state, prev_relation, prev_time, query_embedding, quert_time,
             current_entities, label_action, range_arr, first_step_of_test):

        prev_action_embedding = self.action_encoder(prev_relation, current_entities, prev_time)
        output, new_state = self.policy_step(prev_action_embedding, prev_state)  # output: [B, 4D]
        prev_entity = tf.nn.embedding_lookup(self.entity_lookup_table, current_entities)
        if self.use_entity_embeddings:
            state = tf.concat([output, prev_entity], axis=-1)
        else:
            state = output

        candidate_action_embeddings = self.action_encoder(next_relations, next_entities, next_times)
        state_query_concat = tf.concat([state, query_embedding], axis=-1)
        quert_time = tf.nn.embedding_lookup(self.time_lookup_table, quert_time)
        state_query_time_concat = tf.concat([state_query_concat, quert_time], axis=-1)

        output = self.policy_MLP(state_query_time_concat)
        output_expanded = tf.expand_dims(output, axis=1)
        prelim_scores = tf.reduce_sum(tf.multiply(candidate_action_embeddings, output_expanded), axis=2)

        comparison_tensor = tf.ones_like(next_relations, dtype=tf.int32) * self.rPAD
        mask = tf.equal(next_relations, comparison_tensor)
        dummy_scores = tf.ones_like(prelim_scores) * -99999.0
        scores = tf.where(mask, dummy_scores, prelim_scores)

        action = tf.to_int32(tf.multinomial(logits=scores, num_samples=1))

        label_action = tf.squeeze(action, axis=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=label_action)

        action_idx = tf.squeeze(action)
        chosen_relation = tf.gather_nd(next_relations, tf.transpose(tf.stack([range_arr, action_idx])))
        chosen_time = tf.gather_nd(next_times, tf.transpose(tf.stack([range_arr, action_idx])))

        return loss, new_state, tf.nn.log_softmax(scores), action_idx, chosen_relation, chosen_time

    def __call__(self, candidate_relation_sequence, candidate_entity_sequence, candidate_time_sequence, current_entities,
                 path_label, query_relation, query_time, range_arr, first_step_of_test, T=3, entity_sequence=0):
        self.baseline_inputs = []
        query_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, query_relation)
        state = self.policy_step.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        prev_relation = self.dummy_start_label
        prev_time = tf.constant(np.zeros(self.batch_size, dtype='int32'))

        all_loss = []
        all_logits = []
        action_idx = []

        with tf.variable_scope("policy_steps_unroll") as scope:
            for t in range(T):
                if t > 0:
                    scope.reuse_variables()
                next_possible_relations = candidate_relation_sequence[t]
                next_possible_entities = candidate_entity_sequence[t]
                next_possible_times = candidate_time_sequence[t]

                current_entities_t = current_entities[t]

                path_label_t = path_label[t]  # [B]

                loss, state, logits, idx, chosen_relation, chosen_time = self.step(next_possible_relations,
                                                                              next_possible_entities,
                                                                              next_possible_times,
                                                                              state, prev_relation, prev_time, query_embedding, query_time,
                                                                              current_entities_t,
                                                                              label_action=path_label_t,
                                                                              range_arr=range_arr,
                                                                              first_step_of_test=first_step_of_test)

                all_loss.append(loss)
                all_logits.append(logits)
                action_idx.append(idx)
                prev_relation = chosen_relation
                prev_time = chosen_time

        return all_loss, all_logits, action_idx