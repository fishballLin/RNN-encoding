import sys
import os
import json
import pickle

import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import static_bidirectional_rnn,GRUCell
from tensorflow.contrib.layers import fully_connected, variance_scaling_initializer

from tf_helper.nn import weight, bias, dropout, batch_norm, variable_summary
from tf_helper.nn import gumbel_softmax, attention_decoder, create_opt
from tf_helper.model_utils import get_sequence_length, positional_encoding, compute_likelihood, cosine_similarity

from model.RNN.base_model import RNNBaseModel

class VanillaRNN(RNNBaseModel):
    def build(self,forward_only):
        
        params = self.params
        L, R, D = params.dialogue_size, params.response_size, params.dialogue_counts
        V, W = params.RNN_hidden_size, self.words.vocab_size
        embed_dim = params.embed_dim
        N = params.batch_size

        ## place holder
        #dialogue_ = tf.placeholder('int32', shape=[None, D, L], name='dialogue') ## dialogue input
        dialogue_ = tf.placeholder('int32', shape=[None, L], name='dialogue') ## dialogue input
        true_response_ = tf.placeholder('int32', shape=[None, R], name='true_response') ## decoder target
        false_response_ = tf.placeholder('int32', shape=[None, R], name='false_response')
        
        self.batch_size = tf.shape(true_response_)[0]
        
        dialogue_counts = get_sequence_length(dialogue_)
        true_response_counts = get_sequence_length(true_response_)
        false_response_counts = get_sequence_length(false_response_)
        
        self.is_training = tf.placeholder(tf.bool)

        ## prepare embedding matrix
        with tf.device("/cpu:0"):
            self.embedding_mask = tf.get_variable(name='embedding_mask',
                                                  initializer=[[0.]*embed_dim if i == 0 else [1.]*embed_dim for i in range(W)], 
                                                  dtype=tf.float32,
                                                  trainable=False)
            self.embedding = tf.get_variable('embedding', [W, embed_dim], initializer=tf.contrib.layers.xavier_initializer())

        ## embed dialogue
        dialogue = self.embedding_lookup(dialogue_)#[batch, dialogue, sentence] -> [batch, dialogue, sentence, embed_size]
    
        ## embed response
        true_response = self.embedding_lookup(true_response_)
        false_response = self.embedding_lookup(false_response_)

        num_layers = 1
        ## dialogue encode cell
        dia_encode_fw = tf.contrib.rnn.MultiRNNCell([GRUCell(V) for _ in range(num_layers)])
        dia_encode_bw = tf.contrib.rnn.MultiRNNCell([GRUCell(V) for _ in range(num_layers)])
        ##response encode cell
        res_encode_fw = tf.contrib.rnn.MultiRNNCell([GRUCell(V) for _ in range(num_layers)])
        res_encode_bw = tf.contrib.rnn.MultiRNNCell([GRUCell(V) for _ in range(num_layers)])

        ## build dialogue encoder
        with tf.variable_scope('Dialogue_Encoder') as scope:
            (_,_),(dia_states_fw, dia_states_bw) = tf.nn.bidirectional_dynamic_rnn(dia_encode_fw,
                                                                                   dia_encode_bw,
                                                                                   dialogue,
                                                                                   sequence_length=dialogue_counts,
                                                                                   dtype=tf.float32)
 
            #encoder_output,encoder_state = tf.nn.dynamic_rnn(encoder_cell,dialogue,sequence_length=dialogue_counts,dtype=tf.float32)
            #encoder_state,encoder_output = tf.nn.dynamic_rnn(encoder_cell,dialogue,dtype=tf.float32)
            dia_state = dia_states_fw[0] + dia_states_bw[0]
            
        ## build response encoder
        with tf.variable_scope('Response_Encoder') as scope:
            def response_encoder(response,response_counts,reuse=None):
                with tf.variable_scope('encoder',reuse=reuse):
                    (_,_),(states_fw, states_bw) = tf.nn.bidirectional_dynamic_rnn(res_encode_fw,
                                                                                   res_encode_bw,
                                                                                   response,
                                                                                   sequence_length=response_counts,
                                                                                   dtype=tf.float32)
                return states_fw[0] + states_bw[0]
            true_res_state = response_encoder(true_response,true_response_counts) 
            false_res_state = response_encoder(false_response,false_response_counts,True) 

        with tf.name_scope('Loss'):
            def cosine_similarity(a,b):
                normalize_a = tf.nn.l2_normalize(a,1)        
                normalize_b = tf.nn.l2_normalize(b,1)
                return tf.reduce_sum(tf.multiply(normalize_a,normalize_b),axis=1)

            cosine_true = cosine_similarity(dia_state,true_res_state)
            cosine_false = cosine_similarity(dia_state, false_res_state)
            mean_cosine_true = tf.reduce_mean(cosine_true) 
            mean_cosine_false = tf.reduce_mean(cosine_false) 
            true_loss = tf.losses.mean_squared_error(cosine_true,tf.ones_like(cosine_true,dtype=tf.float32))
            false_loss = tf.losses.mean_squared_error(cosine_false,tf.zeros_like(cosine_false,dtype=tf.float32))
            loss = true_loss+false_loss
        tf.summary.scalar('loss', loss, collections=["SUMM"])
        tf.summary.scalar('cosine_sim_true',mean_cosine_true,collections=['SUMM']) 
        tf.summary.scalar('cosine_sim_false',mean_cosine_false,collections=['SUMM']) 
        ## placeholders
        self.x = dialogue_
        self.y_true = true_response_
        self.y_false = false_response_

        ## output tensors
        self.cosine_true = cosine_true
        self.cosine_false = cosine_false
        self.mean_cosine_true = mean_cosine_true
        self.mean_cosine_false = mean_cosine_false
        self.true_loss = true_loss
        self.false_loss = false_loss
        self.loss = loss
        
        # optimizer ops
        if not forward_only:
            l_rate = self.params.learning_rate
            self.opt_op = create_opt('opt', self.loss, l_rate, self.global_step,decay_steps=10000,clip=5.)

        # merged summary ops
        self.merged_SUMM = tf.summary.merge_all(key='SUMM')
        self.merged_VAR = tf.summary.merge_all(key='VAR_SUMM')
    
    def def_run_list(self,forward_only):
        if not forward_only:
            self.train_list = [self.merged_SUMM,self.merged_VAR,self.global_step,self.opt_op,self.loss]
        self.eval_list = [self.loss,self.mean_cosine_true,self.mean_cosine_false,self.global_step] 
        self.test_match_list = self.cosine_true

    
    def get_postfix(self,ret,mode):
        if mode == 'train':
            return {'loss':ret[0]}
        elif mode == 'eval':
            return {'loss':ret[0],
                    'cos_sim_pos':ret[1],
                    'cos_sim_neg':ret[2]}
        
    
    def save_params(self):
        assert self.action == 'train'
        params = self.params
        filename = os.path.join(self.save_dir, "params.json")
        save_params_dict = {'RNN_hidden_size': params.RNN_hidden_size,
                            'dialogue_size':params.dialogue_size,
                            'response_size':params.response_size,
                            'dialogue_counts':params.dialogue_counts,
                            'arch': params.arch,
                            'task': params.task}
        with open(filename, 'w') as file:
            json.dump(save_params_dict, file, indent=4)

if __name__ == '__main__':
    from util.read_data import read_data
    from util.data_utils import WordTable
