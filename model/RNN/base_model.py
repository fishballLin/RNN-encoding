import sys
import os
import json
import pickle

import numpy as np
import tensorflow as tf

from tqdm import tqdm

from util.data_utils import DataSet 
from util.read_data import tokenize, pad_seq

from abc import abstractmethod
from model.base_model import NNBaseModel
def sigmoid(x):
    return 1 / (1 + np.exp(-1.*x))

class RNNBaseModel(NNBaseModel):
    def __init__(self, words, params, *args):
        
        ## words ##
        self.words = words
        
        super().__init__(params)
    
    @abstractmethod
    def build(self,forward_only):
        pass
   
    @abstractmethod
    def def_run_list(self):
        pass

    def get_feed_dict(self, batch, is_train, mode):
        if mode == 'train' or mode == 'eval':
            assert (mode == 'train' and is_train == True) or (mode == 'eval' and is_train==False)
            feed_dict ={
                self.x: batch[0],
                self.y_true: batch[1],
                self.y_false: batch[2],
                self.is_training: is_train,
            }
        elif mode == 'test':
            assert is_train == False
            y_shape = batch[1].shape
            feed_dict = {
                self.x: np.repeat(batch[0],y_shape[1],axis=0),     
                self.y_true: np.reshape(batch[1],(y_shape[0]*y_shape[1],y_shape[2])),
                self.is_training: False,
            }
        else:
            raise Exception('known mode for RNN base_model get_feed_dict')
        return feed_dict

    @abstractmethod
    def save_params(self):
        pass
    
    def get_batch(self,train_data,full_batch,mode):
        batch = train_data.next_batch(full_batch)
        if mode == 'train' or mode == 'eval':
            false_batch = train_data.get_random_cnt(batch[0].shape[0])
            batch.append(false_batch[1])
        elif mode == 'test':
            pass
        else:
            raise Exception('unknown mode for get_batch:%s'%mode)
        return batch

    @abstractmethod
    def get_postfix(self,ret,mode):
        pass
    
    def train_batch(self, batch):
        feed_dict = self.get_feed_dict(batch, is_train=True, mode='train')
        return self.sess.run(self.train_list, feed_dict=feed_dict)
    
    def eval_batch(self,batch):
        feed_dict = self.get_feed_dict(batch, is_train=False, mode='eval')
        return self.sess.run(self.eval_list, feed_dict=feed_dict)
    
    def test_batch(self,batch,mode):
        if mode == 'match':
            feed_dict = self.get_feed_dict(batch, is_train=False, mode='test')
            return self.sess.run(self.test_match_list, feed_dict=feed_dict)
        elif mode == 'gen' or mode == 'type':
            raise Exception('RNN model do not support generating test  and typing test')
