import sys
import os
import json
import pickle

import numpy as np
import tensorflow as tf

from tqdm import tqdm

from util.data_utils import DataSet 
from util.read_data import tokenize, pad_seq

from abc import ABCMeta, abstractmethod
def sigmoid(x):
    return 1 / (1 + np.exp(-1.*x))

class BaseModel(metaclass=ABCMeta):
    """ Code from mem2nn-tensorflow. """
    def __init__(self, params):
        
        ## dirs ##
        self.save_dir = params.save_dir
        self.load_dir = params.load_dir

        ## set params ##
        self.action = params.action
        self.params = params

    def __del__(self):
        if hasattr(self, "sess"):
            self.sess.close()

    def set_params(self, **kargs):
        self.params = self.params._replace(**kargs)

    
    def train_process(self,train_data,val_data,verbose=False):
        params = self.params
        min_loss = None
        try :
            for i in range(params.num_epochs):
                self.train(train_data,i)
                val_loss = self.eval(val_data,verbose)
                if min_loss is None or val_loss < min_loss:
                    if verbose and min_loss is not None:
                        print ('val_loss %f < min_loss %f'%(val_loss,min_loss))
                    min_loss = val_loss
                    self.save(self.global_step)
                self.train_finish()
        except KeyboardInterrupt:
            #self.save(self.global_step)
            pass
    
    def test_process(self,dialogues,options=None,mode='gen'):
        params = self.params
        if mode == 'gen' or mode == 'type':
            assert options is None
            if type(dialogues) is not list:
                dialogues = [dialogues]
            dialogues_idx = self.words.texts_to_sequences(dialogues)
            for d in dialogues_idx:
                pad_seq(d,params.dialogue_size,truncate='front',value=self.words.word2idx['<PAD>'])
            if mode == 'gen':
                test_data = DataSet(params.batch_size,(dialogues_idx,),shuffle=False,name='generate_data')
            else :
                test_data = DataSet(1,(dialogues_idx,),shuffle=False,name='type_data')

        elif mode == 'match':
            assert options is not None
            ## process dialogue
            dialogues_idx = self.words.texts_to_sequences(dialogues)
            for d in dialogues_idx:
                pad_seq(d,params.dialogue_size,truncate='front',value=self.words.word2idx['<PAD>'])

            ## process options
            options_idx = [self.words.texts_to_sequences(option) for option in options]
            for option in options_idx:
                for o in option:
                    pad_seq(o,params.response_size,truncate='back',value=self.words.word2idx['<PAD>'])
            ## create data set
            test_data = DataSet(1,(dialogues_idx,options_idx),shuffle=False,name='match_test_data')
   
        return self.test(test_data,mode)
    
    @abstractmethod
    def train(self, train_data,epoch):
        pass
    
    @abstractmethod
    def eval(self, eval_data):
        pass
    
    @abstractmethod
    def test(self,test_data,mode):
        pass
    
    def idx2words(self,idx):
        if type(idx) == list or type(idx) == tuple:
            words = []
            for id in idx:
                words.append(self.words.index_to_word(id))
                if id == self.words.word2idx['<EOS>']:
                    break
            words = ' '.join(words)
        elif type(idx) == int:
            words = self.words.index_to_word(idx)
        else:
            raise Exception('Not support type :',type(idx))
        return words

    
   
    def load(self):
        checkpoint = tf.train.get_checkpoint_state(self.load_dir)
        if checkpoint is None:
            print("Error: No saved model found. Please train first.")
            sys.exit(0)
        self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
    
    def save(self, step):
        assert not self.action == 'test'
        print("Saving model to dir %s" % self.save_dir)
        import os
        # self.saver.save(self.sess, os.path.join(self.save_dir, 'run'), self.global_step)
        self.saver.save(self.sess, os.path.join(self.save_dir, 'run'), step)
        self.save_params()
        self.save_words()

    def save_words(self):
        assert self.action == 'train'
        filename = os.path.join(self.save_dir, "words.pickle")
        with open(filename, 'wb') as file:
            pickle.dump(self.words, file)
    
    @abstractmethod
    def save_params(self):
        pass

class NNBaseModel(BaseModel):
    def __init__(self,params):
        super().__init__(params)
        
        ## build model graph ##
        self.graph = tf.Graph()
        '''
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.params.gpu_fraction,
                                    allocator_type ='BFC')
        '''
        #self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = tf.Session(graph=self.graph)
        

        with self.graph.as_default():
            ## global step ##
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            ## epoch
            self.epoch = tf.Variable(0, name='epoch', trainable=False)
            self.next_epoch = tf.assign(self.epoch,self.epoch+1)
            ## validation loss ##
            self.min_validation_loss = tf.Variable(np.inf, name='validation_loss', trainable=False)
            self.new_validation_loss = tf.placeholder('float32', name='new_validation_loss');
            self.assign_min_validation_loss = self.min_validation_loss.assign(self.new_validation_loss).op
            
            if self.action == 'train':
                self.build(forward_only=False)
            else:
                self.build(forward_only=True)
            self.init_op = tf.global_variables_initializer()

        ## init saver ##
        with self.graph.as_default():
            self.saver = tf.train.Saver()

        ## init variables ##
        if not self.load_dir == '':
            print("Loading model ...")
            self.load()
        else:
            print("Init model ...")
            self.sess.run(self.init_op)

        ## summary writer##
        if params.action == 'train':
            self.summary_dir = os.path.join(self.save_dir, 'train_summary')
            self.validation_summary_dir = os.path.join(self.save_dir, 'train_validation_summary')
            self.var_summary_dir = os.path.join(self.save_dir, 'train_var_summary')

            self.summary_writer = tf.summary.FileWriter(logdir=self.summary_dir, graph=self.sess.graph)
            self.validation_summary_writer = tf.summary.FileWriter(logdir=self.validation_summary_dir,
                                                                   graph=self.sess.graph)
            self.var_summary_writer = tf.summary.FileWriter(logdir=self.var_summary_dir, graph=self.sess.graph)

        ## define session run lists ##
        if params.action == 'train':
            self.def_run_list(forward_only = False)
        else:
            self.def_run_list(forward_only = True)
    
    @abstractmethod
    def build(self,forward_only):
        pass
    
    @abstractmethod
    def def_run_list(self,forward_only):
        pass
    
    @abstractmethod
    def get_feed_dict(self, batch, **kwargs):
        pass

    @abstractmethod
    def save_params(self):
        pass
    
    def train(self, train_data,epoch):
        params = self.params
        num_batch = train_data.get_batch_num()
        t=tqdm(range(num_batch),desc='Epoch %d'%epoch,maxinterval=86400, ncols=100)
        for _ in t:
            batch = self.get_batch(train_data,full_batch =True,mode='train')
            summ,var,global_step,_,*ret=self.train_batch(batch)
            
            self.summary_writer.add_summary(summ, global_step)
            self.summary_writer.add_summary(var, global_step)
            postfix = self.get_postfix(ret,mode='train')
            t.set_postfix(postfix)
        
        train_data.reset()
        self.small_testing(train_data)

    def eval(self, eval_data,verbose=False):
        num_batches = eval_data.get_batch_num(full_batch=False)
        tot_loss = 0
        t=tqdm(range(num_batches),desc='Eval',maxinterval=86400, ncols=100)
        for _ in t:
            batch = self.get_batch(eval_data,full_batch = False,mode='eval')
            loss,*ret, global_step = self.eval_batch(batch)
            
            tot_loss += loss*len(batch[0])
           
            fix = [loss]
            fix.extend(ret)
            postfix = self.get_postfix(fix,mode='eval')
            t.set_postfix(postfix)
        
        ## write summary
        ave_loss = tot_loss/eval_data.size
        loss_summary = tf.Summary()
        loss_summary.value.add(tag="loss", simple_value=ave_loss)
        self.validation_summary_writer.add_summary(loss_summary, global_step)
        
        ## reset dataset
        eval_data.reset()
        
        ## testing model on val_data
        self.small_testing(eval_data)
        
        if verbose:
            print ('eval loss : %f'%ave_loss)
        return ave_loss
    
    def test(self,test_data,mode):
        num_batches = test_data.get_batch_num(full_batch=False)
        t=tqdm(range(num_batches),desc='Test %s'%mode,maxinterval=86400, ncols=100)
        test_result = []
        for _ in t:
            batch = self.get_batch(test_data,full_batch =False,mode='test')
            ret = self.test_batch(batch,mode)
            if mode == 'gen' or mode == 'type':
                for i in range(len(batch)):
                    response = self.idx2words([words[i] for words in ret]) 
                    test_result.append(response)
            elif mode == 'match':
                test_result.append(ret)
            else:
                raise Exception('Unknown test mode : %s'%mode)
        test_data.reset()
        return test_result
    
    def get_embedding(self):
        params = self.params
        
        W = self.words.max_num_word
        embed_dim = params.embed_dim
        pad_idx = self.words.word2idx['<PAD>']
        self.embedding_mask = tf.get_variable(name='embedding_mask',
                                              initializer=[[0.]*embed_dim if i == pad_idx  else [1.]*embed_dim for i in range(W)], 
                                              dtype=tf.float32,
                                              trainable=False)
        with tf.device("/cpu:0"):
            if params.use_embedding:
                embedding_matrix = self.words.get_embedding_matrix(embed_dim)
                self.embedding = tf.get_variable('embedding', 
                                                 initializer=embedding_matrix,
                                                 trainable=False,
                                                 dtype=tf.float32)
            else:
                self.embedding = tf.get_variable('embedding', [W, embed_dim], initializer=tf.contrib.layers.xavier_initializer())
        print ('embedding size : ',self.embedding.get_shape())
    
    def embedding_lookup(self,tensor):
        with tf.device("/cpu:0"):
            emb = tf.nn.embedding_lookup(self.embedding,tensor)
            if not  self.params.use_embedding:
                mask = tf.nn.embedding_lookup(self.embedding_mask,tensor)
                emb=emb*mask
        return emb

    def get_batch(self,train_data,full_batch,mode):
        return train_data.next_batch(full_batch)

    def train_finish(self):
        self.sess.run(self.next_epoch)
    
    @abstractmethod
    def get_postfix(self,ret,mode):
        pass

    @abstractmethod
    def train_batch(self,batch,**kwargs):
        pass

    @abstractmethod
    def eval_batch(self,batch,**kwargs):
        pass
    
    def small_testing(self,data):
        pass

    @abstractmethod
    def test_batch(self,batch,**kwargs):
        pass
        
    
