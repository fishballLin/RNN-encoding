#!/usr/bin/python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import json
import pickle
import tensorflow as tf
import time
import argparse
import readline
import numpy as np
from collections import namedtuple
from copy import deepcopy

from util.read_data import read_data,read_test_data
from util.read_data import tokenize
from util.data_utils import WordTable

from model.RNN.vanilla import VanillaRNN
## accessible model ##
MODEL = {'RNN_vanilla':          VanillaRNN}

test_path = 'data/AIFirst_test_problem.txt'

pts_path = './data/format_data/pts_processed'
def load_params_dict(filename):
    with open(filename, 'r') as file:
        params_dict = json.load(file)
    return params_dict
def load_words_dict(filename):
    with open(filename, 'rb') as file:
        words_dict = pickle.load(file)
    return words_dict


def train_normal(model, params, expert_params, lm_params, words, train, val):
    main_model = model(words, params, expert_params, lm_params)
    main_model.pre_train(train, val)
    main_model.save_params()


def test_normal(model, params, expert_params, lm_params, words, test):
    main_model = model(words, params, expert_params, lm_params)
    main_model.decode(test, sys.stdout, sys.stdin, all=False)
    main_model.eval(test, name='Test')


## arguments parser ##
parser = argparse.ArgumentParser(description='dialogue prediction')

# Action and target and arch
parser.add_argument('action', choices=['train', 'test_gen','test_type','test_match'])
parser.add_argument('arch', default = 'RNN',choices=['RNN'])
parser.add_argument('version',default='')

# directory
parser.add_argument('--load_dir', default='')

# training options
parser.add_argument('--task', default='pts',choices=['pts'])
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--learning_rate', default=0.002, type=float)
parser.add_argument('--rl_learning_rate', default=0.0001, type=float)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--acc_period', default=10, type=int)
parser.add_argument('--val_period', default=40, type=int)
parser.add_argument('--gpu_fraction', default=1, type=float)
parser.add_argument('--embed_dim',default=300,type=int)
parser.add_argument('--decay_steps',default=15000,type=int)
parser.add_argument('--use_stopwords',action='store_true')
parser.add_argument('--use_embedding',action='store_true')

# RNN params
parser.add_argument('--RNN_hidden_size',default=128,type=int)

args = parser.parse_args()

model_version= {'RNN'    : ['vanilla','condition','classifier']}
assert args.version in model_version[args.arch], '%s model has no version %s'%(args.arch,args.version)
## main function ##
def main(_):
    ## import main model ##
    main_model_name = '{}_{}'.format(args.arch,args.version)
    if main_model_name in MODEL:
        MainModel = MODEL[main_model_name]
    else:
        raise Exception("Unsupported target-arch pair!")

    ## create save dir ##
    save_dir = os.path.join('save', '{}_{}_{}'.format(args.arch, args.version, args.task))
    if args.arch == 'RNN':
        save_dir += '_%d'%args.RNN_hidden_size
    
    if args.use_embedding:
        save_dir += '_embed'
    if args.use_stopwords:
        save_dir += '_stopwords'
    args.save_dir = save_dir

    
    args.dialogue_size, args.response_size, args.dialogue_counts= 0,0,0
    words = WordTable()
    ## data set ##
    if args.action == 'train':
        if args.task == 'pts':
            path = [pts_path]
        train, words, args.dialogue_size, args.response_size = read_data(path ,args.batch_size,args.use_stopwords)
        val = train.split_dataset(args.val_ratio)
    
        args.dialogue_counts = 1
        print("train data count: {}".format(train.size))
        print('val data count: {}'.format(val.size))
        print("word2idx:", words.vocab_size)
        print("dialogue size: {}".format(args.dialogue_size))
        print("response size: {}".format(args.response_size))

    ## create params ##
    params_dict = vars(args)
    params_class = namedtuple('params_class', params_dict.keys())
    params = params_class(**params_dict)

    ## load params and words from load_dir ##
    if not params.load_dir == '':
        params_filename = os.path.join(params.load_dir, 'params.json')
        load_params = load_params_dict(params_filename)
        if params.action == 'train' and (not load_params['task'] == params.task):
            raise Exception("incompatible task with load model!")
        if (not load_params['arch'] == params.arch):
            raise Exception("incompatible main model with load model!")
        params = params._replace(**load_params)
        
        words_filename = os.path.join(params.load_dir, 'words.pickle')
        words = load_words_dict(words_filename)
    else:
        if tf.gfile.Exists(save_dir):
            tf.gfile.DeleteRecursively(save_dir)
        os.makedirs(save_dir, exist_ok=True)

    ## run action ##
    main_model = MainModel(words, params)
    if args.action == 'train':
        main_model.train_process(train, val,verbose = True)
        main_model.save_params()
    elif args.action == 'test_gen':
        main_model.test_process(test,'gen')
    elif args.action == 'test_type':
        while True:
            inputs = input('> ')
            gen = main_model.test_process(inputs,mode='type')
            print ('> %s'%gen[0])
    elif args.action == 'test_match':
        dialogue,options = read_test_data(test_path)
        probs = main_model.test_process(dialogue,options,mode='match')  
        for index, d in enumerate(dialogue):
            option = options[index]
            print (d)

            for i,o in enumerate(option):
                print ('    (%d) %s : '%(i,o), end='')
                print (probs[index][i],sep=' | ')
            print ('choose : ',np.argmax(probs[index],axis=0))
           
            print('\n===========================')
                
if __name__ == '__main__':
    tf.app.run()
