from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import sys
def get_sequence_length(sequence, scope=None,dtype=tf.int32):
    """
    This is a hacky way of determining the actual length of a sequence that has been padded with zeros.
    """
    '''
    with tf.variable_scope(scope, 'SequenceLength'):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=[-1]))
        length = tf.cast(tf.reduce_sum(used, axis=[-1]), tf.int32)
        return length
    '''
    with tf.variable_scope(scope, 'SequenceLength'):
        used = tf.sign(tf.abs(sequence))
        length = tf.cast(tf.reduce_sum(used, axis=[-1]), dtype)
        return length


def positional_encoding(sentence_size, embedding_size):
    encoding = np.zeros([sentence_size, embedding_size])
    for l in range(sentence_size):
        for v in range(embedding_size):
            encoding[l, v] = (1 - float(l)/sentence_size) - (float(v)/embedding_size)*(1 - 2.0*l/sentence_size)
    return encoding

def compute_likelihood(probs,targets,vocab_size):
    '''
        probs:   tensor with shape [batch_size, sentense_length, vocab_size]
        targets: tensor with shape [batch_size, sentence_length]
    '''
    target_mask = tf.cast(tf.sign(targets),dtype = tf.float32)
    seq_length = get_sequence_length(target_mask,dtype=tf.float32) 

    target = tf.one_hot(targets,vocab_size,dtype = tf.float32) #[N,sentence_length, vocab_size]
    word_probs = tf.reduce_sum(probs * target,axis=2) #[N,sentence_length]
    '''
    print (target.get_shape())
    print(word_probs.get_shape())
    sys.exit()
    word_probs = []
    for index,prob in enumerate(probs):
        target = targets[index] # [N] 
        word_prob = tf.reduce_sum(prob*target,axis=1) # [N]
        word_probs.append(word_prob)
    word_probs = tf.stack(word_probs,axis = 1) # [N,L]
    '''
    word_logprobs = tf.log(word_probs)*target_mask
    likelihood = tf.exp(tf.reduce_sum(word_logprobs,axis=1)/seq_length)

    return likelihood

def cosine_similarity(a,b):
    assert type(a) == type(b),(type(a),' and ',type(b))
    if not isinstance(a,tuple):
        a = (a,)
        b = (b,)
    cos_sim = []
    for i in range(len(a)):
        normalize_a = tf.nn.l2_normalize(a[i],1)        
        normalize_b = tf.nn.l2_normalize(b[i],1)
        cos_sim.append(tf.reduce_sum(tf.multiply(normalize_a,normalize_b),axis=1))
    return tf.reduce_mean(cos_sim)

