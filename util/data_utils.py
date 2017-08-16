# Common data loading utilities.
import pickle
import copy
import os
import numpy as np
import math
import random
import string
from collections import OrderedDict

class DataSet:
    def __init__(self, batch_size, data_list, shuffle=True, name="dataset"):
        assert batch_size <= len(data_list[0]), "batch size cannot be greater than data size."
        self.name = name
        self.data_list = [np.array(data,dtype=np.int32) for data in data_list]
        #self.xs = np.array(xs, dtype=np.int32)
        #self.ys = np.array(ys, dtype=np.int32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.count = len(self.data_list[0])
        self.setup()

    def setup(self):
        self.indexes = list(range(self.count))  # used in shuffling
        self.current_index = 0
        self.num_batches = int(self.count / self.batch_size)
        self.reset()

    def next_batch(self,full_batch = True):
        '''
        if full batch is True, 
            return data with batch size
        else, 
            when the rest data is not enough, still see it as a batch
        '''
        # assert self.has_next_batch(full_batch), "End of epoch. Call 'complete_epoch()' to reset."
        if full_batch :
            from_, to = self.current_index, self.current_index + self.batch_size
        else:
            from_, to = self.current_index, min(self.current_index + self.batch_size,self.count)
        cur_idxs = self.indexes[from_:to]
        ret = [data[cur_idxs] for data in self.data_list]
        #xs = self.xs[cur_idxs]
        #ys = self.ys[cur_idxs]
        self.current_index = to
        #return xs, ys
        return ret

    def get_all(self):
        ret = [data[cur_idxs] for data in self.data_list]
        #return self.xs[self.indexes],  self.ys[self.indexes]
        return ret

    def get_batch_cnt(self, cnt):
        if not self.has_next_batch(cnt):
            self.reset()
        from_, to = self.current_index, self.current_index + cnt
        cur_idxs = self.indexes[from_:to]
        ret = [data[cur_idxs] for data in self.data_list]
        #xs = self.xs[cur_idxs]
        #ys = self.ys[cur_idxs]
        self.current_index += cnt
        #return xs, ys
        return ret

    def get_random_cnt(self,cnt):
        inx = list(range(len(self.indexes)))
        random.shuffle(inx)
        inx = inx[:cnt]
        choices = [self.indexes[i] for i in inx]
        ret = [data[choices] for data in self.data_list]
        '''
        xs = self.xs[choices]
        ys = self.ys[choices]
        return xs, ys
        '''
        return ret

    def split_dataset(self, split_ratio):
        """ Splits a data set by split_ratio.
        (ex: split_ratio = 0.3 -> this set (70%) and splitted (30%))
        :param split_ratio: ratio of train data
        :return: val_set
        """
        end_index = int(self.count * (1 - split_ratio))

        # do not (deep) copy data - just modify index list!
        val_set = copy.copy(self)
        val_set.count = self.count - end_index
        val_set.indexes = list(range(end_index, self.count))
        val_set.num_batches = int(val_set.count / val_set.batch_size)
        val_set.reset()
        self.count = end_index
        self.setup()
        return val_set
    
    def get_batch_num(self, full_batch = True):
        '''
            use this function is better than access num_batches directly
        '''
        if full_batch or self.count%self.batch_size == 0:
            return self.num_batches
        else:
            return self.num_batches + 1
    
    def reset(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indexes)
   
    @property
    def size(self):
        return self.count
   
    def __getitem__(self, key):
        
        # do not (deep) copy data - just modify index list!
        val_set = copy.copy(self)
        if isinstance(key,slice):
            start = 0 if key.start == None else key.start
            stop = self.count if key.stop == None else key.stop
            step = 1 if key.step == None else key.step 
            
            val_set.count = int((stop - start)/step)
            val_set.indexes = [self.indexes[i] for i in range(start,stop,step)]
            val_set.num_batches = int(val_set.count / val_set.batch_size)
            val_set.reset()
            return val_set
        else:
            raise NotImplementedError
        
def read_stopwords(path):
    print ('loading stopwords from %s'%path)
    stopwords_set = set()
    with open(path,'r') as f:
        for line in f:
            line = line.strip('\n')
            stopwords_set.add(line)
    return stopwords_set

class WordTable:
    def __init__(self, embed_size=0,num_words=None,use_stopwords=False):
        self._word_counts = OrderedDict()
        self._word_docs = {}

        self._word2idx = {'<PAD>':0,'<GO>':1,'<UNK>':2,'<EOS>':3}
        self._idx2word = ['<PAD>','<GO>', '<UNK>','<EOS>']  # zero padding will be <eos>
        
        self.all_doc_count = 0.

        self._allow_add = True

        self._num_words = num_words
        self.embed_size = embed_size
            
        self.stopwords_set = set()
        if use_stopwords:
            self.stopwords_set = read_stopwords('data/stopwords.txt')
        self.remove_word = '！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～　. ' 
        self.translator = str.maketrans(self.remove_word,' '*len(self.remove_word))
    def tokenize(self,text):
        ret = []
        text = text.translate(self.translator).strip()
        
        word = ''
        for char in text :
            if char in string.ascii_letters: 
                word += char.lower()
            else:
                if word != '':
                    ret.append(word)
                    word = ''
                ret.append(char)
        if word != '':
            ret.append(word)
        return ret

    def add_vocab_by_texts(self, texts):
        """ Add vocabularies to dictionary. """
        for text in texts:
            self.add_vocab_by_text(text)

    def add_vocab_by_text(self, words):
        if not self.is_allow_add():
            raise Exception('WordTable is not allowed to add new vocab after calling texts_to_sequences function')
        
        words = self.tokenize(words)
        for w in words:
            if w in self._word_counts:
                self._word_counts[w] += 1
            else:
                self._word_counts[w] = 1

        for w in set(words):
            if w in self._word_docs:
                self._word_docs[w] += 1
            else:
                self._word_docs[w] = 1
        self.all_doc_count += 1

    def texts_to_sequences(self,texts):
        if self.is_allow_add():
            wcounts = list(self._word_counts.items())
            wcounts.sort(key=lambda x: x[1], reverse=True)
            sorted_voc = [wc[0] for wc in wcounts]
            
            # note that index 0,1 are reserved, never assigned to an existing word
            self._word2idx.update(dict(list(zip(sorted_voc, list(range(4, len(sorted_voc) + 4))))))
            self._idx2word.extend(sorted_voc)
            
            self.idx2dc = {}
            for w, c in list(self._word_docs.items()):
                self.idx2dc[self._word2idx[w]] = c
            self._allow_add = False
        
        res = []
        for vect in self._texts_to_sequences_generator(texts):
            res.append(vect)
        return res
  
    def _texts_to_sequences_generator(self,texts):
        num_words = self._num_words
        
        for text in texts :
            seq = self.tokenize(text)
            vect = []
            for w in seq :
                if w not in self.stopwords_set:
                    i = self._word2idx.get(w)
                    if i is None or (num_words and i >= num_words):
                        vect.append(self._word2idx['<UNK>'])
                    else:
                        vect.append(i)
            vect.append(self._word2idx['<EOS>'])
            yield vect

    def find_keyterm_by_word(self, words,mode):
        words = self.tokenize(words)
        idx = [self._word2idx.get(w,self._word2idx['<UNK>']) for w in words] 
        idx = self.find_keyterm_by_idx(idx)
        if mode == 'word':
            return self._idx2word[idx]
        elif mode == 'id':
            return idx
        else:
            raise Exception('Unknown mode for WordTable find_keyterm_by_word()')
    
    def find_keyterm_by_idx(self, words):
        doc_counts = np.array([self.idx2dc.get(word, np.inf) for word in words])
        keyterm = words[np.argmin(doc_counts)]
        return keyterm

    def vectorize(self, word):
        """ Converts word to vector.
        :param word: string
        :return: 1-D array (vector)
        """
        vec = self.word2vec.get(word)
        if vec == None:
            vec = self._create_vector(word)
        return vec

    def _create_vector(self, word):
        # if the word is missing from Glove, create some fake vector and store in glove!
        vector = np.random.uniform(-0.25, 0.25, (self.embed_size,))
        self.word2vec[word] = vector
        print("create_vector => %s is missing" % word)
        return vector
    
    def index_to_word(self, index):
        return self._idx2word[index]

    def read_stopwords(self,path):
        stopwords_set = set()
        with open(path,'r') as f:
            for line in f :
                word=line.strip('\n')
                stopwords_set.add(word)
        
        return stopwords_set

    def is_allow_add(self):
        return self._allow_add
   
    def df(self,word):
        return self.idx2dc[self._word2idx[word]] / self.all_doc_count

    def load_word2vec(self,embed_dim):
        path = 'data/wiki_embedding/wiki_vec_d%d'%embed_dim 
        word2vec = {}
        if os.path.exists(path + '.cache'):
            with open(path + '.cache', 'rb') as cache_file:
                word2vec = pickle.load(cache_file)

        else:
            # Load n create cache
            with open(path) as f:
                for line in f:
                    l = line.split()
                    word2vec[l[0]] = [float(x) for x in l[1:]]

            with open(path + '.cache', 'wb') as cache_file:
                pickle.dump(word2vec, cache_file)
        return word2vec

    def get_embedding_matrix(self,embed_dim):
        ## load word2vec
        self.embed_size = embed_dim
        if not hasattr(self,'word2vec'):
            self.word2vec = self.load_word2vec(embed_dim)
        
        ## convert to embedding matrix
        embedding_matrix = np.zeros((self.max_num_word,embed_dim),dtype=np.float32)
        for idx,w in enumerate(self._idx2word[:self.max_num_word]):
            if idx != self._word2idx['<PAD>']:
                embedding_matrix[idx] = self.vectorize(w) 
            
        return embedding_matrix

    @property
    def word2idx(self):
        return self._word2idx
    @property
    def vocab_size(self):
        return len(self._idx2word)
    
    @property
    def max_num_word(self):
        if self._num_words != None:
            return min(len(self._idx2word),self._num_words+4)
        else:
            return len(self._idx2word)

    @property
    def all_word(self):
        return self._idx2word[:]

def load_glove(dim):
    """ Loads GloVe data.
    :param dim: word vector size (50, 100, 200)
    :return: GloVe word table
    """
    word2vec = {}

    path = "data/glove/glove.6B." + str(dim) + 'd'
    if os.path.exists(path + '.cache'):
        with open(path + '.cache', 'rb') as cache_file:
            word2vec = pickle.load(cache_file)

    else:
        # Load n create cache
        with open(path + '.txt') as f:
            for line in f:
                l = line.split()
                word2vec[l[0]] = [float(x) for x in l[1:]]

        with open(path + '.cache', 'wb') as cache_file:
            pickle.dump(word2vec, cache_file)

    print("Loaded Glove data")
    return word2vec
if __name__ == '__main__':
    from read_data import read_data, tokenize
    path = ['../data/format_data/pts_processed','../data/format_data/Gossiping_processed']
    train_data_set, word_table, max_dialogue_length, max_response_length = read_data(path,10,use_stopwords=False)
   
    '''
    all_word = word_table.all_word[3:]
    all_word = [(word,word_table.df(word)) for word in all_word]

    sorted_word = sorted(all_word,key=lambda x:x[1],reverse=True)

    for w in sorted_word:
        print (w)
    '''
    '''
    while True:
        text = input('>')
        print (word_table.find_keyterm_by_word(text,mode = 'word'))
    '''
