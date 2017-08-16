import os
import re
import random
import string
from util.data_utils import DataSet, WordTable
#from data_utils import DataSet, WordTable


ptt_path = './data/Gossiping-QA-Dataset.txt'
test_path = './data/AI_test_data.txt'
english_punc = string.punctuation.translate(str.maketrans('','','.+-'))
chinese_punc = '，？！。：；（）　～'
#punctuation = english_punc + chinese_punc
#translator = str.maketrans(punctuation,' '*(len(punctuation)))
translator = str.maketrans('?! ','？！　')
def tokenize(sentence):
    """
    Tokenize a string by splitting on non-word characters and stripping whitespace.
    """ 
    ret = []
    word = ''
    for char in sentence :
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

def parse_task(all_lines):
    all_task = []
    count = 0
    for line in all_lines:
        count += 1
        print (count,end='\r')
        line = line.translate(translator).strip(' \n')
        try:
            lines = line.split("\t")
        except ValueError:
            print ('%d : %s'%(count,line))
            continue
        #lines = [tokenize(l.strip()) for l in lines]
       
        '''
        for index in range(len(lines)-1):
            l1 = lines[index]
            l2 = lines[index+1]
            if len(l1) > 5 and len(l1) < 70 and len(l2) > 5 and len(l2) < 70 :
                dialogue = (l1,l2)
                dialogues.append(dialogue)
        '''
        all_task.append(lines)
    return all_task 

def get_tokenizer(dialogues, word_table):
    """
    Recover unique tokens as a vocab and map the tokens to ids.
    """
    for texts in dialogues:
        word_table.add_vocab_by_texts(texts)
    '''
    for ask,response in dialogues:
        word_table.add_vocab(*ask)
        word_table.add_vocab(*response)

        #word_table.count_doc(*(response+ask))
    '''
def pad_seq(sequence,length,value=0,truncate=None):
    if (truncate is not None) and len(sequence) > length:
        if truncate == 'back':
            sequence[:] = sequence[:length]
        elif truncate == 'front':
            sequence[:] = sequence[-length:]
    else:
        for _ in range(length - len(sequence)):
            sequence.append(value)
    
    assert truncate == None or len(sequence) == length
def pad_task(task, max_dialogue_length, max_response_length):
    """
    Pad sentences, stories, and queries to a consistence length.
    """
    dialogues = []
    responses = []
    for dialogue,response in task:
        pad_seq(dialogue,max_dialogue_length)
        pad_seq(response,max_response_length)

        dialogues.append(dialogue)
        responses.append(response)
        
        assert len(dialogue) == max_dialogue_length,('%d,%d'%(len(dialogue),max_dialogue_length))
        assert len(response) == max_response_length

    return dialogues,responses

def truncate_task(stories, max_length):
    stories_truncated = []
    for story, query, answer in stories:
        story_truncated = story[-max_length:]
        stories_truncated.append((story_truncated, query, answer))
    return stories_truncated

def tokenize_task(dialogues, word_table):
    """
    Convert all tokens into their unique ids.
    """
    dialogue_ids = []
    max_dialogue_length = 0
    max_response_length = 0
    for d in dialogues:
        ids = word_table.texts_to_sequences(d)
        for index in range(len(ids)-1):
            l1 = ids[index]
            l2 = ids[index+1]
            if len(l1) > 4 and len(l1) <= 70 and len(l2) > 4 and len(l2) <= 70 :
                max_dialogue_length = max(len(l1),max_dialogue_length)
                max_response_length = max(len(l2),max_response_length)
                dialogue_ids.append((l1[:],l2[:]))
    
    return dialogue_ids, max_dialogue_length, max_response_length

def read_data(path,batch_size,use_stopwords):
    word_table = WordTable(use_stopwords=use_stopwords,num_words = None)
    all_train = []
    max_dialogue_length = 0
    max_response_length = 0

    if type(path) is not list:
        path = [path]

    for p in path :
        with open(p,'r') as f:
            print ('reading data from %s'%p)
            train = parse_task(f.readlines())
            #train = truncate_task(train, truncate_length)
            all_train.extend(train)
               
            print ('article size : %d'%len(all_train)) 

    print ('Get word table...')
    get_tokenizer(all_train, word_table)
    print ('Convert to id sequences')
    all_train,max_dialogue_length, max_response_length = tokenize_task(all_train, word_table)
    print ('pad sequences')
    train_dialogues,train_responses = pad_task(all_train, max_dialogue_length, max_response_length)
    
    train_data_set = DataSet(batch_size, (train_dialogues, train_responses), name='train')
    return train_data_set, word_table, max_dialogue_length, max_response_length

def read_test_data(path):
    dialogue = []
    options = []
    with open(path,'r') as f:
        f.readline()
        for line in f:
            line = line.strip(' 　\n').split(',')
            dialogue.append(line[1].replace('\t',' '))
            option = line[2].strip(' 　').split('\t')
            assert len(option) == 6,option
            for o in option:
                assert o != '' ,option
            options.append(option)
    return dialogue,options
if __name__ == '__main__':
    from data_utils import DataSet, WordTable
    #train_data, word_table,max_dialogue_length,max_response_length = read_data(ptt_path,10)
    #print ('train data size : ',train_data.size)
    print(read_test_data(test_path))
