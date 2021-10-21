"""
Created on Tues Jul 23 12:29:48 2019

@author: RyanAlco
"""

from flair.data import Corpus
from flair.models import SequenceTagger
from flair.data import Sentence
#from flair.datasets import ColumnCorpus
import flair.datasets
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings
from typing import List
import nltk, re, pprint
from nltk import word_tokenize


#Perform NER on CoNLL dataset:
columns = {0: 'text', 1: 'pos', 2: 'synt', 3: 'ner'}

data_folder = 'resources'

corpus: Corpus = flair.datasets.ColumnCorpus(data_folder, columns,
                              train_file='eng.train.txt',
                              test_file='eng.testa.txt').downsample(.1)
                              #dev_file='eng.testb.txt')
                             
 
#Perform NER on a different already prepared dataset:                             
#corpus = flair.datasets.WIKINER_ENGLISH().downsample(0.1)  
corpus = text
                              
                            
#Perform flair on Biology dataset:
columns = {0: 'text', 1: 'ner'}
data_folder = 'resources'
corpus: Corpus = flair.datasets.ColumnCorpus(data_folder, columns,
                              train_file='BiologyNer.txt').downsample(.1)                      
                              

#corpus = corpus.downsample(.4)

#len(corpus.train)
#print(corpus.train[0].to_tagged_string('pos'))


#print (corpus.make_tag_dictionary('upos'))
#corpus.make_tag_dictionary('ner')
#corpus.make_label_dictionary()

tag_type = 'ner'

tag_dictionary = corpus.make_tag_dictionary(tag_type = tag_type)

embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings('glove'),
        #PooledFlairEmbeddings('news-forward', pooling = 'min'),
        PooledFlairEmbeddings('news-forward', pooling='min'),
        PooledFlairEmbeddings('news-backward', pooling='min'),
        #PooledFlairEmbeddings('news-backward', pooling = 'min'), 
        ]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings = embedding_types)

from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size = 256,
                                        embeddings = embeddings,
                                        tag_dictionary = tag_dictionary,
                                        tag_type = tag_type)
                                        #use_crf=True)

from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('eng.train.txt',
            #'resources/taggers/example-ner',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=10)



''' Practice with creating word embeddings using glove'''

glove_embedding = WordEmbeddings('glove')
sentence = Sentence('The grass is green .')

# embed a sentence using glove.
glove_embedding.embed(sentence)

# now check out the embedded tokens.
for token in sentence:
    print(token)
    print(token.embedding)





#Beginning NLTK processing on raw biology test

#raw = open('BioTest.txt').read()
#tokens = word_tokenize(raw)

def process(doc):  
    raw = open(doc).read()
    tokens = word_tokenize(raw)
    tagged_words = nltk.pos_tag(tokens)
    print(tagged_words)
    ne_tagged = nltk.ne_chunk(tagged_words, binary = False)
    return ne_tagged
    
    #sentences = nltk.sent_tokenize(document) 
    #print (sentences)
    
    for sent in sentences:
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label'):
                print((c[0] for c in chunk), ' '.join(chunk.label()))
    #print (sentences)
    
    tagger = SequenceTagger.load('ner')
    #sentences = [nltk.word_tokenize(sent) for sent in sentences]
    #sentences = [nltk.pos_tag(sent) for sent in sentences]
    for sent in sentences:
        tagger.predict(sent)
        #sent = str(sent)
        print(sent.to_tagged_string())
    

text = process('BioTest.txt')
