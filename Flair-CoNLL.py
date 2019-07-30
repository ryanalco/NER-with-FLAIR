"""
Created on Tues Jul 23 12:29:48 2019

@author: RyanAlco
"""

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings
from typing import List
#from flair.data import Sentence

columns = {0: 'text', 1: 'pos', 2: 'synt', 3: 'ner'}

data_folder = 'resources'

corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='eng.train.txt',
                              test_file='eng.testa.txt').downsample(.1)
                              #dev_file='eng.testb.txt')

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

from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('eng.testb.txt',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=10)



''' Practice with creating word embeddings using glove'''
'''
glove_embedding = WordEmbeddings('glove')
sentence = Sentence('The grass is green .')

# embed a sentence using glove.
glove_embedding.embed(sentence)

# now check out the embedded tokens.
for token in sentence:
    print(token)
    print(token.embedding)
'''






'''
try:
    with open('/documents/resources', 'r') as fh:
        print ('found')
        # Store configuration file values
except FileNotFoundError:
    print ('not found')
    # Keep preset values
    
'''