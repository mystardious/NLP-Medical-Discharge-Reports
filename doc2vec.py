"""Doc2Vec Experimentation File"""
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import dataread
import classifierutils
import pandas as pd
from sklearn.utils import shuffle

"""Variables"""

headers = ['allergies', 'family_history', 'history_illness', 'social_history']

"""Import data"""

# header --> [header, original, tokenized, tokenized_labelled]
header_corpus = {}

for header in headers:
    header_corpus[header] = {}
    header_corpus[header]['label'] = header
    header_corpus[header]['original'] = pd.Series(
        dataread.read_file(header+'.txt')
    )
    temp = classifierutils.corpus_preprocess(
        header_corpus[header]['original'],
        header
    )
    header_corpus[header]['labelled'] = temp[0]
    header_corpus[header]['labelled_tokenised'] = temp[1]

mixed_labelled = pd.DataFrame()
for value in header_corpus.values():
    mixed_labelled = mixed_labelled.append(value['labelled'])
    
mixed_labelled_tokenised = pd.DataFrame()
for value in header_corpus.values():
    mixed_labelled_tokenised = mixed_labelled_tokenised.append(value['labelled_tokenised'])
# mixed_labelled_tokenised = mixed_labelled_tokenised.

mixed_labelled_tokenised =  mixed_labelled_tokenised.reset_index(drop=True)
mixed_labelled =  mixed_labelled.reset_index(drop=True)

mixed_labelled = shuffle(mixed_labelled)
mixed_labelled_tokenised = shuffle(mixed_labelled)

# Prepare Data for training our Doc2Vec model
no_rows = 300
tagged_data = [TaggedDocument(words = word_tokenize(sample['TEXT'].lower()), tags = [sample['HEADER']]) for index, sample in mixed_labelled.head(no_rows).iterrows()]
test_data = [TaggedDocument(words = word_tokenize(sample['TEXT'].lower())) for index, sample in mixed_labelled.tail(no_rows).iterrows()]
print(test_data)

# Train data

"""Doc2Vec Parameters

dm
    defines the training algorith. 
        dm = 1
            Distributed Memory (PV-DM) - Preserves the word order in document
        dm = 0
            Distributed Bag of Word (PV-DBOW) - No word order preservation.

size
alpha
    Initial Learning Rate
min_alpha
    Learning rate will linearly drop to min_alpha as training progresses
min_count
    Ignores all words with total frequency lower than this

"""

# max_epochs = 100
# alpha = 0.025

# model = Doc2Vec(size=vec_size,
#                 alpha=alpha, 
#                 min_alpha=0.00025,
#                 min_count=1,
#                 dm =1,)

# model = Doc2Vec(vec_size = 50,
#                 min_count = 2, 
#                 epochs = 40,
#                 workers = 4)

# model.build_vocab(tagged_data)

# model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")