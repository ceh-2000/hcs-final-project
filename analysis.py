################ IMPORTS ################

# For preprocessing
import re
import pandas as pd
import numpy as np

# For analysis
import textblob
import gensim
import nltk
import spacy

# For plotting
import seaborn as sns
import matplotlib.pyplot as plt

# For saving
import pickle

################ PREPROCESS TWEETS ################

# Read in all formatted tweets
df = pd.read_csv('formatted_tweets.csv')

# Vectorized operation to remove non-English tweets based on the `en` column
# https://www.geeksforgeeks.org/drop-rows-from-the-dataframe-based-on-certain-condition-applied-on-a-column/
df = df[df['lang'] == 'en']

# Sort based on time stampxx
df = df.sort_values('created_at')

print('Number of tweets after preprocessing: ' + str(len(df)))

counter = 0
tweet_blocks = []
block_text = ''
been_reset = False
tweet_blocks_indices = []

for index, row in df.iterrows():
    cur_time = row['created_at']
    hour = int(re.findall(r'T(\d\d)', cur_time)[0])

    text = row['text']
    text_list = text.split(' ')
    new_text = ' '
    for word in text_list:
        if 'http' not in word:
            new_text += word + ' '

    if hour % 6 == 0:
        if not been_reset:
            tweet_blocks.append(block_text)
            block_text = ''
            been_reset = True
        block_text += ' ' + new_text
    else:
        block_text += ' ' + new_text
        been_reset = False

    tweet_blocks_indices.append(len(tweet_blocks))

# Assign every tweet to a 6-hour tweet block
df['tweet_block'] = tweet_blocks_indices

# df.to_csv('preprocessed_tweets.csv')


################ POLARITY AND SUBJECTIVITY ANALYSIS ################

# Polarity analysis and subjectivity analysis
# From a script I wrote for a workshop: https://www.kaggle.com/clareheinbaugh/dsc-cypher-workshop-nlp-instructor
tweet_block_polarity = [textblob.TextBlob(t).polarity for t in tweet_blocks]
tweet_block_subjectivity = [textblob.TextBlob(t).subjectivity for t in tweet_blocks]

################ PLOT POLARITY ################
plt.figure(figsize = (10, 6))

sns.lineplot(x = np.arange(len(tweet_block_polarity)), y = tweet_block_polarity, linewidth = 3)

# Plot time-ish that ship got stuck
plt.axvline(1, 0, 1, color='r')
plt.annotate('Boat free', xy=(19.2, -0.15), fontsize='x-large')

# Plot time-ish that ship got free
plt.axvline(23, 0, 1, color='g')
plt.annotate('Boat stuck', xy=(1.2, 0.15), fontsize='x-large')

# Plot 0 line
plt.axhline(0, 0, 25, color='#000', linestyle='--')

# Plot x and y labels
plt.ylabel('Polarity', fontdict = {'fontweight' : 'bold'}, fontsize='x-large')
plt.xlabel('6-hour tweet block', fontdict = {'fontweight' : 'bold'}, fontsize='x-large');

# Plot x and y limits
plt.ylim((-0.2, 0.2))

plt.savefig('block_polarity.png')


################ PLOT SUBJECTIVITY ################
plt.figure(figsize = (10, 6))

sns.lineplot(x = np.arange(len(tweet_block_subjectivity)), y = tweet_block_subjectivity, linewidth = 3)

# Plot time-ish that ship got stuck
plt.axvline(1, 0, 1, color='r')
plt.annotate('Boat free', xy=(19.2, 0.8), fontsize='x-large')

# Plot time-ish that ship got free
plt.axvline(23, 0, 1, color='g')
plt.annotate('Boat stuck', xy=(1.2, 0.8), fontsize='x-large')

# Plot x and y labels
plt.ylabel('Subjectivity', fontdict = {'fontweight' : 'bold'}, fontsize='x-large')
plt.xlabel('6-hour tweet block', fontdict = {'fontweight' : 'bold'}, fontsize='x-large');

# Plot x and y limits
plt.ylim((0, 1))

plt.savefig('block_subjectivity.png')
# plt.show()

################ TOPIC MODELING ################

# Container for all texts
texts = []
labels = []

# Let's do some cleaning and prep for analysis using a lemmatizer
# Check for real word and remove conjugation
lemmatizer = nltk.WordNetLemmatizer()

stopwords = nltk.corpus.stopwords.words('english')
print(stopwords)

# CHECK HERE WITH SMALL EXAMPLE
# tweet_blocks = ['hello my name is clare clare', 'hello i love you so much clare', 'hi i think you are cute', 'go away now']

for i in range(0, len(tweet_blocks)):
    tweet_block_no = i
    text = tweet_blocks[i]

    labels.append(tweet_block_no)

    tokenized_text = nltk.word_tokenize(text)

    refined = [lemmatizer.lemmatize(word.lower()) for word in tokenized_text if
               word.isalnum() and word not in stopwords]

    texts.append(refined)

# Actually using gensim: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

# Gensim expects to run on a list of integers, so we must respect this
id2word = gensim.corpora.Dictionary(texts)

# Gensim uses bag of words for topic modeling, so we need to turn our corpus
# into a list of tuples each representing a word found in our original document
corpus = [id2word.doc2bow(text) for text in texts]

print([[(id2word[id], freq) for id, freq in cp] for cp in corpus])

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

print(lda_model.print_topics(num_words=10))

# Topic outputs:
# [(0, '0.003*"canal" + 0.003*"suez" + 0.002*"ship" + 0.002*"http" + 0.001*"stuck" + 0.001*"every" + 0.001*"the" + 0.001*"container" + 0.001*"given" + 0.001*"amp"'),
# (1, '0.006*"suez" + 0.006*"http" + 0.006*"canal" + 0.004*"ship" + 0.002*"stuck" + 0.002*"the" + 0.001*"every" + 0.001*"blocked" + 0.001*"container" + 0.001*"suezcanal"'),
# (2, '0.004*"canal" + 0.004*"suez" + 0.003*"http" + 0.003*"ship" + 0.002*"stuck" + 0.001*"the" + 0.001*"container" + 0.001*"blocked" + 0.001*"i" + 0.001*"amp"'),
# (3, '0.001*"canal" + 0.001*"http" + 0.001*"suez" + 0.001*"ship" + 0.000*"stuck" + 0.000*"container" + 0.000*"blocked" + 0.000*"every" + 0.000*"the" + 0.000*"boat"'),
# (4, '0.003*"suez" + 0.003*"canal" + 0.003*"http" + 0.002*"ship" + 0.001*"stuck" + 0.001*"container" + 0.001*"blocked" + 0.001*"the" + 0.001*"given" + 0.001*"every"'),
# (5, '0.011*"canal" + 0.007*"suez" + 0.005*"http" + 0.004*"ship" + 0.002*"stuck" + 0.002*"the" + 0.001*"boat" + 0.001*"blocking" + 0.001*"cargo" + 0.001*"breaking"'),
# (6, '0.024*"suez" + 0.022*"canal" + 0.015*"http" + 0.007*"egypt" + 0.006*"would" + 0.005*"port" + 0.005*"yacht" + 0.005*"the" + 0.004*"ship" + 0.004*"if"'),
# (7, '0.003*"canal" + 0.003*"http" + 0.003*"suez" + 0.003*"ship" + 0.001*"stuck" + 0.001*"the" + 0.001*"container" + 0.001*"world" + 0.001*"blocked" + 0.001*"suezcanal"'),
# (8, '0.002*"http" + 0.002*"canal" + 0.002*"suez" + 0.001*"ship" + 0.001*"stuck" + 0.001*"container" + 0.001*"blocked" + 0.001*"the" + 0.000*"every" + 0.000*"i"'),
# (9, '0.070*"suez" + 0.069*"canal" + 0.056*"http" + 0.033*"ship" + 0.020*"stuck" + 0.015*"the" + 0.009*"i" + 0.009*"boat" + 0.008*"blocking" + 0.007*"container"'),
# (10, '0.031*"empire" + 0.029*"canal" + 0.028*"suez" + 0.017*"taiwan" + 0.016*"british" + 0.016*"big" + 0.015*"crisis" + 0.015*"beijing" + 0.015*"america" + 0.014*"may"'),
# (11, '0.010*"http" + 0.010*"suez" + 0.009*"canal" + 0.006*"ship" + 0.003*"stuck" + 0.003*"container" + 0.002*"blocked" + 0.002*"the" + 0.002*"every" + 0.002*"suezcanal"'),
# (12, '0.002*"canal" + 0.002*"suez" + 0.002*"http" + 0.002*"ship" + 0.001*"every" + 0.001*"container" + 0.001*"blocked" + 0.001*"stuck" + 0.001*"amp" + 0.001*"suezcanal"'),
# (13, '0.003*"canal" + 0.002*"suez" + 0.002*"http" + 0.002*"ship" + 0.001*"stuck" + 0.001*"container" + 0.001*"the" + 0.001*"every" + 0.001*"amp" + 0.001*"blocked"'),
# (14, '0.000*"canal" + 0.000*"http" + 0.000*"suez" + 0.000*"ship" + 0.000*"stuck" + 0.000*"blocked" + 0.000*"container" + 0.000*"the" + 0.000*"amp" + 0.000*"every"'),
# (15, '0.049*"http" + 0.034*"ship" + 0.031*"canal" + 0.026*"suez" + 0.025*"day" + 0.024*"every" + 0.024*"amp" + 0.023*"sideways" + 0.023*"blocked" + 0.019*"world"'),
# (16, '0.004*"canal" + 0.004*"http" + 0.004*"suez" + 0.002*"ship" + 0.001*"stuck" + 0.001*"container" + 0.001*"the" + 0.001*"every" + 0.001*"blocked" + 0.001*"amp"'),
# (17, '0.064*"http" + 0.046*"ship" + 0.037*"canal" + 0.033*"container" + 0.029*"suez" + 0.026*"blocked" + 0.025*"stuck" + 0.020*"given" + 0.020*"suezcanal" + 0.017*"ever"'),
# (18, '0.051*"http" + 0.038*"canal" + 0.037*"ship" + 0.034*"suez" + 0.024*"blocked" + 0.023*"container" + 0.021*"every" + 0.019*"amp" + 0.017*"stuck" + 0.015*"the"'),
# (19, '0.007*"suez" + 0.007*"canal" + 0.005*"ship" + 0.005*"http" + 0.002*"stuck" + 0.002*"the" + 0.002*"container" + 0.002*"blocked" + 0.001*"suezcanal" + 0.001*"day"')]

################ ENTITY EXTRACTION ################
# Again, using this workshop I created: https://www.kaggle.com/clareheinbaugh/dsc-cypher-workshop-nlp-instructor

def get_entities(article_text):
    nlp = spacy.load('en_core_web_sm')

    doc = nlp(article_text)

    named_ents = {}

    for entity in doc.ents:
        if entity.label_ not in named_ents:  # If we haven't seen this label yet, add to our dictionary
            named_ents[entity.label_] = []
        named_ents[entity.label_].append(entity)  # Keep track of the entity
    return named_ents


all_entities = []

for text in tweet_blocks:
    entities = get_entities(text)
    refined = {}

    # These entities make the most sense
    refined['PERSON'] = list(set([str(i) for i in entities.get('PERSON')]))
    refined['NORP'] = list(set([str(i) for i in entities.get('NORP')]))
    refined['ORG'] = list(set([str(i) for i in entities.get('ORG')]))
    refined['LOC'] = list(set([str(i) for i in entities.get('LOC')]))

    all_entities.append(refined)

print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
print(all_entities)

entity_file = open('entities', 'ab')
pickle.dump(all_entities, entity_file)
entity_file.close()