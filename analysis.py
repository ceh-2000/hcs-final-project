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

tweet_blocks.append(block_text)

# Assign every tweet to a 6-hour tweet block
df['tweet_block'] = tweet_blocks_indices

# df.to_csv('preprocessed_tweets.csv')


################ POLARITY AND SUBJECTIVITY ANALYSIS ################

# Polarity analysis and subjectivity analysis
# From a script I wrote for a workshop: https://www.kaggle.com/clareheinbaugh/dsc-cypher-workshop-nlp-instructor
tweet_block_polarity = [textblob.TextBlob(t).polarity for t in tweet_blocks]
tweet_block_subjectivity = [textblob.TextBlob(t).subjectivity for t in tweet_blocks]

for tweet in tweet_block_subjectivity:
    print(tweet)


################ PLOT POLARITY ################
# plt.figure(figsize = (5, 5))
# plt.tight_layout()
#
# sns.lineplot(x = np.arange(len(tweet_block_polarity)), y = tweet_block_polarity, linewidth = 3)
#
# # Plot time-ish that ship got stuck
# plt.axvline(1, 0, 1, color='r')
# plt.annotate('Boat free', xy=(15.6, -0.15), fontsize='x-large')
#
# # Plot time-ish that ship got free
# plt.axvline(23, 0, 1, color='g')
# plt.annotate('Boat stuck', xy=(1.2, 0.15), fontsize='x-large')
#
# # Plot 0 line
# plt.axhline(0, 0, 25, color='#000', linestyle='--')
#
# # Plot x and y labels
# plt.ylabel('Polarity', fontdict = {'fontweight' : 'bold'}, fontsize='x-large')
# plt.xlabel('6-hour tweet block', fontdict = {'fontweight' : 'bold'}, fontsize='x-large');
#
# # Plot x and y limits
# plt.ylim((-0.2, 0.2))
#
# BIGGER_SIZE = 150
#
# plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)
#
#
# plt.savefig('block_polarity.png', bbox_inches = "tight")
# plt.show()


################ PLOT SUBJECTIVITY ################
plt.figure(figsize = (5, 5))
plt.tight_layout()

sns.lineplot(x = np.arange(len(tweet_block_subjectivity)), y = tweet_block_subjectivity, linewidth = 3)

# Plot time-ish that ship got stuck
plt.axvline(1, 0, 1, color='r')
plt.annotate('Boat free', xy=(15.6, 0.15), fontsize='x-large')

# Plot time-ish that ship got free
plt.axvline(23, 0, 1, color='g')
plt.annotate('Boat stuck', xy=(1.2, 0.15), fontsize='x-large')

# Plot 0 line
plt.axhline(0, 0, 25, color='#000', linestyle='--')

# Plot x and y labels
plt.ylabel('Subjectivity', fontdict = {'fontweight' : 'bold'}, fontsize='x-large')
plt.xlabel('6-hour tweet block', fontdict = {'fontweight' : 'bold'}, fontsize='x-large');

# Plot x and y limits
plt.ylim((0, 1))

BIGGER_SIZE = 150

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


plt.savefig('block_subjectivity.png', bbox_inches = "tight")
plt.show()

################ TOPIC MODELING ################
#
# # Container for all texts
# texts = []
# labels = []
#
# # Let's do some cleaning and prep for analysis using a lemmatizer
# # Check for real word and remove conjugation
# lemmatizer = nltk.WordNetLemmatizer()
#
# stopwords = nltk.corpus.stopwords.words('english')
# print(stopwords)
#
# # CHECK HERE WITH SMALL EXAMPLE
# # tweet_blocks = ['hello my name is clare clare', 'hello i love you so much clare', 'hi i think you are cute', 'go away now']
#
# for i in range(0, len(tweet_blocks)):
#     tweet_block_no = i
#     text = tweet_blocks[i]
#
#     labels.append(tweet_block_no)
#
#     tokenized_text = nltk.word_tokenize(text)
#
#     refined = [lemmatizer.lemmatize(word.lower()) for word in tokenized_text if
#                word.isalnum() and word not in stopwords]
#
#     texts.append(refined)
#
# # Actually using gensim: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
#
# # Gensim expects to run on a list of integers, so we must respect this
# id2word = gensim.corpora.Dictionary(texts)
#
# # Gensim uses bag of words for topic modeling, so we need to turn our corpus
# # into a list of tuples each representing a word found in our original document
# corpus = [id2word.doc2bow(text) for text in texts]
#
# print([[(id2word[id], freq) for id, freq in cp] for cp in corpus])
#
# lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                            id2word=id2word,
#                                            num_topics=20,
#                                            random_state=100,
#                                            update_every=1,
#                                            chunksize=100,
#                                            passes=10,
#                                            alpha='auto',
#                                            per_word_topics=True)
#
# print(lda_model.print_topics(num_words=10))
#
# # Topic outputs:
# # [(0, '0.003*"canal" + 0.003*"suez" + 0.002*"ship" + 0.002*"http" + 0.001*"stuck" + 0.001*"every" + 0.001*"the" + 0.001*"container" + 0.001*"given" + 0.001*"amp"'),
# # (1, '0.006*"suez" + 0.006*"http" + 0.006*"canal" + 0.004*"ship" + 0.002*"stuck" + 0.002*"the" + 0.001*"every" + 0.001*"blocked" + 0.001*"container" + 0.001*"suezcanal"'),
# # (2, '0.004*"canal" + 0.004*"suez" + 0.003*"http" + 0.003*"ship" + 0.002*"stuck" + 0.001*"the" + 0.001*"container" + 0.001*"blocked" + 0.001*"i" + 0.001*"amp"'),
# # (3, '0.001*"canal" + 0.001*"http" + 0.001*"suez" + 0.001*"ship" + 0.000*"stuck" + 0.000*"container" + 0.000*"blocked" + 0.000*"every" + 0.000*"the" + 0.000*"boat"'),
# # (4, '0.003*"suez" + 0.003*"canal" + 0.003*"http" + 0.002*"ship" + 0.001*"stuck" + 0.001*"container" + 0.001*"blocked" + 0.001*"the" + 0.001*"given" + 0.001*"every"'),
# # (5, '0.011*"canal" + 0.007*"suez" + 0.005*"http" + 0.004*"ship" + 0.002*"stuck" + 0.002*"the" + 0.001*"boat" + 0.001*"blocking" + 0.001*"cargo" + 0.001*"breaking"'),
# # (6, '0.024*"suez" + 0.022*"canal" + 0.015*"http" + 0.007*"egypt" + 0.006*"would" + 0.005*"port" + 0.005*"yacht" + 0.005*"the" + 0.004*"ship" + 0.004*"if"'),
# # (7, '0.003*"canal" + 0.003*"http" + 0.003*"suez" + 0.003*"ship" + 0.001*"stuck" + 0.001*"the" + 0.001*"container" + 0.001*"world" + 0.001*"blocked" + 0.001*"suezcanal"'),
# # (8, '0.002*"http" + 0.002*"canal" + 0.002*"suez" + 0.001*"ship" + 0.001*"stuck" + 0.001*"container" + 0.001*"blocked" + 0.001*"the" + 0.000*"every" + 0.000*"i"'),
# # (9, '0.070*"suez" + 0.069*"canal" + 0.056*"http" + 0.033*"ship" + 0.020*"stuck" + 0.015*"the" + 0.009*"i" + 0.009*"boat" + 0.008*"blocking" + 0.007*"container"'),
# # (10, '0.031*"empire" + 0.029*"canal" + 0.028*"suez" + 0.017*"taiwan" + 0.016*"british" + 0.016*"big" + 0.015*"crisis" + 0.015*"beijing" + 0.015*"america" + 0.014*"may"'),
# # (11, '0.010*"http" + 0.010*"suez" + 0.009*"canal" + 0.006*"ship" + 0.003*"stuck" + 0.003*"container" + 0.002*"blocked" + 0.002*"the" + 0.002*"every" + 0.002*"suezcanal"'),
# # (12, '0.002*"canal" + 0.002*"suez" + 0.002*"http" + 0.002*"ship" + 0.001*"every" + 0.001*"container" + 0.001*"blocked" + 0.001*"stuck" + 0.001*"amp" + 0.001*"suezcanal"'),
# # (13, '0.003*"canal" + 0.002*"suez" + 0.002*"http" + 0.002*"ship" + 0.001*"stuck" + 0.001*"container" + 0.001*"the" + 0.001*"every" + 0.001*"amp" + 0.001*"blocked"'),
# # (14, '0.000*"canal" + 0.000*"http" + 0.000*"suez" + 0.000*"ship" + 0.000*"stuck" + 0.000*"blocked" + 0.000*"container" + 0.000*"the" + 0.000*"amp" + 0.000*"every"'),
# # (15, '0.049*"http" + 0.034*"ship" + 0.031*"canal" + 0.026*"suez" + 0.025*"day" + 0.024*"every" + 0.024*"amp" + 0.023*"sideways" + 0.023*"blocked" + 0.019*"world"'),
# # (16, '0.004*"canal" + 0.004*"http" + 0.004*"suez" + 0.002*"ship" + 0.001*"stuck" + 0.001*"container" + 0.001*"the" + 0.001*"every" + 0.001*"blocked" + 0.001*"amp"'),
# # (17, '0.064*"http" + 0.046*"ship" + 0.037*"canal" + 0.033*"container" + 0.029*"suez" + 0.026*"blocked" + 0.025*"stuck" + 0.020*"given" + 0.020*"suezcanal" + 0.017*"ever"'),
# # (18, '0.051*"http" + 0.038*"canal" + 0.037*"ship" + 0.034*"suez" + 0.024*"blocked" + 0.023*"container" + 0.021*"every" + 0.019*"amp" + 0.017*"stuck" + 0.015*"the"'),
# # (19, '0.007*"suez" + 0.007*"canal" + 0.005*"ship" + 0.005*"http" + 0.002*"stuck" + 0.002*"the" + 0.002*"container" + 0.002*"blocked" + 0.001*"suezcanal" + 0.001*"day"')]
#
# ################ ENTITY EXTRACTION ################
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

# [{'PERSON': ['Jake Sullivan', 'Antony Blinken', 'Biden', 'Yang Jiechi'], 'NORP': ['Chinese'], 'ORG': ['National Security', "Possibly America's", 'State', 'EU', 'Ethiopia &', 'Abay'], 'LOC': ['Europe', 'the Suez Canal', 'Gulf', 'Suez Canal']}, {'PERSON': ['Freyberg', 'Nasser', 'Knowles Carter', 'Khan', 'Biden', 'H.M.T Cameronia', 'Charles de Gaulle', 'Canal'], 'NORP': ['British'], 'ORG': ['Piraeus', 'the New Zealander Division', 'Den Helder', 'Pak-China friendship(65)and', '5th Brigade', 'Non-Petroleum Ships'], 'LOC': ['the Strait of Hormuz', 'the Arabian Gulf', 'Suez Canal', 'Mediterranean', 'the Horn of Africa', 'the Suez Canal']}, {'PERSON': ['Canal', 'Nasser', 'Khan', 'km.151'], 'NORP': ['British'], 'ORG': ['the Red Sea Ports Authority', 'Port Clearance', 'Update', 'Pak-China friendship(65)and', 'UTC'], 'LOC': ['Suez', 'Suez\xa0Canal', 'Suez Canal', 'Mediterranean', 'the Suez Canal', 'Red   ']}, {'PERSON': ['km.151', 'Anthony Eden', 'Ever Given', 'White', 'Seuss', 'Suez Canal', '@Maersk', 'Julianne Cona/', 'Mitch McConnell', 'Evergreenlines'], 'NORP': ['Americans', 'Egyptians', 'Russian', 'Syrians', 'American', 'Saudi', 'Israeli'], 'ORG': ['Mediterranean &', 'the Yellow Fleet', '@typesfast @BlackstoneVG', '’ve', 'GlobalSystem', 'the Red Sea Ports Authority', 'KSA', 'Evergreen Marine', 'Sentinel #', '@roberttuttle &', 'Latest', 'Suez', 'House', "BBC News - Egypt's", 'Evergree', 'Suez Canal', 'Senate', 'Evergreen', 'IDK', 'Port Clearance', 'Megaship Blocks All Traffic', 'UTC', 'Clacket Lane Services'], 'LOC': ['Africa', 'Suez', 'Suez\xa0Canal', 'Suez Canal', 'Big Ben', 'the Suez   ', 'Mediterranean', 'The Suez Canal', 'Asia', 'Horn of Africa', 'Red Sea', 'Cape', 'the Suez Canal', 'Pacific', 'Europe']}, {'PERSON': ['Evergreen Mega', 'Ever Green', 'Anthony Eden', 'George Segal', 'Mitch McConnell', 'Evergreenlines', 'Matt Gaetz', 'Backlog                        Me', 'Sobieski', 'Suez-Max', 'Rod Wave', '@RebelLoki', 'Suez Canal', 'Capsule - Javaica Recipes', 'Eisenhower', 'Tony Bradley', "Paul Wall's", '🤔', 'Someone Austin', 'Lenin', '@Quasarcasm47'], 'NORP': ['Taiwanese', 'Russian', 'Egyptian', '@Pixelfish', 'Panamanian', 'Polish', 'Saudi', 'Americans', 'English'], 'ORG': ['Mediterranean &', 'Enron', 'EVERGIVEN', '🚢', 'EVERGREEN', 'EverGiven', 'The Ever Green', 'Evergiven', 'BBC News', 'HRC’s', 'Wild Red Mimosa Pudica', 'Suez Canal Blockage Set', 'SuezCanal', 'GIANT', 'Shipbuilding Industry Prospering', 'WIDEN THE GODDAMN SUEZ CANAL', 'Maersk Denver', 'TIL', 'Suez Canal Clogged', 'the Cinnamon Toast Crunch', 'KSA', '@Agortitz', 'Ripple Through Global Energy Market', 'Evergreen Marine', 'Suez Canal &', 'the Suez Canal Authority', 'Canal', 'Global Supply Chain Hit', 'Marine Academy', 'Latest', 'BreakingNews', '@roberttuttle &', 'AIS', 'House', 'Gigantic', 'Amazon', 'Senate', 'Evergreen', 'Suez Canal', 'Suez Canal Blockage After Demand-Driven Sell-Off -   ', 'Evergreen Marine Corp', 'Mariners', 'GIF', 'ShrimpGate', 'Suez Canal Traffic Block', '@Maersk', 'An EVERGREEN SHIPPING CONTAINER', 'Cinnamon Toast Crunch', 'Suez Canal @Wikipedia', 'Leaked'], 'LOC': ['Africa', 'East', 'Elbe River', 'Suez', "Brexit Britain's", 'Nile', 'Suez Canal', 'Earth', 'Mediterranean', 'Asia-Europe', 'the Suez Canal’', 'Horn of Africa', 'Red Sea', 'Oil Swings', 'the Suez Canal']}, {'PERSON': ['🚢🚢🚢', 'Ever Given', 'Netanyahu', 'Marc Rich', 'Evergreen Cargo', 'Suez Canal', 'Rotterdam', 'Sisi', 'Container Ship Gets', 'Canal', 'Wish', 'Lough Atalia', 'Jeff Hendrick', 'Hillary Clintons', 'Port Said'], 'NORP': ['THREAD', 'Eurasian', 'Taiwanese', 'republicans', 'Russian', 'Egyptian', 'Panamanian', 'Iranian', 'European', 'Saudi', 'Israeli'], 'ORG': ['Mediterranean &', 'The World for Sale', 'the Evergreen Fleet', 'Huge Container Ship Involuntarily Blocks Suez Canal', 'EVERGIVEN', 'Suez Canal Authority', 'Sumed', 'Evergiven', 'Navigation', "Evergreen Marine's", 'HRC’s', 'Suez Canal Blockage', 'SuezCanal', 'THE CRASH SITE', 'NEW 🚨', 'Justice', 'ULCS', 'HRCs Secret Service', 'TimPeel', 'Strong European Data', 'KSA', '#', 'Evergreen Marine', 'Suez Canal &', 'Evergreen Line', 'Canal', 'Jack Lynch Tunnel', 'Reopens', 'GAC', 'Ashkelon-Eilat', 'Guardian', 'Shell', 'Latest', 'RodWave', 'the Eiffel Tower', '@roberttuttle &', 'Suez', 'StupidAnalogues', 'The El Ferdan Railway Bridge', 'AIS', '@kuku27 Huge', 'Suez Canal', 'Evergreen', 'Topanga', '🚢 One', 'BBC', 'RoadWatch', 'Bank of the #', '@Yahoo', 'TL', 'BLOCKED', 'Suez Canal (Planet Labs', '🤯', 'Cargo', 'RIP', 'Leaked'], 'LOC': ['Asia', 'The Suez Canal', 'Mideast', 'North', 'the Red Sea', 'Horn of Africa', 'Red Sea', '#Asia', 'Hoping Europe', 'Suez', 'Suez Canal Blocked', 'Suez Canal', 'Earth', 'Mediterranean', 'Africa', 'East', 'the Med Sea', 'the Suez Canal', 'Europe', '#Suez']}, {'PERSON': ['Stephen Donnelly', "Brian Lilley's", 'Daniel Blackstone', 'Adams Morgan', '@Theothebald', 'Chang Evergreen', 'Austin Powers', 'Grant Shapps', 'Twitter', 'Gamal Abdel Nasser’s', 'Mitch McConnell', 'Jim GW', 'Hillary', 'Ty Pennington', 'LARGEST', 'Biden', 'Karl Marx', 'Tigray\n🔹', 'Tesla', 'Shooter', 'Harris', 'Johnson', 'Suez Canal', 'Rotterdam', 'Osama Rabie', 'Richard Meade', 'Clintons', 'Ever Given', 'Paul Stanley', "Hillary Clinton's", 'Dan Snyder', 'Cargo'], 'NORP': ['Marxists', '❓', '@planetlabs', '@LloydsList', 'Bahraini'], 'ORG': ['Mediterranean &', 'HRC', 'EVERGREEN', 'Suez Canal Authority', 'EVERGIVEN', 'SuezCanal', 'Tokyo Drift', 'ULCS', 'NEW 🚨', 'Ford', '#VIDEO Large', 'Bitcoin #', "the Sitmar Line's", '⏩ Traffic', 'UPDATE Tug', 'Merkel Red', 'NSFW', 'KSA', 'Lloyd’s List', 'Suez Canal &', 'GMT', 'the Suez Canal Authority', 'WeChat', 'Canal', 'GAC', 'Giant', 'The Block #Mimi', 'Latest', 'CTS/L', 'Suez', 'the Eiffel Tower', "Traffic Can't Pass", 'RISE', 'AIS', 'Army', 'Amazon Malala', 'Gigantic', 'Amazon', 'Suez Canal', 'Evergreen', 'BBC News -', 'FILL', '🚢 One', '@mercoglianos', 'Clumsy', 'Rory', 'the #Suez Canal', 'Bank of the #', 'Parallel Parking', '@davidhorovitz', 'TSMC &', 'QAnon', 'Suez Canal (Planet Labs', 'Ships Carrying Commodities Stuck', 'The Ultra Large Container Ship', 'Cargo', 'UN', 'Lebanon’s', 'CTW'], 'LOC': ['Asia', 'The Suez Canal', 'the Suez   ', 'the Red Sea', 'Regina', 'Horn of Africa', 'Red Sea', 'the Eifel Tower', 'the Suez   Suez Canal', 'Suez', 'Suez Canal', 'Earth', 'Mediterranean', 'Africa', 'the Sinai Peninsula', 'the Suez   Satellite', 'Oil', 'the Suez Canal', 'Europe']}, {'PERSON': ['David Dobrik', 'Austin Powers', 'Grant Shapps', 'Sean Maitland', 'Hillary Clinton', 'Microsoft Tesla', 'Suez Canal\xa0', 'Bitcoin', 'Dr Sal Mercogliano', 'Suez Canal', 'Rep Norman', 'Karl Marx Bill Clinton', 'AZ Vaccine', 'Deja Vu', 'Ever Given', 'Ben Adryl', 'Spelling Bee', 'Bob', 'Biden Prez'], 'NORP': ['Egyptian', 'Dutch'], 'ORG': ['Crown', 'EVERGIVEN', "THE WORLD'S TRADE GOES THROUGH THE CANAL", 'Videos/', 'Ships', 'SuezCanal', 'Oxford', 'Blocked - News &', 'Guts Media', 'Flem Candango', 'WIDEN THE GODDAMN SUEZ CANAL', "Evergreen Marine Corp's", 'the Ever Globe', '/Minute', 'The World', 'Suez Canal &', 'FBI', '@nameshiv', 'Giant', 'Sinn Fein', 'GAC', 'HMS Howe', 'Latest', 'Suez', 'Jackass', "BBC News - Egypt's", 'the British Pacific Fleet', 'QAnon Claims Stuck Suez Canal Ship', 'Suez Canal', 'Satellite Imagery', 'Evergreen', 'Suez Canal STILL', 'BBC', 'the #Suez Canal', '#GlobalSupplyChains', 'Fast &', 'YM Fountain', 'Sport Golf MLB Boston', 'QAnon', 'Netflix', 'Aircraft Carrier', 'Fox/Newsmax/Trump', 'Cargo', 'Rails', 'NFT', 'Traffic Children - Newsweek'], 'LOC': ['the Suez Canal\n\nForgive', 'the Suez   Photos', 'Suez', 'Suez Canal Blocked', 'the Suez   🚢 Suez Canal', 'Suez Canal', 'Earth', 'the Suez   ', 'Mediterranean', 'the Southern Hemisphere', 'Asia', 'The Suez Canal', 'the Red Sea', 'Red Sea', 'the Suez   AM news', 'the Suez Canal', 'Europe']}, {'PERSON': ['Samir', 'Brexit', 'Add Newsweek', 'Austin Powers', 'Ezzat Adel', 'Skyscraper', 'THERES SPACE', 'Stuck', 'Canal', 'Joe Flacco', 'Stranded Japanese-Owned', '@SpeakingSatan', 'Baby Jessica', 'Suez Canal', '@lib_crusher', 'Lookout Guy', 'Crew', 'Chris Grayling', 'Ted Cruz', '🤔', 'Bill Clinton', 'Bernhard Schulte Shipmanagement', 'Stacey Abrams’s', 'RV', 'Lieutenant Dan'], 'NORP': ['THREAD', 'French', 'Taiwanese', 'Egyptians', 'American', 'Egyptian', '♀', 'German', 'Palestinians', 'British'], 'ORG': ['Mediterranean &', 'Honda', 'Lowry', 'EVERGIVEN', 'Suez Canal Authority', 'NBA', 'Navigation', 'SuezCanal', 'Nike', 'Global Market Update\nFollow the Link Below For the Complete Story', 'WIDEN THE GODDAMN SUEZ CANAL', 'THREAD', 'Another Suez Canal Authority', 'Cargo Ship EVERGREEN', 'AAA', '#', 'Evergreen Marine', 'Suez Canal &', 'Canal', 'Giant', 'Global-Market Cues', 'Suez', 'Jimmy Fallon Pities the Suez Canal’s\xa0‘Dockblocker’', 'AIS', 'Samsung', 'Amazon', 'Satellite Imagery', 'Evergreen', 'Suez Canal Traffic Snarled After Megamax Boxship', 'SpongeBob', '🚢🚢🚢🚢 Factbox', 'Lloyds List', 'the #Suez Canal', 'TSMC &', 'Joburg', 'BSM', 'BLOCKED', '@nzherald Satellite', 'MIL', 'Cargo', 'UN', 'Jimmy Fallon'], 'LOC': ['Stranded Suez', 'Asia', 'the Suez   Tension', 'Atlantic', 'The Suez Canal', 'the Indian Ocean', 'the Suez   ', 'the Suez   Stranded Suez', 'the Red Sea', 'Horn of Africa', 'Red Sea', 'Suez', 'Suez Canal Blocked', 'Suez Canal', 'Earth', 'Mediterranean', 'Africa', 'the Suez Canal\n\nForgive', 'the Suez', 'the Suez Canal', 'Europe']}, {'PERSON': ['Clinton', 'Grant Shapps', 'Austin Powers', 'Mitch McConnell', 'Hillary Clinton', 'Outer Ring Road', 'Covid ‘', 'Nick Sloane', 'Skyscraper', '📸', 'Michael Joseph', 'al', 'Biden', 'John Lewis', 'Stuck', 'Canal', 'Oscar', 'Container Ship - YouTube', 'App Sample', 'Hancock', 'Unstick', 'Bojo', 'JeffHK', 'Suez Canal Blocked', 'Leth Agencies', 'Suez Canal', 'Rotterdam', 'Ever Given', 'Beloveds', 'Nasser', 'Stacey Abrams’s', 'Greed', 'Stephen Colbert Spins'], 'NORP': ['Chinese', 'Taiwanese', 'French', 'Dutch', 'Egyptians', '@MikkelsenDean', 'Capitalism', 'Egyptian', 'Republicans', 'Middle Eastern', 'Japanese', 'Panamanian', '@LloydsList', 'British', 'Arabs', 'Arab', 'Israeli'], 'ORG': ['UK &', 'Lowry', 'BugsBunnychallenge', 'OXY', 'Suez Canal Authority', 'EVERGIVEN', '🚢', 'ITE', 'The Egyptian Suez Canal Authority', 'NBA', 'Vessels', 'TIM', 'the SuezCanal Authority', 'Inshallah', 'Evergiven', 'UST', 'SuezCanal', 'NEWEST', 'Fed Speakers\n- Suez Canal', 'WIDEN THE GODDAMN SUEZ CANAL', 'THREAD', 'BBNaija', 'Ancestors', 'the US Navy', 'TIL', 'BOE', 'Oil Falls', 'EVER GIVEN', 'LNG', 'KSA', 'Smit', '👉    Oil Up', 'Al Jazeera', 'Suez Canal &', 'the Suez Canal Authority', '👉', 'SC', 'Giant', 'The Suez Canal Container Ship Drifting World Championship', 'The Suez-Canal\n\n💡', 'Eurasian Land Route', 'BreakingNews', 'Suez', 'the Eiffel Tower', 'Suez Canal Timelapse | Life at Sea on', 'Suez Canal', 'Evergreen', 'Satellite Imagery', '#SuezCanal authority', 'SCA', 'Cat Vibing To Ievan Polkka', 'Anthony Eden &', 'RutoSpeaks', 'Clumsy', 'USD', 'MRO', 'UPDATE', 'APP', 'Joburg', 'QAnon', 'LOGO', 'Tinubu || Suez Canal', 'Cargo', 'The #SuezCanal Authority', 'Airtel/Telkom', 'Royal Marine Commando', 'SPY $', 'Evergreen Marine Corp.', 'the Maritime Museum', 'Suez Canal\n'], 'LOC': ['the Suez   Tugs', 'Black', 'Asia', 'Atlantic', 'the Suez   Me', 'the Indian Ocean', 'the Suez   ', 'the Red Sea', 'Red Sea', 'Suez', "the Suez   '", 'the Suez   Clients', 'Suez Canal', 'Earth', 'Mediterranean', 'a Middle East', 'Africa', 'East', 'Suez Canal - Evergreen', 'Arctic Sea Route', 'the Suez   Men', 'the Suez Canal', 'the Sini Peninsula', 'Europe']}, {'PERSON': ['Smit Salvage', 'Ever Green', 'Matt Hancock', 'Qanon', 'Nefud', 'Boring Canal', 'Lesson', 'Austin Powers', 'Hillary Clinton', 'Bright Khumalo -', 'Elon Musk', 'Trains', 'Damn', 'Capella', 'Biden', '\u2066@LloydsList\u2069', 'Refloat Container Ship', 'Stuck', 'QUEUE', 'Aaron Gordon', 'Michael Gove', 'Unstick', 'Sissi', 'Bartolomeu Dias', 'Kyle Lowry', 'BusinessWrap', 'Suez Canal', 'Rotterdam', 'Chris Grayling', 'JFK', 'Can Get Stuck', '🤔', 'Musk', 'Nasser', 'Dan Scavino', 'Bernoulli', 'Bernhard Schulte Shipmanagement', 'Mike Mulligan', '😹'], 'NORP': ['Eurasian', 'French', 'Dutch', 'Egyptians', 'Indian', 'Tankers', 'Canadian', 'Russian', 'Egyptian', 'Japanese', 'German', '😜', 'Traffic', 'Pléiades', 'Israeli'], 'ORG': ['SuezCanal\nVoice', 'Global Food Trade', 'SuezCanal - Estimates', 'Ever Green', 'UK &', 'Frédéric Auguste Bartholdi', '@CostasParis @CloudberryRoque', 'Suez Canal Traffic', 'EVERGIVEN', 'Disraeli', 'Abu Dhabi 📹', 'Navigation', 'Ships', 'SuezCanal', 'Fed Speakers\n- Suez Canal', 'Honey', 'Cargo Ship Blocking Waterway', 'Suez Canal Crews Continue Efforts', 'BBC News - Suez Canal', 'LNG', 'AZ', '👉    ', '#', '@CEGltd', 'Suez Canal &', 'Al Jazeera', 'AFP', 'the Suez Canal Authority', 'China Inc.', 'ON IT', 'Fagradalsfjall', 'Capitol Hill', 'Latest', 'Suez', 'the Eiffel Tower', 'HRC USSS', 'Wikipedia', 'Suez Canal', 'Evergreen', '@CNBC', 'Cargo Ship Blocking Waterway\xa0', 'Copernicus', 'Airbus', 'Joburg', 'Tikait &', 'QAnon', 'the United Nations', 'Bernhard Schulte Shipmanagement', 'Tension', 'The Suez Canal Authority', 'Cargo', 'Hildog', 'The #SuezCanal Authority', 'SPY $', 'Trump’s', 'Operations', 'Vestact Asset Management \n\nReserve Bank'], 'LOC': ['Africa', 'East', 'Suez Backlog Grows', 'Suez', 'the Suez Canal\n\nForgive', 'the Indian Ocean', 'Cape', 'Suez Canal', 'Earth', 'the Suez   ', 'Asia', 'the Red Sea', 'the Suez Suez Canal Authority', 'The Suez Canal', 'the Suez Canal', 'the Sini Peninsula', 'Europe']}, {'PERSON': ['Christopher Grayling', 'Reach Me', 'Tom', 'Jim Crowe', '🚢', 'Patel', 'Beautiful Boaters', 'Grant Shapps', 'Austin Powers', 'Joe Biden', 'Hillary Clinton', 'Xerxes', 'Hillary', 'Rose George', 'Idk', 'Gab', 'LeBron James', 'Biden', 'Stuck', 'Apple Apps Store', 'Morrison', 'Dr Kimberly Tam', 'Aurora Intel', 'Goliath', 'Dick Ship Latest', 'Unstick', 'Johnson', 'Pierre Poilievre', 'David', 'Helmsman', 'Suez Canal', 'Jim Eagle', 'Container Ship', 'Chris Grayling', 'Trump', 'MCBOATFACE', 'Trudeau', 'Hrrrrnnggh Captain', 'Bernhard Schulte Shipmanagement', 'Stacey Abrams’s', 'Meghan McCain', 'HarambeeStars Shaffie', '@TeamMessi @Argentina @Cristiano'], 'NORP': ['Australian', 'Eurasian', 'Dutch', 'French', 'Egyptians', 'Canadian', 'Atlantic/Pacific', 'Egyptian', 'Japanese', 'British', '😜', 'Pléiades', 'Israeli', 'Tryna'], 'ORG': ['😵', '@NetHistorian', 'UK &', 'IQAX', 'The Suez Canal &', 'SS', 'EVERGIVEN', 'EVERGREEN', 'BBC News', 'Disraeli', 'AAPL', '@Lee_in_Iowa', '@tomhanks', 'SuezCanal', 'Trafalgar Group', 'Big Schedules', 'Honey', 'Learn', 'Macmillan', 'LNG', '#SuezCanal Authority', 'MLN', "the Suez Canal Authority's", 'Reuters', '@MDX_PSSFaculty', 'Suez Canal &', 'the Financial Times', 'the Suez Canal Authority', 'Canal', "PS 5's", 'Trump America', 'Guardian', 'Giant', 'GAC', 'Satellite Images Show Huge Traffic Jam', 'Sedgwick', 'CCAMLR', 'Suez', 'HRC USSS', 'CAT', '@AuroraIntel', '@komonews', 'Import of Oil!', 'Suez Canal', 'Evergreen', 'Senate', 'NewZealand', '@CNBC', 'BarstoolRundown', '@InstantAirtime &', 'TOM', 'the Army Corps of Engineers', 'Airbus', 'Tikait &', 'NAT', 'QAnon', '@Lloydslisted Much', '@globalnews &', '#AI', 'GIF', 'Country', 'Tension', 'The Suez Canal Authority', 'THE SUEZ CANAL &', 'Cargo', 'Kent', 'CNN', 'RT', 'EASY', 'Trump’s', 'Operations'], 'LOC': ['Africa', 'East', 'the Suez Canal Boat', 'Suez', 'the Indian Ocean', 'the Sini Peninsula', 'Suez\xa0Canal', 'Northern Sea Route', 'Suez Canal', 'Earth', 'the Suez   ', 'Asia', 'the Red Sea', 'Red Sea', 'The Suez Canal', 'the Suez Canal', 'Suez Can', 'Cape']}, {'PERSON': ['Chris Christie', 'Pimple Popper', '⬆', 'Satellogic', 'Annastacia Palaszczuk', 'Background', 'Queensland', 'Elon Musk', 'Ghislaines', 'Capella Space', '📸', 'Learn', 'Bogey', 'Biden', 'Stuck', 'Satellite Photo', 'Moller Maersk', 'Natasha Frost', 'Unstick', 'Suez Canal', 'Rotterdam', '@Bloomberg', 'Boskalis', 'COVID-19', 'Ever Given', 'Hrrrrnnggh Captain', 'Nat'], 'NORP': ['Chapmans', 'Eurasian', 'Dutch', 'Indian', 'Canadian', 'Russian', 'Egyptian', 'Avengers', 'Japanese', 'Yantian', 'Arabs', '😜', '♀', 'Pléiades', 'Tugboats'], 'ORG': ['PanamaCanal', 'Global Food Trade', 'Ever Green', 'RT', 'EVERGIVEN', 'Suez Canal Authority', 'Blocking', 'Inshallah', 'TIM', 'Ships', 'SuezCanal', 'NEW 🚨', 'EverGreen', 'THREAD', 'the World Billions of Dollars', 'TIL', 'IHI Jet Service', 'Mrs Nirmala Sitaraman', 'LNG', 'NYMEX', 'Brain', '#', 'Suez Canal &', "Suez Canal's", 'Giant Container Ship Blocks Suez Canal', 'the Suez Canal Authority', 'Canal', '🚢 That’s', 'Guardian', 'SaudiArabia', 'Trolls', 'PH &', 'Suez', 'NYT', '@Mamurudi', 'The Coffee Shop', 'Suez Canal', 'Evergreen', 'BBC News -', 'Cat Vibing To Ievan Polkka', 'Evergreen Marine Corp', 'HURRY', 'AO3 &', 'Satellogic’s', 'the Army Corps of Engineers', 'Airbus', 'Tikait &', '#China &', 'TV1', 'Texas AG', 'The Suez Canal Authority', 'IRL', 'RV', 'RIP', 'Cargo', 'GOP', 'Yale University'], 'LOC': ['Arctic', 'the SUEZ CANAL', 'Asia', 'The Suez Canal', 'the Indian Ocean', 'Suez\xa0Canal', 'the Suez   ', 'Regent Park', 'the Red Sea', 'Suez', 'Suez Canal', 'Earth', 'Mediterranean', 'the Gulf of Suez', 'Africa', 'the Suez Canal 😩', 'Persian Gulf', 'Cape', 'the Suez Canal', 'Europe']}, {'PERSON': ['Werner Herzog', 'Chris Christie', 'Ron Howard', 'Satellogic', 'Laura K.   ', 'Austin Powers', 'Ross', 'Mark Wahlherg', 'Ford', 'Andheri Traffic', 'Elon Musk', 'Mark McGowan', 'Colin Barnett', 'Ben Affleck', 'Capella', 'Learn', 'Wut', 'Jessie Yeung', '📸 Pléiades', 'Heed', 'Netanyahu', 'Suez Canal', 'Nene Hatun', 'Aim', 'Wellerman', 'Built', '@bjpmaha', '🤔', 'Yer da', 'Dylan', 'Biden Lil Wayne', '@AGENTBRAVO2', 'Mussolini', 'Stephen Colbert Spins'], 'NORP': ['Pléiades', 'Taiwanese', 'Dutch', 'Boskalis', 'Indian', 'Italian', 'French', 'Egyptian', 'Japanese', 'Capetonians', 'Arabs', 'European', 'Saudi', 'British', 'Nigerians', 'South Korean'], 'ORG': ['DM', 'Frédéric Auguste Bartholdi', 'GHG', 'EverGiven', 'EVERGIVEN', 'YouTube', 'EVERGREEN', 'Inshallah', 'Navigation', 'Ships', 'African Fiction', 'Caterpillar', 'SuezCanal', '👉    Blocking the Suez Canal', 'the League of Nations', 'EverGreen', '@UNECE', 'LNG', '#', 'the Suez Canal Authority', '👉', 'Bunnings', 'Wingardium Leviosa', 'The Dutch Smit Salvage', 'Trolls', 'Moller Maersk', 'Giant', 'SUEZ CANAL AUTHORITY', 'Latest', '@AJEnglish', 'Suez', 'Turkish Transport Minister', 'BreakingNews', 'Isime Davido', 'Panama Canal YouTube', 'VHR', '@TheTerminal Full', 'Suez Canal', 'Evergreen', 'Knight', 'the #Evergiven', 'BYOB - Boat', '@BIMCONews &', 'Photos &', 'Satellogic’s', 'the Multi-Modal Transport', 'Summat', '@Steinbergf @EnriqueFeas', 'Airbus', 'WEF', '@MessageAnnKoh &', 'Ever Giving &', 'The Suez Canal Authority', 'Cargo', 'EU', 'USDA', 'THE EVERGREEN BARGE', 'CNN', 'UN', 'Operations'], 'LOC': ['Nile', 'a Suez Canal', '#Europe', 'Asia', 'The Suez Canal', 'the Suez   ', 'The Suez Canal Ship of Theseus', 'the Red Sea', 'Suez', "the Suez Canal's", 'the Suez   Good morning', 'Cape Town', 'Suez Canal', 'the Suez Suez Canal Authority', 'the Suez Canal Ship Said', 'the Mediterranean Sea', 'the Gulf of Suez', 'Africa', 'East', 'Persian Gulf', 'the Suez Canal', 'Europe']}, {'PERSON': ['Werner Herzog', 'Chris Christie', 'Suzano SA', 'Amos Harel', 'jack', 'Robin Bullock Told', 'Ford', 'Elon Musk', 'DAWG', 'Al-Jaz', '🤣', '📸 Pléiades', '❌', '@DougJBalloon', 'Stepbrother', 'Narendra', 'Netanyahu', 'ba3d', 'Suez Canal', 'Project Veritas', 'Putin', 'Jen Psaki', 'Gets Stuck - The', 'Heli', 'Gamal Abdel Nasser', 'Erica Nlewedim Enisa', 'Doug Ford', 'Richard Citizen', 'Bernhard Schulte Shipmanagement', 'Explainer', 'Kate Toynbee'], 'NORP': ['Romanian', '@CuyperHans', 'Brazilian', 'French', 'Dutch', 'Republicans', 'Egyptian', '😔', '👉Results', 'British', 'European', 'Jewish', 'Pléiades', 'Saudi', 'Israeli'], 'ORG': ['@Reuters @MofaJapan_jp Blockage', 'Bloomberg News', 'Global Food Trade', '@Kadriilham &', 'FleetMon #', 'EVERGIVEN', 'The Yellow Fleet:', 'RecruiterHumor', '@INTLMAV', 'British &', 'Cloudy Radar &', 'THE BLOCKAGE LASTS', 'Ships', 'Evergreen Ship', 'U S BLOCKING South Atlantic', 'VESSELS', 'SuezCanal', '👉Sudan', 'CSLA', 'FIEO', 'EASY', 'FOREVER', 'LNG', 'GMV', 'Simulation of EVER GIVEN Accident', 'Evergreen Marine', 'GPL', '#', 'SuezCanel', 'Shipping Ministry', 'the Suez Canal Authority', 'U.S.SUBS BLOCKING ENTRANCE TO', 'Canal', 'Moller Maersk', 'US Navy', 'Massive Pirate Siege of Waiting Suez Canal', 'Suez', 'SeaShanty', 'Panama Canal YouTube', 'Amazon', 'MTV', 'Evergreen', 'Suez Canal', 'INDIANS', 'WH Press Sec', 'BYOB - Boat', 'RIP', 'Coke', 'LIQUEFIED NATURAL GAS', 'ATTEMPT', 'Airbus', 'APEDA', '@EZaharievaMFA', 'Gutter Gas Power', 'Suez Blockade', '#AI', 'The White House', 'Suez Canal Update', 'Pleiades', 'The Suez Canal Authority', 'Cargo', 'EU WARSHIPS', 'USDA', 'CNN', 'RT', 'NEW PHOTO:', 'CBS News', 'the Northeast Passage', '#EverGiven'], 'LOC': ['Africa', 'East', 'Suez', "the Suez Canal's", 'Middle East Minute', 'EUROPE', 'a Suez Canal', 'the Suez Canal Blocked', 'Suez Canal', 'the Suez   ', 'Asia', 'the Red Sea', 'the Suez Canal', 'West coast', 'Europe', 'the Gulf of Suez']}, {'PERSON': ['Lesson', '🚢', 'Noah', 'Patrick Star', 'Hillary Clinton', 'Elon Musk', 'Bloated', 'Capella', 'Andy', 'Bulldozer', '🤣', '📸 Pléiades', 'Biden', 'Austin', 'Unstick', 'Suez Canal', 'Oprah', 'Marcus Rashford', '🤔', 'Retweet', '🤔🚢 We'], 'NORP': ['Canadian', 'French', 'Asian', 'Egyptians', 'African', 'American', 'Egyptian', 'Filipino', 'North American', 'Israeli', 'Saudi', 'British', 'Sea-Doo', 'Europeans'], 'ORG': ['UK &', 'EVERGIVEN', 'HRC', 'Suez Canal Authority', 'U S BLOCKING South Atlantic', 'Charlbury', 'Godzilla’s', 'SuezCanal', 'the White House', 'NeverAgain', 'BRP', '#SuezCanal', 'Gouache', 'America News', 'Appropriate', '#', 'the Suez Canal Authority', 'U.S.SUBS BLOCKING ENTRANCE TO', 'ACAB', 'Toilet Paper', 'Avoid Future', 'US Border crisis \n', 'Suez', '@GChesterman', 'Suez Canal', 'Evergreen', '🚢 Experts', 'Blue Wave', "Lloyd's List", 'AnimalWelfare', 'Social History', 'TikTok', 'Deliveroo', 'Taiwan News', 'Re-Float', 'Black Widow', 'QAnon', '🤯', 'Plot Twist', 'IKEA', 'Disney/Marvel', 'Times', 'Secret Service', 'Tension', 'EU WARSHIPS', 'Cargo', 'CNN', 'NFT', 'The Suez Canal Authority', 'Economic Security', 'India Chalks'], 'LOC': ['Africa', 'the Mediterranean Sea', 'Suez', "the Suez Canal's", 'North America', 'Tweet', 'Suez Canal', 'Asia', 'the Pacific Rim', 'the Red Sea', 'Java Sea\n\nHoping', 'South China Sea', 'The Suez Canal', 'the Suez Canal', 'West coast', 'Europe']}, {'PERSON': ['Filibuster', 'Eskom', 'Jim Crowe', 'Noah', 'US Border', 'Patrick Star', 'Left Twitter', 'Confer', 'Dreadships', 'Elon Musk', '📸', 'Capella', 'Michael Bay', 'Charles de Gaulle', 'Biden', 'Yukito Higaki', 'Canal', 'Karl Marx', 'Aurora Intel', 'Salim', 'Narendra', 'Yogendra', 'Biden Presser', 'Netanyahu', 'Suez Canal', 'Jim Eagle', 'Glad', '😩', 'Queue', 'Samaj Hyderabad', 'Magneto', 'Shoei Kisen', 'Jen Psaki', 'Vessels Approaching Suez Canal', 'Bal Narendra', 'Wedged', 'Mark Wahlberg', 'Retweet', 'the Suez Canal'], 'NORP': ['Western', 'French', 'Indian', 'Russian', 'Egyptian', 'Somali', 'Republican', 'Japanese', '♀', 'Australians', 'Pléiades', 'Democrat', 'British', 'Indians'], 'ORG': ['CCP', 'TimesUp #', 'WD40', 'IQAX', 'EVERGIVEN', 'Ships', 'Cloudy Radar &', 'U S BLOCKING South Atlantic', 'World Trade Center', 'Evergreen Ship', 'SuezCanal', 'a SHIT CANAL AND VERY', 'Gulf of Oman &', 'Florida Boat Copies Suez Canal Cargo Ship', 'Ninety Percent of Everything', 'Sat', '#', 'the Suez Canal Authority', 'ACAB', 'U.S.SUBS BLOCKING ENTRANCE TO', 'Navy', 'MDA', 'NATO', 'Fox News', '@AuroraIntel', 'Amazon', 'Toilet', 'Senate', 'Evergreen', 'Suez Canal', 'BlankPolitik', 'HIMARS Training &', "Lloyd's List", 'the Kobayashi Maru', 'Darinder Moody', 'NPR', 'Airbus', 'Gutter Gas Power', 'QAnon', '🤯', 'Plot Twist', 'The White House', 'Global', 'EU WARSHIPS', 'The Suez Canal Authority', 'CNN', 'NFT', 'WTF American', '#EverGiven'], 'LOC': ['Asia', 'Atlantic', 'The Suez Canal', 'the Indian Ocean', 'the Red Sea', 'Red Sea', 'the Suez CANAL', 'Suez', "the Suez Canal's", 'the Suez Canal - Ever Given', 'Arabian Sea', 'Suez Canal', 'Mediterranean', 'Africa', 'the Floating Republic', 'the Rio Grande River', 'Strait Transits', 'the Suez Canal', 'West coast', 'Europe']}, {'PERSON': ['Werner Herzog', 'Nah', 'Lesson', 'Chris Hemsworth\n\n- DSAI', 'Secret Weapon', 'Joe Biden', '\u2066@RoryWSJ\u2069', 'Boaty McStuckface', 'Scott Morrison', 'Uri Geller', 'Sentinel1', 'Zahrani', 'Canal', 'Kpler', 'JeffHK', 'Suez Canal', 'Keithy McKeithface', 'Marcus Rashford', 'Samaj Hyderabad', 'Ever Given', 'Gamal Abdel Nasser', 'Wedged', 'Jane McDonald', 'Retweet', 'Frac Spread Count -', '@AliVelshi', 'Satyagraha'], 'NORP': ['African', 'Pakistani', 'Egyptian', 'Japanese', '@LloydsList', 'Arabs', 'BRITISH', 'British', 'Pacific Asian', 'Indians'], 'ORG': ['Suez Canal Blockage Could', 'Marine Life', 'RIP Anthony Eden', 'Times of\xa0India', 'EVERGIVEN', 'Britons', 'EVERGREEN', '@MSNBC', 'RedSea', 'U S BLOCKING South Atlantic', 'SuezCanal', 'ACS', 'Suez Canal \nEVERGREEN', 'Boeing', "Lloyd's", 'Allianz', 'Global Trade Gets Rerouted With Suez Canal', 'World Trade', 'Silk Board Junction', '@Coretrayn', 'Suez Canal &', 'Shipping Route 2', 'Salvager', 'U.S.SUBS BLOCKING ENTRANCE TO', 'ACAB', 'Canal', 'the Suez Canal Company', 'Chadwell Canal', 'Latest', 'Texas @PrimaryVision', 'Suez', 'Latest Suez', 'Suez Canal', 'Evergreen', 'PVN', 'Cat Vibing To Ievan Polkka', 'Govt', 'the Kobayashi Maru', 'BigData', 'the #Suez Canal', 'TikTok', 'Deliveroo', 'Tension', 'NCAA', 'EU WARSHIPS', 'Cargo', 'the Suez Canal &', '#EverGiven'], 'LOC': ['#Suez   Stop', 'the Persian Gulf', 'Asia', 'The Suez Canal', '1/South', 'the Red Sea', 'Red Sea', 'the Suez CANAL', 'Suez', "the Suez Canal's", 'Northern Europe', 'Cape Town', 'Suez Canal', 'Earth', 'Mediterranean', 'Africa', 'Suez Canal Game', 'the Suez Canal', 'West coast', 'Europe']}, {'PERSON': ['Sinn Féin', 'Liberia', 'Al-Sisi', 'McConnell', 'Marwa', 'Ross', 'Rose George', '🇮Burundi', 'Biden', 'Mom', 'Crypto Messi', 'Op-Ed', 'Tucker', 'Canal', 'JFK Jr', 'Truck', 'Boat Stuck', 'Hillary Clintons', 'Salim', 'Saddam Hussein', '😉 good morning', 'Suez Canal', 'Chris Graying', 'Sentinel5p', 'HILLARY CLINTON', 'CricTracker', 'Ever Given', 'Barry Chuckle', 'Bar Lev', 'Bernhard Schulte Shipmanagement', 'Ever Given’s', 'Mario Kart', 'Fredericton', 'Adventuremice', 'DeBarge'], 'NORP': ['Dutch', 'French', 'Egyptians', 'Egyptian', 'Canadians', 'Japanese', 'Suez Canal', '@LloydsList', 'European', 'Israeli', 'British', 'Indians'], 'ORG': ['Stunning New Photos', 'EVERGIVEN', 'Suez Canal Authority', 'HFO', 'EVERGREEN', 'Ships', 'Giant Container', 'Sark/Grammys/Suez Canal', 'Shit', 'SuezCanal', 'BULLET', 'MiddleEast', 'Suez Canal \nEVERGREEN', 'Diana &', 'Rescuers', '🚢 Freight', 'Sinai', 'Evergreen Tweet', 'Legal Topic', 'ATMs/riri/#AReeceTurns24', 'Lesotho\n', 'Seychelles\n', '#', 'Suez Canal Authorities', 'Salvager', 'the Suez Canal Authority', 'ACAB', 'Canal', 'Chadwell Canal', '@Foone', 'Suez', 'Bellerin/Northern Nigeria', 'Suez Canal', 'Evergreen', 'SCA', 'Sierra Leone', "Lloyd's List", 'Govt', 'SEA', 'Blockage Cleared', 'Egypt &', 'Crowley', 'EVER', 'Turkish Satellite', 'SuezCanal &', 'RPF', 'Cargo', '🐭🧀💥\n\nInk &', '#EverGiven'], 'LOC': ['Asia', 'Atlantic', 'the Middle East', 'the Indian Ocean', 'Suez Canal\xa0', 'blokingthe Suez Canal', 'the Red Sea', "Suez Canal '", 'Suez', "the Suez Canal's", 'Suez Canal', 'Mediterranean', 'Kashmir', 'Africa', 'the Sinai Peninsula', "the Suez Canal '", 'Gambia', 'Northern Sea Route   ', 'the Suez Canal', 'the Suez Canal - Ever Given', 'Europe']}, {'PERSON': ['Anthony Eden', 'McConnell', 'Mitch McConnell', 'Matt Damon', 'Sticky Joe Biden', 'Important Stuff', 'Peter Stevenson', 'Wilder', '🤣', 'Biden', 'Svyataya Anna', 'Canal', 'Dunce', 'Moab', 'Suez Canal', 'Livestock', 'Kashgar', 'Keithy McKeithface', 'Limbaugh', 'Satanic', 'Trump', 'Ted Cruz', 'Uri Gellar', 'Mahmoud Khaled', 'Jason Derulo', '🤔', '100s', 'Retweet', 'Georgy Brusilov', 'Floaty McFloatface'], 'NORP': ['Chinese', 'Dutch', 'republicans', 'African', 'Japanese', 'Jewish', 'Pléiades', 'British', 'Moabites'], 'ORG': ['Newsmax', 'EVERGREEN', 'Suez Canal Authority', 'EVERGIVEN', 'MSC', 'Tantallon', 'SuezCanal', 'Ship Blocking Suez Canal Moves Slightly', 'The Turkish Navy SF', 'TIL', 'Global Supply Chains', '#', 'PSA', 'Salvager', 'ACAB', "Picturing Foucault's", 'U.S Navy Offers', 'Genesis', '🤯🤯🤯🤯🤯🤯🤯🤯🤯🤯', 'the Eiffel Tower', 'DigitalGlobe', 'Suez Canal', 'Evergreen', 'the New Suez Canal', 'Maersk &', 'BBC', 'Trump', 'Blockage Cleared', 'CV', 'the #Suez Canal', 'Giant Sexy Animals', 'HMS Newport', 'Wreaking Havoc', 'Airbus', 'JSIL', 'EVER', 'QAnon', 'Breaking News', 'SAT', 'Cargo', 'EU', 'Maritime Casualty', 'GOP'], 'LOC': ['Asia', 'Atlantic', 'The Suez Canal', 'Mideast', 'the Indian Ocean', 'the Suez   ', 'Rift Valley', 'the Red Sea', 'Northern Sea', 'the Northern Sea Route', 'Suez', 'the Eiffel Tower', "the Suez Canal's", 'Suez Canal', 'Earth', 'Mediterranean', 'the Gulf of Suez', 'Africa', 'the Christian Old Testament', 'Cape', 'the Suez Canal', 'Europe']}, {'PERSON': ['Brexit', 'Strong Winds', 'McConnell', 'Derrick Henry', 'Mitch McConnell', 'Barry', 'Idk', 'Hydroxychloroquine', 'Hrc', 'Karl Marx', 'Truck', 'Rudy Giuliani', 'Jacinda Ardern', 'Chuck Norris', 'Marx', 'Suez Canal', 'Osama Rabie', 'Satanic', '@Marzofaraz', 'Nasser', '100s', '@samirsinh189 @LeoUmanah', 'Retweet', '😇'], 'NORP': ['Chinese', 'Canadian', 'French', 'Indian', 'Dutch', 'American', 'Egyptian', 'Japanese', '@LloydsList', '♀', 'Conservatives', 'Ngannou', 'Stoic', 'European', 'British'], 'ORG': ['Suez Canal Blockage Could', '@Nicochan33', 'EVERGIVEN', 'EVERGREEN', 'Suez Canal Authority', 'Ships', 'StuckShip', 'SriLanka', 'Suez Canal Authority Hopes High Tide', 'Suez Canal Blockage', 'SuezCanal', 'INSTC', 'the US Navy', 'UPDATE - Latest', 'PSA', 'Salvager', '#Liverpool', 'the Suez Canal Authority', 'the Maxar World View-2 Satellite', 'SNL', 'US Navy', 'Latest', 'U.S. Navy Sending Team', "Harvey's", 'QED', 'Suez Canal', 'Evergreen', 'Time', "Lloyd's List", "U-Turn'", 'BRI', 'CV', 'KamalaToe Harris', 'WW', 'SuezCrisis', 'JSIL', 'Breaking News', 'QAnon', 'El Sisi', 'TPLF', '@AnglicisedArab @Cyberia', 'UNLOADED', 'ChabaharPort', 'Vessels 🚢', 'Maritime Casualty', 'EU', 'the Suez Canal &'], 'LOC': ['Suez', 'the Gulf of Oman', "the Suez Canal's", 'Arctic', "the Suez Can'tal", 'the Gulf of Suez', 'the Northern Sea', 'the Arabian Sea', 'Suez Canal', 'the Suez   ', 'Mediterranean', 'Earth', 'Kashmir', 'the Red Sea', 'Red Sea', 'the Suez Canal', 'the Mediterranean Sea', 'Beacon']}, {'PERSON': ['@7tine76', 'Desert Wind Blew Global Trade Off Course - Bloomberg', 'McConnell', 'Mitch McConnell', 'Carlo Magno', 'Harambe', 'Rick', 'Capella', 'Remy Tumin', '@PawpadsGaming @Wolfen_Eson', 'Udaya Gammanpila', 'Biden', 'Sisi', 'Nevermind Reeling', 'Stuck', '@KudSverchkov', 'Doug Ford’s', 'Prayut Chan-o', 'Mississauga', 'Suez Canal', '@SuezDiggerGuy Yay', 'Jason Kenney', 'Coir', 'HILLARY CLINTON', 'Trump', 'Tony', 'Suez Canal Issue SOLVED', 'Jeremiah M. Bogert', '🤔', 'Jane McDonald', 'Jr', 'Abdel Fattah al-Sisi'], 'NORP': ['Chinese', 'Pléiades', 'Dutch', 'Italian', 'Russian', 'Egyptian', 'American', '@SPCRIST1', 'Republicans', '@LloydsList', 'Stoic', 'Arab', 'European'], 'ORG': ['RIP Gandalf', 'ChinaJoe', 'Suez Canal Authority', 'Canal de Suez', 'EVERGIVEN', '@tomhanks', 'MarineTraffic', 'SuezCanal', 'VesselFinder', 'RTE', 'DC Islamabad', 'UPDATE - Latest', 'Reuters', 'PSA', 'the Suez Canal Authority', 'SC', 'Suez Canal Live Updates', 'Spotify', 'Global Trade', 'AP', 'NYT', 'China &', 'PARKOUR', 'Suez Canal', 'Evergreen', 'SCA', "Lloyd's List", 'BBC', 'Blockage Cleared', 'CV', 'Suez Canal ‘Traffic Jam’', 'AmericaFirst', 'Airbus', 'JSIL', 'Breaking News', '@NYFEX', 'The Suez Canal Authority', 'EU', 'Alp Guard', 'Parker'], 'LOC': ["the Suez Canal's", 'Suez', '@CopernicusEU East coast', 'Suez Canal', 'the Suez   ', 'the Suez Canal -  ', 'THE RED SEA BEFORE GETTING STUCK', 'Cape', 'the Suez Canal']}, {'PERSON': ['jack', 'Lo', 'McConnell', 'Mitch McConnell', 'Birx', 'Lo-lee-ta', 'Jim Jordan', 'Lee', 'Biden', 'Mashhour Dredge', 'Abdel Fattah Al-Sisi', 'Oscar', 'Karl Marx', 'Truck', 'Steve Bannon', 'Sissi', 'Sudarsan Raghavan', 'Suez Canal', 'Chris Cillizza', 'Nippy', 'Jennifer', 'Yo', 'Trump', 'Offer', 'Jen Psaki', '🤔', 'Jason Derulo', 'Boromir', 'Cheniere', '@JohnDombroski8', 'Floyd', 'Lindsey Graham', '@AlsisiOfficial', 'Derrick Henry', '@AliVelshi', 'El-Sisi'], 'NORP': ['Chinese', 'Egyptians', 'Tankers', 'Russian', 'Egyptian', 'Jewish', 'Jalali', 'Americans', 'Muslims'], 'ORG': ['My New Anxiety Dream Where I’m Blocking the Suez Canal - McSweeney’s', 'SkySat', 'the Lloyd’s List', 'Shell/BG', 'EVERGIVEN', '@tomhanks', 'SAS', 'Evergreen Ship', 'U S BLOCKING South Atlantic', 'SuezCanal', '@Marmel', 'Gouache', 'Honey', 'INSTC', '👁', 'Challenger 3 Tanks', '@RNDYGFFE', 'Urine Geller', 'ToiletPaper', '#', 'Evergreen Marine', 'Canal Live Updates', 'the Suez Canal Authority', '🚢 😤', 'U.S.SUBS BLOCKING ENTRANCE TO', 'NuezCanal', 'SNL', 'Suez', 'PARKOUR', 'Suez Canal', 'Evergreen', 'Iranian Embassy', 'Microsoft', 'ANWAR', '@CNBC', 'Suez Canal - The Wall Street Journal', 'Ninety Percent', 'EconomicRecovery', 'BBC', 'Reroute', 'Chinese Motorway days', 'SuezCrisis', 'Breaking News', 'QAnon', 'Pentagon', 'The Suez Canal Authority’s', 'SHIPS', 'EU WARSHIPS', 'Cargo', 'Grape Nuts', 'FRIEND'], 'LOC': ['Suez', "the Suez Canal's", 'West coast', '👏👏', "the Suez Canal'", 'Suez Canal\n\n- Former', 'Suez Canal', 'the Nuez Canal', 'The Suez Canal', 'the Suez Canal', 'the Suez Canal - Ever Given', 'Europe']}, {'PERSON': ['Iwaizumi Hajime', 'Big Booty Boat', 'McConnell', 'Mitch McConnell', 'Peter Berdowski', 'Desert Wind Blew', 'Birx', 'Jim Jordan', '🤣', 'Oscar', 'Butterfly', 'Mashhour', 'Normalize 👏', 'Steve Bannon', 'Marx', 'Julian Felipe Reef', 'Rita Ora', 'Marie Kondo', '😉😉\n\nNina', 'Suez Canal', 'Jack', 'Rose', 'Osama Rabie', 'Trump', 'Miku', 'Johnny Knoxville', 'Dex', 'Lindsey Graham', 'Jane McDonald', '@Tinashe'], 'NORP': ['Chinese', 'South American', 'Russian', 'Egyptian', 'Republican', '🐭', 'Southern', 'Saudi'], 'ORG': ['Whitsun Reef', 'Suez Canal Authority', 'EVERGIVEN', 'Aladdin', '@MSNBC', 'Ships', 'SuezCanal', 'M4 Newport', 'NEW 🚨', 'Penis Captivus', 'The New York Times', 'General Organization For Veterinary Services', 'Daily News Egypt', 'the Cinnamon Toast Crunch', 'Suez Canal’s', 'LNG', 'the Defense Department', 'Vocaloid', '#', '🚨🚨🚨', 'Punta Cana', 'LPG', 'YT Link &', 'LONGER', '💰🚢💰', 'SaudiArabia', 'Suez', '✌', 'Suez Canal', 'Microsoft', 'Evergreen', 'Senate', 'ANWAR', 'I.e', 'Daffy', 'Twice', 'KamalaToe Harris', 'BSM', 'Breaking News', 'XXV', 'IKEA', 'the Nebraska State Fair', 'The Suez Canal Authority', 'Cargo', 'CNN', '👏', 'Pentagon'], 'LOC': ['the South China Sea', 'Suez', 'Suez\xa0Canal', 'North Star', 'middle East', '👏👏', 'Suez Canal\n\n- Former', 'Suez Canal', 'New England', 'Microsoft Flight', 'Engels', 'the Suez Canal', 'Beacon']}, {'PERSON': ['🚢', 'McConnell', 'Mitch McConnell', 'Peter Berdowski', 'Scott Morrison', 'Tinashe', 'Birx', 'Sentinel1', 'Vesselfinder', 'Mohab Mamish', 'Jim Jordan', 'Stuck', 'Ian', 'Oscar', 'Steve Bannon', 'Inch Cape', 'Suez Canal', 'Glad', 'Stanley Yelnats', 'Leslie Knope', 'Trump', 'Brb', 'Ever Given', 'Hulk', 'Aquaman', '400m', 'Lindsey Graham', 'Sergey Kud-Sverchkov', 'Stranded Cargo', 'Bill Walton'], 'NORP': ['Gibraltar', 'Canadian', 'Dutch', 'Democrats', 'Russian', 'Egyptian', 'Saudi', 'Conservatives'], 'ORG': ['FINALLY', 'Pac-12', 'EVERGIVEN', '🚢', 'Ships', 'Hamdullah', 'the Levees Break (2006', 'Suez Canal Blockage', '@CNET', '@Amradib', 'Ford', 'SuezCanal', 'VesselFinder', 'Inch Cape Shipping Services', 'XRP', 'LNG', 'Reuters', '#', 'UNSTUCK &', 'Al Jazeera', 'LPG', 'GMT', 'the #Suez Canal\n\n⬇', 'the Suez Canal Authority', 'Canal', '💰🚢💰', '@nameshiv', 'GAC', 'Giant', 'HBO', 'Godzilla', 'Suez', 'PUT', 'DIshwashers', 'Lil Nas X', 'Suez Canal', 'Microsoft', 'Evergreen', 'TrafficJam', 'BYOB - Boat', 'Thant’s', "Al Kharid's", '♥', 'AnimalWelfare', 'REUTERS', 'THE EVER GREEN SHIP', 'Latest AIS', 'ClimaCell', 'Trafic', 'the Nebraska State Fair', 'The Suez Canal Authority', 'Cargo', 'the Egyptian Suez Canal Authority', '👏👏👏'], 'LOC': ['Africa', 'Suez', 'Suez Canal\n\n- Former', 'the Suez Gulf', 'Suez Canal', 'Mediterranean', 'New England', 'the Suez Canal’s', 'Asia', 'Microsoft Flight', 'the Red Sea', 'the Central Lake', 'the Suez Canal - The Washington Post', 'Red Sea', 'Suez @GoogleNews', 'The Suez Canal', 'the Suez Canal', 'Europe']}, {'PERSON': ['@bouta_nt', "Container Ship's", 'Evan Peters', 'Ill', 'Dhruv Rathee', '🚢', 'McConnell', 'Mitch McConnell', 'Peter Berdowski', 'Boris Johnson', '📸', 'Ferry', 'Jensen Ackles', 'Rita Ora', 'Ryan Murphy', 'Suez Canal', 'Beechcroft', 'Glad', 'Osama Rabie', 'BreakingNews|#قناة_السويس', 'Ever Given', 'Gorilla Glue', 'Ambedkar', 'Evergreen|#قناة_السويس', '🤔', 'AnimalAg', 'Mehbooba Mufti', '@GoogleNews', 'Sergey Kud-Sverchkov', 'Alert', 'Diane Abbot'], 'NORP': ['Chinese', 'Taiwanese', 'Indian', 'Dutch', 'Russian', 'Egyptian', 'Panamanian', 'Indonesian', 'Movie', '@LexBlog'], 'ORG': ['LiveExport', 'Videos', 'RESUME', 'MASHHOUR', 'DHRadio', 'Suez Canal Authority', 'EVERGIVEN', 'Port Authority', 'Canal de Suez', 'Evergiven', 'Quick', 'Aladdin', 'Ships', 'SHIELD', 'Hamdullah', 'SAS', 'SuezCanal', 'VesselFinder', 'M4 Newport', 'the Indian Economy', 'Challenger 3 Tanks', 'OECD', 'NEW 🚨 Forget', 'Suez Canal Drift', 'Pisces Mercury', 'UNSTUCK &', 'SUEZ CANAL OPERATIONS IS', 'Suez Canal &', 'SuezCanel', 'the Suez Canal Authority', '👉', '💰🚢💰', 'Maersk', 'Container', 'Guardian', 'Ernest Bevin College', 'Suez', 'Stealers Wheel', 'Crews', 'the Suez Canal Authority 👏', 'Suez Canal', 'Evergreen', 'Archegos Capital', 'Bond', 'Open Source Intelligence (OSINT): Tracking the Ever Given Cargo Ship', 'SCA', 'TrafficJam', "Lloyd's List", 'SupplyChains', '♥', 'Fault For Blocking the Suez Canal', 'FREED', 'REUTERS', 'CV', 'Evergreen Marine Corporation’s Ever Given', 'EG', 'Modi Government', 'International Travel', 'The Freedom House', 'Gemini', 'ATH', 'SuezCrisis', 'Latest AIS', 'Breaking News', '#ExpressExplained', 'US Futures', 'Suez Update', 'The Suez Canal Authority', 'Cargo', 'The Sunday Times: the Ever Given', 'Eygpt', 'RT', 'the Suez Canal &'], 'LOC': ['Suez', 'Suez\xa0Canal', 'Mars', 'Suez Canal', 'Earth', 'the Suez   ', 'Mediterranean', 'Red Seas', 'Asia', 'Microsoft Flight', 'Cardiff', 'Hudson', 'The Suez Canal', 'the Suez Canal', 'Europe']}, {'PERSON': ['ICYMI', 'Jamie Lee Curtis', 'Dhruv Rathee', 'Liza Minnelli', 'McConnell', 'Mitch McConnell', 'Katie', 'Peter Berdowski', 'Sputnik', 'Richard Ayoade', 'Boris Johnson', 'Antifa', 'Sarah', 'Marcelo Bielsa', 'd=√(x₁', 'Fauci', 'Derek Chauvin,#GeorgeFloyd', 'Liberan Canal de Suez', 'Bol Bol', 'Jack Wittels', 'Lazio', 'Rupert Friend', 'Mashhour', 'Shoei Kisen Kaisha', 'Hue Jackson', 'Joyner Lucas', 'Taika Waititi', 'Joe Manchin', 'Suez Canal', 'Derek Chauvin', 'Glad', 'Baraka I.', 'Joy', 'Trump', 'George Floyd\n', 'Bonnie Piesse', 'Man Falls', 'Ever Given', '⅛', 'Jerry Nadler', 'Randy Orton', 'Mac Jones', 'Padma Vibhishan', 'Fitch', 'Coach Taylor', 'Trudeau', 'Diane Abbot', 'Ann Koh', 'Paul Wall'], 'NORP': ['Dutch', 'Indian', 'Boskalis', 'American', 'Egyptian', 'Today Egyptians', 'Herculean'], 'ORG': ['Suez Canal 😊', '💐', 'Singhu', 'EVERGIVEN', 'YouTube', 'Suez Canal Authority', '🎥', '@MSNBC', 'SHIELD', 'Ships', 'Dhruv', 'Suez Canal #', 'Kenya Coast Guard', 'MarineTraffic', 'SuezCanal', 'Pepsi', 'NPO Radio', 'XL', '#Evergreen Congratulations', 'The Egyptian Team of the Tug', 'Evergreen Marine’s', 'GOP', 'Allianz', 'Liverpool Cathedral', '👏', 'Reuters', 'UNSTUCK &', 'AI', 'the Suez Canal Authority', 'Giant', 'BreakingNews', 'the Bitter Lakes', 'THE BIG CHIPPER', 'Visa', 'Suez Canal', 'Evergreen', 'SCA', '#URGENT |', '💐👏', 'REUTERS', 'FREED', 'GoI', 'Romania &', 'the Kenya Navy', 'International Travel', 'TRUE', 'Mafia', 'The Suez Canal Authority', 'Cargo', 'Birx,HIPAA,Satanic Panic', 'RT', '@UNCTAD ’s'], 'LOC': ['Suez Canals', 'the Suez   Suez Canal', 'Suez', "the Suez Canal's", 'Americas', 'Suez Canal', 'Mediterranean', 'the Suez   ', 'Red Sea', 'the Suez Canal', 'the West Philippine Sea', '#Suez']}, {'PERSON': ['Jamie Lee Curtis', 'Evan Peters', 'Abdel Hamid', 'Kurz', 'Brexit', 'YM WISH', 'Dhruv Rathee', '🚢', 'McConnell', 'Mitch McConnell', 'Joe Biden', 'Tinubu', 'Carlo Magno', 'Nintendo', 'Flight Simulator', 'Tugboat', '🤣', "H. H. Holmes' Murder Castle", 'Fauci', 'N’Golo Kante', 'Pls RT', 'Churchill', 'Mashhour', 'Bar Mitzvah', 'George Floyd', 'Truck', '🌏', 'Rita Ora', 'Ryan Murphy', 'Mega McIlroy', 'Suez Canal', 'Izzat Adel\n', 'Glad', 'Gainer', 'Osama Rabie', 'Joy', 'Trump', 'Mostafa Mahmoud\nPort', '🤔', 'Donald Trump', 'Paul Wall'], 'NORP': ['Boskalis', 'Indian', 'Egyptians', 'American', 'Egyptian', 'Austrian', 'British', 'English', 'Indians', 'Tugboats'], 'ORG': ['Ever Green', 'US Mail', 'Suez Canal Authority', 'EVERGIVEN', 'Evergiven', '#Suez Canal Authority', 'Fair', 'The Egyptian Team of the Tug', 'UniversityChallenge', 'LNG', 'Reuters', 'Femi Adesina', '#', 'LPG', 'Maersk', 'Guardian', 'Container House', 'BreakingNews', 'Suez', 'the Bitter Lakes', 'Wikipedia', 'Visa', 'Suez Canal', 'Evergreen', 'Microsoft', 'Senate', 'The #Evergiven', 'SCA', '♥', 'the #Suez Canal', 'Google Search', 'International Travel', 'Breaking News', 'deTermination', 'Daily Mail', 'Ever Globe', 'The Suez Canal Authority', 'Cargo', 'Kano Iniesta', 'GOP'], 'LOC': ['Suez', 'the Suez\xa0Canal', 'Suez Canal', 'Earth', 'the Suez   ', 'Microsoft Flight', 'the Red Sea', 'the Suez Canal']}]