# For preprocessing
import re
import pandas as pd
import numpy as np

# For analysis
import textblob

# For plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Read in all formatted tweets
df = pd.read_csv('formatted_tweets.csv')

# Vectorized operation to remove non-English tweets based on the `en` column
# https://www.geeksforgeeks.org/drop-rows-from-the-dataframe-based-on-certain-condition-applied-on-a-column/
df = df[df['lang'] == 'en']

# Sort based on time stampxx
df = df.sort_values('created_at')

print('Number of tweets after preprocessing: '+str(len(df)))

counter = 0
tweet_blocks = []
block_text = ''
been_reset = False
tweet_blocks_indices = []

for index, row in df.iterrows():
    cur_time = row['created_at']
    hour = int(re.findall(r'T(\d\d)', cur_time)[0])
    if hour % 6 == 0:
        if not been_reset:
            tweet_blocks.append(block_text)
            block_text = ''
            been_reset = True
        block_text += ' '+row['text']
    else:
        block_text += ' '+row['text']
        been_reset = False

    tweet_blocks_indices.append(len(tweet_blocks))

# Assign every tweet to a 6-hour tweet block
df['tweet_block'] = tweet_blocks_indices
