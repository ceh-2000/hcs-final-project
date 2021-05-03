import json
import glob
import pandas as pd
import requests
import time

def get_api_key():
    with open("keys.txt") as file:
        api_key = file.read()
    return api_key

def get_full_tweet(tweet_id):
    headers = {"Authorization": "Bearer {}".format(get_api_key())}
    url = "https://api.twitter.com/2/tweets?ids="+tweet_id
    response = requests.request("GET", url, headers=headers)
    time.sleep(5)
    print(response)
    return response.json()

def get_param(cur_param, tweet):
    value_to_return = ''
    if cur_param == 'public_metrics_like_count':
        try:
            value_to_return = tweet['public_metrics']['like_count']
        except:
            pass
    elif cur_param == 'public_metrics_retweet_count':
        try:
            value_to_return = tweet['public_metrics']['retweet_count']
        except:
            pass
    elif cur_param == 'public_metrics_reply_count':
        try:
            value_to_return = tweet['public_metrics']['reply_count']
        except:
            pass
    elif cur_param == 'text':
        try:
            tweet_type = tweet['referenced_tweets'][0]['type']
            if(tweet_type=='retweeted'):
                value_to_return = get_full_tweet(tweet['referenced_tweets'][0]['id'])['data'][0]['text']
                print(value_to_return)
        except:
            pass

        if(len(value_to_return) == 0):
            try:
                value_to_return = tweet['text']
            except:
                pass
    elif cur_param == 'referenced_tweets_type':
        try:
            value_to_return = tweet['referenced_tweets'][0]['type']
        except:
            pass
    elif cur_param == 'referenced_tweets_id':
        try:
            value_to_return = tweet['referenced_tweets'][0]['id']
        except:
            pass
    else:
        try:
            value_to_return = tweet[cur_param]
        except:
            pass
    return value_to_return

def get_new_entry(tweet):
    columns = ['author_id', 'created_at', 'lang', 'id', 'public_metrics_like_count', 'public_metrics_retweet_count', 'public_metrics_reply_count', 'text', 'referenced_tweets_type', 'referenced_tweets_id']
    observation = []
    for param in columns:
        observation.append(get_param(param, tweet))

    df = pd.DataFrame([observation], columns=columns)
    return df

if __name__ == "__main__":
    # Use glob to get all tweets in the "saved_tweets" directory that are .json files
    columns = ['author_id', 'created_at', 'lang', 'id', 'public_metrics_like_count', 'public_metrics_retweet_count', 'public_metrics_reply_count', 'text', 'referenced_tweets_type', 'referenced_tweets_id']
    df = pd.DataFrame([], columns=columns)

    all_files = glob.glob("./saved_tweets/*.json")
    for filename in all_files:
        with open(filename, "r", encoding="utf8") as file:
            json_file = json.loads(file.read())
        if(json_file.get('data')):
            for tweet in json_file['data']:
                df2 = get_new_entry(tweet)
                df = pd.concat([df, df2], axis=0)
                df.to_csv('tweets2.csv', index=False)
        else:
            print("WELLO")

    df.to_csv('tweets2.csv', index=False)
