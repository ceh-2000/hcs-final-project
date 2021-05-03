import requests
import json


def get_api_key():
    with open('keys.txt') as file:
        api_key = file.read()
    print(api_key)
    return '' #api_key


def make_request(query, tweet_fields, start_time, end_time, max_results, api_key):
    headers = {"Authorization": "Bearer {}".format(api_key)}

    url = "https://api.twitter.com/2/tweets/search/recent?query={}&tweet.fields={}&start_time={}&end_time={}&max_results={}".format(
        query,
        tweet_fields,
        start_time,
        end_time,
        max_results,
    )
    response = requests.request("GET", url, headers=headers)

    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


# Twitter query
# Consider adding: place_country:GB, has:geo (so we ensure a place mention)
query = "(suez canal) OR (suezcanal) OR (suezcanal)"

max_results = 100

# Pull twitter fields including tweet id, text, coordinates, user, retweet count, time
# tweet_fields = "text,id,geo,lang,created_at,referenced_tweets,author_id"
tweet_fields = "attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,in_reply_to_user_id,lang,possibly_sensitive,public_metrics,referenced_tweets,reply_settings,source,text,withheld"

# Iterate through each 24 hour period collecting and saving tweets since the crisis started.
for day in range(23, 30):
    for hour in range(0, 24):
        # Daylight savings baby. Check here: https://www.utctime.net/
        start_time = "2021-03-"+str(day)+"T"+str(hour)+":00:00-04:00"

        end_time = "2021-03-"+str(day)+"T"+str(hour)+":59:59-04:00"

        # Language of tweets
        # lang = 'en'

        # Call our function to make the request to Twitter
        json_response = make_request(query=query, tweet_fields=tweet_fields, start_time=start_time, end_time=end_time,
                                     max_results=max_results, api_key=get_api_key())

        # Print the JSON out nicely
        pretty_json = json.dumps(json_response, indent=4, sort_keys=True)
        print(pretty_json)

        with open("saved_tweets/"+start_time+".json", "a+", encoding="utf8") as wf:
            wf.write(pretty_json)





####################################################################################
# Questions for Professor Vierthaler:
# Extract tweets about Suez canal and discuss impacts on economy via sentiment analysis and topic modeling?
# Allowed to extract only English tweets
