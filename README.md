## Hacking Chinese Studies: Final Project

### Tweet collection
Collecting up to 100 tweet per hour from just before the crisis until it ended. 
Tweets saved to individual JSON files with the time in the filename. 
If retweeted, pull the full original tweet.

### Additional Twitter query parameters
- `max_results`: parameters that specify how many tweets should be returned: A number between 10 and the system limit (currently 100). By default, a request response will return 10 results.
- `start_time`: The oldest UTC timestamp (from most recent seven days) from which the Tweets will be provided. (YYYY-MM-DDTHH:mm:ssZ (ISO 8601/RFC 3339).)
- `end_time`: The newest, most recent UTC timestamp to which the Tweets will be provided. (YYYY-MM-DDTHH:mm:ssZ (ISO 8601/RFC 3339).

### Relevant Links
- Set up new app here: https://developer.twitter.com/en/portal/dashboard
- Python + Twitter Tutorial: https://towardsdatascience.com/searching-for-tweets-with-python-f659144b225f
- Recent search Twitter documentation: https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-recent#tab2

### Preprocessing steps
- Read tweets into Pandas dataframe
- Extract out only English tweets
- Order tweets based on `created_at` column

### Sentiment analysis steps
- Split into six hour blocks of time combining all text in that time frame
- Perform sentiment analysis on each block and plot the results over time

### Topic modeling and entity extraction on each of the tweet blocks