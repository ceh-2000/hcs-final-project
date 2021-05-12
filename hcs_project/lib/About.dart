import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'CenterPanel.dart';
import 'TweetBlock.dart';

class About extends StatefulWidget {
  @override
  _About createState() => _About();
}

class _About extends State<About> {
  ///////////////////////////////////////////////////
  // Constructors, initializers, and manage states

  Color background2 = Color.fromRGBO(254, 255, 255, 1.0);
  Color background1 = Color.fromRGBO(222, 242, 241, 1.0);
  Color color1 = Color.fromRGBO(58, 175, 169, 1.0);
  Color color2 = Color.fromRGBO(43, 122, 120, 1.0);
  Color textColor = Color.fromRGBO(23, 37, 42, 1.0);

  GlobalKey<CenterPanelState> _keyCenterPanel = GlobalKey();
  GlobalKey<CenterPanelState> _keyBottomPanel = GlobalKey();

  int _index = 0;

  List<TweetBlock> _finalTweetBlocks = [];

  @override
  void initState() {}

  _About() {}

  void _launchURL(String url) async =>
      await canLaunch(url) ? await launch(url) : throw 'Could not launch $url';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        backgroundColor: background1,
        body: ListView(
          children: [
            Stack(children: [
              Positioned(
                  child: Container(
                      padding: EdgeInsets.all(20),
                      child: Column(
                        children: [
                          Row(
                            children: [
                              Flexible(
                                flex: 1,
                                child: Hero(
                                    tag: 'more',
                                    child: Icon(
                                      Icons.bookmark,
                                      size: 100,
                                      color: color1,
                                    )),
                              ),
                              Flexible(
                                  flex: 1,
                                  child: Text(
                                    'About This Project',
                                    textAlign: TextAlign.left,
                                    style: TextStyle(
                                        fontSize: 50,
                                        fontWeight: FontWeight.bold),
                                  ))
                            ],
                          ),
                          Text("\nConcept",
                              style: TextStyle(
                                  color: textColor,
                                  fontSize: 22,
                                  fontWeight: FontWeight.bold)),
                          Text(
                              "My favorite museum exhibitions are interactive. I wanted my project to feel like an interactive timeline that museum curators would project on a giant screen. Little kids would click through the timeline at the bottom while their older siblings would read the tweets and giggle at the curse words and memes. Adults would see the connection to digital humanities and hopefully appreciate a visually appealing way to step through an event like the blocking of the Suez Canal explored through tweets.",
                              style: TextStyle(color: textColor, fontSize: 18)),
                          Text("\nData Collection",
                              style: TextStyle(
                                  color: textColor,
                                  fontSize: 22,
                                  fontWeight: FontWeight.bold)),
                          Text(
                              "The world watched as tug boats, tides, and digging tried to free the Ever Given. Much of this discourse played out on Twitter, where many used the ship as a metaphor for everything from boyfriends to political figures like Mitch McConnell. Twitter has a policy that users can only access “recent” tweets via their API up to seven days after the tweets are posted. I collected one-hundred tweets every hour from March 23, 2021 at 12 am EST to March 29, 2021 at 7 pm EST. I gathered tweet data including text, time created, ID, type, and language. Most of the tweets were retweeted which only have the first 144 characters of the text, but I had the original tweets’s IDs thanks to the `referenced_tweet` parameter. I wrote another Python script to extract the full tweet and put the tweets into a pandas dataframe which I extracted to a CSV. I ran this script as a job overnight in Cloudera, because I could only send one request every 5 seconds and there were thousands of retweets to rehydrate.",
                              style: TextStyle(color: textColor, fontSize: 18)),
                          Text("\nPreprocessing",
                              style: TextStyle(
                                  color: textColor,
                                  fontSize: 22,
                                  fontWeight: FontWeight.bold)),
                          Text(
                              "I read all tweets, including rehydrated retweets, into a pandas dataframe. Then, I extracted only English tweets by filtering on the `en` tag. Next, I ordered the tweets chronologically using the `created_at` column. This made it possible to split the tweets into six-hour blocks of time called tweet blocks. Six hours made the most sense because it was enough data to shrink the overall number of tweet data blocks from over 10,000 tweets to twenty-eight, but it was not too general because trends were still obvious in the subjectivity and polarity analysis. In order to group into six-hour blocks, I needed to extract the hour from the timestamp. I used regular expressions to match on the date and hour pattern. Then, I created a new column in my dataframe to assign each tweet to a tweet block, which made it possible to pinpoint the beginning and end of the crisis according to Twitter discourse.",
                              style: TextStyle(color: textColor, fontSize: 18)),
                          Text("\nAnalysis",
                              style: TextStyle(
                                  color: textColor,
                                  fontSize: 22,
                                  fontWeight: FontWeight.bold)),
                          Text(
                              "First, I conducted subjectivity and polarity analysis for every tweet block using `TextBlob`. Using `matplotlib` and `seaborn`, I plotted these generated values over time. One challenge I faced was determining how to mark the crisis start and end, but this was solved using the `annotate` function from `matplotlib` and the `axvline` function to draw the red and green vertical lines exhibited on the plots.\n\nNext, I wanted to extract entities from each tweet block. Initially I extracted all entities, but many of the entity types were not relevant to these tweets including works of art, language, and percent. I settled on people, nationalities/groups, organizations, and locations (which includes bodies of water). An interesting result was that many of the person entities were political figures including Joe Biden, Mitch McConnell, Kamala Harris, and Hillary Clinton. I was curious to see if they would form a topic when I performed topic modeling on the data.\n\nI tried topic modelling with `gensim` and `MALLET`, but the generated topics were all just variations of the phrase “Suez Canal,” and thus not very meaningful. I decided not to include these results in the website, but they can be found commented out in the `analysis.py` file in the GitHub repository for this project.",
                              style: TextStyle(color: textColor, fontSize: 18)),
                          Text("\nWebsite Construction",
                              style: TextStyle(
                                  color: textColor,
                                  fontSize: 22,
                                  fontWeight: FontWeight.bold)),
                          Text(
                              "I wrote my own Flutter website to display my analysis findings. I copied my original design exactly and was honestly surprised that I could get everything to work.\n\nI had the most trouble allowing the clickable bottom panel to control which tweet block was displayed in the center panel. I spent over two hours researching how to allow parent widgets to send information to a particular child widget until I realized I need to remove an underscore. This allowed me to assign a global key to the Center Panel widget to call the `updateIndex` function in `CenterPanelState`. I sent information from the child widget Bottom Panel to the parent widget Main using callback functions, which was a new experience for me.\n\nIt was also my first time implementing animations. The transition from the `Main` page to this `About` page uses a hero animation.\n\nIn the past I have created websites using raw HTML, JavaScript, and CSS. Using Flutter for web produced, in my opinion, a much cleaner website because it uses some premade widgets and a consistent Material design scheme. I would definitely make another website with Flutter, and next time I will bypass the challenges related to parent/child widget communication.",
                              style: TextStyle(color: textColor, fontSize: 18)),
                          Text("\nResources",
                              style: TextStyle(
                                  color: textColor,
                                  fontSize: 22,
                                  fontWeight: FontWeight.bold)),
                          InkWell(
                              child: new Text('Github repository for this project',  style: TextStyle(color: Colors.blueAccent, fontSize: 18)),
                              onTap: () => launch('https://github.com/ceh-2000/hcs-final-project')
                          ),
                          InkWell(
                              child: new Text('Global Times article',  style: TextStyle(color: Colors.blueAccent, fontSize: 18)),
                              onTap: () => launch('https://www.globaltimes.cn/page/202103/1219660.shtml')
                          ),
                          InkWell(
                              child: new Text('Britannica Suez Canal overview',  style: TextStyle(color: Colors.blueAccent, fontSize: 18)),
                              onTap: () => launch('https://www.britannica.com/topic/Suez-Canal/The-economy')
                          ),
                          InkWell(
                              child: new Text('New York Times article',  style: TextStyle(color: Colors.blueAccent, fontSize: 18)),
                              onTap: () => launch('https://www.nytimes.com/2021/03/25/world/middleeast/suez-canal-container-ship.html?auth=login-google')
                          ),
                          InkWell(
                              child: new Text('Twitter API dashboard',  style: TextStyle(color: Colors.blueAccent, fontSize: 18)),
                              onTap: () => launch('https://developer.twitter.com/en/portal/dashboard')
                          ),
                          InkWell(
                              child: new Text('Python + Twitter tutorial',  style: TextStyle(color: Colors.blueAccent, fontSize: 18)),
                              onTap: () => launch('https://towardsdatascience.com/searching-for-tweets-with-python-f659144b225f')
                          ),
                          InkWell(
                              child: new Text('Recent Search tweets',  style: TextStyle(color: Colors.blueAccent, fontSize: 18)),
                              onTap: () => launch('https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-recent#tab2')
                          ),
                          InkWell(
                              child: new Text('Flutter for web introduction',  style: TextStyle(color: Colors.blueAccent, fontSize: 18)),
                              onTap: () => launch('https://medium.com/flutter-community/flutter-for-web-building-a-portfolio-website-3e9865710efe')
                          ),
                          InkWell(
                              child: new Text('Parent/child widget interactions',  style: TextStyle(color: Colors.blueAccent, fontSize: 18)),
                              onTap: () => launch('https://medium.com/flutter-community/flutter-communication-between-widgets-f5590230df1e')
                          ),
                          InkWell(
                              child: new Text('Hero animations',  style: TextStyle(color: Colors.blueAccent, fontSize: 18)),
                              onTap: () => launch('https://flutter.dev/docs/development/ui/animations/hero-animations')
                          ),
                          InkWell(
                              child: new Text('Publish Flutter web app with Firebase hosting',  style: TextStyle(color: Colors.blueAccent, fontSize: 18)),
                              onTap: () => launch('https://medium.com/flutter/must-try-use-firebase-to-host-your-flutter-app-on-the-web-852ee533a469')
                          ),
                          InkWell(
                              child: new Text('TextBlob documentation',  style: TextStyle(color: Colors.blueAccent, fontSize: 18)),
                              onTap: () => launch('https://textblob.readthedocs.io/en/dev/')
                          ),
                          InkWell(
                              child: new Text('spaCy documentation',  style: TextStyle(color: Colors.blueAccent, fontSize: 18)),
                              onTap: () => launch('https://spacy.io/api/doc')
                          ),
                          InkWell(
                              child: new Text('seaborn documentation',  style: TextStyle(color: Colors.blueAccent, fontSize: 18)),
                              onTap: () => launch('https://seaborn.pydata.org/')
                          ),
                          InkWell(
                              child: new Text('matplotlib documentation',  style: TextStyle(color: Colors.blueAccent, fontSize: 18)),
                              onTap: () => launch('https://matplotlib.org/stable/contents.html')
                          ),
                          InkWell(
                              child: new Text('Gensim documentation',  style: TextStyle(color: Colors.blueAccent, fontSize: 18)),
                              onTap: () => launch('https://radimrehurek.com/gensim/')
                          ),
                          InkWell(
                              child: new Text('MALLET documentation',  style: TextStyle(color: Colors.blueAccent, fontSize: 18)),
                              onTap: () => launch('http://mallet.cs.umass.edu/')
                          ),
                          InkWell(
                              child: new Text('Developer Student Club at W&M workshop',  style: TextStyle(color: Colors.blueAccent, fontSize: 18)),
                              onTap: () => launch('https://www.kaggle.com/clareheinbaugh/dsc-cypher-workshop-nlp-instructor')
                          ),
                        ],
                      ))),
              Positioned(
                  top: 10,
                  right: 15,
                  child: IconButton(
                    iconSize: 40,
                    color: color2,
                    icon: Icon(Icons.cancel),
                    onPressed: () {
                      Navigator.pop(context);
                    },
                  ))
            ])
          ],
        ));
  }
}
