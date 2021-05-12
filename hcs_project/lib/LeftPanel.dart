import 'dart:async';

import 'package:flutter/material.dart';

class LeftPanel extends StatefulWidget {
  @override
  _LeftPanel createState() => _LeftPanel();
}

class _LeftPanel extends State<LeftPanel> {
  ///////////////////////////////////////////////////
  // Constructors, initializers, and manage states

  Color background2 = Color.fromRGBO(254, 255, 255, 1.0);
  Color background1 = Color.fromRGBO(222, 242, 241, 1.0);
  Color color1 = Color.fromRGBO(58, 175, 169, 1.0);
  Color color2 = Color.fromRGBO(43, 122, 120, 1.0);
  Color textColor = Color.fromRGBO(23, 37, 42, 1.0);

  @override
  void initState() {}

  _LeftPanel() {}

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        backgroundColor: background1,
        body: Container(
            margin: const EdgeInsets.all(15.0),
            padding: const EdgeInsets.all(3.0),
            decoration: BoxDecoration(
                color: background2,
                border: Border.all(color: color2, width: 6.0)),
            child: Center(
              child: Padding(
                  padding: EdgeInsets.all(20.0),
                  child: SingleChildScrollView(
                      child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          crossAxisAlignment: CrossAxisAlignment.center,
                          children: <Widget>[
                        Row(
                          children: [
                            Icon(Icons.menu_book_rounded,
                                  size: 40, color: color1),
                            SizedBox(width: 15.0),
                            Text('Overview',
                                style: TextStyle(
                                    color: textColor,
                                    fontSize: 20,
                                    fontWeight: FontWeight.bold)),
                          ],
                        ),
                        SizedBox(height: 15),
                        SizedBox(
                            width: double.infinity,
                            child: Container(
                                decoration: BoxDecoration(
                                    color: background2,
                                    border:
                                        Border.all(color: color2, width: 3.0)),
                                child: Center(
                                    child: Image(
                                  image: AssetImage(
                                      "assets/images/suez_canal.jpeg"),
                                )))),
                        SizedBox(height: 15),
                        SizedBox(
                          width: double.infinity,
                          child: Text('Project Background',
                              style: TextStyle(
                                  color: textColor,
                                  fontSize: 15,
                                  fontWeight: FontWeight.bold)),
                        ),
                        SizedBox(
                          height: 15.0,
                        ),
                        SizedBox(
                          width: double.infinity,
                          child: Text(
                              'I analyze Twitter data mentioning the Suez Canal crisis, a shipping blockage that harmed the global economy and cost billions of dollars per day from March 23-29, 2021. I performed sentiment analysis, topic modeling, and entity extraction and display my findings as an interactive timeline and time series plots in a Flutter-built website.\n\nThis project is appealing because of the economic and political tensions surrounding the Suez Canal crisis and because I have never worked with tweet data or Flutter for the web before.',
                              style: TextStyle(color: textColor, fontSize: 15)),
                        ),
                        SizedBox(height: 15),
                        SizedBox(
                          width: double.infinity,
                          child: Text(
                              'Crisis Explained and Connection to China',
                              style: TextStyle(
                                  color: textColor,
                                  fontSize: 15,
                                  fontWeight: FontWeight.bold)),
                        ),
                        SizedBox(
                          height: 15.0,
                        ),
                        SizedBox(
                          width: double.infinity,
                          child: Text(
                              'The Suez Canal connects the Mediterrean Sea to the Indian Ocean via a 120-mile man-made channel. Global politics have played out in the Suez Canal since it was built via French colonialism in 1859. When Egypt tried to reclaim their independence and control over the canal, France, Britain, and Israel invaded and called on one of the two world superpowers, the United States, for backing. When the United States refused to aid British efforts, a potential hot spot in the Cold War between the U.S. and the Soviet Union fizzled out.\n\nThe economic importance of the canal underscores all political motives to control the canal. Approximately 10% of all shipping goes through the canal which translates to about ten billion dollars per day. When the 200,000 ton Japanese ship the Ever Given traveling from China to Europe blocked all canal traffic, experts predicted that the world would feel the economic effects of this maritime trade shutdown for a long time after the ship was dislodged.\n\nThe importance of Chinese exports to the global economy was underscored through this crisis. Europe and America were reminded of their reliance on China for goods.',
                              style: TextStyle(color: textColor, fontSize: 15)),
                        ),
                            SizedBox(height: 15),
                            SizedBox(
                              width: double.infinity,
                              child: Text(
                                  'Why Twitter',
                                  style: TextStyle(
                                      color: textColor,
                                      fontSize: 15,
                                      fontWeight: FontWeight.bold)),
                            ),
                            SizedBox(
                              height: 15.0,
                            ),
                            SizedBox(
                              width: double.infinity,
                              child: Text(
'The world watched as tug boats, tides, and digging tried to free the Ever Given. Much of this discourse played out on Twitter, where many used the ship as a metaphor for everything from boyfriends to political figures like Mitch McConnell. Tweets can be analyzed using Natural Language Processing tools. Tweet analysis and the findings displayed here show the power of digital humanities.',
                                      style: TextStyle(color: textColor, fontSize: 15)),
                            ),
                      ]))),
            )));
  }
}
