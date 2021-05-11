import 'dart:async';
import 'package:csv/csv.dart';
import 'package:flutter/material.dart';

import 'TweetBlock.dart';

class CenterPanel extends StatefulWidget {
  TweetBlock currentTweetBlock;

  CenterPanel({required this.currentTweetBlock});

  @override
  _CenterPanel createState() => _CenterPanel(currentTweetBlock);
}

class _CenterPanel extends State<CenterPanel> {
  ///////////////////////////////////////////////////
  // Constructors, initializers, and manage states

  Color background2 = Color.fromRGBO(254, 255, 255, 1.0);
  Color background1 = Color.fromRGBO(222, 242, 241, 1.0);
  Color color1 = Color.fromRGBO(58, 175, 169, 1.0);
  Color color2 = Color.fromRGBO(43, 122, 120, 1.0);
  Color textColor = Color.fromRGBO(23, 37, 42, 1.0);

  // Initialize default tweet block
  TweetBlock _currentTweetBlock = TweetBlock(0.1, 0.2, 'March 23 00:00-06:00', 0, '0.png',
      ['hello', 'what is up', 'nothing'], ['hello', 'tweet', 'tweet']);

  @override
  void initState() {}

  _CenterPanel(currentTweetBlock) {
    _currentTweetBlock = currentTweetBlock;

  }

  String formatEntities(List<String> entities) {
    String stringToReturn = '';
    int counter = 0;
    entities.forEach((String entity) {
      if (counter != entities.length - 1) {
        stringToReturn += entity + '\n';
      } else {
        stringToReturn += entity;
      }
      counter += 1;
    });
    return stringToReturn;
  }

  String formatTweets(List<String> tweets) {
    String stringToReturn = '';
    int counter = 0;
    tweets.forEach((String tweet) {
      if (counter != tweets.length - 1) {
        stringToReturn += tweet + '\n\n';
      } else {
        stringToReturn += tweet;
      }
      counter += 1;
    });
    return stringToReturn;
  }

  Widget tweetBlockCard(TweetBlock tweetBlock) {
    return Column(
      children: <Widget>[
        Flexible(
            flex: 1,
            child: Row(
              children: <Widget>[
                Flexible(
                    flex: 1,
                    child: Container(
                        margin: const EdgeInsets.all(5.0),
                        padding: const EdgeInsets.all(3.0),
                        decoration: BoxDecoration(
                            color: background2,
                            border: Border.all(color: color2, width: 3.0)),
                        child: Center(
                            child: Text(tweetBlock.getDate(),
                                style: TextStyle(
                                    fontSize: 20,
                                    fontWeight: FontWeight.bold))))),
                Flexible(
                    flex: 1,
                    child: Container(
                        margin: const EdgeInsets.all(5.0),
                        padding: const EdgeInsets.all(3.0),
                        decoration: BoxDecoration(
                            color: background2,
                            border: Border.all(color: color2, width: 3.0)),
                        child: Center(
                            child: Text(
                                'Polarity: ' +
                                    tweetBlock.getPolarity().toString() +
                                    '\nSubjectivity: ' +
                                    tweetBlock.getSubjectivity().toString(),
                                style: TextStyle(fontSize: 20)))))
              ],
            )),
        Flexible(
            flex: 3,
            child: Row(
              children: <Widget>[
                Flexible(
                    flex: 2,
                    child: Container(
                        margin: const EdgeInsets.all(5.0),
                        decoration: BoxDecoration(
                            color: background2,
                            border: Border.all(color: color2, width: 3.0)),
                        child: Center(
                          child: Image(
                            image: AssetImage(
                                'assets/images/' + tweetBlock.getImageURL()),
                          ),
                        ))),
                Flexible(
                    flex: 1,
                    child: Container(
                        margin: const EdgeInsets.all(5.0),
                        padding: const EdgeInsets.all(3.0),
                        decoration: BoxDecoration(
                            color: background2,
                            border: Border.all(color: color2, width: 3.0)),
                        child: Center(
                            child: Text(
                                formatEntities(tweetBlock.getEntities())))))
              ],
            )),
        Flexible(
            flex: 2,
            child: Row(
              children: <Widget>[
                Flexible(
                    flex: 1,
                    child: Container(
                        margin: const EdgeInsets.all(5.0),
                        padding: const EdgeInsets.all(3.0),
                        decoration: BoxDecoration(
                            color: background2,
                            border: Border.all(color: color2, width: 3.0)),
                        child: Center(
                            child: Text(formatTweets(tweetBlock.getTweets())))))
              ],
            )),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        backgroundColor: background1,
        body: Container(
            margin: const EdgeInsets.all(10.0),
            padding: const EdgeInsets.all(3.0),
            decoration: BoxDecoration(
                color: background2,
                border: Border.all(color: color2, width: 6.0)),
            child: Center(
                child: Padding(
              padding: EdgeInsets.all(10.0),
              child: tweetBlockCard(_currentTweetBlock),
            ))));
  }
}
