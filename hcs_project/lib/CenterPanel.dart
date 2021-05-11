import 'dart:async';
import 'package:csv/csv.dart';
import 'package:flutter/material.dart';

import 'TweetBlock.dart';

class CenterPanel extends StatefulWidget {
  CenterPanel(
      {Key? key,
      required this.tweetBlocks,
      required this.index,
      required this.onChanged})
      : super(key: key);

  List<TweetBlock> tweetBlocks;
  final ValueChanged<int> onChanged;
  int index;

  @override
  CenterPanelState createState() => CenterPanelState(tweetBlocks);
}

class CenterPanelState extends State<CenterPanel> {
  ///////////////////////////////////////////////////
  // Constructors, initializers, and manage states

  Color background2 = Color.fromRGBO(254, 255, 255, 1.0);
  Color background1 = Color.fromRGBO(222, 242, 241, 1.0);
  Color color1 = Color.fromRGBO(58, 175, 169, 1.0);
  Color color2 = Color.fromRGBO(43, 122, 120, 1.0);
  Color textColor = Color.fromRGBO(23, 37, 42, 1.0);

  int _index = 0;
  List<TweetBlock> _tweetBlocks = [];

  @override
  void initState() {}

  CenterPanelState(tweetBlocks) {
    _tweetBlocks = tweetBlocks;
  }

  void updateIndex(int index) {
    setState(() {
      _index = index;
    });
  }

  void _handleTapLeft() {
    print('Left');
    if (widget.index > 0) {
      widget.onChanged(-1);
    }
  }

  void _handleTapRight() {
    print('Right');
    if (widget.index < 27) {
      widget.onChanged(1);
    }
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
            flex: 7,
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
            flex: 5,
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
        Flexible(
            flex: 1,
            child: Row(
              children: [
                ElevatedButton(
                  child: Text('Left'),
                  onPressed: () {
                    _handleTapLeft();
                  },
                ),
                ElevatedButton(
                  child: Text('Right'),
                  onPressed: () {
                    _handleTapRight();
                  },
                )
              ],
            ))
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
              child: tweetBlockCard(_tweetBlocks[_index]),
            ))));
  }
}
