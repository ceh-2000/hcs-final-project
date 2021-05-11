import 'dart:convert';

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

  String formatIndividualEntities(List<dynamic> entityList){
    String finalString = '\n';
    entityList.forEach((element) {
      finalString += '- '+element+'\n';
    });
    return finalString;
  }

  Widget formatEntities(String entities) {
    var parsedJson = json.decode(entities);
    String stringToReturn = parsedJson.toString();

    return ListView(
      children: <Widget>[
        Container(
          padding: EdgeInsets.all(5.0),
          color: background2,
          child: Center(
              child: Text("Entities",
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold))),
        ),
        Container(
          padding: EdgeInsets.all(5.0),
          color: background1,
          child: Text(
            "People, including fictional: "+formatIndividualEntities(parsedJson["PERSON"]),
            textAlign: TextAlign.left,
          ),
        ),
        Container(
          padding: EdgeInsets.all(5.0),
          color: background2,
          child: Text(
            "Nationalities or religious or political groups: "+formatIndividualEntities(parsedJson["NORP"]),
            textAlign: TextAlign.left,
          ),
        ),
        Container(
            padding: EdgeInsets.all(5.0),
            color: background1,
            child: Text(
              "Companies, agencies, institutions, etc.: "+formatIndividualEntities(parsedJson["ORG"]),
              textAlign: TextAlign.left,
            )),
        Container(
            padding: EdgeInsets.all(5.0),
            color: background2,
            child: Text(
              "Non-GPE locations, mountain ranges, bodies of water: "+formatIndividualEntities(parsedJson["LOC"]),
              textAlign: TextAlign.left,
            )),
      ],
    );
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
                        decoration: BoxDecoration(
                            color: background2,
                            border: Border.all(color: color2, width: 3.0)),
                        child: Center(
                            child: formatEntities(tweetBlock.getEntities()))))
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
                        decoration: BoxDecoration(
                            color: background2,
                            border: Border.all(color: color2, width: 3.0)),
                        child: Center(
                            child: ListView(
                          children: <Widget>[
                            Container(
                              padding: EdgeInsets.all(5.0),
                              color: background2,
                              child: Center(
                                  child: Text("Tweet Sample",
                                      style: TextStyle(
                                          fontSize: 20,
                                          fontWeight: FontWeight.bold))),
                            ),
                            Container(
                              padding: EdgeInsets.all(5.0),
                              color: background1,
                              child: Text(
                                tweetBlock.getTweets()[0],
                                textAlign: TextAlign.left,
                              ),
                            ),
                            Container(
                              padding: EdgeInsets.all(5.0),
                              color: background2,
                              child: Text(
                                tweetBlock.getTweets()[1],
                                textAlign: TextAlign.left,
                              ),
                            ),
                            Container(
                                padding: EdgeInsets.all(5.0),
                                color: background1,
                                child: Text(
                                  tweetBlock.getTweets()[2],
                                  textAlign: TextAlign.left,
                                )),
                          ],
                        ))))
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
            margin: const EdgeInsets.all(15.0),
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
