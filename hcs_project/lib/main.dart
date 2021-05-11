import 'dart:html';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:csv/csv.dart';

import 'LeftPanel.dart';
import 'CenterPanel.dart';
import 'RightPanel.dart';
import 'BottomPanel.dart';
import 'TweetBlock.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Suez Canal Tweets',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: Main(),
    );
  }
}

class Main extends StatefulWidget {
  @override
  _Main createState() => _Main();
}

class _Main extends State<Main> {
  ///////////////////////////////////////////////////
  // Constructors, initializers, and manage states

  Color background2 = Color.fromRGBO(254, 255, 255, 1.0);
  Color background1 = Color.fromRGBO(222, 242, 241, 1.0);
  Color color1 = Color.fromRGBO(58, 175, 169, 1.0);
  Color color2 = Color.fromRGBO(43, 122, 120, 1.0);
  Color textColor = Color.fromRGBO(23, 37, 42, 1.0);

  @override
  void initState() {}

  _Main() {}

  // Read in list with flutter
  // https://www.kindacode.com/article/flutter-load-and-display-content-from-csv-files/
  Future<List<TweetBlock>> _loadCSV() async {
    final _rawData = await rootBundle.loadString('assets/files/tweet_blocks.csv');
    List<List<dynamic>> listData = CsvToListConverter().convert(_rawData);


    List<TweetBlock> listOfTweetBlocks = [];
    int counter = 0;
    listData.forEach((List<dynamic> row){
      if(counter > 0 && counter < 29){
        int index = row[0].toInt();
        String date = row[1].toString();
        double polarity = row[2].toDouble();
        double subjectivity = row[3].toDouble();
        String imageURL = index.toString()+'.png';
        List<String> entities = row[5].split(',');
        List<String> tweets = [];
        tweets.add(row[6].toString());
        tweets.add(row[7].toString());
        tweets.add(row[8].toString());

        listOfTweetBlocks.add(TweetBlock(polarity, subjectivity, date, index, imageURL, entities, tweets));
      }
      counter += 1;
    });

    return listOfTweetBlocks;
  }

  @override
  Widget build(BuildContext context) {

    return Scaffold(
        backgroundColor: background1,
        body: FutureBuilder(
            future: _loadCSV(),
            builder: (BuildContext context, AsyncSnapshot<List<TweetBlock>> tweetBlocks) {

              if(tweetBlocks.data == null){
                return CircularProgressIndicator();
              }
              else{
                List<TweetBlock> finalTweetBlocks = tweetBlocks.data!;

                return Center(
                  child: Padding(
                      padding: EdgeInsets.all(20.0),
                      child: Column(children: <Widget>[
                        Flexible(
                            flex: 6,
                            child: Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                crossAxisAlignment: CrossAxisAlignment.center,
                                children: <Widget>[
                                  Flexible(flex: 2, child: LeftPanel()),
                                  Flexible(flex: 3, child: CenterPanel(currentTweetBlock: finalTweetBlocks[26])),
                                  Flexible(flex: 2, child: RightPanel())
                                ])),
                        Flexible(flex: 1, child: BottomPanel())
                      ])),
                );
              }

            }));
  }
}
