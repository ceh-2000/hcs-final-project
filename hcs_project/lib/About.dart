import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:csv/csv.dart';

import 'LeftPanel.dart';
import 'CenterPanel.dart';
import 'RightPanel.dart';
import 'BottomPanel.dart';
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        backgroundColor: background1,
        body: ListView(
          children: [
            Stack(children: [
              Positioned(
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
                            'About this project',
                            style: TextStyle(
                                fontSize: 50, fontWeight: FontWeight.bold),
                          ))
                    ],
                  )
                ],
              )),
              Positioned(
                  top: 10,
                  right: 15,
                  child: IconButton(
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
