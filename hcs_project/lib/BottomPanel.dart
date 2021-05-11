import 'dart:async';

import 'package:flutter/material.dart';

import 'TweetBlock.dart';

class BottomPanel extends StatefulWidget {
  BottomPanel(
      {Key? key,
      required this.tweetBlocks,
      required this.index,
      required this.onChanged})
      : super(key: key);

  List<TweetBlock> tweetBlocks;
  final ValueChanged<int> onChanged;
  int index;

  @override
  BottomPanelState createState() => BottomPanelState(tweetBlocks);
}

class BottomPanelState extends State<BottomPanel> {
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

  BottomPanelState(tweetBlocks) {
    _tweetBlocks = tweetBlocks;
  }

  void updateIndex(int index) {
    setState(() {
      _index = index;
      print(_index);
    });
  }

  String _formateDateTime(String datetime) {
    String finalString = '';
    List<String> listdatetime = datetime.split(' ');
    finalString += listdatetime[0] + ' ' + listdatetime[1] + '\n';
    List<String> splitagain = listdatetime[3].split('\-');
    finalString += splitagain[0] + '-\n' + splitagain[1];

    return finalString;
  }

  Widget _getTimelineWidgets() {
    return SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: Row(
            children: _tweetBlocks
                .map((item) => Container(
                    margin: EdgeInsets.all(5.0),
                    child: MaterialButton(
                        hoverColor:  _index == item.getIndex() ? color1 : background1,
                        color: _index == item.getIndex() ? color1 : background2,
                        onPressed: () {
                          widget.onChanged(item.getIndex());
                          updateIndex(item.getIndex());
                        },
                        child: Padding(
                            padding: EdgeInsets.all(5.0),
                            child: Center(
                                child: Text(
                              _formateDateTime(item.getDate()),
                              textAlign: TextAlign.center,
                                  style: TextStyle(color: textColor),
                            ))))))
                .toList()));
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
          child: Center(child: _getTimelineWidgets())),
    );
  }
}
