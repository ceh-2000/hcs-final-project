import 'dart:async';

import 'package:flutter/material.dart';

class RightPanel extends StatefulWidget {
  @override
  _RightPanel createState() => _RightPanel();
}

class _RightPanel extends State<RightPanel> {
  ///////////////////////////////////////////////////
  // Constructors, initializers, and manage states

  Color background2 = Color.fromRGBO(254, 255, 255, 1.0);
  Color background1 = Color.fromRGBO(222, 242, 241, 1.0);
  Color color1 = Color.fromRGBO(58, 175, 169, 1.0);
  Color color2 = Color.fromRGBO(43, 122, 120, 1.0);
  Color textColor = Color.fromRGBO(23, 37, 42, 1.0);

  @override
  void initState() {}

  _RightPanel() {}

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
                  child: new SingleChildScrollView(
                      child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          crossAxisAlignment: CrossAxisAlignment.center,
                          children: <Widget>[
                        SizedBox(
                          width: double.infinity,
                          child: Text('Trends Over Time',
                              style: TextStyle(
                                  color: textColor,
                                  fontSize: 20,
                                  fontWeight: FontWeight.bold)),
                        ),
                        SizedBox(
                          height: 15.0,
                        ),
                        SizedBox(
                          width: double.infinity,
                          child: Text(
                              'All trends are derived from a score assigned to all tweets in a six-hour period referred to as a tweet block. The following two plots show tweet block polarity and subjectivity over time.',
                              style: TextStyle(color: textColor, fontSize: 15)),
                        ),
                        SizedBox(
                          height: 15.0,
                        ),
                        SizedBox(
                          width: double.infinity,
                          child: Text('Tweet Block Subjectivity',
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
                          child: Container(
                              decoration: BoxDecoration(
                                  color: background2,
                                  border:
                                  Border.all(color: color2, width: 3.0)),
                              child: Center(
                                  child: Image(
                                    image: AssetImage(
                                        "assets/images/block_subjectivity.png"),
                                  )))
                        ),
                        SizedBox(
                          height: 15.0,
                        ),
                        SizedBox(
                          width: double.infinity,
                          child: Text('Analysis',
                              style: TextStyle(color: textColor, fontSize: 15)),
                        ),
                        SizedBox(
                          height: 15.0,
                        ),
                        SizedBox(
                          width: double.infinity,
                          child: Text('Tweet Block Polarity',
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
                            child: Container(
                                decoration: BoxDecoration(
                                    color: background2,
                                    border:
                                        Border.all(color: color2, width: 3.0)),
                                child: Center(
                                    child: Image(
                                  image: AssetImage(
                                      "assets/images/block_polarity.png"),
                                )))),
                        SizedBox(
                          height: 15.0,
                        ),
                        SizedBox(
                          width: double.infinity,
                          child: Text('Analysis',
                              style: TextStyle(color: textColor, fontSize: 15)),
                        )
                      ]))),
            )));
  }
}
