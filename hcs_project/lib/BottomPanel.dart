import 'dart:async';

import 'package:flutter/material.dart';

class BottomPanel extends StatefulWidget {
  @override
  _BottomPanel createState() => _BottomPanel();
}

class _BottomPanel extends State<BottomPanel> {
  ///////////////////////////////////////////////////
  // Constructors, initializers, and manage states

  Color background2 = Color.fromRGBO(254, 255, 255, 1.0);
  Color background1 = Color.fromRGBO(222, 242, 241, 1.0);
  Color color1 = Color.fromRGBO(58, 175, 169, 1.0);
  Color color2 = Color.fromRGBO(43, 122, 120, 1.0);
  Color textColor = Color.fromRGBO(23, 37, 42, 1.0);

  @override
  void initState() {}

  _BottomPanel() {}

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
                  child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: <Widget>[
                        Text('Interactive Timeline',
                            style: TextStyle(color: textColor, fontSize: 20)),
                      ]))),
        ));
  }
}
