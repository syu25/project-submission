import 'dart:core';
import 'package:cse474demo/screens/homePage.dart';
import 'package:flutter/material.dart';
import 'package:fluttertoast/fluttertoast.dart';

class addPage extends StatefulWidget {
  @override
  _addPageState createState() => _addPageState();
}

class _addPageState extends State<addPage> {
  String _currentSearch = '';
  String _currentPrice = '';
  String _currentPrediction = '';
  List<String> listOfStock = ['AAPL', 'MSFT', 'AMZN', 'AMD', 'NVDA', 'DIS'];
  List<double> listOfCurrentPrice = [291.02, 176.76, 2308.98, 50.78, 287.59, 102.14];
  List<double> listOfPrediction = [282.31, 180.12, 2322.64, 47.84, 291.49, 111.56];

  @override
  Widget build(BuildContext context) {
    String newStock = '';
    double newPrice = 0.0;
    double newPrediction = 0.0;
    return Scaffold(
      appBar: AppBar(
        title: Text('Search for new stock'),
        actions: [
          RaisedButton(
            child: Icon(Icons.done),
            color: Colors.blue,
            textColor: Colors.white,
            onPressed: () {
              Fluttertoast.showToast(
                  msg: "DIS is added to watchlist!",
                  toastLength: Toast.LENGTH_SHORT,
                  gravity: ToastGravity.BOTTOM,
                  backgroundColor: Colors.red,
                  textColor: Colors.white,
                  fontSize: 16.0
              );
              Navigator.pushAndRemoveUntil(context, MaterialPageRoute(builder: (context) => homePage(listOfStock, listOfCurrentPrice, listOfPrediction)), ModalRoute.withName('/homePage'));
            },
          )
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Column(
          children: [
            TextField(
              textAlign: TextAlign.center,
              decoration: new InputDecoration(hintText: 'Enter company name or stock code', ),
              onChanged: (value) { _currentSearch = value;},
            ),
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: RaisedButton(
                child: Text('Search', style: TextStyle(fontSize: 20),),
                onPressed: () {
                  setState(() {
                    _currentPrice = '102.14';
                    _currentPrediction = '111.56';
                  });
                },
              ),
            ),
            Card(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: <Widget>[
                  ListTile(
                    title: Text(_currentSearch),
                    subtitle: Text(_currentPrice),
                  ),
                  ButtonBar(
                    children: <Widget>[
                      Text(_currentPrediction == '' ? '' : 'Prediction:', style: TextStyle(fontSize: 20),),
                      Text(_currentPrediction, style: TextStyle(color: Colors.green, fontSize: 20),)
                    ],
                  ),
                ],
              ),
            )
          ],
        ),
      ),
    );
  }
}
