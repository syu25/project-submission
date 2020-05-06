import 'dart:core';
import 'package:flutter/material.dart';

import 'addPage.dart';

class homePage extends StatefulWidget {
  List<String> listOfStock;
  List<double> listOfCurrentPrice;
  List<double> listOfPrediction;

  homePage(this.listOfStock, this.listOfCurrentPrice, this.listOfPrediction);

  @override
  _homePageState createState() => _homePageState(listOfStock, listOfCurrentPrice, listOfPrediction);
}

class _homePageState extends State<homePage> {
  String _currentStock = 'AAPL';
  List<String> listOfStock;
  double _currentPrice = 291.02;
  List<double> listOfCurrentPrice;
  double _currentPrediction = 282.31;
  List<double> listOfPrediction;

  _homePageState(this.listOfStock, this.listOfCurrentPrice,
      this.listOfPrediction);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          title: Text('Price Prediction'),
          actions: [
            RaisedButton(
              child: Icon(Icons.add),
              color: Colors.blue,
              textColor: Colors.white,
              onPressed: () => Navigator.push(context, new MaterialPageRoute(builder: (context) => new addPage())),
            )
          ],
        ),
        body: ListView.builder(
            itemCount: listOfStock.length,
            itemBuilder: (context, index) {
              return Card(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: <Widget>[
                    ListTile(
                      title: Text(listOfStock.elementAt(index)),
                      subtitle: Text(listOfCurrentPrice.elementAt(index).toString()),
                    ),
                    ButtonBar(
                      children: <Widget>[
                        Text('Prediction:', style: TextStyle(fontSize: 20),),
                        Text(listOfPrediction.elementAt(index).toString(), style: TextStyle(color: listOfPrediction.elementAt(index) > listOfCurrentPrice.elementAt(index) ? Colors.green : Colors.red, fontSize: 20),)
                      ],
                    ),
                  ],
                ),
              );
        }),
//        Center(
//          child: Column(
//            mainAxisAlignment: MainAxisAlignment.center,
//            children: [
//              Padding(
//                padding: const EdgeInsets.fromLTRB(8.0, 24.0, 8.0, 12.0),
//                child: Row(
//                  mainAxisAlignment: MainAxisAlignment.spaceAround,
//                  children: [
//                    Text('Pick a stock', style: TextStyle(fontSize: 20),),
//                    new DropdownButton<String>(
//                      focusColor: Colors.white,
//                      value: _currentStock == '' ? listOfStock.elementAt(0) : _currentStock,
//                      items: listOfStock.map((String value) {
//                        return new DropdownMenuItem<String>(
//                          value: value,
//                          child: new Text(value, textAlign: TextAlign.center,),
//                        );
//                      }).toList(),
//                      onChanged: (value) => setState(() {
//                        _currentStock = value;
//                        _currentPrice = listOfCurrentPrice.elementAt(listOfStock.indexOf(_currentStock));
//                        _currentPrediction = listOfPrediction.elementAt(listOfStock.indexOf(_currentStock));
//                      }) ,
//                    ),
//                  ],
//                ),
//              ),
//              Padding(
//                padding: const EdgeInsets.fromLTRB(8.0, 24.0, 8.0, 24.0),
//                child: Row(
//                  mainAxisAlignment: MainAxisAlignment.spaceAround,
//                  children: [
//                    Text('Current Price: ', style: TextStyle(fontSize: 20), ),
//                    Text(listOfCurrentPrice.elementAt(listOfStock.indexOf(_currentStock)).toString(), style: TextStyle(color: Colors.blue, fontSize: 20),)
//                  ],
//                ),
//              ),
//              Padding(
//                padding: const EdgeInsets.fromLTRB(8.0, 24.0, 8.0, 24.0),
//                child: Row(
//                  mainAxisAlignment: MainAxisAlignment.spaceAround,
//                  children: [
//                    Text('Prediction:', style: TextStyle(fontSize: 20), ),
//                    Text(listOfPrediction.elementAt(listOfStock.indexOf(_currentStock)).toString(), style: TextStyle(color: _currentPrediction > _currentPrice ? Colors.green : Colors.red, fontSize: 20),)
//                  ],
//                ),
//              )
//            ],
//          ),
//        ),
      );
  }
}
