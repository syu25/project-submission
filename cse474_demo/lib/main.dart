import 'package:cse474demo/screens/homePage.dart';
import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  List<String> listOfStock = ['AAPL', 'MSFT', 'AMZN', 'AMD', 'NVDA'];
  List<double> listOfCurrentPrice = [291.02, 176.76, 2308.98, 50.78, 287.59];
  List<double> listOfPrediction = [282.31, 180.12, 2322.64, 47.84, 291.49];

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'CSE474 Demo',
      theme: ThemeData(
        // This is the theme of your application.
        //
        // Try running your application with "flutter run". You'll see the
        // application has a blue toolbar. Then, without quitting the app, try
        // changing the primarySwatch below to Colors.green and then invoke
        // "hot reload" (press "r" in the console where you ran "flutter run",
        // or simply save your changes to "hot reload" in a Flutter IDE).
        // Notice that the counter didn't reset back to zero; the application
        // is not restarted.
        primarySwatch: Colors.blue,
        // This makes the visual density adapt to the platform that you run
        // the app on. For desktop platforms, the controls will be smaller and
        // closer together (more dense) than on mobile platforms.
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: new homePage(listOfStock, listOfCurrentPrice, listOfPrediction),
    );
  }
}
