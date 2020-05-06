let request1 = new XMLHttpRequest();
request1.open('GET', 'http://127.0.0.1:5000/head1');
request1.send();
request1.onload = function (){
    document.getElementById('head1').innerHTML = request1.response
}

let request2 = new XMLHttpRequest();
request2.open('GET', 'http://127.0.0.1:5000/desc1');
request2.send();
request2.onload = function (){
    document.getElementById('desc1').innerHTML = request2.response
}

let request3 = new XMLHttpRequest();
request3.open('GET', 'http://127.0.0.1:5000/null1');
request3.send();
request3.onload = function (){
    document.getElementById('null1').innerHTML = request3.response
}

let request4 = new XMLHttpRequest();
request4.open('GET', 'http://127.0.0.1:5000/null2');
request4.send();
request4.onload = function (){
    document.getElementById('null2').innerHTML = request4.response
}

let request5 = new XMLHttpRequest();
request5.open('GET', 'http://127.0.0.1:5000/head2');
request5.send();
request5.onload = function (){
    document.getElementById('head2').innerHTML = request5.response
}

let request6 = new XMLHttpRequest();
request6.open('GET', 'http://127.0.0.1:5000/worst');
request6.send();
request6.onload = function (){
    document.getElementById('worst').innerHTML = request6.response
}

let request7 = new XMLHttpRequest();
request7.open('GET', 'http://127.0.0.1:5000/best');
request7.send();
request7.onload = function (){
    document.getElementById('best').innerHTML = request7.response
}

// let request8 = new XMLHttpRequest();
// request8.open('GET', 'http://128.0.0.1:5000/result1');
// request8.send();
// request8.onload = function (){
//     document.getElementById('result1').innerHTML = request8.response
// }
//
// let request9 = new XMLHttpRequest();
// request9.open('GET', 'http://127.0.0.1:5000/result2');
// request9.send();
// request9.onload = function (){
//     document.getElementById('result2').innerHTML = request9.response
// }
//
// let request10 = new XMLHttpRequest();
// request10.open('GET', 'http://127.0.0.1:5000/result3');
// request10.send();
// request10.onload = function (){
//     document.getElementById('result3').innerHTML = request10.response
// }

let request11 = new XMLHttpRequest();
request11.open('GET', 'http://127.0.0.1:5000/error1');
request11.send();
request11.onload = function (){
    document.getElementById('error1').innerHTML = request11.response
}

let request12 = new XMLHttpRequest();
request12.open('GET', 'http://127.0.0.1:5000/error2');
request12.send();
request12.onload = function (){
    document.getElementById('error2').innerHTML = request12.response
}

let request13 = new XMLHttpRequest();
request13.open('GET', 'http://127.0.0.1:5000/error3');
request13.send();
request13.onload = function (){
    document.getElementById('error3').innerHTML = request13.response
}

let request14 = new XMLHttpRequest();
request14.open('GET', 'http://127.0.0.1:5000/error4');
request14.send();
request14.onload = function (){
    document.getElementById('error4').innerHTML = request14.response
}

let request15 = new XMLHttpRequest();
request15.open('GET', 'http://127.0.0.1:5000/error5');
request15.send();
request15.onload = function (){
    document.getElementById('error5').innerHTML = request15.response
}

let request16 = new XMLHttpRequest();
request16.open('GET', 'http://127.0.0.1:5000/error6');
request16.send();
request16.onload = function (){
    document.getElementById('error6').innerHTML = request16.response
}
