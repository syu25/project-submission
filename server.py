from flask import Flask
from flask_cors import CORS, cross_origin
from main import *
app = Flask(__name__)
cors = CORS(app)

@app.route('/head1')
@cross_origin()
def head1():
    return get_head1()

@app.route('/desc1')
@cross_origin()
def desc1():
    return get_desc1()

@app.route('/null1')
@cross_origin()
def null1():
    return get_null1()

@app.route('/null2')
@cross_origin()
def null2():
    return get_null2()

@app.route('/head2')
@cross_origin()
def head2():
    return get_head2()

@app.route('/worst')
@cross_origin()
def worst():
    return get_worst()

@app.route('/best')
@cross_origin()
def best():
    return get_best()

@app.route('/result1')
@cross_origin()
def result1():
    return get_result1()

@app.route('/error1')
@cross_origin()
def error1():
    return get_error1()

@app.route('/result2')
@cross_origin()
def result2():
    return get_result2()

@app.route('/error2')
@cross_origin()
def error2():
    return get_error2()

@app.route('/result3')
@cross_origin()
def result3():
    return get_result3()

@app.route('/error3')
@cross_origin()
def error3():
    return get_error3()

@app.route('/error4')
@cross_origin()
def error4():
    return get_error4()

@app.route('/error5')
@cross_origin()
def error5():
    return get_error5()

@app.route('/error6')
@cross_origin()
def error6():
    return get_error6()
