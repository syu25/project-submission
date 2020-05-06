import sklearn
from sklearn.linear_model import RidgeCV
from sklearn import preprocessing
import process as pr
from sklearn.model_selection import train_test_split
import graphresult
import pickle

def ridgeCV_regressor(data, rows, samples=25, saveImg=False, saveM=False):
   
    #formats and shuffles data
    df = pr.format_input(data, rows)
    df = sklearn.utils.shuffle(df)

    X = df.drop("age", axis=1).values
    X = preprocessing.scale(X)
    y = df["age"].values

    #split the training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    #train the machine
    ridCV = RidgeCV()
    ridCV.fit(X_train, y_train)

    #test the machines accuracy and show a prediction
    y_pred = ridCV.predict(X_test)
    accuracy = ridCV.score(X_test, y_test)

    #option to save the machine
    if saveM:
        with open('machines/rcv.pickle', 'wb') as f:
            pickle.dump(ridCV, f)

    #graph the results
    title = "Ridge-Cross-Validation-Regression-Accuracy"
    graphresult.graph_me(y_pred, y_test, title, accuracy, samples=samples, save=saveImg)


