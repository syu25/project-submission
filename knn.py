import sklearn
from sklearn import neighbors, preprocessing
import process as pr
from sklearn.model_selection import train_test_split
import graphresult
import pickle


def knn_regressor(data, rows, samples=25, saveImg=False, saveM=False):

    #formats and shuffles data
    df = pr.format_input(data, rows)
    df = sklearn.utils.shuffle(df)

    #prepare data for training and testing
    X = df.drop("age", axis=1).values
    X = preprocessing.scale(X)
    y = df["age"].values

    #split the training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    #train the machine
    knnReg = neighbors.KNeighborsRegressor()
    knnReg.fit(X_train, y_train)

    #test the machines accuracy and show a prediction
    y_pred = knnReg.predict(X_test)
    accuracy = knnReg.score(X_test, y_test)

    #option to save the machine
    if saveM:
        with open('machines/knnregression.pickle', 'wb') as f:
            pickle.dump(knnReg, f)

    #graph the results
    title = "K-Nearest-Neighbors-Regression"
    graphresult.graph_me(y_pred, y_test, title, accuracy, samples=samples, save=saveImg)