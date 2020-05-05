import sklearn
from sklearn import preprocessing
import process as pr
import graphresult
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle




def linear_regressor(data, rows, samples=25, saveImg=False, saveM=False):

    #formats and shuffles data
    df = pr.format_input(data, rows)
    df = sklearn.utils.shuffle(df)

    
    X = df.drop("age", axis=1).values
    X = preprocessing.scale(X)
    y = df["age"].values

    #split the training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    #train the machine
    lineReg = LinearRegression(n_jobs=-1)
    lineReg.fit(X_train, y_train)

    #test the machines accuracy and show a prediction
    y_pred = lineReg.predict(X_test)
    accuracy = lineReg.score(X_test, y_test)

    #option to save the machine
    if saveM:
        with open('machines/linearregression.pickle', 'wb') as f:
            pickle.dump(lineReg, f)
            
    #graph the results
    title = "Linear-Regression-Accuracy"
    graphresult.graph_me(y_pred, y_test, title, accuracy, samples=samples, save=saveImg)


