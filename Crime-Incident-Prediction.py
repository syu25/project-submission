import csv
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.formula.api as stats

# Milestone 3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


current_file_count = 0
# validation_file = -1
colnames = ["incident_id", "case_number", "incident_datetime", "incident_type_primary", "incident_description",
            "clearance_type", "address_1", "address_2", "city", "state", "zip", "country", "latitude", "longitude",
            "created_at", "updated_at", "location", "hour_of_day", "day_of_week", "parent_incident_type",
            "Council Districts", "Police Districts", "Zip Codes", "Tracts", "Block Groups", "Blocks"]


def main():
    # Take the files and sort them by date to create new sorted csvs
    with open("Crime_Incidents.csv") as f:
        next(f)
        reader = csv.reader(f)
        sort = sorted(reader, key=lambda row: sort_by_date(row[2]))

    # Writes to a single, sorted file.
    generate_crime_incident_sort(sort)

    # Partitions, Milestone 2
    partitions = list(partition(sort, 100))
    partition_writer(partitions)

    model_generator()


def visualize(data_frame, surface1, surface2, surface3, title):
    # https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title(title)
    ax.scatter(data_frame['Longitude'], data_frame['Latitude'], data_frame['Date'], c='red', marker='o', alpha=0.5)
    ax.plot_surface(surface1, surface2, surface3.reshape(surface1.shape), color='b', alpha=0.3)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Date')


def vis(X_test, y_test, prediction, title):
    data = np.asarray(y_test)
    Z = np.asarray(X_test)
    X = data[:, 0]
    Y = data[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title(title)
    ax.scatter(Z, X, Y, c='red', marker='o', alpha=0.5)
    yt = np.asarray(prediction)
    x = yt[:, 0]
    y = yt[:, 1]
    # ax.plot(x.flatten(), y.flatten(), Z.flatten(),  c="blue", linewidth=2)
    ax.plot(Z.flatten(), x.flatten(), y.flatten(), c="blue", linewidth=2)
    ax.set_ylim(bottom=35)
    ax.set_zlim(top=-60)
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Latitude')
    ax.set_xlabel('Date')


# Writes multiple csv files by partitioning the sorted crime incidents by date
def partition_writer(partitions):
    file_count = 0
    for lst in partitions:
        with open("Crime_Incidents_Sorted" + str(file_count) + ".csv", 'w', newline='') as f:
            writer = csv.writer(f)
            for line in lst:
                writer.writerow(line)
        file_count += 1


# Writes to a single, sorted file. Previously tested and now commented out for reference
def generate_crime_incident_sort(sort):
    with open("Crime_Incidents_Sorted.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        for line in sort:
            writer.writerow(line)


# handler for model generation
def model_generator():
    # Milestone 2: temporary
    # obtain current file
    mil2 = pd.read_csv("Crime_Incidents_Sorted" + str(current_file_count) + ".csv", names=colnames)
    m2lat = mil2.latitude.tolist()
    m2long = mil2.longitude.tolist()
    m2pos = list(zip(m2lat, m2long))
    m2date = list(map(lambda x: time_convert(sort_by_date(x)), mil2.incident_datetime.tolist()))

    # Milestone 3: Failed model
    data = pd.read_csv("Crime_Incidents_Sorted.csv", names=colnames)

    # Split data again into training and validastion
    lat = data.latitude.tolist()
    long = data.longitude.tolist()
    pos = list(zip(lat, long))
    date = list(map(lambda x: time_convert(sort_by_date(x)), data.incident_datetime.tolist()))

    # hold out validation
    X_train, X_test, y_train, y_test = train_test_split(date, pos, test_size=0.2)

    print("X_train: ")
    print(X_train)
    print("X_test: ")
    print(X_test)

    print("y_train: ")
    print(y_train)
    print("y_test: ")
    print(y_test)

    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)

    # Milestone 2:
    regression_milestone_2(m2pos, m2date)

    # Failed Milestone 3:
    regression(X_train, X_test, y_train, y_test)

    # Milestone 3: Multi Output Regression Hold-out validation
    multi_output_regression(X_train, X_test, y_train, y_test)

    # Milestone 3: Multi Output Regression K-Fold cross validation
    k_fold_multi_output_regression(date, pos)

    # Show all figures
    plt.show()


def generate_graph(data_frame, inp):
    surfacex, surfacey = np.meshgrid(np.linspace(data_frame.Longitude.min(), data_frame.Longitude.max(), 100),
                                     np.linspace(data_frame.Latitude.min(), data_frame.Latitude.max(), 100))
    temp_frame = pd.DataFrame({'Longitude': surfacex.ravel(), 'Latitude': surfacey.ravel()})
    post = np.array(inp.predict(exog=temp_frame))
    print("POST:")
    print(post)
    return surfacex, surfacey, post


def time_convert(dt):
    return 10000 * dt.year + 100 * dt.month + dt.day


def partition(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def sort_by_date(date):
    if isinstance(date, float) or date == '':
        date = "01/17/2020 02:00:00 PM"
    return datetime.strptime(date, "%m/%d/%Y %I:%M:%S %p")


# Milestone 2 ols model
def regression_milestone_2(pos, date):
    data_frame = pd.DataFrame(pos, columns=['Longitude', 'Latitude'])
    data_frame['Date'] = pd.Series(date)

    model = stats.ols(formula='Date ~ Longitude + Latitude', data=data_frame)

    preprocessed = model.fit()

    surfacex, surfacey, processed = generate_graph(data_frame, preprocessed)

    visualize(data_frame, surfacex, surfacey, processed, "Milestone 2: OLS")


# Milestone 3 - Failed model
# Continuation of using stats models ols from Milestone 2
# Hold out validation, since it is an easier implementation, showing the major concern with the
# data set and chosen training model.
def regression(X_train, X_test, y_train, y_test):
    data_frame = pd.DataFrame(y_train, columns=['Longitude', 'Latitude'])
    data_frame['Date'] = pd.Series(X_train)

    # Generate testing data frame
    data_frame_pre = pd.DataFrame(y_test, columns=['Longitude', 'Latitude'])
    data_frame_pre['Date'] = pd.Series(X_test)

    model = stats.ols(formula='Date ~ Longitude + Latitude', data=data_frame)

    preprocessed = model.fit()
    print(preprocessed.summary())

    predictions = preprocessed.predict(data_frame_pre)
    
    print(predictions)

    surfacex, surfacey, processed = generate_graph(data_frame, preprocessed)

    visualize(data_frame, surfacex, surfacey, processed, "Milestone 3: OLS")


# Milestone 3
# Attempting multiple output regression with hold-out validation
def multi_output_regression(X_train, X_test, y_train, y_test):
    lr = LinearRegression()
    X_train = np.reshape(X_train, (-1, 1))
    X_test = np.reshape(X_test, (-1,1))

    model = lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)

    # Notice how the output is in the right dimensions now
    # Since the output dimension is clarified
    print("Sample output: ", predictions[0])

    # Terrible accuracy
    print(model.score(X_test, y_test))

    # Visualizes model
    vis(X_test, y_test, predictions, "Milestone 3: Multi-Output Regression w/ Hold-out Validation")
    # plt.show()


# Milestone 3
# Attempting multiple output regression with k-fold cross validation
def k_fold_multi_output_regression(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    lr = LinearRegression()
    X = np.reshape(X, (-1, 1))
    kf = KFold(n_splits=4)
    i = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = lr.fit(X_train, y_train)
        print(model.score(X_test, y_test))
        predictions = lr.predict(X_test)
        vis(X_test, y_test, predictions, "Milestone 3: Multi-Output Regression w/ K-Fold Cross Validation, Split " + str(i))
        i = i+1


if __name__ == "__main__":
    main()
