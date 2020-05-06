#!/usr/bin/env python
# coding: utf-8

# In[52]:


#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import datetime
import time
from matplotlib import rcParams
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
import tensorflow as tf
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout
import xgboost as xgb

# to install sklearn: pip install -U scikit-learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import copy


# In[53]:


all_stocks = pd.read_csv("./data/all_stocks_2006-01-01_to_2018-01-01.csv",parse_dates = ["Date"])


# In[54]:


# all_stocks.head()
all_stocks1 = copy.deepcopy(all_stocks.head())

def get_head1():
    return all_stocks1.head().to_html()


all_stocks2 = copy.deepcopy(all_stocks.describe())

def get_desc1():
    return all_stocks2.describe().to_html()


# #Data Cleaning:
# 1.Outlier Check
# 2.Fill missing values

# In[55]:


sns.boxplot(y="Open",data=all_stocks)
plt.title("Open Price")
plt.show()


# In[56]:


sns.boxplot(y="Close",data=all_stocks)
plt.title("Close Price")
plt.show()


# In[57]:


sns.boxplot(y="Low",data=all_stocks)
plt.title("Low Price")
plt.show()


# In[58]:


sns.boxplot(y="High",data=all_stocks)
plt.title("High Price")
plt.show()


# In[59]:


for i in range(0,100,10):
    var = all_stocks['Open'].values
    var = np.sort(var,axis=None)
    print("{} percentile value is {}".format(i,var[int(len(var)*float(i)/100)]))
print("100 percentile values is ",var[-1])


# In[60]:


for i in range(90,100,1):
    var = all_stocks['Open'].values
    var = np.sort(var,axis=None)
    print("{} percentile value is {}".format(i,var[int(len(var)*float(i)/100)]))
print("100 percentile values is ",var[-1])


# In[61]:


all_stocks[all_stocks['Open']>1200]


# In[62]:


for i in range(0,100,10):
    var = all_stocks['Close'].values
    var = np.sort(var,axis=None)
    print("{} percentile value is {}".format(i,var[int(len(var)*float(i)/100)]))
print("100 percentile values is ",var[-1])


# In[63]:


for i in range(0,100,10):
    var = all_stocks['High'].values
    var = np.sort(var,axis=None)
    print("{} percentile value is {}".format(i,var[int(len(var)*float(i)/100)]))
print("100 percentile values is ",var[-1])


# In[64]:


for i in range(0,100,10):
    var = all_stocks['Low'].values
    var = np.sort(var,axis=None)
    print("{} percentile value is {}".format(i,var[int(len(var)*float(i)/100)]))
print("100 percentile values is ",var[-1])


# In[65]:


all_stocks.isnull().sum()


# In[66]:


all_stocks[all_stocks.Open.isnull()]

all_stocks3 = copy.deepcopy(all_stocks[all_stocks.Open.isnull()])

def get_null1():
    return all_stocks3[all_stocks3.Open.isnull()].to_html()


# In[68]:


rng = pd.date_range(start='2006-01-01', end='2018-01-01', freq='B')
rng[~rng.isin(all_stocks.Date.unique())]


# In[69]:


all_stocks.groupby('Name').count().sort_values('Date', ascending=False)['Date']


# In[70]:


gdf = all_stocks[all_stocks.Name == 'AABA']
cdf = all_stocks[all_stocks.Name == 'CAT']
cdf[~cdf.Date.isin(gdf.Date)]


# In[71]:


# Total number of companies
all_stocks.Name.unique().size


# In[72]:


all_stocks.groupby('Date').Name.unique().apply(len)


# In[73]:


all_stocks.set_index('Date', inplace=True)

#Backfill `Open` column
values = np.where(all_stocks['2017-07-31']['Open'].isnull(), all_stocks['2017-07-28']['Open'], all_stocks['2017-07-31']['Open'])
all_stocks['2017-07-31']= all_stocks['2017-07-31'].assign(Open=values.tolist())

values = np.where(all_stocks['2017-07-31']['Close'].isnull(), all_stocks['2017-07-28']['Close'], all_stocks['2017-07-31']['Close'])
all_stocks['2017-07-31']= all_stocks['2017-07-31'].assign(Close=values.tolist())

values = np.where(all_stocks['2017-07-31']['High'].isnull(), all_stocks['2017-07-28']['High'], all_stocks['2017-07-31']['High'])
all_stocks['2017-07-31']= all_stocks['2017-07-31'].assign(High=values.tolist())

values = np.where(all_stocks['2017-07-31']['Low'].isnull(), all_stocks['2017-07-28']['Low'], all_stocks['2017-07-31']['Low'])
all_stocks['2017-07-31']= all_stocks['2017-07-31'].assign(Low=values.tolist())

all_stocks.reset_index(inplace=True)
all_stocks[all_stocks.Date == '2017-07-31']
all_stocks4 = copy.deepcopy(all_stocks[all_stocks.Date == '2017-07-31'])

def get_null2():
    return all_stocks4[all_stocks4.Date == '2017-07-31'].to_html()

# In[75]:


missing_data_stocks = ['CSCO','AMZN','INTC','AAPL','MSFT','MRK','GOOGL', 'AABA']
columns = all_stocks.columns.values


# In[78]:


for stock in missing_data_stocks:
    tdf = all_stocks[(all_stocks.Name == stock) & (all_stocks.Date == '2014-03-28')].copy()
    tdf.Date = '2014-04-01'
    pd.concat([all_stocks, tdf])
print("Complete")


# In[80]:


all_stocks[(all_stocks.Name == 'CSCO') & (all_stocks.Date == '2014-04-01')]


# In[81]:


all_stocks[all_stocks.Open.isnull()]


# In[82]:


all_stocks = all_stocks[~((all_stocks.Date == '2012-08-01') & (all_stocks.Name == 'DIS'))]


# In[84]:


all_stocks.isnull().sum()


# In[85]:


#We predict the average value of all the 4 prices present in the data, so we create a new column called "avgPrice"
all_stocks["avgPrice"] = (all_stocks['High']+all_stocks['Low']+all_stocks['Open']+all_stocks['Close'])/4


# In[86]:


head2 = copy.deepcopy(all_stocks.head())
def get_head2():
    return head2.to_html()


# In[87]:


stock_names = all_stocks.Name.unique()
day_prices = all_stocks[all_stocks.Date == all_stocks.Date.min()].avgPrice
price_mapping = {n : c for n, c in zip(stock_names, day_prices)}
base_mapping = np.array(list(map(lambda x : price_mapping[x], all_stocks['Name'].values)))
all_stocks['Growth'] = all_stocks['avgPrice'] / base_mapping - 1
all_stocks.Growth.describe()


# In[88]:


sample_dates = pd.date_range(start='2006-01-01', end='2018-01-01', freq='B')
year_end_dates = sample_dates[sample_dates.is_year_end]
year_end_dates
worst_stocks = all_stocks[all_stocks.Date == all_stocks.Date.max()].sort_values('Growth').head(5)
best_stocks = all_stocks[all_stocks.Date == all_stocks.Date.max()].sort_values('Growth', ascending=False).head(5)
ws = worst_stocks.Name.values
bs = best_stocks.Name.values
tdf = all_stocks.copy()
tdf = all_stocks.set_index('Date')
tdf[tdf.Name.isin(ws)].groupby('Name').Growth.plot(title='Historical trend of worst 5 stocks of 2017', legend=True)
worst_stocks
def get_worst():
    return worst_stocks.to_html()

# In[89]:


tdf[tdf.Name.isin(bs)].groupby('Name').Growth.plot(title='Historical trend of best 5 stocks of 2017', legend=True)
best_stocks
def get_best():
    return best_stocks.to_html()


# In[90]:


corr = all_stocks.pivot('Date', 'Name', 'Growth').corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr)


# In[91]:


#Reference: http://www.insightsbot.com/augmented-dickey-fuller-test-in-python/
#Performing Augmented Dickey Fuller test to see if the data is stationary
class StationarityTests:
    def __init__(self, significance=.05):
        self.SignificanceLevel = significance
        self.pValue = None
        self.isStationary = None

    def ADF_Stationarity_Test(self, timeseries, printResults = True):
        #Dickey-Fuller test:
        adfTest = adfuller(timeseries, autolag='AIC')

        self.pValue = adfTest[1]

        if (self.pValue<self.SignificanceLevel):
            self.isStationary = True
        else:
            self.isStationary = False

        if printResults:
            dfResults = pd.Series(adfTest[0:4], index=['ADF Test Statistic','P-Value','# Lags Used','# Observations Used'])
            #Add Critical Values
            for key,value in adfTest[4].items():
                dfResults['Critical Value (%s)'%key] = value
            print('Augmented Dickey-Fuller Test Results:')
            print(dfResults)


# #divide the data into train and test data based on time
# #data between the years 2006 to 2016 is taken as train data and the data from 2017 to 2018 is taken as test data

# In[92]:


#we predict the avgPrice for "Amazon"
all_stocks_amazon = all_stocks[all_stocks['Name']=='AMZN']


# In[93]:


mask = (all_stocks_amazon['Date'] >= '2006-01-01') & (all_stocks_amazon['Date']<'2017-01-01')
all_stocks_amazon_train_arima = all_stocks_amazon[mask]
all_stocks_amazon_test_arima = all_stocks_amazon.loc[~mask]
print("Percentage of train data: {}".format((len(all_stocks_amazon_train_arima)/len(all_stocks_amazon))*100))
print("Percentage of test data: {}".format((len(all_stocks_amazon_test_arima)/len(all_stocks_amazon))*100))
plt.title("Amazon Price data")
plt.plot(all_stocks_amazon["avgPrice"])


# In[94]:


#check if the amazon data is stationary
#From the results it is found that data is not stationary
sTest = StationarityTests()
sTest.ADF_Stationarity_Test(all_stocks_amazon["avgPrice"], printResults = True)
print("Is the time series stationary? {0}".format(sTest.isStationary))


# In[95]:


plot_acf(all_stocks_amazon["avgPrice"])


# In[96]:


plot_pacf(all_stocks_amazon["avgPrice"])
plt.show()


# In[97]:


#index the data with date to perform arima model
all_stocks_amazon_train_arima = all_stocks_amazon_train_arima.set_index('Date')
all_stocks_amazon_test_arima = all_stocks_amazon_test_arima.set_index('Date')
all_stocks_amazon_train_arima.index = pd.DatetimeIndex(all_stocks_amazon_train_arima.index).to_period('D')
all_stocks_amazon_test_arima.index = pd.DatetimeIndex(all_stocks_amazon_test_arima.index).to_period('D')


# In[98]:


#Fit the arima model on amazon data
model = ARIMA(all_stocks_amazon_train_arima['avgPrice'],order=(2,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())
#model_fit.plot_predict(start = 2517,end = 3020)
#plt.show()

def get_result1():
    return model_fit.summary().to_html()


# In[99]:


residuals = DataFrame(model_fit.resid)
#print(residuals)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())


# In[100]:


#forecast the data in test set
fc, se, conf = model_fit.forecast(251, alpha=0.05)
fc_series = pd.Series(fc, index=all_stocks_amazon_test_arima.index)
lower_series = pd.Series(conf[:, 0], index=all_stocks_amazon_test_arima.index)
upper_series = pd.Series(conf[:, 1], index=all_stocks_amazon_test_arima.index)


# In[101]:


mape_arima = (mean_absolute_error(all_stocks_amazon_test_arima.avgPrice, fc_series))/(sum(all_stocks_amazon_test_arima.avgPrice)/len(all_stocks_amazon_test_arima.avgPrice))*100
rmse_arima = np.sqrt(np.sum((fc_series-all_stocks_amazon_test_arima.avgPrice)**2))
print("Mean absolute percentage error for Arima model is:",round(mape_arima,2))
#print("RMSE for Arima model is:",round(rmse_arima,2))

def get_error1():
    return str(round(mape_arima,2))
# In[102]:


fp = fc_series
ap = all_stocks_amazon_test_arima.avgPrice
pdf = pd.DataFrame(data={'Actual': ap, 'Predicted': fp})
ax = pdf.plot()


# LSTM model to predict stock price for amazon data

# In[103]:


#create test and train data for LSTM
amazon_data_lstm = all_stocks_amazon[['Date','avgPrice']]
training_set_amazon_lstm = amazon_data_lstm[amazon_data_lstm.Date.dt.year != 2017].avgPrice.values
test_set_amazon_lstm = amazon_data_lstm[amazon_data_lstm.Date.dt.year == 2017].avgPrice.values
print("Training set size: ",training_set_amazon_lstm.size)
print("Test set size: ", test_set_amazon_lstm.size)


# In[104]:


#scale the data to perform LSTM model
scaler = MinMaxScaler()
training_set_amazon_lstm_scaled = scaler.fit_transform(training_set_amazon_lstm.reshape(-1, 1))


# In[105]:


#functions to create training and test data to build LSTM model
##Previous 30 days avgPrice values are taken as features to predict the avgPrice value of a given day
def create_train_data(training_set_amazon_lstm_scaled):
    X_train, y_train = [], []
    for i in range(30, training_set_amazon_lstm_scaled.size):
        X_train.append(training_set_amazon_lstm_scaled[i-30: i])
        y_train.append(training_set_amazon_lstm_scaled[i])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train, y_train

def create_test_data():
    X_test,y_test = [],[]
    inputs = amazon_data_lstm[len(amazon_data_lstm) - len(test_set_amazon_lstm) - 30:].avgPrice.values
    inputs = scaler.transform(inputs.reshape(-1, 1))
    for i in range(30, test_set_amazon_lstm.size+30):
        X_test.append(inputs[i - 30: i, 0])
        y_test.append(inputs[i,0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test,y_test


# In[106]:


#create train and test data
X_train_amazon_lstm, y_train_amazon_lstm = create_train_data(training_set_amazon_lstm_scaled)
X_test_amazon_lstm,y_test_amazon_lstm = create_test_data()


# In[107]:


print(X_train_amazon_lstm.shape)
print(y_train_amazon_lstm.shape)


# In[108]:


def create_simple_model():
    model = Sequential()
    model.add(LSTM(units = 10, return_sequences = False, input_shape = (X_train_amazon_lstm.shape[1], 1)))
    #model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    return model

def compile_and_run(model, epochs=50, batch_size=64):
    model.compile(metrics=['accuracy'], optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train_amazon_lstm, y_train_amazon_lstm, epochs=epochs, batch_size=batch_size, verbose=3)
    return history

def plot_metrics(history):
    metrics_df = pd.DataFrame(data={"loss": history.history['loss']})
    metrics_df.plot()

def make_predictions(X_test_amazon_lstm, model):
    y_pred = model.predict(X_test_amazon_lstm)
    final_predictions = scaler.inverse_transform(y_pred)
    fp = np.ndarray.flatten(final_predictions)
    ap = np.ndarray.flatten(test_set_amazon_lstm)
    pdf = pd.DataFrame(data={'Actual': ap, 'Predicted': fp})
    ax = pdf.plot()
    return final_predictions


# In[109]:


simple_model = create_simple_model()
history = compile_and_run(simple_model, epochs=20)


# In[110]:


plot_metrics(history)


# In[111]:


simple_model.summary()

def get_result2():
    return simple_model.summary().to_html()


# In[112]:


y_pred = make_predictions(X_test_amazon_lstm, simple_model)


# In[113]:


mape_lstm_simple = (mean_absolute_error(test_set_amazon_lstm, y_pred))/(sum(test_set_amazon_lstm)/len(test_set_amazon_lstm))*100
rmse_lstm_simple = np.sqrt(np.sum((test_set_amazon_lstm-y_pred)**2))
print("Mean absolute percentage error for Simple LSTM model is:",round(mape_lstm_simple,2))
#print("RMSE for Simple LSTM model is:",round(rmse_lstm_simple,2))

def get_error2():
    return str(round(mape_lstm_simple,2))


# In[114]:


#Build a Deep NN
from keras.layers.normalization import BatchNormalization
def create_dl_model():
    model = Sequential()

    #Adding the LSTM layers
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train_amazon_lstm.shape[1], 1)))
    #model.add(Dropout(0.2))
    #model.add(BatchNormalization())
    model.add(LSTM(units = 50, return_sequences = True))
    #model.add(Dropout(0.2))
    #model.add(BatchNormalization())
    model.add(LSTM(units = 50, return_sequences = True))
    #model.add(Dropout(0.2))
    #model.add(BatchNormalization())
    model.add(LSTM(units = 50))
    #model.add(Dropout(0.2))
    #model.add(BatchNormalization())

    # Adding the output layer
    model.add(Dense(units = 1))
    return model

dl_model = create_dl_model()
dl_model.summary()
history = compile_and_run(dl_model, epochs=20)
plot_metrics(history)

def get_result3():
    return dl_model.summary().to_html()


# In[115]:


y_pred_dnn = make_predictions(X_test_amazon_lstm, dl_model)


# In[116]:


mape_lstm_dnn = (mean_absolute_error(test_set_amazon_lstm, y_pred_dnn))/(sum(test_set_amazon_lstm)/len(test_set_amazon_lstm))*100
rmse_lstm_dnn = np.sqrt(np.sum((test_set_amazon_lstm-y_pred_dnn)**2))
print("Mean absolute percentage error for Deep Neural network model is:",round(mape_lstm_dnn,2))
# print("RMSE for Deep Neural network model is:",round(rmse_lstm_dnn,2))
def get_error3():
    return str(round(mape_lstm_dnn,2))

# In[117]:


from keras.layers.normalization import BatchNormalization
def create_dl_model():
    model = Sequential()

    #Adding the LSTM layers
    model.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train_amazon_lstm.shape[1], 1)))
    #model.add(Dropout(0.2))
    #model.add(BatchNormalization())
    model.add(LSTM(units = 100, return_sequences = True))
    #model.add(Dropout(0.2))
    #model.add(BatchNormalization())
    model.add(LSTM(units = 100, return_sequences = True))
    #model.add(Dropout(0.2))
    #model.add(BatchNormalization())
    model.add(LSTM(units = 100))
    #model.add(Dropout(0.2))
    #model.add(BatchNormalization())

    # Adding the output layer
    model.add(Dense(units = 1))
    return model

dl_model = create_dl_model()
dl_model.summary()
history = compile_and_run(dl_model, epochs=20)
plot_metrics(history)


# In[118]:


y_pred_dnn_100 = make_predictions(X_test_amazon_lstm, dl_model)


# In[119]:


mape_lstm_dnn_100 = (mean_absolute_error(test_set_amazon_lstm, y_pred_dnn_100))/(sum(test_set_amazon_lstm)/len(test_set_amazon_lstm))*100
rmse_lstm_dnn = np.sqrt(np.sum((test_set_amazon_lstm-y_pred_dnn)**2))
print("Mean absolute percentage error for Deep Neural network model is:",round(mape_lstm_dnn_100,2))
print#("RMSE for Deep Neural network model is:",round(rmse_lstm_dnn,2))


# In[120]:


#Create train and test data to perfrom Linear Regression, Random Forest regression and XGBoost Regression
#Previous 30 days avgPrice values are taken as features to predict the avgPrice value of a given day
def create_train_data(training_set_scaled):
    X_train, y_train = [], []
    for i in range(30, training_set_scaled.size):
        X_train.append(training_set_scaled[i-30: i])
        y_train.append(training_set_scaled[i])
    # Converting list to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train, y_train

def create_test_data():
    X_test = []
    inputs = amazon_data_lstm[len(amazon_data_lstm) - len(test_set_amazon_lstm) - 30:].avgPrice.values
    inputs = inputs.reshape(-1, 1)
    for i in range(30, test_set_amazon_lstm.size+30): # Range of the number of values in the training dataset
        X_test.append(inputs[i - 30: i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    return X_test


# In[121]:


#linear regression on amazon data
X_train_lin,y_train_lin = create_train_data(training_set_amazon_lstm)
X_test_lin = create_test_data()
#print(X_train_lin.shape)

from sklearn.linear_model import LinearRegression
lr_reg=LinearRegression().fit(X_train_lin, y_train_lin)

y_pred = lr_reg.predict(X_test_lin)
lr_test_predictions = [value for value in y_pred]
y_pred = lr_reg.predict(X_train_lin)
lr_train_predictions = [value for value in y_pred]

fp = lr_test_predictions
ap = np.ndarray.flatten(test_set_amazon_lstm)
pdf = pd.DataFrame(data={'Actual': ap, 'Predicted': fp})
ax = pdf.plot()


# In[122]:


mape_lr = (mean_absolute_error(test_set_amazon_lstm, lr_test_predictions))/(sum(test_set_amazon_lstm)/len(test_set_amazon_lstm))*100
rmse_lr = np.sqrt(np.sum((test_set_amazon_lstm-lr_test_predictions)**2))
print("Mean absolute percentage error for linear regression model is:",round(mape_lr,2))
#print("RMSE for linear regression model is:",round(rmse_lr,2))

def get_error4():
    return str(round(mape_lr,2))

# In[123]:


#Random Forest Regression on amazon data
regr1 = RandomForestRegressor(max_features='sqrt',min_samples_leaf=4,min_samples_split=3,n_estimators=40)
regr1.fit(X_train_lin, y_train_lin)

y_pred = regr1.predict(X_test_lin)
rndf_test_predictions = [value for value in y_pred]
y_pred = regr1.predict(X_train_lin)
rndf_train_predictions = [value for value in y_pred]

fp = rndf_test_predictions
ap = np.ndarray.flatten(test_set_amazon_lstm)
pdf = pd.DataFrame(data={'Actual': ap, 'Predicted': fp})
ax = pdf.plot()


# In[124]:


mape_rndf = (mean_absolute_error(test_set_amazon_lstm, rndf_test_predictions))/(sum(test_set_amazon_lstm)/len(test_set_amazon_lstm))*100
rmse_rndf = np.sqrt(np.sum((test_set_amazon_lstm-rndf_test_predictions)**2))
print("Mean absolute percentage error for Random Forests is:",round(mape_rndf,2))
#print("RMSE for Random Forests is:",round(rmse_rndf,2))

def get_error5():
    return str(round(mape_rndf,2))
# In[125]:


#XGBoost Regression on amazon data
x_model = xgb.XGBRegressor(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=3,
 min_child_weight=3,
 gamma=0,
 subsample=0.8,
 reg_alpha=200, reg_lambda=200,
 colsample_bytree=0.8,nthread=4)
x_model.fit(X_train_lin, y_train_lin)

y_pred = x_model.predict(X_test_lin)
xgb_test_predictions = [value for value in y_pred]
y_pred = x_model.predict(X_train_lin)
xgb_train_predictions = [value for value in y_pred]


# In[126]:


fp = xgb_test_predictions
ap = np.ndarray.flatten(test_set_amazon_lstm)
pdf = pd.DataFrame(data={'Actual': ap, 'Predicted': fp})
ax = pdf.plot()


# In[127]:


mape_xgb = (mean_absolute_error(test_set_amazon_lstm, xgb_test_predictions))/(sum(test_set_amazon_lstm)/len(test_set_amazon_lstm))*100
rmse_xgb = np.sqrt(np.sum((abs(test_set_amazon_lstm-xgb_test_predictions)**2)))
print("Mean absolute percentage error for XGBoost regression is:",round(mape_xgb,2))
#print("RMSE for XGBoost regression is:",round(rmse_xgb,2))

def get_error6():
    return str(round(mape_xgb,2))


# In[ ]:
