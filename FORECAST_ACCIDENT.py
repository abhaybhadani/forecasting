#!/usr/bin/env python
# coding: utf-8

# Download the “Monatszahlen Verkehrsunfälle” Dataset from the München Open Data Portal. Here you see the number of accidents for specific categories per month. Important are the first 5 columns:
# 
# 1. Category
# 2. Accident-type (insgesamt means total for all subcategories)
# 3. Year
# 4. Month
# 5. Value
# 
# 
# Your goal would be to visualise historically the number of accidents per category (column1). 
# 
# The dataset currently contains values until the end of 2020. 
# 
# Create an application that forecasts the values for:
# 
# Category: 'Alkoholunfälle'
# 
# Type: 'insgesamt
# 
# Year: '2021'
# 
# Month: '01'

# In[1]:

import matplotlib.pyplot as plt  # plots
import numpy as np  # vectors and matrices
import pandas as pd  # tables and data manipulations
import datetime as dt

import warnings  # `do not disturbe` mode
from itertools import product  # some useful functions

import statsmodels.api as sm
import statsmodels.tsa.api as smt
from tqdm import tqdm_notebook
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
from flask import Flask, jsonify, request
import json


warnings.filterwarnings("ignore")

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'


# In[2]:


# sns.set()


# In[3]:


pd.options.display.max_rows = 500


# In[4]:


df = pd.read_csv('data/dataset1.csv')


# In[5]:


df.rename(columns={'MONATSZAHL': 'Category', 'AUSPRAEGUNG': 'Accident_Type', 'JAHR':'YEAR', 'MONAT':'MONTH',
                   'WERT':'VALUE'}, inplace=True)
df=df[['Category','Accident_Type','YEAR','MONTH','VALUE']]


# In[6]:


df.shape


# In[7]:


df.Category.unique()


# In[8]:


df.Accident_Type.unique()


# In[9]:


df_1 = df[(df.Category=='Alkoholunfälle')&(df.Accident_Type=='insgesamt')&(df.MONTH != 'Summe')].copy()
df_1.shape


# In[10]:


df_1.head()


# In[11]:


df_1.tail()


# In[12]:



df_1["MONTH"] = pd.to_numeric(df_1["MONTH"])
df_1['DT'] = pd.to_datetime(df_1["MONTH"], format='%Y%m')
df_1 = df_1.sort_values('DT')


# In[13]:


df_1.tail()


# In[14]:


sdate = dt.date(df_1.DT.min().year,df_1.DT.min().month,df_1.DT.min().day) 
edate = dt.date(df_1.DT.max().year,df_1.DT.max().month,df_1.DT.max().day) + pd.DateOffset(months=1)


# In[15]:


sdate


# In[16]:


edate


# In[17]:


pd.date_range(sdate, edate, freq='m')


# In[18]:


df_1['DT'] = pd.date_range(sdate, edate, freq='m')


# In[19]:


df_1.set_index('DT', inplace=True)


# In[20]:


accident_values = df_1[(df_1.MONTH <= 202012)].VALUE

accident_values.plot(figsize=(20,5))


# In[21]:


# p = plt.plot(accident_values.index,accident_values.values)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:


def tsplot(y, lags=None, figsize=(12, 7), style="bmh"):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)   # Plot the accident values
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title(
            "Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}".format(p_value)
        )
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax) # Plot the autocorrelation
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax) # Plot the partial autocorrelation
        plt.tight_layout()


# In[23]:


df_1[(df_1.MONTH <= 202012)].VALUE


# In[24]:


tsplot(df_1[(df_1.MONTH <= 202012)].VALUE, lags=60)


# In[ ]:





# In[25]:


# Importing everything from above

from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_squared_log_error, median_absolute_error,
                             r2_score)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[26]:


ads_diff = df_1[(df_1.MONTH <= 202012)].VALUE - df_1[(df_1.MONTH <= 202012)].VALUE.shift(6)

ads_diff.head(10)


# In[27]:


tsplot(ads_diff[6:], lags=60)


# In[ ]:





# In[28]:


# setting initial values and some bounds for them
ps = range(2, 5)
d = 1
qs = range(2, 5)
Ps = range(0, 2)
D = 1
Qs = range(0, 2)
s = 12 

# creating list with all the possible combinations of parameters
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)


# In[29]:


parameters_list


# In[30]:


def optimizeSARIMA(parameters_list, d, D, s):
    """
        Return dataframe with parameters and corresponding AIC
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order 
        s - length of season
    """

    results = []
    best_aic = float("inf")

    for param in tqdm_notebook(parameters_list):
        # we need try-except because on some combinations model fails to converge
        try:
            model = SARIMAX(
                df_1[(df_1.MONTH <= 202012)].VALUE,
                order=(param[0], d, param[1]),
                seasonal_order=(param[2], D, param[3], s),
            ).fit(disp=-1)
        except:
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ["parameters", "aic"]
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by="aic", ascending=True).reset_index(drop=True)

    return result_table


# In[31]:


TRAIN=False


# In[32]:


# %%time
model_filename="SARIMA_best_model.pkl"

if TRAIN==True:
    result_table = optimizeSARIMA(parameters_list, d, D, s)
    result_table.head()
    p, q, P, Q = result_table.parameters[0]

    # set the parameters that give the lowest AIC

    best_model = sm.tsa.statespace.SARIMAX(df_1[(df_1.MONTH <= 202012)].VALUE, 
                                           order=(p, d, q), 
                                           seasonal_order=(P, D, Q, s)).fit(disp=-1)
    print(best_model.summary())
    pickle.dump(best_model, open(model_filename, 'wb'))
    
    
else:
    best_model = pickle.load(open(model_filename, 'rb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[33]:


tsplot(best_model.resid[24 + 1 :], lags=60)


# In[34]:


def plot_forecast(series, model, n_steps):
    """
        Plots model vs predicted values
        series - dataset with timeseries
        model - fitted SARIMA model
        n_steps - number of steps to predict in the future
        
    """
    # adding model values
    data = series.copy()
    data.columns = ["actual"]
    data["sarima_model"] = model.fittedvalues
    
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    data["sarima_model"][: s + 1] = np.NaN

    # forecasting on n_steps forward
    forecast = model.predict(start=data.shape[0], end=data.shape[0] + n_steps)
    forecast = data.sarima_model.append(forecast)
    
    forecast.index = pd.date_range(data.index.min(), data.index.max() + pd.DateOffset(months=n_steps+1), freq='m')
    
    plt.figure(figsize=(20, 8))
    plt.plot(forecast, color="r", label="model")
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color="lightgrey")
    plt.plot(data.actual, label="actual")
    plt.xlabel('YEAR')
    plt.ylabel('Number of Accidents')
    plt.title('Actual vs Forecasted :  Alkoholunfalle and insgesamt case')
    plt.legend()
    plt.grid(False)
    plt.savefig('images/alkoholunfalle_insgesamt_predicted.jpg')
    return (pd.DataFrame(forecast))


# In[35]:


df_forecast = plot_forecast(df_1[(df_1.MONTH <= 202012)][['VALUE']], best_model, 11)
df_forecast.columns=['pred_values']


# In[36]:


df_forecast.tail()


# In[37]:



df_forecast['year'] = df_forecast.index.year
df_forecast['month'] = df_forecast.index.month
df_forecast.reset_index(inplace=True)


# In[ ]:





# In[38]:


year=2021
month=10
values = df_forecast[(df_forecast.year==year) & (df_forecast.month==month)]['pred_values'].values[0]
print(values)


# In[ ]:





# In[ ]:





# In[39]:


## Make an API


# In[ ]:


from flask import Flask, jsonify, request

# creating a Flask app
app = Flask(__name__)



@app.route('/predict', methods = ['GET', 'POST']) 
def load_predict():
    if(request.method == 'POST'):
        year=int(request.args.get('year'))
        month=int(request.args.get('month'))
        print(year, month)
        if (year>=2000) & (year<=2021) & (month>=1) & (month <=12):
            print('Inside if')
            values = df_forecast[(df_forecast.year==year) & (df_forecast.month==month)]['pred_values'].values[0]
            return json.dumps({'prediction':values})
        else:
            return json.dumps({'prediction':'Values not in range'})
    

if __name__ == '__main__':
    app.run(debug=False, port=5000, host='0.0.0.0')





