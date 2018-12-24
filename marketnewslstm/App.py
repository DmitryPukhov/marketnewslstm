#####################################
# Libraries
#####################################

######################################
# Project modules
######################################
from marketnewslstm import MarketPrepro, NewsPrepro, JoinedPreprocessor, JoinedGenerator, ModelFactory, \
    TrainValTestSplit, Predictor

# Common libs
import pandas as pd
import numpy as np
import sys
import os
import os.path
import random
from pathlib import Path
from time import time
from itertools import chain

# Image processing
import imageio
import skimage
import skimage.io
import skimage.transform
# from skimage.transform import rescale, resize, downscale_local_mean

# Charts
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# ML
import scipy
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
# from xgboost import XGBClassifier
# from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, \
    RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import TimeseriesGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, LSTM, Embedding
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
import tensorflow

#####################################
# Settings
#####################################
plt.style.use('seaborn')
# Set random seed to make results reproducable
np.random.seed(42)
tensorflow.set_random_seed(42)
os.environ['PYTHONHASHSEED'] = '42'
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 200)

input_dir = '../input'
print(os.listdir("../input"))
market = pd.read_csv(input_dir + '/marketdata_sample.csv')
news = pd.read_csv(input_dir + '/news_sample.csv')

# !!! Hack
news.time = pd.to_datetime('2007-02-01 23:35')
# Restrict datetime to date
# news.time = pd.to_datetime(news.time.astype('datetime64').dt.date, utc=True)
news.time = news.time.astype('datetime64').dt.date
market.time = market.time.astype('datetime64').dt.date

# Split to train, validation and test
toy = True
if toy:
    sample_size = 10000
else:
    sample_size = 500000
train_idx, val_idx, test_idx = TrainValTestSplit.train_val_test_split(market, sample_size)

# Create preprocessors
market_prepro = MarketPrepro()
market_prepro.fit(train_idx, market)
news_prepro = NewsPrepro()
news_prepro.fit(train_idx, news)
prepro = JoinedPreprocessor(market_prepro, news_prepro)

# prediction_prepro = PredictionPreprocessor(prepro, market_prepro, news_prepro)
# x = prediction_prepro.get_X_with_lookback(market, news, 4,2)


# Train data generator instance
join_generator = JoinedGenerator(prepro, train_idx, market, news)
val_generator = JoinedGenerator(prepro, val_idx, market, news)
print('Generators created')

# Create and train model
model = ModelFactory.lstm_128(len(market_prepro.feature_cols) + len(news_prepro.feature_cols))
model.load_weights("best_weights.h5")
print(model.summary())
#ModelFactory.train(model, toy, join_generator, val_generator)

# Predict
predictor = Predictor( prepro, market_prepro, news_prepro, model, ModelFactory.look_back, ModelFactory.look_back_step)
y_pred, y_test = predictor.predict_idx(test_idx, market, news)

y_pred = predictor.predict(market, news)

plt.plot(y_pred)
plt.plot(y_test)
plt.legend("pred", "test")
plt.show()

# get_merged_Xy(train_idx.sample(5), market, pd.DataFrame([],columns=news.columns)).head()
print('The end')
