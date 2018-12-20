import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class JoinedPreprocessor:
    """
    Join market with news and preprocess
    """

    def __init__(self, market_prepro, news_prepro):
        self.market_prepro = market_prepro
        self.news_prepro = news_prepro

    def get_X(self, market, news):
        """
        Returns preprocessed market + news
        :return: X
        """
        # Market row
        market_X = self.market_prepro.get_X(market)
        # One row in news contains many asset codes. Extend it to news_X with one asset code - one row
        news_idx = self.news_prepro.news_idx(news)
        news_X = self.news_prepro.get_X(news_idx, news)
        news_X.time = news_X.time.astype('datetime64')
        # X = market X + news X
        X = market_X.merge(news_X, how='left', on=['time', 'assetCode'], left_index=True)
        X = X.fillna(0)
        X = X[self.market_prepro.feature_cols + self.news_prepro.feature_cols]
        return X

    def get_Xy(self, idx, market, news, is_train=False, is_raw_y=False):
        """
        Returns preprocessed features and labels for given indices
        """
        # Get market data for index
        market_df = market.loc[idx.index]
        # We can remove bad data in train
        if is_train:
            market_df = self.market_prepro.fix_train(market_df)
        market_Xy = self.market_prepro.get_X(market_df)
        # Get news data for index
        news_X = self.news_prepro.get_X(idx, news)
        #news_X.time = pd.to_datetime(news_X.time, utc=True)
        news_X.time = news_X.time.astype('datetime64')
        # Merge and return
        Xy = market_Xy.merge(news_X, how='left', on=['time', 'assetCode'], left_index=True)
        Xy = Xy.fillna(0)
        X = Xy[self.market_prepro.feature_cols + self.news_prepro.feature_cols]
        y = self.market_prepro.get_y(market_df, is_raw_y)

        return X, y

    def with_look_back(self, X, y, look_back, look_back_step):
        """
        Add look back window values to prepare dataset for LSTM
        """
        look_back_fixed = look_back_step * (look_back // look_back_step)
        # Fill look_back rows before first
        first_xrow = X.values[0]
        first_xrow.shape = [1, X.values.shape[1]]
        first_xrows = np.repeat(first_xrow, look_back_fixed, axis=0)
        X_values = np.append(first_xrows, X.values, axis=0)

        if y is not None:
            first_yrow = y.values[0]
            first_yrow.shape = [1, y.values.shape[1]]
            first_yrows = np.repeat(first_yrow, look_back_fixed, axis=0)
            y_values = np.append(first_yrows, y.values, axis=0)

        # for i in range(0, len(X) - look_back + 1):
        X_processed = []
        y_processed = []
        for i in range(look_back_fixed , len(X_values)):
            # Add lookback to X
            x_window = X_values[i - (look_back_fixed//look_back_step)*look_back_step:i+1:look_back_step, :]
            X_processed.append(x_window)
            # If input is X only, we'll not output y
            if y is None:
                continue
            # Add lookback to y
            y_window = y_values[i - (look_back_fixed//look_back_step)*look_back_step:i+1:look_back_step, :]
            y_processed.append(y_window)
        # Return Xy for train/test or X for prediction
        if y is not None:
            #return np.array(X_processed), np.array(y_processed)
            return np.array(X_processed), y.values
        else:
            return np.array(X_processed)

# #Market and news preprocessor instance
# prepro = JoinedPreprocessor(market_prepro, news_prepro)
# print('Preprocessor created, but not fit yet')
