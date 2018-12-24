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

    def with_look_back_Xy(self, X, y, look_back, look_back_step):
        """
        Add look back window values to X, y
        """
        X_processed = self.with_look_back(X, look_back , look_back_step)
        y_processed = self.with_look_back(y, look_back, look_back_step)
        return X_processed, y_processed

    def with_look_back(self, df, look_back, look_back_step, start_pos=0):
        """
        Add look back window values to X or y to prepare dataset for LSTM
        """
        # look_back should be multiple of look_back_step. Fix if not
        look_back_fixed = look_back_step * (look_back // look_back_step)

        # If not enough rows before start_pos for look back, fill them with repeatable start row.
        if start_pos < look_back_fixed:
            first_rows_num = look_back_fixed - start_pos
            # Fill look_back rows before first
            first_row = df.values[0]
            first_row.shape = [1, df.values.shape[1]]
            first_rows = np.repeat(first_row, first_rows_num, axis=0)
            # After inserting rows, start pos should point to the same row
            start_pos_fixed = look_back_fixed
            values = np.append(first_rows, df.values, axis=0)
        else:
            # Ok, enough data before start pos for look back
            values = df.values
            start_pos_fixed = start_pos

        processed = []

        for i in range(start_pos_fixed, len(values)):
            # Add lookback to X
            window = values[i - (look_back_fixed // look_back_step) * look_back_step:i + 1:look_back_step, :]
            processed.append(window)

        # Return Xy for train/test or X for prediction
        return np.array(processed)


# #Market and news preprocessor instance
# prepro = JoinedPreprocessor(market_prepro, news_prepro)
# print('Preprocessor created, but not fit yet')
