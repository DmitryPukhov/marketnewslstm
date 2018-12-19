import pandas as pd
from sklearn.model_selection import train_test_split


class TrainValTestSplit:

    @staticmethod
    def train_val_test_split(market, size):
        """
        Get train, validation, test sample indices - time, assetCode, market index in original market df
        @return: train, validation, test df.  Columns - time, assetCode, market_index, news_index
        """
        market_idx = market[['assetCode', 'time']]
        start_date = pd.datetime(2000, 1, 1).date()
        market_idx = market_idx.loc[market_idx.time >= start_date] \
            .sort_values(by=['time', 'assetCode']) \
            .tail(size).copy()

        # Split to train, validation and test
        train_idx, test_idx = train_test_split(market_idx, shuffle=False, random_state=24)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.1, shuffle=False, random_state=24)

        return train_idx, val_idx, test_idx
