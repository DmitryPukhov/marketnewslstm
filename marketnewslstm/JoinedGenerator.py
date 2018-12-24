import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class JoinedGenerator:
    """
    Keras standard approach to generage batches for model.fit_generator() call.
    """

    def __init__(self, prepro, idx, market, news):
        """
        @param preprocessor: market and news join preprocessor
        @param market: full loaded market df
        @param news: full loaded news df
        @param index_df: df with assetCode and time of train or validation market data. Batches will be taken from them.
        """
        self.market = market
        self.prepro = prepro
        self.news = news
        self.idx = idx

    def flow_lstm(self, batch_size, is_train, look_back, look_back_step):
        """
        Generate batch data for LSTM NN
        Each cycle in a loop we yield a batch for one training step in epoch.
        """
        while True:
            # Get market indices of random assets, sorted by assetCode, time.
            batch_idx = self.get_random_assets_idx(batch_size)

            # Get X, y data for this batch, containing market and news, but without look back yet
            X, y = self.prepro.get_Xy(batch_idx, self.market, self.news, is_train)
            # Add look back data to X, y
            X = self.prepro.with_look_back(X, look_back, look_back_step)
            yield X, y

    def get_random_assets_idx(self, batch_size):
        """
        Get random asset and it's last market data indices.
        Repeat for next asset until we reach batch_size.
        """
        asset_codes = self.idx['assetCode'].unique().tolist()

        # Insert first asset
        asset = np.random.choice(asset_codes)
        asset_codes.remove(asset)
        #asset = 'ADBE.O'
        batch_index_df = self.idx[self.idx.assetCode == asset].tail(batch_size)

        return batch_index_df.sort_values(by=['assetCode', 'time'])

