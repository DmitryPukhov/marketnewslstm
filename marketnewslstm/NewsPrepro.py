from itertools import chain
import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, \
    RobustScaler
import scipy

# Common libs
import pandas as pd
import numpy as np
from itertools import chain
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, \
    RobustScaler


class NewsPrepro:
    """
    Aggregate news by day and asset. Normalize numeric values.
    """
    news_cols_numeric = ['urgency', 'takeSequence', 'wordCount', 'sentenceCount', 'companyCount',
                         'marketCommentary', 'relevance', 'sentimentNegative', 'sentimentNeutral',
                         'sentimentPositive', 'sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H',
                         'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H',
                         'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D', 'volumeCounts7D']

    feature_cols = news_cols_numeric

    def fit(self, idx, news):
        """
        Fit preprocessing scalers, encoders on given train df.
        @param idx: index with time, assetCode
        """
        # Save indices[assetCode, time, news_index] for all news
        self.all_news_idx = self.news_idx(news)

        # Get news only related to market idx
        news_idx = idx.merge(self.all_news_idx, on=['assetCode', 'time'], suffixes=['_idx', ''])[
            ['news_index', 'assetCode', 'time']]
        news_train_df = news_idx.merge(news, left_on='news_index', right_index=True, suffixes=['_idx', ''])[
            self.news_cols_numeric]

        # Numeric data normalization
        self.numeric_scaler = StandardScaler()
        news_train_df.fillna(0, inplace=True)

        # Fit scaler
        self.numeric_scaler.fit(news_train_df)

    def get_X(self, idx, news):
        """
        Preprocess news for asset code and time from given index
        """
        news_idx = idx.merge(self.all_news_idx, on=['assetCode', 'time'], suffixes=['_idx', ''])[
            ['news_index', 'assetCode', 'time']]
        news_df = news_idx.merge(news, left_on='news_index', right_index=True, suffixes=['_idx', ''])[
            ['time', 'assetCode'] + self.news_cols_numeric]
        news_df = self.aggregate_news(news_df)

        return self.safe_fix(news_df)

    def safe_fix(self, news_df):
        """
        Scale, fillna
        """
        # Normalize, fillna etc without removing rows.
        news_df.fillna(0, inplace=True)
        if not news_df.empty:
            news_df[self.news_cols_numeric] = self.numeric_scaler.transform(news_df[self.news_cols_numeric])
        return news_df


    def news_idx(self, news):
        """
        Get asset code, time -> news id
        :param news:
        :return:
        """

        # Fix asset codes (str -> list)
        asset_codes_list = news['assetCodes'].str.findall(f"'([\w\./]+)'")

        # Expand assetCodes
        assetCodes_expanded = list(chain(*asset_codes_list))

        assetCodes_index = news.index.repeat(asset_codes_list.apply(len))
        assert len(assetCodes_index) == len(assetCodes_expanded)
        df_assetCodes = pd.DataFrame({'news_index': assetCodes_index, 'assetCode': assetCodes_expanded})

        # Create expanded news (will repeat every assetCodes' row)
        #        df_expanded = pd.merge(df_assetCodes, news, left_on='level_0', right_index=True)
        df_expanded = pd.merge(df_assetCodes, news[['time']], left_on='news_index', right_index=True)
        # df_expanded = df_expanded[['time', 'assetCode'] + self.news_cols_numeric].groupby(['time', 'assetCode']).mean()

        return df_expanded

    def with_asset_code(self, news):
        """
        Update news index to be time, assetCode
        :param news:
        :return:
        """
        if news.empty:
            if 'assetCode' not in news.columns:
                news.columns = news.columns + 'assetCode'
            return news

        # Fix asset codes (str -> list)
        news['assetCodesList'] = news['assetCodes'].str.findall(f"'([\w\./]+)'")

        # Expand assetCodes
        assetCodes_expanded = list(chain(*news['assetCodesList']))

        assetCodes_index = news.index.repeat(news['assetCodesList'].apply(len))
        assert len(assetCodes_index) == len(assetCodes_expanded)
        df_assetCodes = pd.DataFrame({'level_0': assetCodes_index, 'assetCode': assetCodes_expanded})

        # Create expanded news (will repeat every assetCodes' row)
        #        df_expanded = pd.merge(df_assetCodes, news, left_on='level_0', right_index=True)
        df_expanded = pd.merge(df_assetCodes, news, left_on='level_0', right_index=True)
        df_expanded = df_expanded[['time', 'assetCode'] + self.news_cols_numeric].groupby(['time', 'assetCode']).mean()

        return df_expanded

    def aggregate_news(self, df):
        """
        News are rare for an asset. We get mean value for 10 days
        :param df:
        :return:
        """
        if df.empty:
            return df

        # News are rare for the asset, so aggregate them by rolling period say 10 days
        rolling_days = 10
        df_aggregated = df.groupby(['assetCode', 'time']).mean().reset_index(['assetCode', 'time'])
        df_aggregated = df_aggregated.groupby('assetCode') \
            .rolling(rolling_days, on='time') \
            .apply(np.mean, raw=False) \
            .reset_index('assetCode')
        #df_aggregated.set_index(['time', 'assetCode'], inplace=True)
        return df_aggregated
#
#
# # Create instance for global usage
# news_prepro = NewsPrepro()
# print('news_prepro created')
