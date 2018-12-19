from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, \
    RobustScaler
import pandas as pd


class MarketPrepro:
    """
    Standard way to generate batches for model.fit_generator(generator, ...)
    Should be fit on train data and used on all train, validation, test
    """
    # Features
    assetcode_encoded = []
    time_cols = ['year', 'week', 'day', 'dayofweek']
    numeric_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
                    'returnsClosePrevMktres1', 'returnsOpenPrevMktres1', 'returnsClosePrevRaw10',
                    'returnsOpenPrevRaw10', 'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
    feature_cols = ['assetCode_encoded'] + time_cols + numeric_cols

    # Labels
    label_cols = ['returnsOpenNextMktres10']

    def __init__(self):
        self.cats = {}
        self.numeric_scaler = StandardScaler()

    def fit(self, market_train_idx, market):
        """
        Fit preprocessing scalers, encoders on given train df.
        Store given indices to generate batches_from.
        @param market_train_df: train data to fit on
        """
        market_train_df = market.loc[market_train_idx.index].copy()
        # Clean bad data. We fit on train dataset and it's ok to remove bad data
        market_train_df = self.fix_train(market_train_df)

        # Extract day, week, year from time
        market_train_df = self.prepare_time_cols(market_train_df)
        # Fit for numeric and time
        # self.numeric_scaler = QuantileTransformer()
        self.numeric_scaler.fit(market_train_df[self.numeric_cols + self.time_cols])

        # Fit asset encoding
        self.encode_asset(market_train_df, is_train=True)

    def fix_train(self, train_df):
        """
        Remove bad data. For train dataset only
        """
        # Remove strange cases with close/open ratio > 2
        max_ratio = 2
        train_df = train_df[(train_df['close'] / train_df['open']).abs() <= max_ratio].loc[:]
        # Fix outliers etc like for test set
        train_df = self.safe_fix(train_df)
        return train_df

    def safe_fix(self, df):
        """
        Fill na, fix outliers. Safe for test dataset, no rows removed.
        """
        # Fill nans
        df[self.numeric_cols] = df[self.numeric_cols].fillna(0)
        # Fix outliers
        df[self.numeric_cols] = df[self.numeric_cols].clip(df[self.numeric_cols].quantile(0.01),
                                                           df[self.numeric_cols].quantile(0.99), axis=1)
        return df

    def get_X(self, df):
        """
        Preprocess and return X without y
        """
        df = df.copy()
        df = self.safe_fix(df)

        # Add day, week, year
        df = self.prepare_time_cols(df)
        # Encode assetCode
        df = self.encode_asset(df)
        # Scale numeric features and labels

        df = df.set_index(['assetCode', 'time'], drop=False)
        df[self.numeric_cols + self.time_cols] = self.numeric_scaler.transform(
            df[self.numeric_cols + self.time_cols].astype(float))

        # print(df.head())
        # Return X
        return df[self.feature_cols]

    def get_y(self, df, is_raw_y=False):
        if is_raw_y:
            return df[self.label_cols]
        else:
            return (df[self.label_cols] >= 0).astype(float)

    def encode_asset(self, df, is_train=False):
        def encode(assetcode):
            """
            Encode categorical features to numbers
            """
            try:
                # Transform to index of name in stored names list
                index_value = self.assetcode_encoded.index(assetcode) + 1
            except ValueError:
                # If new value, add it to the list and return new index
                self.assetcode_encoded.append(assetcode)
                index_value = len(self.assetcode_encoded)

            # index_value = 1.0/(index_value)
            index_value = index_value / (self.assetcode_train_count + 1)
            return (index_value)

        # Store train assetcode_train_count for use as a delimiter for test data encoding
        if is_train:
            self.assetcode_train_count = len(df['assetCode'].unique()) + 1

        df['assetCode_encoded'] = df['assetCode'].apply(lambda assetcode: encode(assetcode))
        return (df)

    @staticmethod
    def prepare_time_cols(df):
        """
        Extract time parts, they are important for time series
        """
        df['year'] = pd.to_datetime(df['time']).dt.year
        # Maybe remove month because week of year can handle the same info
        df['day'] = pd.to_datetime(df['time']).dt.day
        # Week of year
        df['week'] = pd.to_datetime(df['time']).dt.week
        df['dayofweek'] = pd.to_datetime(df['time']).dt.dayofweek
        return df

#
# # Create instance for global usage
# market_prepro = MarketPrepro()
# print('market_prepro created')
