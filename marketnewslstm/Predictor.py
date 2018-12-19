import pandas as pd


class Predictor:
    """
    Predict for test data or real prediction
    """

    def __init__(self, prepro, market_prepro, news_prepro, model, look_back, look_back_step):
        self.prepro = prepro
        self.market_prepro = market_prepro
        self.news_prepro = news_prepro
        self.model = model
        self.look_back = look_back
        self.look_back_step = look_back_step

    def predict_idx(self, pred_idx, market, news):
        """
        Predict for test from indices
        Returns:
            predicted labels, true y
        """
        # Get preprocessed X, y
        X_test, y_test = self.prepro.get_Xy(pred_idx, market, news, is_train=False, is_raw_y=True)
        # Add there look back rows for LSTM
        X_test, y_test = self.prepro.with_look_back(X_test, y_test, look_back=self.look_back, look_back_step=self.look_back_step)
        y_pred = self.model.predict(X_test) * 2 - 1
        return y_pred, y_test

    def get_X(self, market, news):
        """
        For given indices, get market data and join news by date and asset code.
        Return normalized features.
        """
        # Preprocess market X
        market_X = self.market_prepro.get_X(market)

        # Preprocess news X and add
        news = self.news_prepro.with_asset_code(news)
        news = self.news_prepro.aggregate_news(news)
        news_X = self.news_prepro.safe_fix(news)
        news_X.time = pd.to_datetime(news_X.time, utc=True)
        news_X.time = news_X.time.astype('datetime64')
        X = market_X.merge(news_X, how='left', on=['time', 'assetCode'], left_index=True)

        # Some market data can be without news, fill nans
        X[self.market_prepro.feature_cols + self.news_prepro.feature_cols] = X[
            self.market_prepro.feature_cols + self.news_prepro.feature_cols].fillna(0)

        # Return features market + news from joined df
        features = X[self.market_prepro.feature_cols + self.news_prepro.feature_cols]
        return features

    def get_X_with_lookback(self, market, news, look_back, look_back_step):
        """
        Get preprocessed data with look backs for lstm
        """
        X = self.get_X(market, news)
        X = self.prepro.with_look_back(X, None, look_back, look_back_step)
        return X
