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

    def predict(self, market, news):
        """
        Predict from new received market and news data.
        :return: predicted y
        """
        X = self.prepro.get_X(market, news)
        X = self.prepro.with_look_back(X, None, self.look_back, self.look_back_step)
        y = self.model.predict(X) * 2 - 1
        return y

    def predict_idx(self, pred_idx, market, news):
        """
        Predict for test from indices
        :return:
            predicted y, ground truth y
        """
        # Get preprocessed X, y
        X_test, y_test = self.prepro.get_Xy(pred_idx, market, news, is_train=False, is_raw_y=True)
        # Add there look back rows for LSTM
        X_test, y_test = self.prepro.with_look_back(X_test, y_test, look_back=self.look_back,
                                                    look_back_step=self.look_back_step)
        y_pred = self.model.predict(X_test) * 2 - 1
        return y_pred, y_test
