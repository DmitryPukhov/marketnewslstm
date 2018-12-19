from marketnewslstm import MarketPrepro, NewsPrepro, JoinedPreprocessor, JoinedGenerator
import pandas as pd
import unittest


class JoinedPreprocessorTest(unittest.TestCase):

    # def setUp(self):
    # Create test market, news, idx
    # self.market = pd.DataFrame({'assetCode': ['asset1', 'asset1', 'asset2', 'asset2'],
    #                             'time': [pd.Timestamp('2018-01-01'), pd.Timestamp('2018-01-02'),
    #                                      pd.Timestamp('2018-01-01'), pd.Timestamp('2018-01-02')],
    #                             'open': [1, 2, 3, 4],
    #                             'close': [1, 2, 3, 4]},
    #                            columns=['assetCode', 'time', 'open', 'close'],
    #                            index=[11, 12, 13, 14])
    # self.news = pd.DataFrame({'assetCode': ['asset1', 'asset1', 'asset2', 'asset2'],
    #                           'time': [pd.Timestamp('2018-01-01'), pd.Timestamp('2018-01-02'),
    #                                    pd.Timestamp('2018-01-01'), pd.Timestamp('2018-01-02')]},
    #                          columns=['assetCode', 'time'],
    #                          index=[21, 22, 23, 24])
    # self.idx = pd.DataFrame({'market_index': [11, 12, 13, 14],
    #                          'news_index': [21, 22, 23, 24]})
    # Create tested class instance
    # self.market_prepro = MarketPrepro()
    # self.news_prepro = NewsPrepro()
    # self.prepro = JoinedPreprocessor(self.market_prepro, self.news_prepro)
    # #self.prepro.fit(self.idx, self.market, self.news)
    # self.generator = JoinedGenerator(None, self.prepro, self.market, self.news)

    def test_with_lookback(self):
        # Call
        prepro = JoinedPreprocessor(None, None)
        X = pd.DataFrame({'id': [1, 2, 3, 4, 5, 6], 'name': ['name1', 'name2', 'name3', 'name4', 'name5', 'name6']})
        y = pd.DataFrame({'value': [11, 12, 13, 14, 15, 6]})
        X_res, y_res = prepro.with_look_back(X, y, 5, 1)

        # Assert
        # Shape: 6 rows, 6 is value + lookback 1, 2 columns
        self.assertEqual((6, 6, 2), X_res.shape)
        # Each result lookback should contain previous window
        for i in range(5, 4, -1):
            self.assertEqual(X[:6].values.tolist(), X_res[i].tolist())
        # Look back for first element should be the first element repeated
        for i in range(0, 6):
            self.assertEqual(X.values[0].tolist(), X_res[0, i].tolist())

    def test_with_lookback_1(self):
        # Call
        prepro = JoinedPreprocessor(None, None)
        X = pd.DataFrame({'id': [1, 2, 3, 4, 5], 'name': ['name1', 'name2', 'name3', 'name4', 'name5']})
        y = pd.DataFrame({'value': [11, 12, 13, 14, 15]})
        X_res, y_res = prepro.with_look_back(X, y, 1, 1)

        # Assert
        # Shape: 5 rows, 2 is lookback 1 plus the value itself, 2 columns
        self.assertEquals((5, 2, 2), X_res.shape)
        # Each result lookback should contain previous window
        for i in range(4, 1, -1):
            self.assertEqual(X[i - 1:i + 1].values.tolist(), X_res[i].tolist())
        # Look back for first element should be the first element repeated
        for i in range(0, 2):
            self.assertEqual(X.values[0].tolist(), X_res[0, i].tolist())

    def test_with_lookback_3_step2(self):
        # Call
        prepro = JoinedPreprocessor(None, None)
        X = pd.DataFrame({'id': [1, 2, 3, 4, 5], 'name': ['name1', 'name2', 'name3', 'name4', 'name5']})
        y = pd.DataFrame({'value': [11, 12, 13, 14, 15]})
        X_res, y_res = prepro.with_look_back(X, y, 3, 2)
        # Assert
        # Shape: 5 rows, 2 is value + lookback 1, 2 columns
        self.assertEquals((5, 2, 2), X_res.shape)
        # Each result lookback should contain previous window
        for i in range(4, 2, -1):
            self.assertEqual(X[i - 2:i + 1:2].values.tolist(), X_res[i].tolist())
        # Look back for first element should be the first element repeated
        for i in range(0, 2):
            self.assertEqual(X.values[0].tolist(), X_res[0, i].tolist())


if __name__ == '__main__':
    unittest.main()
