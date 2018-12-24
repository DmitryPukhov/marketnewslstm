from marketnewslstm import MarketPrepro, NewsPrepro, JoinedPreprocessor, JoinedGenerator
import pandas as pd
import unittest


class JoinedPreprocessorTest(unittest.TestCase):

    def test_with_lookback(self):
        # Call
        prepro = JoinedPreprocessor(None, None)
        X = pd.DataFrame({'id': [1, 2, 3, 4, 5, 6], 'name': ['name1', 'name2', 'name3', 'name4', 'name5', 'name6']})
        X_res = prepro.with_look_back(X, 5, 1)

        # Assert
        # Shape: 6 rows, 6 is value + lookback 1, 2 columns
        self.assertEqual((6, 6, 2), X_res.shape)
        # Each result lookback should contain previous window
        for i in range(5, 4, -1):
            self.assertEqual(X[:6].values.tolist(), X_res[i].tolist())
        # Look back for first element should be the first element repeated
        for i in range(0, 6):
            self.assertEqual(X.values[0].tolist(), X_res[0, i].tolist())

    def test_with_lookback_10_step_5(self):
        # Call
        prepro = JoinedPreprocessor(None, None)
        X = pd.DataFrame({'id': [1], 'name': ['name1']})
        X_res = prepro.with_look_back(X, 10, 5)

        # Assert
        # Shape: 3 rows, 2 is lookback 1 plus the value itself, 2 columns
        self.assertEquals((1, 3, 2), X_res.shape)


    def test_with_lookback_1(self):
        # Call
        prepro = JoinedPreprocessor(None, None)
        X = pd.DataFrame({'id': [1, 2, 3, 4, 5], 'name': ['name1', 'name2', 'name3', 'name4', 'name5']})
        X_res = prepro.with_look_back(X, 1, 1)

        # Assert
        # Shape: 5 rows, 2 is lookback 1 plus the value itself, 2 columns
        self.assertEquals((5, 2, 2), X_res.shape)
        # Each result lookback should contain previous window
        for i in range(4, 1, -1):
            self.assertEqual(X[i - 1:i + 1].values.tolist(), X_res[i].tolist())
        # Look back for first element should be the first element repeated
        for i in range(0, 2):
            self.assertEqual(X.values[0].tolist(), X_res[0, i].tolist())

    def test_with_lookback_3_step2_Xy(self):
        # Call
        prepro = JoinedPreprocessor(None, None)
        X = pd.DataFrame({'id': [1, 2, 3, 4, 5], 'name': ['name1', 'name2', 'name3', 'name4', 'name5']})
        X_res = prepro.with_look_back(X, 3, 2)
        # Assert
        # Shape: 5 rows, 2 is value + lookback 1, 2 columns
        self.assertEquals((5, 2, 2), X_res.shape)
        # Each result lookback should contain previous window
        for i in range(0, 2):
            self.assertEqual(X.values[0].tolist(), X_res[0, i].tolist())

    def test_with_lookback_4_step1_last(self):
        X = pd.DataFrame({'id': [1, 2, 3, 4, 5], 'name': ['name1', 'name2', 'name3', 'name4', 'name5']})
        prepro = JoinedPreprocessor(None, None)

        # Look back window 1,2,3,4 for the last row 5
        X_res = prepro.with_look_back(X, 4, 1, X.index.size - 1)

        # 1 - only row 5 should be returned,
        # 5 - look back 1,2,3,4 + row 5
        # 2 - index and one column as in input
        self.assertEquals(X_res.shape, (1, 5, 2))
        # Check items in look back by their ids
        look_back_ids = X_res[0, :, 0]
        self.assertListEqual([1, 2, 3, 4, 5], look_back_ids.tolist())

    def test_with_lookback_5_step1_last(self):
        X = pd.DataFrame({'id': [1, 2, 3, 4, 5], 'name': ['name1', 'name2', 'name3', 'name4', 'name5']})
        prepro = JoinedPreprocessor(None, None)

        X_res = prepro.with_look_back(X, 5, 1, X.index.size - 1)

        # 1 - only row 5 should be returned,
        # 6 - look back 1,1,2,3,4 + row 5
        # 2 - index and one column as in input
        self.assertEqual(X_res.shape, (1, 6, 2))

        look_back_ids = X_res[0, :, 0]
        self.assertListEqual([1, 1, 2, 3, 4, 5], look_back_ids.tolist())

    def test_with_lookback_4_step2_last(self):
        X = pd.DataFrame({'id': [1, 2, 3, 4, 5], 'name': ['name1', 'name2', 'name3', 'name4', 'name5']})
        prepro = JoinedPreprocessor(None, None)

        # Look back window 1,3 for the last row 5
        X_res = prepro.with_look_back(X, 4, 2, X.index.size - 1)

        # 1 - only row 5 should be returned,
        # 3 - look back 1,3 + row 5
        # 2 - index and one column as in input
        self.assertEqual(X_res.shape, (1, 3, 2))

        # Check
        look_back_ids = X_res[0, :, 0]
        # 1, 3 lookback and 5 - last item
        self.assertListEqual([1, 3, 5], look_back_ids.tolist())

    def test_with_lookback_4_step5_last(self):
        X = pd.DataFrame({'id': [1, 2, 3, 4, 5], 'name': ['name1', 'name2', 'name3', 'name4', 'name5']})
        prepro = JoinedPreprocessor(None, None)

        # Look back step > look back means no look back at all
        # Returns only last row
        X_res = prepro.with_look_back(X, 4, 5, X.index.size - 1)

        # 1 - only row 5 should be returned,
        # 1 - no look  back + row 5
        # 2 - index and one column as in input
        self.assertEqual(X_res.shape, (1, 1, 2))

        # Check
        look_back_ids = X_res[0, :, 0]
        # no lookback and 5 - last item
        self.assertListEqual([5], look_back_ids.tolist())


if __name__ == '__main__':
    unittest.main()
