import unittest
from unittest.mock import patch

import pandas as pd

from westminster_seat_model import WestminsterSeatModel


class TestWestminsterSeatModel(unittest.TestCase):

    @patch('pandas.read_csv')
    def setUp(self, mock_read_csv):
        # Mock the pd.read_csv method to return specific DataFrames for testing
        self.vote_share_df = pd.DataFrame({
            'Election Year': [2019, 2017, 2015, 2010, 2005],
            'Conservative': [43.63, 42.30, 36.80, 36.10, 32.40],
            'Labour': [32.08, 40.00, 30.40, 29.00, 35.20],
            'Scottish National Party': [3.88, 3.00, 4.70, 1.70, 1.50],
            'Liberal Democrats': [11.55, 7.40, 7.90, 23.00, 22.00],
            'Plaid Cymru': [0.48, 0.50, 0.60, 0.60, 0.60],
            'Green Party': [2.70, 1.60, 3.90, 1.00, 1.10],
            'UKIP/Brexit Party/Reform UK': [2.01, 1.80, 12.60, 3.10, 2.20],
            'Other': [3.67, 3.40, 3.10, 5.50, 5.00]
        })
        self.english_seat_share_df = pd.DataFrame({
            'Election Year': [2019, 2017, 2015, 2010, 2005],
            'Conservative': [65.73, 55.72, 59.85, 55.72, 36.67],
            'Labour': [33.77, 42.59, 38.65, 35.83, 54.06],
            'Scottish National Party': [0, 0, 0, 0, 0],
            'Liberal Democrats': [1.31, 1.50, 1.13, 8.07, 8.88],
            'Plaid Cymru': [0, 0, 0, 0, 0],
            'Green Party': [0.19, 0.19, 0.19, 0.19, 0.00],
            'UKIP/Brexit Party/Reform UK': [0, 0, 0.19, 0, 0],
            'Other': [0, 0, 0, 0, 0.38]
        })
        self.scottish_seat_share_df = pd.DataFrame({
            'Election Year': [2019, 2017, 2015, 2010, 2005],
            'Conservative': [10.17, 22.03, 1.69, 1.69, 1.69],
            'Labour': [1.69, 11.86, 1.69, 69.49, 69.49],
            'Scottish National Party': [81.36, 59.32, 94.92, 10.17, 10.17],
            'Liberal Democrats': [6.78, 6.78, 1.69, 18.64, 18.64],
            'Plaid Cymru': [0, 0, 0, 0, 0],
            'Green Party': [0, 0, 0, 0, 0],
            'UKIP/Brexit Party/Reform UK': [0, 0, 0, 0, 0],
            'Other': [0, 0, 0, 0, 0]
        })
        self.welsh_seat_share_df = pd.DataFrame({
            'Election Year': [2019, 2017, 2015, 2010, 2005],
            'Conservative': [35.0, 20.0, 27.5, 20.0, 7.5],
            'Labour': [55.0, 70.0, 62.5, 65.0, 72.5],
            'Scottish National Party': [0, 0, 0, 0, 0],
            'Liberal Democrats': [0, 0, 2.5, 7.5, 10.0],
            'Plaid Cymru': [10.0, 10.0, 7.5, 7.5, 7.5],
            'Green Party': [0, 0, 0, 0, 0],
            'UKIP/Brexit Party/Reform UK': [0, 0, 0, 0, 0],
            'Other': [0, 0, 0, 0, 2.5]
        })

        mock_read_csv.side_effect = [self.vote_share_df, self.english_seat_share_df, self.scottish_seat_share_df, self.welsh_seat_share_df]
        self.model = WestminsterSeatModel()

    def test_set_speakers_constituency_nation(self):
        self.model.set_speakers_constituency_nation("scotland")
        self.assertEqual(self.model._speakers_constituency_nation, "scotland")

    def test_predict_seats(self):
        vote_share_df = self.vote_share_df.drop(columns=["Election Year"])
        predicted_seats = self.model.predict_seats(
            number_of_english_seats=533,
            number_of_scottish_seats=59,
            number_of_welsh_seats=40,
            vote_share_df=vote_share_df
        )
        self.assertIsInstance(predicted_seats, pd.DataFrame)
