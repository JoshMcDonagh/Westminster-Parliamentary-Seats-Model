import unittest

import pandas as pd
from sklearn.pipeline import Pipeline

from src.seat_model import fit_polynomial_regression, SeatModel


class TestFitPolynomialRegression(unittest.TestCase):
    def setUp(self):
        # Setup code to create a sample DataFrame and Series for testing
        self.X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })
        self.y = pd.Series([1, 4, 9, 16, 25])  # Quadratic relationship for simplicity

    def test_fit_polynomial_regression_returns_pipeline(self):
        # Test that the function returns a Pipeline object
        model = fit_polynomial_regression(self.X, self.y)
        self.assertIsInstance(model, Pipeline)

    def test_model_is_fitted(self):
        # Test that the model is fitted correctly
        model = fit_polynomial_regression(self.X, self.y)
        self.assertIsNotNone(model.named_steps['linearregression'].coef_)
        self.assertIsNotNone(model.named_steps['linearregression'].intercept_)

    def test_model_coefficients(self):
        # Test that the model coefficients are correct for a simple polynomial case
        model = fit_polynomial_regression(self.X, self.y, degree=2)
        actual_coefficients = model.named_steps['linearregression'].coef_
        expected_coefficients = [0.08108108, -0.08108108, 0.81981982, -0.33333333, -0.15315315]
        for actual, expected in zip(actual_coefficients, expected_coefficients):
            self.assertAlmostEqual(actual, expected, places=5)


class TestSeatModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the data once for all tests
        cls.vote_share_df = pd.read_csv("data/british_vote_share_data.csv")
        cls.english_seat_share_df = pd.read_csv("data/english_seat_share_data.csv")
        cls.scottish_seat_share_df = pd.read_csv("data/scottish_seat_share_data.csv")
        cls.welsh_seat_share_df = pd.read_csv("data/welsh_seat_share_data.csv")

    def test_seat_model_england(self):
        model = SeatModel("england", self.english_seat_share_df, self.vote_share_df)
        vote_share_df = self.vote_share_df.drop(columns=["Election Year"])
        predicted_seats = model.predict_seats(total_number_of_seats=533,
                                              vote_share_df=vote_share_df)
        self.assertIsInstance(predicted_seats, pd.DataFrame)

    def test_seat_model_scotland(self):
        model = SeatModel("scotland", self.scottish_seat_share_df, self.vote_share_df)
        vote_share_df = self.vote_share_df.drop(columns=["Election Year"])
        predicted_seats = model.predict_seats(total_number_of_seats=59,
                                              vote_share_df=vote_share_df)
        self.assertIsInstance(predicted_seats, pd.DataFrame)

    def test_seat_model_wales(self):
        model = SeatModel("wales", self.welsh_seat_share_df, self.vote_share_df)
        vote_share_df = self.vote_share_df.drop(columns=["Election Year"])
        predicted_seats = model.predict_seats(total_number_of_seats=40,
                                              vote_share_df=vote_share_df)
        self.assertIsInstance(predicted_seats, pd.DataFrame)
