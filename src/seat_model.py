import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


def fit_polynomial_regression(X, y, degree=3):
    """
    Fit a polynomial regression model to the given data.

    Parameters:
    X (pd.DataFrame): The feature matrix (vote shares).
    y (pd.Series): The target vector (seat shares).
    degree (int): The degree of the polynomial.

    Returns:
    Pipeline: The fitted polynomial regression model.
    """
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    pipeline = make_pipeline(polynomial_features, linear_regression)
    pipeline.fit(X, y)
    return pipeline


class SeatModel:
    """
    A class to model seat allocation based on vote shares using linear regression.

    Attributes:
    _previous_vote_share_df (pd.DataFrame): Previous vote share data.
    _previous_seat_share_df (pd.DataFrame): Previous seat share data.
    _party_models (dict): Dictionary to store the linear regression models for each party.
    """

    def __init__(self, nation_name: str, previous_vote_share_df: pd.DataFrame, previous_seat_share_df: pd.DataFrame) -> None:
        """
        Initialize the SeatModel with historical data and fit models for each party.

        Parameters:
        nation_name (str): String containing the name of the nation this seat model models.
        previous_vote_share_df (pd.DataFrame): DataFrame containing previous vote share data.
        previous_seat_share_df (pd.DataFrame): DataFrame containing previous seat share data.
        """
        self._nation_name = nation_name
        self._previous_vote_share_df = previous_vote_share_df.drop(columns=["Election Year"])
        self._previous_seat_share_df = previous_seat_share_df.drop(columns=["Election Year"])
        self._party_models = {}

        # Fit a linear regression model for each party
        for party_name, party_data in self._previous_seat_share_df.items():
            if nation_name == "england":
                model = fit_polynomial_regression(self._previous_vote_share_df, party_data, 3)
                self._include_snp = False
                self._include_pc = False
            elif nation_name == "scotland":
                model = fit_polynomial_regression(self._previous_vote_share_df, party_data, 5)
                self._include_snp = True
                self._include_pc = False
            elif nation_name == "wales":
                model = fit_polynomial_regression(self._previous_vote_share_df, party_data, 3)
                self._include_snp = False
                self._include_pc = True
            else:
                raise Exception("Invalid nation name provided")

            self._party_models[party_name] = model

    def predict_seats(self, total_number_of_seats: int, vote_share_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the number of seats for each party based on vote shares.

        Parameters:
        total_number_of_seats (int): Total number of seats to be allocated.
        vote_share_df (pd.DataFrame): DataFrame containing vote share data for the prediction.

        Returns:
        pd.DataFrame: DataFrame containing the predicted number of seats for each party.
        """
        seat_share_predictions = {}
        seat_share_total = 0.0

        # Predict the seat share for each party using the corresponding party models
        for party_name, model in self._party_models.items():
            if party_name != "Scottish National Party" and party_name != "Plaid Cymru":
                predicted_seat_share = model.predict(vote_share_df)
            elif party_name == "Scottish National Party" and self._include_snp:
                predicted_seat_share = model.predict(vote_share_df)
            elif party_name == "Plaid Cymru" and self._include_pc:
                predicted_seat_share = model.predict(vote_share_df)
            else:
                predicted_seat_share = 0
            seat_share_total += predicted_seat_share
            seat_share_predictions[party_name] = predicted_seat_share

        seat_predictions = {}

        # Calculate the number of seats for each party based on the associated seat share
        for party_name, party_seat_share_prediction in seat_share_predictions.items():
            seat_predictions[party_name] = total_number_of_seats * (party_seat_share_prediction / seat_share_total)
        seat_predictions_df = pd.DataFrame(seat_predictions).round().clip(lower=0)

        return seat_predictions_df
