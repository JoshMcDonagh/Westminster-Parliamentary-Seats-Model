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


def adjust_row_to_int_preserving_sum(row):
    """
    Adjust the values in a row of float values to integers while preserving the sum.

    Parameters:
    row (pd.Series): A row of float values.

    Returns:
    list: A list of integers with the same sum as the original float values.
    """
    float_list = row.values
    target_sum = round(sum(float_list))

    # Initial rounding of the float values
    rounded_ints = [int(round(num)) for num in float_list]

    # Calculate the initial error
    current_sum = sum(rounded_ints)
    error = target_sum - current_sum

    if error == 0:
        return rounded_ints

    # Calculate the difference between the float and the rounded integer
    differences = [(abs(num - round(num)), i) for i, num in enumerate(float_list)]

    # Sort differences by the absolute value of the difference (descending order)
    differences.sort(reverse=True, key=lambda x: x[0])

    # Adjust the largest differences, ensuring no negative values
    for i in range(abs(error)):
        index = differences[i % len(differences)][1]
        if error > 0:
            rounded_ints[index] += 1
        elif error < 0 and rounded_ints[index] > 0:
            rounded_ints[index] -= 1

    return rounded_ints


def dataframe_floats_to_ints_preserving_sum(df):
    """
    Convert float values in a DataFrame to integers while preserving the sum for each row.

    Parameters:
    df (pd.DataFrame): The DataFrame containing float values.

    Returns:
    pd.DataFrame: A DataFrame with integer values preserving the sum for each row.
    """
    # Apply the function to each row
    int_df = df.apply(adjust_row_to_int_preserving_sum, axis=1, result_type='expand')

    # Ensure the column names are preserved
    int_df.columns = df.columns
    return int_df


class SeatModel:
    """
    A class to model seat allocation based on vote shares using polynomial regression.

    Attributes:
    _previous_vote_share_df (pd.DataFrame): Previous vote share data.
    _previous_seat_share_df (pd.DataFrame): Previous seat share data.
    _party_models (dict): Dictionary to store the polynomial regression models for each party.
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

        # Fit a polynomial regression model for each party
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
                predicted_seat_share = 0.0

            if not isinstance(predicted_seat_share, int) and not isinstance(predicted_seat_share, float):
                predicted_seat_share = predicted_seat_share[0]

            if predicted_seat_share < 0.0:
                predicted_seat_share = 0.0

            seat_share_total += predicted_seat_share
            seat_share_predictions[party_name] = predicted_seat_share

        seat_predictions = {}

        # Calculate the number of seats for each party based on the associated seat share
        for party_name, party_seat_share_prediction in seat_share_predictions.items():
            value = total_number_of_seats * (party_seat_share_prediction / seat_share_total)
            seat_predictions[party_name] = [value]

        pd.set_option('display.max_columns', None)

        # Convert seat_predictions to DataFrame
        df = pd.DataFrame.from_dict(seat_predictions).clip(lower=0)
        df = dataframe_floats_to_ints_preserving_sum(df)
        return df