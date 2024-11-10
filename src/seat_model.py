import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.multioutput import MultiOutputRegressor


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


def fit_adjustment_model(X, y, n_estimators=500, max_depth=20, min_samples_leaf=5, max_features='sqrt', degree=5):
    """
    Fit a machine learning model to adjust seat shares based on interactions.

    Parameters:
    X (pd.DataFrame): The feature matrix (initial seat shares).
    y (pd.DataFrame): The target matrix (historical adjusted seat shares).
    n_estimators (int): Number of trees in the forest.
    max_depth (int): The maximum depth of the trees.
    min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
    max_features (str or int): The number of features to consider when looking for the best split.
    degree (int): The degree of polynomial features to capture non-linear interactions.

    Returns:
    Pipeline: The fitted adjustment model.
    """
    # Polynomial features to add interaction terms
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)

    # Random Forest with modified parameters
    base_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )
    multi_output_model = MultiOutputRegressor(base_model)

    # Pipeline with polynomial features and the multi-output random forest
    adjustment_model = Pipeline([
        ('polynomial_features', polynomial_features),
        ('multi_output_model', multi_output_model)
    ])

    adjustment_model.fit(X, y)
    return adjustment_model


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
    A class to model seat allocation based on vote shares with interaction effects.

    Attributes:
    _previous_vote_share_df (pd.DataFrame): Previous vote share data.
    _previous_seat_share_df (pd.DataFrame): Previous seat share data.
    _party_models (dict): Dictionary to store the polynomial regression models for each party.
    _adjustment_model (MultiOutputRegressor): Model to adjust initial seat shares based on interactions.
    """

    def __init__(self, nation_name: str, previous_vote_share_df: pd.DataFrame, previous_seat_share_df: pd.DataFrame) -> None:
        """
        Initialize the SeatModel with historical data and fit models for each party.

        Parameters:
        nation_name (str): The nation this seat model models (e.g., "england").
        previous_vote_share_df (pd.DataFrame): DataFrame with historical vote shares.
        previous_seat_share_df (pd.DataFrame): DataFrame with historical seat shares.
        """
        self._nation_name = nation_name
        self._previous_vote_share_df = previous_vote_share_df.drop(columns=["Election Year"])
        self._previous_seat_share_df = previous_seat_share_df.drop(columns=["Election Year"])
        self._party_models = {}

        # Fit a polynomial regression model for each party
        for party_name, party_data in self._previous_seat_share_df.items():
            model = fit_polynomial_regression(self._previous_vote_share_df, party_data, degree=3)
            self._party_models[party_name] = model

        # Fit the adjustment model on historical initial seat shares to final adjusted seat shares
        initial_seat_shares = pd.DataFrame({party: model.predict(self._previous_vote_share_df) for party, model in self._party_models.items()})
        self._adjustment_model = fit_adjustment_model(initial_seat_shares, self._previous_seat_share_df)

    def predict_seats(self, total_number_of_seats: int, vote_share_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the number of seats for each party based on vote shares with interactions.

        Parameters:
        total_number_of_seats (int): Total number of seats to allocate.
        vote_share_df (pd.DataFrame): DataFrame with vote share data for prediction.

        Returns:
        pd.DataFrame: DataFrame with predicted seats for each party.
        """
        # Step 1: Generate initial seat share predictions for each party independently
        initial_seat_shares = {party: model.predict(vote_share_df)[0] for party, model in self._party_models.items()}
        initial_seat_shares_df = pd.DataFrame([initial_seat_shares])

        # Step 2: Use the adjustment model to modify initial predictions based on interactions
        adjusted_seat_shares = self._adjustment_model.predict(initial_seat_shares_df)
        adjusted_seat_shares_df = pd.DataFrame(adjusted_seat_shares, columns=initial_seat_shares_df.columns)

        # Normalize to ensure the total seat shares match the specified total number of seats
        seat_share_total = adjusted_seat_shares_df.values.sum()
        adjusted_seat_shares_df = adjusted_seat_shares_df.apply(lambda x: total_number_of_seats * (x / seat_share_total))

        # Convert to integers while preserving the total seat count
        adjusted_seat_shares_df = dataframe_floats_to_ints_preserving_sum(adjusted_seat_shares_df)

        return adjusted_seat_shares_df
