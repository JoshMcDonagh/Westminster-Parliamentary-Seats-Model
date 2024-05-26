import pandas as pd

from seat_model import SeatModel


class WestminsterSeatModel:
    """
    A class to model the allocation of seats in the Westminster Parliament based on vote shares.

    Attributes:
    _previous_vote_share_df (pd.DataFrame): DataFrame containing previous vote share data.
    _seat_models (dict): Dictionary storing SeatModel instances for each nation.
    _speakers_constituency_nation (str): The nation of the Speaker's constituency.
    """

    def __init__(self) -> None:
        """
        Initialize the WestminsterSeatModel with historical data for each nation.
        """
        self._previous_vote_share_df = pd.read_csv("data/british_vote_share_data.csv")
        self._seat_models = {
            "england": SeatModel(self._previous_vote_share_df, pd.read_csv("data/english_seat_share_data.csv")),
            "scotland": SeatModel(self._previous_vote_share_df, pd.read_csv("data/scottish_seat_share_data.csv")),
            "wales": SeatModel(self._previous_vote_share_df, pd.read_csv("data/welsh_seat_share_data.csv"))
        }

        self._speakers_constituency_nation = "england"

    def set_speakers_constituency_nation(self, speakers_constituency_nation: str) -> None:
        """
        Set the nation of the Speaker's constituency to adjust the total seat count.

        Parameters:
        speakers_constituency_nation (str): The nation of the Speaker's constituency.
        """
        self._speakers_constituency_nation = speakers_constituency_nation

    def predict_seats(self,
                      number_of_english_seats: int,
                      number_of_scottish_seats: int,
                      number_of_welsh_seats: int,
                      vote_share_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the number of seats for each party based on vote shares for each nation.

        Parameters:
        number_of_english_seats (int): Total number of seats in England.
        number_of_scottish_seats (int): Total number of seats in Scotland.
        number_of_welsh_seats (int): Total number of seats in Wales.
        vote_share_df (pd.DataFrame): DataFrame containing vote share data for the prediction.

        Returns:
        pd.DataFrame: DataFrame containing the predicted number of seats for each party.
        """
        overall_seat_predictions = None
        seat_totals = {
            "england": number_of_english_seats,
            "scotland": number_of_scottish_seats,
            "wales": number_of_welsh_seats
        }

        # Generate and accumulate the seats generated for each British nation
        for nation_name, seat_model in self._seat_models.items():
            # Adjust the seat total if this nation has the Speaker's constituency
            if nation_name == self._speakers_constituency_nation:
                seat_totals[nation_name] = seat_totals[nation_name] - 1

            # Predict the seat allocation for the nation
            seat_model_prediction = seat_model.predict_seats(seat_totals[nation_name], vote_share_df)

            # Sum the predictions across nations
            if overall_seat_predictions is None:
                overall_seat_predictions = seat_model_prediction
            else:
                overall_seat_predictions = overall_seat_predictions.add(seat_model_prediction)

        return pd.DataFrame(overall_seat_predictions)
