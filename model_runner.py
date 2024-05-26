import pandas as pd

from westminster_seat_model import WestminsterSeatModel

# Main block to execute the script
if __name__ == "__main__":
    # Create an instance of WestminsterSeatModel
    westminster_model = WestminsterSeatModel()

    # Define the number of seats for each nation
    number_of_english_seats = 543
    number_of_scottish_seats = 57
    number_of_welsh_seats = 32

    # Load the input vote share data from a CSV file
    input_vote_share_df = pd.read_csv("input.csv")

    # Predict the number of seats for each party based on the input vote shares
    predicted_seats_df = westminster_model.predict_seats(number_of_english_seats,
                                                      number_of_scottish_seats,
                                                      number_of_welsh_seats,
                                                      input_vote_share_df)

    # Save the predicted seats to an output CSV file
    predicted_seats_df.to_csv("output.csv", index=False)
