# Westminster Parliamentary Seats Model

This repository contains a model to predict the allocation of parliamentary seats in the Westminster system based on vote shares. The model uses polynomial regression to fit historical data and make predictions for future elections. Note that the model only generates a seat prediction for each party in England, Scotland, and Wales. It does not generate a prediction for Northern Ireland.

## Directory Structure

```sh
Westminster-Parliamentary-Seats-Model
├── data
│ ├── british_vote_share_data.csv
│ ├── english_seat_share_data.csv
│ ├── scottish_seat_share_data.csv
│ ├── welsh_seat_share_data.csv
├── src
│ ├── init.py
│ ├── seat_model.py
│ ├── westminster_seat_model.py
├── tests
│ ├── init.py
│ ├── test_seat_model.py
│ ├── test_westminster_seat_model.py
├── .gitignore
├── input.csv
├── model_runner.py
├── output.csv
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```sh
git clone https://github.com/JoshMcDonagh/Westminster-Parliamentary-Seats-Model.git
```

2. Navigate to the repository directory:
```sh
cd Westminster-Parliamentary-Seats-Model
```

3. Install the required Python packages:
```sh
pip install -r requirements.txt
```

## Usage

1. Place your vote share data in the 'input.csv' file.

2. Run the model using the 'model_runner.py' script:
```sh
python model_runner.py
```

3. The predicted seat allocations will be saved in the 'output.csv' file.

## Data

The 'data' directory contains historical vote share and seat share data for:

- England

- Scotland

- Wales

## Tests

To run the tests, use the following command:
```sh
python -m unittest discover -s tests
```

## Model Description

- 'SeatModel' - this class fits a polynomial regression model to historical vote and seat share data for a given region.

- 'WestminsterSeatModel' - this class uses instances of 'SeatModel' to predict seat allocations for England, Scotland, and Wales based on input vote shares.

## License

This project is licensed under the MIT License.