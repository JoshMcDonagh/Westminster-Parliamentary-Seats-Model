# Westminster Parliamentary Seats Model

This repository contains a model to predict the allocation of parliamentary seats in the Westminster system based on vote shares. The model uses polynomial regression to fit historical data and make predictions for future elections.

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

