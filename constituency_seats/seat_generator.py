from constituency_seats.seat import Seat
from political_parties import parties


def get_previous_election_vote_shares():
    return {
        parties.get_party("conservative"): 43.63,
        parties.get_party("labour"): 32.08,
        parties.get_party("liberal democrats"): 11.55,
        parties.get_party("scottish national party"): 3.88,
        parties.get_party("green party of england and wales"): 2.61,
        parties.get_party("reform uk"): 2.01,
        parties.get_party("dup"): 0.76,
        parties.get_party("sinn féin"): 0.57,
        parties.get_party("plaid cymru"): 0.48,
        parties.get_party("alliance"): 0.42,
        parties.get_party("sdlp"): 0.37,
        parties.get_party("ulster unionist"): 0.29,
        parties.get_party("yorkshire"): 0.09,
        parties.get_party("scottish greens"): 0.09,
        parties.get_party("speaker"): 0.08,
        parties.get_party("other"): 1.09
    }

def get_previous_election_seats():
    return {
        parties.get_party("conservative"): 365,
        parties.get_party("labour"): 202,
        parties.get_party("liberal democrats"): 11,
        parties.get_party("scottish national party"): 48,
        parties.get_party("green party of england and wales"): 1,
        parties.get_party("reform uk"): 0,
        parties.get_party("dup"): 8,
        parties.get_party("sinn féin"): 7,
        parties.get_party("plaid cymru"): 4,
        parties.get_party("alliance"): 1,
        parties.get_party("sdlp"): 2,
        parties.get_party("ulster unionist"): 0,
        parties.get_party("yorkshire"): 0,
        parties.get_party("scottish greens"): 0,
        parties.get_party("speaker"): 1,
        parties.get_party("other"): 0
    }

def generate_seats():
    previous_election_seats = get_previous_election_seats()
    seats = []

    for party, seats_won in get_previous_election_seats().items():
        for i in range(seats_won):
            seat = Seat(i, {party: 50.0})
            seats.append(seat)
            print(seat.prev_party_winner.get_name())

generate_seats()
