class Seat:
    def __init__(self, constituency_id, prev_election_vote_shares):
        self.constituency_id = constituency_id
        self.prev_election_vote_shares = prev_election_vote_shares
        self.prev_party_winner = max(self.prev_election_vote_shares, key=self.prev_election_vote_shares.get)



