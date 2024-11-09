from typing import List
from typing import Dict


class Seat:
    def __init__(self, parties_list: List[str]) -> None:
        self.vote_share_sensitivities = {party: 1.0 for party in parties_list}

    def get_sensitivities_as_dict(self) -> Dict[str, float]:
        return self.vote_share_sensitivities

    def set_sensitivities_with_dict(self, new_sensitivities_dict: Dict[str, float]) -> None:
        self.vote_share_sensitivities = new_sensitivities_dict