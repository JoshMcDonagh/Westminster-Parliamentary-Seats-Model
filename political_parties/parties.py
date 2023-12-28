from political_parties.party import Party

parties = {
    "conservative": Party("conservative"),
    "labour": Party("labour"),
    "liberal democrats": Party("liberal democrats"),
    "scottish national party": Party("scottish national party"),
    "green party of england and wales": Party("green party of england and wales"),
    "reform uk": Party("reform uk"),
    "dup": Party("dup"),
    "sinn féin": Party("sinn féin"),
    "plaid cymru": Party("plaid cymru"),
    "alliance": Party("alliance"),
    "sdlp": Party("sdlp"),
    "ulster unionist": Party("ulster unionist"),
    "yorkshire": Party("yorkshire"),
    "scottish greens": Party("scottish greens"),
    "speaker": Party("speaker"),
    "other": Party("other")
}


def get_party(party_name):
    return parties[party_name]
