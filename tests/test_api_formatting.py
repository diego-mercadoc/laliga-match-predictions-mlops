import pandas as pd

from api import _prediction_rows


def test_prediction_rows_normalize_probabilities_and_add_intervals():
    source = pd.DataFrame(
        [
            {
                "Anfitrion": "Team A",
                "Adversario": "Team B",
                "Fecha": "2024-08-17",
                "Sedes": 1,
                "xG(tm)": 2.0,
                "xG(opp)": 1.0,
                "PrgC(tm)": 80.0,
                "PrgC(opp)": 64.0,
                "% de TT(tm)": 30.0,
                "% de TT(opp)": 20.0,
                "TklG(tm)": 24.0,
                "TklG(opp)": 18.0,
                "Err(tm)": 2.0,
                "Err(opp)": 1.0,
            }
        ]
    )

    rows = _prediction_rows(source, [[0.2, 0.3, 0.5]])

    assert len(rows) == 1
    assert rows[0].Probabilidad_Victoria == 0.5
    assert rows[0].Probabilidad_Empate == 0.3
    assert rows[0].Probabilidad_Derrota == 0.2
    assert rows[0].Goles_Predichos_Local_CI_Lower < rows[0].Goles_Predichos_Local_CI_Upper
