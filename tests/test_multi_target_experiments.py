import pandas as pd

from services.multi_target_experiments import CLASSIFICATION_TARGETS, REGRESSION_TARGETS


def test_multi_target_definitions_cover_result_goals_corners_and_cards():
    matches = pd.DataFrame(
        [
            {
                "FTR": "H",
                "FTHG": 2,
                "FTAG": 1,
                "HC": 6,
                "AC": 5,
                "HY": 3,
                "AY": 2,
                "HR": 0,
                "AR": 1,
            }
        ]
    )

    assert CLASSIFICATION_TARGETS["home_win"](matches).iloc[0] == 1
    assert CLASSIFICATION_TARGETS["over_0_5_goals"](matches).iloc[0] == 1
    assert CLASSIFICATION_TARGETS["over_2_5_goals"](matches).iloc[0] == 1
    assert CLASSIFICATION_TARGETS["under_5_5_goals"](matches).iloc[0] == 1
    assert CLASSIFICATION_TARGETS["both_teams_score"](matches).iloc[0] == 1
    assert CLASSIFICATION_TARGETS["total_corners_over_9_5"](matches).iloc[0] == 1
    assert CLASSIFICATION_TARGETS["total_yellow_cards_over_4_5"](matches).iloc[0] == 1
    assert CLASSIFICATION_TARGETS["any_red_card"](matches).iloc[0] == 1
    assert REGRESSION_TARGETS["total_goals"](matches).iloc[0] == 3
    assert REGRESSION_TARGETS["total_corners"](matches).iloc[0] == 11
    assert REGRESSION_TARGETS["total_yellow_cards"](matches).iloc[0] == 5
