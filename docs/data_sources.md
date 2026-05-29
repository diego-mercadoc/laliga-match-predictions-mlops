# Data Sources And Feature Logic

## Accepted Sources

### Football-Data.co.uk

- URL pattern: `https://www.football-data.co.uk/mmz4281/{season}/SP1.csv`
- League file: Spanish La Liga, `SP1`.
- Used for: final scores, shots, shots on target, corners, fouls, cards, and betting market odds.
- Validation: each refresh writes `artifacts/multi_season/data_source_report.json` with row counts, date ranges, missing required columns, duplicate fixtures, result row counts, market-column presence, and closing-odds coverage.

Football-Data is the primary source because its CSVs are stable, historical, machine-readable, and include both match statistics and odds aligned to fixtures.

Pinnacle/PS direct columns are intentionally excluded from market-consensus features because Football-Data warns that Pinnacle public API delivery became unreliable from July 2025. The pipeline uses consensus/maximum odds plus Bet365 columns instead.

### ClubElo

- URL: `http://api.clubelo.com`
- Status: approved as an optional future source.
- Intended use: external pre-match club-strength ratings.
- Integration note: keep it behind a team-name mapping layer before using it in training. The current pipeline already maintains an internal Elo state from prior match results to avoid provider name drift.

## Leakage Rules

- Match results, shots, cards, corners, and Elo updates are added to team state only after the feature row for that match is created.
- Season table position is computed from points before the match.
- Head-to-head features use only previous meetings.
- Rolling form windows use only prior matches.
- Market odds are treated as pre-match information. Closing odds are useful for backtesting near kickoff; if predictions must be made days earlier, use the same feature contract with unavailable market columns imputed or switch to opening odds only.

## Feature Families

- Long-run team strength: points per match, goal rates, shot rates, corner/card rates, win/draw/loss rates.
- Current-season state: season points per match, goal difference per match, shot efficiency, conversion, and prior rank.
- Recent form: 3, 5, and 10-match windows for points, goal differential, shots on target, and corners.
- Fixture context: home/away split, rest days, matchday, promoted-team proxy.
- Ratings: internal Elo before the match, away Elo, and Elo difference with home advantage.
- Matchup history: prior head-to-head points, goal differential, and sample size.
- Market consensus: normalized home/draw/away implied probabilities, overround, entropy, source count, and over/under 2.5 probabilities.
