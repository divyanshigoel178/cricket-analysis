# 📁 model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import joblib

# STEP 1: Load Data
matches = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")

# STEP 2: Clean Data
matches = matches[(matches['dl_applied'] == 0) & 
                  (matches['result'] == 'normal') & 
                  (matches['winner'].notnull())]

valid_match_ids = matches['id'].unique()
deliveries = deliveries[(deliveries['match_id'].isin(valid_match_ids)) & 
                        (deliveries['inning'] == 2)]

# STEP 3: Merge and Sort
merged_df = deliveries.merge(matches[['id', 'team1', 'team2', 'winner', 'venue', 'Season']],
                             left_on='match_id', right_on='id')
merged_df = merged_df.sort_values(by=['match_id', 'over', 'ball'])

# STEP 4: Feature Engineering
processed_data = []
for match_id, match_data in merged_df.groupby('match_id'):
    target = match_data['total_runs'].sum()
    chasing_team = match_data['batting_team'].iloc[0]
    bowling_team = match_data['bowling_team'].iloc[0]
    winner = match_data['winner'].iloc[0]
    venue = match_data['venue'].iloc[0]
    season = str(match_data['Season'].iloc[0])  # Convert season to string

    current_runs = 0
    wickets = 0

    for i, row in match_data.iterrows():
        current_runs += row['total_runs']
        if pd.notna(row['player_dismissed']):
            wickets += 1

        balls_bowled = ((row['over'] - 1) * 6 + row['ball'])
        balls_left = 120 - balls_bowled
        runs_left = target - current_runs
        wickets_left = 10 - wickets
        overs_done = balls_bowled / 6
        run_rate = current_runs / overs_done if overs_done > 0 else 0
        required_run_rate = (runs_left * 6 / balls_left) if balls_left > 0 else 0

        result = 1 if chasing_team == winner else 0

        processed_data.append({
            'match_id': match_id,
            'batting_team': chasing_team,
            'bowling_team': bowling_team,
            'venue': venue,
            'season': season,
            'runs_left': runs_left,
            'balls_left': balls_left,
            'wickets_left': wickets_left,
            'total_runs': current_runs,
            'run_rate': run_rate,
            'required_run_rate': required_run_rate,
            'result': result
        })

model_data = pd.DataFrame(processed_data)


# STEP 5: Encode Categorical Columns
X = model_data.drop(columns=['result', 'match_id'])
y = model_data['result']
categorical_cols = ['batting_team', 'bowling_team', 'venue', 'season']

ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

X_encoded = ct.fit_transform(X)

# STEP 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# STEP 7: Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

model_data.to_csv("model_data.csv", index=False)


# Save model and transformer
joblib.dump(model, 'ipl_model.pkl')
joblib.dump(ct, 'transformer.pkl')

# Save categories used for dropdown restriction
allowed_values = model_data[['batting_team', 'bowling_team', 'venue', 'season']].drop_duplicates()
allowed_values.to_csv("model_input_template.csv", index=False)
