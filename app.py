import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load model, transformer, and allowed input values
model = joblib.load("ipl_model.pkl")
transformer = joblib.load("transformer.pkl")
allowed_df = pd.read_csv("model_input_template.csv")

# Unique dropdown values
batting_teams = sorted(allowed_df['batting_team'].unique())
bowling_teams = sorted(allowed_df['bowling_team'].unique())
venues = sorted(allowed_df['venue'].unique())
seasons = sorted(allowed_df['season'].unique())

# UI
st.set_page_config(page_title="IPL Win Predictor", layout="wide")
st.title("🏏 IPL 2nd Innings Win Predictor")
st.markdown("### Predict the winning probability based on current match stats.")

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Batting Team", batting_teams)
    bowling_team = st.selectbox("Bowling Team", [team for team in bowling_teams if team != batting_team])
    venue = st.selectbox("Venue", venues)
    season = st.selectbox("Season", seasons)

with col2:
    runs_left = st.number_input("Runs Left", min_value=0)
    balls_left = st.number_input("Balls Left", min_value=1, max_value=120)
    wickets_left = st.number_input("Wickets Left", min_value=0, max_value=10)
    total_runs = st.number_input("Runs Scored So Far", min_value=0)

# Calculate derived metrics
if balls_left > 0:
    overs_done = (120 - balls_left) / 6
    run_rate = total_runs / overs_done if overs_done > 0 else 0
    required_run_rate = (runs_left * 6) / balls_left
else:
    st.warning("Balls left must be greater than 0 to calculate prediction.")
    st.stop()

# Prediction button
if st.button("Predict Win Probability"):
    input_df = pd.DataFrame([{
        'batting_team': batting_team,
        'bowling_team': bowling_team,
        'venue': venue,
        'season': season,
        'runs_left': runs_left,
        'balls_left': balls_left,
        'wickets_left': wickets_left,
        'total_runs': total_runs,
        'run_rate': run_rate,
        'required_run_rate': required_run_rate
    }])

    encoded_input = transformer.transform(input_df)
    win_prob = model.predict_proba(encoded_input)[0][1] * 100
    lose_prob = 100 - win_prob

    # Plotly chart
    fig = go.Figure(go.Bar(
        x=[win_prob, lose_prob],
        y=[batting_team, bowling_team],
        orientation='h',
        marker=dict(
            color=['#00cc96', '#ef553b']
        ),
        text=[f"{win_prob:.2f}%", f"{lose_prob:.2f}%"],
        textposition='outside'
    ))
    fig.update_layout(title="🏆 Win Probability", xaxis_title="Probability (%)", yaxis_title="Team", height=400)

    st.plotly_chart(fig, use_container_width=True)
