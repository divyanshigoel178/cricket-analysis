# ✅ save_template.py
import pandas as pd

# Load full processed model data from training (same features used in Streamlit)
model_data = pd.read_csv("model_data.csv")  # You need to save this in model_training.py

# Extract allowed values for dropdowns
allowed_values = model_data[['batting_team', 'bowling_team', 'venue', 'season']].drop_duplicates()

# Save to template for UI dropdowns
allowed_values.to_csv("model_input_template.csv", index=False)
print("✅ Dropdown template saved successfully.")
