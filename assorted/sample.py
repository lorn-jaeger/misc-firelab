import pandas as pd
import random

# Load the original CSV
df = pd.read_csv("fires.csv")

# Check we have at least 200 entries
if len(df) < 200:
    raise ValueError("Not enough fires to sample 200 entries.")

# Randomly sample 200 fires
sampled_df = df.sample(n=200, random_state=42).reset_index(drop=True)

# Write the result to final.csv
sampled_df.to_csv("final.csv", index=False)

print("final.csv written with 200 random fires.")
