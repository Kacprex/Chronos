import pandas as pd
df = pd.read_csv(r"c:\Users\kacpe\Documents\chronos\data\GM\gm_games.csv", nrows=5)
print(df.columns.tolist())
print(df["Result"].head())
print(df["pgn"].iloc[0][:200])
