# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# A exécuter pour créer substrat.csv

# %%
import pandas as pd
df1 = pd.read_csv("substrat_part1.csv")
df2 = pd.read_csv("substrat_part2.csv")
df = pd.concat([df1, df2])
df.set_index("Unnamed: 0", inplace=True)
df.index.name = None
df.to_csv("../substrat.csv")

# %% [markdown]
# A exécuter pour substrat1.csv et substrat2.csv

# %%
import pandas as pd
df = pd.read_csv("../substrat.csv").drop(columns=["Unnamed: 0"])
mask = df.index < len(df) / 2
df1 = df[mask]
df2 = df[~mask]
df1.to_csv("substrat_part1.csv")
df2.to_csv("substrat_part2.csv")
