# make_public_full.py
import json
from pathlib import Path

import pandas as pd

in_path = "dist_v1/merged_final.parquet"
pub_out = "dist_v1/merged_public.parquet"
full_out = "dist_v1/merged_full.parquet"

df = pd.read_parquet(in_path)


# parse list-like strings back to lists if needed
def as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.strip().startswith("["):
        try:
            return json.loads(x)
        except:
            return []
    return [] if pd.isna(x) or x is None else [str(x)]


for c in ["emotions", "themes", "themes_50"]:
    if c in df.columns:
        df[c] = df[c].map(as_list)

# full copy (keep everything)
Path(full_out).parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(full_out, index=False)

# public: drop poem text and long interpretation to be license-safe
keep_pub = [
    "split",
    "poem_id",
    "author",
    "title",
    "source",
    "emotions",
    "primary_emotion",
    "sentiment",
    "themes",
    "themes_50",
]
df_pub = df[[c for c in keep_pub if c in df.columns]].copy()
df_pub.to_parquet(pub_out, index=False)

print("public rows:", len(df_pub), "full rows:", len(df))
