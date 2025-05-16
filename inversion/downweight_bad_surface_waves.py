import pandas as pd
import json

# Read the file, assuming it is space-separated and has variable-width columns
df = pd.read_csv("misfit_details.txt", sep="\s+")
df["w_misfit"] = df["weight"] * df["misfit"]

df = df[df["component"].isin(["L", "R"])]
percentile_80 = df["w_misfit"].quantile(0.80)
print("80th percentile w_misfit:", percentile_80)

df_sorted = df.sort_values(by="w_misfit", ascending=False)

# Load the JSON data
with open("surf_waves.json") as f:
    surf_wave = json.load(f)

# Optional: strip and upper-case names to avoid mismatches
misfit_map = df.set_index("sta_name")["w_misfit"].to_dict()

for entry in surf_wave:
    station_name = entry["name"]
    misfit = misfit_map.get(station_name)
    if misfit is not None and misfit > percentile_80:
        entry["trace_weight"] = 0.0

with open("surf_waves.json", "w") as f:
    json.dump(surf_wave, f, indent=4)


print(df_sorted)
df_sorted.to_csv("misfit_sorted.csv", index=False)
