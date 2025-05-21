import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser(
    description="downweight waveforms in WASP inversion based on misfit"
)
parser.add_argument(
    "--type",
    choices=("surf", "body"),
    help="type of waveform to process",
    required=True,
    type=str,
)
args = parser.parse_args()


if args.type == "surf":
    fname = "surf_waves.json"
    comp_dict = {"SH": "L", "BHZ": "R"}
elif args.type == "body":
    components = ["P", "SH"]
    fname = "tele_waves.json"
    comp_dict = {"SH": "SH", "BHZ": "P"}

df = pd.read_csv("misfit_details.txt", sep="\s+")
df["w_misfit"] = df["weight"] * df["misfit"]
df_sorted = df.sort_values(by="w_misfit", ascending=False)
print(df_sorted)
df_sorted.to_csv("misfit_sorted.csv", index=False)


df = df[df["component"].isin(comp_dict.values())]
percentile_50 = df["misfit"].quantile(0.50)

threshold = 1.6
n_downweight = (df["misfit"] > threshold * percentile_50).sum()
print("signals to be downweighted: ", n_downweight)
print(f"out of {len(df)} ({n_downweight/len(df)*100:.1f}%)")
if n_downweight:
    with open(fname) as f:
        surf_wave = json.load(f)

    for entry in surf_wave:
        station_name = entry["name"]
        component = comp_dict[entry["component"]]
        row = df[(df["sta_name"] == station_name) & (df["component"] == component)]
        if row["misfit"].values[0] > threshold * percentile_50:
            entry["trace_weight"] = 0.0
            print(row)

    with open(fname, "w") as f:
        json.dump(surf_wave, f, indent=4)
    print(f"done update {fname}")
