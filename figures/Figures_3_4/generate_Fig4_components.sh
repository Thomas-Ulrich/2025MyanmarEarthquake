#!/bin/bash
# offsets
set -xe
best=0080
python ../2025MyanmarEarthquake/figures/figure_offsets/compare_offsets.py extracted_output/dyn_0 offsets.csv $best
#redyn metrics fault-offsets extracted_output/dyn_0 --bestmodel 0080 Myanmar_sentinel2_offset_latlon_v2.csv

# moment rate
ln -sf ../2025MyanmarEarthquake/figures/moment_rate/MomentRateObs/ .
python ../2025MyanmarEarthquake/figures/moment_rate/plot_moment_rate.py extracted_output --best $best 

#NPW
python ../2025MyanmarEarthquake/figures/figure_NPW/plot_NPW_comparison.py ../2025MyanmarEarthquake/figures/figure_NPW/strong_motion_data/ extracted_output $best

#CCTV
python ../2025MyanmarEarthquake/figures/CCTV_analysis/compare_CCTV.py extracted_output/dyn_0 --best $best --align --plot

#CCTV strike-slip vs dip slip
python ../2025MyanmarEarthquake/figures/CCTV_analysis/compare_CCTV.py extracted_output/dyn_$best --align --strike_vs

mkdir -p figures_for_fig4
mv figures/comparison_offset_all_models.svg figures_for_fig4
mv figures/moment_rate.svg figures_for_fig4
mv figures/NPW_comparison.svg figures_for_fig4
mv figures/CCTV_comparison.svg figures_for_fig4
mv CCTV_comparison_strike_vs_dip_slip.svg figures_for_fig4

realpath figures_for_fig4
