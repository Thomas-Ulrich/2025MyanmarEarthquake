# offsets
python ../rapid-earthquake-dynamics/dynworkflow/compare_offset.py extracted_output/dyn_0 --bestmodel 0080 Myanmar_sentinel2_offset_latlon_v2.csv

# moment rate
ln -s ../2025MyanmarEarthquake/figures/moment_rate/MomentRateObs/ .
python ../2025MyanmarEarthquake/figures/moment_rate/plot_moment_rate.py extracted_output --plot_ensemble --best 0080

#NPW
python ../2025MyanmarEarthquake/figures/figure_NPW/plot_NPW_comparison_MM.py ../2025MyanmarEarthquake/figures/figure_NPW/strong_motion_data/ extracted_output 0080

#CCTV
python ../2025MyanmarEarthquake/figures/CCTV_analysis/compare_CCTV.py extracted_output/dyn_0 --best 0080 --align --plot

#CCTV strike-slip vs dip slip
python ../2025MyanmarEarthquake/figures/CCTV_analysis/compare_CCTV.py extracted_output/dyn_0080 --align --strike_vs
