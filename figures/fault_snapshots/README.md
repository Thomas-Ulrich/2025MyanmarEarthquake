# Fault Snapshot Generation

## Steps to generate snapshots:

1. Extract parameter files from the yaml file:
   ```bash
   python SeisSol/preprocessing/science/read_ini_fault_parameter.py output/fl33-fault.xdmf yaml_files/fault_0010_coh0.25_1_B0.9_C0.2_R0.8.yaml --par "mu_s,d_c,T_s,T_d" --output fault_parameters
   ```

2. Extract only the last time step (optional):
   ```bash
   seissol_output_extractor output/dyn_0010_coh0.25_1_B0.9_C0.2_R0.8-fault.xdmf --var ASl RT Vr PSR --time "i-1" --add last_myanmar
   ```

3. Generate the plot:
   ```bash
   ./plot_fault_snapshot.sh ../../../dyn_0010_coh0.25_1_B0.9_C0.2_R0.8last_myanmar-fault.xdmf ../../../fault10_parameters.xdmf param
   ```
