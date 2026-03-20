# Dynamic Rupture Workflow for the Mw7.8 2025 Myanmar Earthquake

This guide outlines the steps for running the dynamic rupture workflow to
 characterize the rupture dynamics of the Mw7.8 2025 Myanmar earthquake

---

## 1. Installing the Workflow

Install the `rapid-earthquake-dynamics` package at version `v0.4.2`:

```bash
pip install git+https://github.com/Thomas-Ulrich/rapid-earthquake-dynamics.git@v0.4.2
```

## 3. Prepare the Earthquake Scenario

Run the setup script to retrieve earthquake information and generate the files
 needed to construct an ensemble of dynamic rupture models from a kinematic
finite fault model:

```bash
redyn init --config input_config_myanmar_m11.yaml
```

This will create a directory containing all necessary input files.

## 4. Run the Workflow on HPC (e.g., superNG)

Copy the generated folder to your HPC system (e.g., superNG) and execute the
 workflow using:

```bash
git clone --branch v0.4.2 https://github.com/Thomas-Ulrich/rapid-earthquake-dynamics.git
sh rapid-earthquake-dynamics/run_full_workflow_supercomputer.sh
```

This script will launch a sequence of srun jobs to:

1. Compute the stress change from the finite fault model

2. Generate input files for the ensemble of dynamic rupture models

3. Run all dynamic rupture simulations

4. Validate the models against available observations
