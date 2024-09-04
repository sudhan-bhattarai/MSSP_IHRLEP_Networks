# Journal paper
*S. Bhattarai and Y. Song, "Multi-stage Stochastic Programming for Integrated Network Optimization in Hurricane Relief Logistics and Evacuation Planning", Accepted for Publication at Networks, 2024.*

# Dependencies

This project requires the following Python libraries:

- pandas
- numpy
- itertools
- matplotlib
- geopandas
- pgeocode
- shapely
- contextily
- geopy
- sklearn
- gurobipy

# General Workflow

1. **Data Collection and Network Setup**: Gather initial data and establish logistics network parameters for all instances. Important files associated with this module are *data_initial_inputs.py* and *data_process_network_data.py*.
   
2. **Forecast Error Analysis**: Analyze forecast error data to estimate Autoregressive model parameters for each error type (track, along, cross, intensity). An important file associated with this module is *data_forecast_error_data_analysis.py*.

3. **Forecast Error Scenario Generation**: Use the estimated Autoregressive model parameters to create forecast error scenarios. Construct a Markov Chain model for track, along, and cross errors using discretized error samples and transition probabilities. An important file associated with this module is *data_forecast_error_scenarios.py*.

4. **Demand Estimation and Scenario Mapping**: Map the forecast error scenarios to estimate demand for out-of-sample scenarios, in-sample scenarios for two-stage models, and the discretized Markov Chain model. Important files associated with this module are *data_demand_estimation.py*, and *data_main.py*.

5. **Solving the models**: Solve an optimization model using the data for a specific instance and configuration. Important files associated with this module are *sddp.py*, and *two_stage.py*.

# Commands
Creating instances or solving the models can be done by giving appropriate commands to main.py. A brief details of the meaning of arguments can be found with: *python main.py --help*

## Task: create data
To create an instance: *python main.py --task create_data* should be used. Some examples are:
- *python main.py --task create_data --data_opt 1 --n_oos 1000 --ST_track 5*: create all data related to forecast error scenarios with 1000 out-of-samples, and MC models for forecast errors. The number of MC states used at the last stage is 5.
- *python main.py --task create_data --data_opt 2 --hurricane Ian --instance 3 --n_oos 100*: create all data related to demand estimation of hurricane Ian for instance 3. Map first 100 out-of-sample error scenarios to demand.
- *python main.py --task create_data --data_opt 3 --hurricane Florence --instance 1*: create all data related to logistics parameters of hurricane Florence for instance 1.
- *python main.py --task create_data --data_opt 4 --hurricane Florence --instance 1 --n_oos 1000 --ST_track 15*: create all data required to solve all models, starting from forecast errors (will be overridden for different instances picked), for instance 1 of hurricane Florence.

## Task: solve models
To solve models: *python main.py --task solve* should be used. Some examples are:
- *python main.py --task solve --hurricane Florence --model mssp --method bb --instance 3 --eval both --gfact 200 --pfact 300 --ffact 5 --purchase_cost 5 --time_limit_train 21600 --time_limit_test 3600 --n_oos 1000 --n_UB_samples 10000 --oos_heur 1 --first_stg_opt 1 --delay 2*: using naive branch and bound (without lazy constraints) to solve MSSP for instance 3 (I=J=10) of hurricane Florence.
  - *--eval both*: after solving the MSSP model, get out-of-sample cost on (1) test samples from the MC model and (2) true OOS samples from AR-1 models.
  - *--gfact 200 --pfact 300 --ffact 5 --purchase_cost 5*: cost configurations
  - *--time_limit_train 21600 --time_limit_test 3600*: time limit to run SDDP and to do out-of-sample testing for --eval option, respectively.
  - *--n_oos 1000*: 1000 out-of-samples to test the solution on.
  - *--n_UB_samples 10000*: 10000 random samples used to compute statistical upper bound.
  - *--oos_heur 1*: the heuristic used to conduct the true OOS test. Not applicable if --eval is mc_tree.
  - *--first_stg_opt 1*: the first-stage problem is MILP, i.e., not all SPs are open at the start.
  - *--delay 2*: delayed opening of SPs is not allowed.

- *python main.py --task solve --hurricane Ian --model 2ssp --method bc --instance 3 --eval oos --gfact 200 --pfact 300 --ffact 5 --purchase_cost 5 --time_limit_train 21600 --time_limit_test 3600 --n_oos 1000 --first_stg_opt 2*: using branch and bound with lazy cuts to solve 2SSP for instance 3 (I=J=10) of hurricane Ian.
  - *--eval oos*: solve 2SSP using in-samples from AR-1 model and get out-of-sample cost on the true OOS samples from AR-1 models.
  - *--first_stg_opt 2*: the first-stage problem is continuous, i.e., all SPs are open at the start.

# File Organization

- All Python scripts are located in the parent directory.
- The input data for the optimization model is organized in the `Data/` directory.

## Data Directory

- **problem_size_opt.csv**: Describes the problem size with the number of demand points (I) and shelter points (J).
- **us_GIS.json**: Contains the geometry of the US map and state boundaries.
- **numeric_inputs.csv**: Includes user-defined numeric inputs for various modules. Key variables include:
  - `landfall_tol`: Landfall zone threshold in miles from the coastline.
  - `x_max`: Maximum distance from the hurricane position to observe demand.
  - `y_max`: Maximum cross-directional distance in the study region for the Florence case.
  - `GFact`, `PFact`, `HFact`, `FFact`, `FVarFact`: Cost factors for emergency, penalty, holding, and fixed costs, respectively.
  - `phi`: Number of relief items per evacuee per period.
  - `alpha`, `beta`: Transportation cost factors for relief items and evacuee transport.
  - `INVE_FACT`, `INVR_FACT`: Cost factors for evacuee and relief item inventories.
  - `cut_tol`: Threshold for cut violation.
  - `n_itr_lb_rate`: Iterations to compute the lower bound improvement rate.
  - `lb_tol`: Lower bound improvement threshold to stop SDDP.
  - `Xmiles`, `Ymiles`: Miles per degree of longitude and latitude.
  - `DP_DIST_TOL`: Distance threshold from the coastline for demand zones in the Florence case.
  - `n_realization`: Number of AR-1 error model realizations per period.
  - `S_T_intensity`, `S_T_track`, `S_T_along`, `S_T_cross`: Discretization states for intensity, track, along, and cross errors.
  - `S`: Number of scenarios in the two-stage stochastic programming model.
  - `cat_5_speed`: Wind speed for a Category 5 hurricane.
  - `n_oos`: Number of out-of-sample scenarios.
  - `T`, `T_max`: Planning horizon periods for deterministic and random landfall cases.
  - `max_itr`, `max_itr_sddp_rerun`: Maximum iterations for SDDP and Benders, and rerun iterations for SDDP.
  - `n_UB_samples`: Number of sample paths for computing statistical upper bound.

### Forecast Error Data (`/Data/Forecast error/`)

Contains data related to forecast errors, starting with AR-1 parameter estimation:

- **12hr_avg**: Average 12-hour forecast error from the historical database.
- **correlation_along**: Correlation coefficients for along error between periods.
- **eps_grid.json**: Epsilon grid with 100 realizations per period for all errors.
- **oos_errors_along**: Out-of-sample along errors sampled using the AR-1 model.
- **ST_along**: Number of Markov Chain states for along error.
- **transition_prob_along_t0**: Transition probability matrix for along error at t=0.

### Florence Case Study Data (`/Data/Florence/`)

Contains data specific to the Florence case study:

- **DP_all_ZIPs**: ZIP codes with population and location data in the risk zone.

#### Instance-Specific Data (`/Data/Florence/Instance_I{I}_J{J}/`)

Instance-specific data for various combinations of demand points (I) and shelter points (J):

- **DP**: Demand points for a given instance (e.g., I = 3).
- **SP**: Shelter points for a given instance (e.g., J = 3).
- **c_{}**: Logistics cost data.
- **cat_scen**: Out-of-sample hurricane category scenarios.
- **ST**: Number of Markov Chain states per period.
- **DF_MSSP_t10**: Demand factors at t=10 for all Markov Chain states.
- **PI_MSSP_t9**: Transition probability matrix at t=9.
- **SAMPLES_OOS_MC_LATTICE**: Samples for testing MSSP and two-stage models.
- **oos_demand_t10**: Out-of-sample demand data at t=10.
- **DF_OOS_MC_LATTICE_t10**: Demand factors for out-of-sample scenarios.
- **p**: Probability for all scenarios in S.

## Results Directory

The results are stored at *~/Results/{--hurricane}/instance{--instance}/ff{--ffact}_gf{--gfact}_pf{--pfact}_pcost{--pcost}/file_name.ext* for the respective commands.
- algorithm solutions are saved by using the method names: bb, bc, sddp etc.
- the test sample costs are named as 'eval'.

# Miscellaneous Notes
- File *helper.py* includes functions that are rather repetative and of general purpose. It is imported at different modules as per needed
- Files *results_analysis.py* and *plot_gis.py* are used to create summary of the results and plots after getting the results. These files are executed rather individually as they are not incorporated in *main.py*.
- File *commands.py* is used to create argument inputs to *main.py* for data creation and solving the models. File *commands_defaults.csv* has the default values of arguments on *commands.py*.
- command --landfall is useless when --hurricane is specified. It is only used to indicate the landfall characterization of the chosen case study.

# Contact
Please email your questions or comments to *sudhanb@clemson.edu*
