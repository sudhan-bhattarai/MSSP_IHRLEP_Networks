
# Libraries (excluding in-built)

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

## Data description
Data folder has all input data. Results folder has all results.
Data folder has the following files and folders:

### file: numeric_inputs.csv 
It has all numeric inputs (user defined) given to different modules of the 
code. The following is the description of each variable:

- landfall_tol: Number of miles from the coastline in the sea 
that is a landfall zone
- x_max: Maximum distance from hurricane position to observe some demand.
Demand = 0 for DPs whose distance from hurricane position is more than x_max.
- y_max: Maximum distance in cross- direction of study region from two
extremes of coast line in Florence case study.
- GFact: Emergency cost factor by which $c^G_j$ is determined 
- PFact: Penalty cost factor by which $c^{PE}_j$ is determined 
- HFact: Cost factor by which $c^H_j$ is determined
- FFact: Cost factor by which $c^F_j$ is determined
- FVarFact: Cost factor by which $c^f_j$ is determined (fixed cost paid per 
period for active SPs)
- phi: Number of relief items (demand) per evacuee per period
- alpha: Distance to transportation cost factor for relief items
- beta: Distance to transportation cost factor for evacuee transportation
- INVE_FACT: Cost factor by which $c^{invE}_j$ is determined
- INVR_FACT: Cost factor by which $c^{invR}_j$ is determined
- cut_tol: Threshold value of cut violation to add cuts
- n_itr_lb_rate: Number of past iterations to look at, to compute LB improve
rate 
- lb_tol: Threshold value of LB improvement over n_itr_lb_rate iterations 
(to stop SDDP if threshold is exceeded)
- Xmiles: Number of miles per degree of longitude
- Ymiles: Number of miles per degree of latitude
- DP_DIST_TOL: Distance from coastline upto which all ZIP codes are considered
in demand zone (only for Florence)
- n_realization: Number of realizations of $\epsilon_t$ recombined tree
 per period of $AR-1$ error models
- S_T_intensity: Number of states for intensity error (SS scale of hurricane
category (1-5) + 0)
- S_T_track: Number of track discretization states to use at $T$
- S_T_along: Number of along error discretization states to use at $T_{max}$
- S_T_cross: Number of cross error discretization states to use at $T_{max}$
- S: Number of scenarios in 2SSP model
- cat_5_speed: Wind Speed of a category 5 Hurricane
- n_oos: Number of Out-of-sample scenarios to generate/read
- T: Number of periods in the planning horizon in Determinstic landfall case
- T_max: Number of periods in the planning horizon in Random landfall case 
(Maximum number of periods for all samples, models, case studies)
- max_itr: Maximum number of iterations to run SDDP/Benders
- max_itr_sddp_rerun: Number of iterations to rerun SDDP for the same 
first-stage solution if a cut is not violated (used in callback method)
- n_UB_samples: Number of sample paths to use in computing statistical UB

### file: problem_size_opt.csv
A specific instance refers to a number of DPs and SPs. Run the model for the
given instance.

### file: us_GIS.json
Geometry of US map and states

#### main.py commands


usage: main.py [-h] [--task {solve,data}] [--hurricane {Florence,Ian}]
                   [--data_opt {1,2,3}] [--model {2ssp,rh,mssp}]
                   [--delay {1,2}] [--method {bb,bc,ext}]
                   [--instance {1,2,3,4,5}] [--eval {oos,mc_tree,both}]
                   [--first_stg_opt {1,2}] [--scen SCEN] [--gfact GFACT]
                   [--pfact PFACT] [--ffact FFACT] [--irfact IRFACT]
                   [--iefact IEFACT] [--timelimit TIMELIMIT]
                   [--maxItrSDDP MAXITRSDDP] [--n_UB_samples N_UB_SAMPLES]
                   [--uboption UBOPTION] [--n_oos N_OOS]

optional arguments:
  -h, --help            show this help message and exit
  --task {solve,data}   solve: to solve models; data: to create instances
                        (default: solve)
  --hurricane {Florence,Ian}
                        Name of storm (Florence or Ian) (default: Florence)
  --data_opt {1,2,3}    1 if creating forecast error data as well; 2 if only
                        creating other data3 if only creating logistics
                        parameters but not demand (default: 2)
  --model {2ssp,rh,mssp}
                        Model to solve (2ssp, rh, or mssp) (default: mssp)
  --delay {1,2}         1 if delayed facilities opening allowed in the first-
                        stage MILP; 2 if only allowed to open at t=0 (default:
                        1)
  --method {bb,bc,ext}  Algorithm: bb (branch & bound); bc (branch & cut); ext
                        (extended model (2ssp only)) (default: bb)
  --instance {1,2,3,4,5}
                        Instance index (default: 1)
  --eval {oos,mc_tree,both}
                        Type of sample paths (OOS or from MC tree) to evaluate
                        models on (default: oos)
  --first_stg_opt {1,2}
                        1 if the first-stage problem is a mixed-integer or,2
                        if all SPs are open at the beginning and the first-
                        stage problem is a continuous LP (default: 1)
  --scen SCEN           Number of in-sample scenarios fo SLAM (default: 100)
  --gfact GFACT         Emergency cost factor (default: 5)
  --pfact PFACT         Penalty cost factor (default: 20)
  --ffact FFACT         Fixed cost factor (default: 2)
  --irfact IRFACT       Relief items inventory cost factor (default: 0.5)
  --iefact IEFACT       Evacuees inventory cost factor (default: 0.5)
  --timelimit TIMELIMIT
                        Time limit (in sec) for algorithms (default: 3600)
  --maxItrSDDP MAXITRSDDP
                        Max number of iterations to re-run SDDP (default: 500)
  --n_UB_samples N_UB_SAMPLES
                        Number of sample to use for statistical UB (default:
                        100)
  --uboption UBOPTION   Compute UB at every itr (opt 1) of BB or only at last
                        (opt 2) (default: 2)
  --n_oos N_OOS         Number of out-of-samples to evaluate the model on
                        (default: 100)

## folder: Forecast error
All data related to forecast errors. The first step is to analyze forecast
errors. The results of the "forecast error analysis" are AR-1 parameters
estimated for forecast errors: "track" (absolute great circle), "along", 
"cross" and "intensity", correlation, and plots. 

- 12hr_avg: The average error at 12h forecast (computed from historical 
forecast error database.)

- correlation_along: correlation coefficients of "along" error between 
different periods (from historical forecast error database)

- eps_grid.json: Epsilon grid created with 100 realizations per period for 
all errors. Epsilon is the white noise of AR-1 model.

- oos_errors_along: Out-of-sample errors sampled using AR-1 model and random
sampling of errors from eps_grid_along.

- ST_along: Number of MC states used for "along" error at every period

- transition_prob_along_t0: Transition probability matrix of "along" error at
t=0. Transition from discretized "along" error states at t=0 to states at t=1.


## Florence

- DP_all_ZIPs: Has all ZIP codes with respective population and location that
are in risk zone. We can cluster them together to get any I and J

### Folder Instance_I3_J3: 
All inputs to the model for |I| = 3 and |J| = 3

- DP: Demand points such that |I| = 3
- SP: Shelter Points such that |J| = 3
- c_{}: logistics cost

- cat_scen: Out-of-sample paths of hurricane categories that represent the
intensity. The wind speed is converted to categories.

- ST: Number of MC states at each period (ST = ST_intensity * ST_track)

- DF_MSSP_t10: Demand factor at t=10 for all MC states at t=10. 
-PI_MSSP_t9: Transition probability matrix at t=9
-SAMPLES_OOS_MC_LATTICE: Samples created from MC lattice to later use for 
testing MSSP and 2SSP models.

-oos_demand_t10: Out-of-sample demand data at t=10

-DF_OOS_MC_LATTICE_t10: Demand factors of SAMPLES_OOS_MC_LATTICE
-p: 1/|S| for all s \in S

### Folder Results_I3_J3: All results for Instance_I3_J3

#### Folder Ff0.8_Gf5_Pf10_IRf1.5_IEf1.5: Results associated with data when 
Fixed cost (F), Emergency cost (G), Penalty cost (P), Relief items inventory
cost (IR), Evacuee inventory (holding of evacuees) cost (IE) are augmented by
factors Ff=0.8, Gf=5, Pf=10, IRf=1.5, IEf=1.5, respectively.

- mssp_b&b : LB, LB change rate over last 100 iteration, computational time,
statistical UB (min, 25%, average, 75%, max) over 100 forward samples.

# Ian
- p_int: Transition probability matrix of intensity samples (in terms of 
hurricane category)
- forecast_slope_angles: Slope of forecast at every period. Need this to
add along- and cross- errors and create scenarios.

## Folder Instances: 
All parameters (demand and logistics) related to Ian case for a fixed I and J.
- test_samples_from_recombined_tree: Same as SAMPLES_OOS_MC_LATTICE in 
case of Florence but a different name.
- ts: Landfall period of out-of-samples 
- ETS_MSSP_t13: Expected lanndfall period of MC states at t=13

### Folder Results: 
Results associated with the Instance.

                      
Note: 
- To create instance, we only need --hurricane and --instance commandas
- --method = "bc" for --model="mssp" is incomplete
- Size of instances (1, 2, ...) can be found at /Data/User Input/problem_size.csv

### MSSP_1st_stage_MILP.py

Data : Python class (data structure that has all the inputes to the models)

class SDDP **kwargs:

#### Function: makeModels
Returns a dictionary.
Every _item_ of the dictionary has a _key_ as a tuple (t, s) where t is the period and s is the index of the MC state
at that period. At the first stage, define inetegr variables. At the last stage, set $\theta=0$. For $0<t<T$, define
initial models with continuous variables. The _value_ of every item is a python class **Model**. Class Model is defined for 
every node of the MSSP recombined tree which has the following attribites:
- Model.t: time period of the associated node in the recombined tree
- Model.s: MC state index of the associated node in the recombined tree at the respective t
- Model.m: Gurobi model for the node
- Model.vars: dictionary of the variables of Model.m
- Model.constrs: dictionary of the constraints of Model.m
- Model.constrs["cuts"]: list of cuts added to Model.m. Many constraints have initial values at RHS
which is updated later in forward and backward pass using frunction updateRHS
- Model.cuts_RHS: at any time, list of RHS of cutting planes added so far in SDDP 
(used to calculate the cuts using the dual information)

#### Function updateRHS
For a given state variable solution from the parent node, update the RHS of contraints of the
children nodes.

#### Function: generateCut
For the given model variables of a parent node, dual coefficients from a children node, RHS of 
the cuts added until the current iteration to the chilren node, returns the cutting plane.
The following flags are used:
- T_minus_1: if True, then the current node to generate the cut for is at T-1. Since we do not have
cuts at T and hence no RHS of cuts at T.
- option: if 1, use the forward pass solution to generate cut. Compute AX of the cutting plane and
later in backward pass compute the y-intercept as: $Z^* - A*\hat{X}$. Using option 1,
we can only add cut to the nodes that were smapled in the forward pass. If option = 2, generate cut
using the dual coefficients. Using option 2, we can generate cut for all nodes but we can only
check cut violation for the nodes sampled in the forward pass. For nodes not sampled in the forward pass,
we can not check violation and hence directly add the cuts.

#### Function: forwardPass
Flags:
- getUB: Is forward pass is used to get the statistical UB
- solve_root: True if we are solving model at t=0. It is set to False if for a cut generated
from backward pass at t=0 at an iteration is not violated. In such case, we want to run SDDP
from t=1 using the same solution at t=0 from the previous iteration.

#### Function: backwardPass
For some nodes, $\mathcal{Q}$ computed is zero. This is because the children nodes 
with positive transition probabilities from these nodes have zero demand and hence zero objective.

At t=0, if a cut is not violated, then run forward pass without solving the root node (t=0) and 
then the backward pass. Check at every iteration if the cut is violated at t=0. If not,
run SDDP for the same $\hat{X}_0$ until the maximum number of iterations is reached.


#### Data description
- demand_mssp: [t][(\xi_a, \xi_c, \xi_i)] = list(demand)