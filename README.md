# electric-taxi-mobility
Modeling the mobility of electric taxi.

## 1. Getting Start
### 1.1 Dependency
The project uses `Python>=3.7`

You need install following python package: `pandas`, `numpy`, `sklearn`, `xgboost`, `pyyaml`.
### 1.2 Usage
Preliminary: you need set the "project_path" in "config.yaml" as project root path.
#### 1.2.1 Mobility Modeling & Data Generation
**Stage 1** preprocessing
1. to extract charging event, from raw trajectories and charging station info, run 
`s1_preprocessing/charging_event_extraction/charging_event_extraction.py`
2. to cluster pick-up or drop-off hotspots, run `...`
3. to conduct the GPS trajectory map matching, please refer to the `easysam/IVMM` (search it in GitHub) project.

**Stage 2** mobility modeling
1. to update transition model, run `s2_mobility/transition.py`.
2. to update whether to charge model, run `s2_mobility/charging_behavior/whether_to_charge_data_preparation.py` and 
`s2_mobility/charging_behavior/whether_to_charge.py`, in turn.
3. to update where to charge model, run `s2_mobility/charging_behavior/where_to_charge_data_preparation.py` and 
`s2_mobility/charging_behavior/where_to_charge.py`, in turn.
4. to update rest pattern, run `s2_mobility/rest_pattern.py`.

**Stage 3** data generation
1. `generation.py`

#### 1.2.2 Other Purpose


## 2. For developer
### 2.1 Variable meaning

`o_t`, `o_l`: timestamp and pick-up hotspot label for origin of transaction (OD).

`d_t`, `d_l`: timestamp and drop-off hotspot label for destination of transaction (OD).

`source_d_t`, `source_d_l`: timestamp and drop-off hotspot label for destination of last transaction of charging event (CE).