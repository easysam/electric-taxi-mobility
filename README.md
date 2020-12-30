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
### 2.1 Utils
#### 2.1.1 coord Transform

```bash
python ./coord_transform/coord_converter.py -i ./coord_transform/charging_station_bd.csv -o ./coord_transform/charging_station_wsg.csv -t b2g -n lng -a lat
```

the result is in the coord_transform file folder, we transform BD-09  coordinate to WGS-84  coordinate 

#### 2.1.2 python crawler

run spider/XXX-APP/extract_station_meta.py

We use python crawler to obtain charging station information on mobile phone app, the information for the charging station we want is stored in a folder named station_meta, and the web files required to extract information are stored in a folder called chargeApp.

## 3 Documentation
### 3.1 Variable meaning

`o_t`, `o_l`: timestamp and pick-up hotspot label for origin of transaction (OD).

`d_t`, `d_l`: timestamp and drop-off hotspot label for destination of transaction (OD).

`source_d_t`, `source_d_l`: timestamp and drop-off hotspot label for destination of last transaction of charging event (CE).

### 3.2 Codes
#### 3.2.1 s1_preprocessing
##### `transactions_processing_14.py`
Cluster pick-up and drop-off events for taxi in 2017.4

Output
1. data/od/fod_w_14f_hs.csv: full od with 14 taxi hotspot label
2. data/hotspot/f14p.list_of_dict: full pick-up hotspots for 14 taxi
3. data/hotspot/f14d.list_of_dict: full drop-off hotspots for 14 taxi
##### `transactions_processing_17.py`
Cluster pick-up and drop-off events for taxi in 2017.6
#### 3.2.2 s2_mobility
#### 3.2.3 s3_generation
