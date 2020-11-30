# electric-taxi-mobility
Modeling the mobility of electric taxi.

## 1. Getting Start
### 1.1 Dependency
You need install following python package: `pandas`, `numpy`, `pyyaml`.
### 1.2 Usage
#### 1.2.1 Mobility Modeling & Data Generation
1. you need set the "project_path" in "config.yaml" as project root path.

#### 1.2.2 Other Purpose


## 2. For developer
### 2.1 Variable meaning

`o_t`, `o_l`: timestamp and pick-up hotspot label for origin of transaction (OD).

`d_t`, `d_l`: timestamp and drop-off hotspot label for destination of transaction (OD).

`source_d_t`, `source_d_l`: timestamp and drop-off hotspot label for destination of last transaction of charging event (CE).