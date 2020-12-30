## Data Setting for Train and Validation
For training stage, we select od pairs with demand exceeding 10
, w.r.t only EV taxis in 2014.07, as the instances to build training data set.

For validation stage, we select od pairs with demand exceeding 10
,w.r.t all taxis in 2014.07, as the instances to build validation
data set.

## Method
### Utility Model
Predict a `EV Taxi willingness score vector` For O_i, using a 
utility model. Elements in the `EV taxi willingness score vector` 
correspond to the D_j in **D**. Element-wise multiply the 
`EV Taxi willingness score vector` and the corresponding
`taxi transition probability distribution`, followed by a
normalization function, e.g. `Softmax`, to produce the prediction
result of `EV taxi transition probability distribution`.

The utility model implement can be XGBoost or MLP.

#### XGBoost

#### MLP
### CNN
## Implement
### Utility Model
#### XGBoost
#### MLP
### CNN
