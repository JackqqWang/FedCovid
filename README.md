# Code for the Federated COVID-19 Vaccine Side Effect Prediction

We implemented the code of FedCovid.

## Files

The "data" folder contains sample training and testing data. The data sample data has been added noise due to the data sharing policy of IQVIA

The main file is __federated_learning.py__. 
Backbones are in __neural_nets.py__.
The paremater setting and adjustment is __option.py__.

## Requirements

- Python 3.9.7

- Pytorch 1.9.1

- Pytorch-metric-learning 1.2.1

## Parameters

- define_loss = ours, our defined loss

- warmup_round = 200, different values for hyperparameter study

- active_rate = 0.4, active client ratio for each communication round

- weight_list, weights before each compoment of the designed loss

- lr_stra = 1, enable the adaptive learning rate strategy

- backbone: CNN, LSTM, Transformer, biLSTM

- aggregation: weighted_mean, mean

- strategy: central_to_all, all_state

## How to run

Note: This code is written in Python3.
```
python federated_learning.py
```