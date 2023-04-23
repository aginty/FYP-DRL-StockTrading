
## FYP - DRL for Stock Trading

The file structure of this repository is as follows:

(Note the FinRL & stablebaselines3 are large codebases. Only files to which
modifications were made are detailed in the file structure below)

    1. DRL
        |- finrl
            |- agents
                |- stablebaselines3
                    |- models.py
            |- applications
            |- meta
                |- env_stock_trading.py
        |- stable_baselines3
            |- ppo
                |- policies.py
                |- ppo.py
            |- common
                |- torch_layers.py
                |- policies.py
                |- on_policy_algorithm.py
        |- train.py
        |- trade.py
        |- models

    2. Forecast-Decision-Rule
        |- stock_price_forecasting
        |- decision_rule_trading
        |- models

    3. Datasets
    4. Results


* All datasets used for training and testing are available as .csv files in `Datasets`
* All models for which results are presented in the report are saved in `DRL/models` and `Forecast-Decision-Rule/models` respectively
* Results for all models are available as .csv files in `Results`
  There are 4 csv results files for each model

	*  performance: details portfolio evalutaion metrics

	*  account_balance: details changes in account balance during the testing period

	*  raw_actions: the actions requested by the agent

	*  df_actions: the actions executed in the environment 


Training a DRL agent:

```
./DRL/train.py -ticker="IBM" -train_file="Datasets/IBM_train.csv" -trade_file="Datasets/IBM_trade.csv" 
-model_version="v1" -lr=0.001 -timesteps=500000 -ts=1 -scale=1 -ts_fe=0
```

The above code will train an agent on the IBM stock data. The parameters ts, scale, ts_fe are boolean values which can be set to 0 (False) or 1 (True).

* ts: represent the state of the environment as a univariate time series (True/False)
* scale: scale the features that represent the environment (True/False)
* ts_fe: use a time series feature extractor in the policy and value function approximators (True/False)

Note: All models included in this repository were trained using a learning rate of 0.001 and 500000 timesteps.




        

