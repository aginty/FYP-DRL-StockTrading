
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
        1. performance: details portfolio evalutaion metrics

        2. account_balance: details changes in account balance during the testing period

        3. raw_actions: the actions requested by the agent

        4. df_actions: the actions executed in the environment 




        

