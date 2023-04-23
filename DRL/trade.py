import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.meta.data_processor import DataProcessor

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint

import sys
sys.path.append("../FinRL")

import itertools

from finrl import config
from finrl import config_tickers
import os
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)
check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])


'''
TRAIN_START_DATE = "2021-01-01"
# TRAIN_END_DATE = '2019-12-31'
# TRADE_START_DATE = '2020-01-01'
TRADE_END_DATE = "2023-12-31"

df = YahooDownloader(start_date = TRAIN_START_DATE,
                     end_date = TRADE_END_DATE,
                     ticker_list = ["TWTR"]).fetch_data() 

'''

#V-data
df = pd.read_csv("halfV_vanilla.csv")
df = df.reset_index()


fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = INDICATORS,
                    use_vix=False,
                    use_turbulence=False,
                    user_defined_feature = False)

processed = fe.preprocess_data(df)


list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
combination = list(itertools.product(list_date,list_ticker))

processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date','tic'])

processed_full = processed_full.fillna(0)

trade = processed_full.reset_index()

stock_dimension = 1
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

#PPO

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000, 
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}


from stable_baselines3 import PPO
model = PPO.load("Models/halfV_simulated/PPO_v2_lr0-0001.zip", print_system_info=True)


'''
#TD3
env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000, 
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}


from stable_baselines3 import TD3
model = TD3.load("Models/TWTR/v1.zip", print_system_info=True)

'''

e_trade_gym = StockTradingEnv(df = trade, **env_kwargs) #turbulence_threshold = 70,risk_indicator_col='vix',


df_account_memory, df_actions, actions = DRLAgent.DRL_prediction(
    model=model, 
    environment = e_trade_gym)


perf_stats_all = backtest_stats(account_value=df_account_memory)
perf_stats_all = pd.DataFrame(perf_stats_all)
print("ppo:", perf_stats_all)

actions = [list(action[0])[0] for action in actions]
min_action = min(actions)
max_action = max(actions)
mean_action = np.mean(actions)
count_unique_action = len(set(actions))


raw_actions = pd.DataFrame({"raw_actions": actions})


# perf_stats_all.to_csv("Models/TWTR/v2_performance.csv")
# df_actions.to_csv("Models/TWTR/v2_df_actions.csv")
# raw_actions.to_csv("Models/TWTR/v2_raw_actions.csv")
df_account_memory.to_csv("Models/halfV_simulated/PPO_v2_lr0-0001_account_memory.csv")