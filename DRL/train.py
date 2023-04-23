import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

import torch as th
from torch import nn
from typing import Dict, List, Tuple, Type, Union

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv, OneDTSStockTradingEnv
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

import argparse
# #===================================================================
# #Argparser Set Up
# #===================================================================
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-ticker", "--ticker", help="stock ticker symbol")
parser.add_argument("-train_file", "--train_file")
parser.add_argument("-trade_file", "--trade_file")
parser.add_argument("-model_version", "--model_version", help="batch size for training")
parser.add_argument("-lr", "--lr", type=float, help="learning rate for training")
parser.add_argument("-timesteps", "--timesteps", type=float, help="number of timesteps for training")
parser.add_argument("-ts", "--time_series", type=float, help="should the environment be represented as a 1D closing price time series")
parser.add_argument("-scale", "--scale", type=float, help="should scaling be applied to features that represent the state of the environment")
parser.add_argument("-ts_fe", "--time_series_fe", type=float, help="should time series feature extraction be used")
args = parser.parse_args()


train = pd.read_csv(args.train_file).reset_index()[["date", "open", "high", "low", "close", "volume", "tic", "day"]]
trade = pd.read_csv(args.trade_file).reset_index()[["date", "open", "high", "low", "close", "volume", "tic", "day"]]

print(train.head(5))

fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = INDICATORS,
                    use_vix=False,
                    use_turbulence=False,
                    user_defined_feature = False)


train = fe.preprocess_data(train)
trade = fe.preprocess_data(trade)


def process(processed):
    list_ticker = processed["tic"].unique().tolist()
    list_date =list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])

    processed_full = processed_full.fillna(0)
    return processed_full

train = process(train).reset_index()
trade = process(trade).reset_index()

stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

if bool(args.scale):
    print("is scaling")

    close_min = min(train["close"])
    close_max = max(train["close"])

    inds_mins = []
    inds_maxs = []
    for i in range(len(INDICATORS)):
        ind = INDICATORS[i]
        inds_mins.append(min(train[ind]))
        inds_maxs.append(max(train[ind]))

else:
    close_min = close_max = inds_mins = inds_maxs = None
        
   
        
class TSFENetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        ts_net = [nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU()]
        self.ts_extractor = nn.Sequential(*ts_net).to(device="cuda")
        
        self.out_layer = nn.Linear(16, 16)
        mlp = [nn.Linear(12, 16), nn.ReLU(), self.out_layer, nn.ReLU()]
        self.network = nn.Sequential(*mlp).to(device="cuda")
    
    
    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
            
        other = features.unsqueeze(1)[:, :, 0:2].squeeze(1)

        ts = features.unsqueeze(1)[:, :, 2:].squeeze(1)
        trend = th.where(ts.squeeze(1)[:,9] > ts.squeeze(1)[:,0], 1, -1).unsqueeze(1)
        ts_features = self.ts_extractor(ts)
        policy_in = th.cat((other, trend, ts_features), 1)
       
        return self.network(features)
    
        
if bool(args.time_series):
    print("is time series")

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000, 
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "min_ts": close_min,
        "max_ts": close_max,
        "ts_variable": "close",
        "day": 10
    }
    
    if bool(args.time_series_fe):
    
        policy_kwargs = {"actor_network": TSFENetwork(), "critic_network": TSFENetwork()}
    
    else:
        policy_kwargs = {}
    
    e_train_gym = OneDTSStockTradingEnv(df = train, **env_kwargs)
    e_trade_gym = OneDTSStockTradingEnv(df = trade, **env_kwargs)

    

else:
    
    env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000, #100000, #1000000, #
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space_dim": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "close_min": close_min,
    "close_max": close_max,
    "inds_mins": inds_mins,
    "inds_maxs": inds_maxs,
    }

    policy_kwargs = {}
    e_train_gym = StockTradingEnv(df = train, **env_kwargs)
    e_trade_gym = StockTradingEnv(df = trade, **env_kwargs)


      
env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))


agent = DRLAgent(env = env_train)
PPO_PARAMS = {
    "n_steps": 2048, 
    "ent_coef": 0.01,
    "learning_rate": args.lr, 
    "batch_size": 128
}


model_ppo = agent.get_model("ppo", model_kwargs = PPO_PARAMS, tensorboard_log = TENSORBOARD_LOG_DIR, policy_kwargs = policy_kwargs)

trained_ppo = agent.train_model(model=model_ppo, 
                         tb_log_name="ppo",
                         total_timesteps=args.timesteps) 

saveloc = "Models/" + args.ticker + args.model_version
trained_ppo.save(saveloc)



##TRADING

df_account_memory, df_actions, actions = DRLAgent.DRL_prediction(model=trained_ppo, 
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


saveloc = "Results/" + args.ticker + args.model_version

loc1 = saveloc + "_performance.csv"
perf_stats_all.to_csv(loc1)
loc2 = saveloc + "_df_actions.csv"
df_actions.to_csv(loc2)
loc3 = saveloc + "_raw_actions.csv"
raw_actions.to_csv(loc3)
loc4 = saveloc + "_account_memory.csv"
df_account_memory.to_csv(loc4)