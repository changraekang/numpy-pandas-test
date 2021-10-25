import numpy as np
from numpy.lib import tracemalloc_domain
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import sqlalchemy as db

csv_bream_length =  pd.read_csv('C:/workspace/pythonwork/fish/bream_length.csv')
bream_length = csv_bream_length.to_numpy().reshape(-1)
csv_bream_weight =  pd.read_csv('C:/workspace/pythonwork/fish/bream_weight.csv')
bream_weight = csv_bream_weight.to_numpy().reshape(-1)
csv_smelt_length =  pd.read_csv('C:/workspace/pythonwork/fish/smelt_length.csv')
smelt_length = csv_smelt_length.to_numpy().reshape(-1)
csv_smelt_weight =  pd.read_csv('C:/workspace/pythonwork/fish/smelt_weight.csv')
smelt_weight = csv_smelt_weight.to_numpy().reshape(-1)

bream_data = np.column_stack((bream_length,bream_weight))
smelt_data = np.column_stack((smelt_length,smelt_weight))







fish_length = np.hstack((bream_length,smelt_length))
fish_weight = np.hstack((bream_weight,smelt_weight))                
fish_target = np.concatenate((np.ones(34), np.zeros(13)))

fish = np.column_stack((fish_length, fish_weight))

index = np.arange(47)
np.random.shuffle(index)

train_input = fish[index[:34]] 
train_length = fish_length[index[:34]] 
train_weight = fish_weight[index[:34]] 
train_target = fish_target[index[:34]] 

train_data = np.column_stack((train_input,train_target))


test_input = fish[index[34:]] 
test_lenght = fish_length[index[34:]] 
test_weight = fish_weight[index[34:]] 
test_target = fish_target[index[34:]] 


test_data = np.column_stack((test_input,test_target))



engine = db.create_engine("mariadb+mariadbconnector://python:python1234@127.0.0.1:3306/pythondb")
train_dataFrame = pd.DataFrame(train_data, columns=["fish_len", "fish_wei", "target"])
test_dataFrame = pd.DataFrame(test_data, columns=["fish_len", "fish_wei", "target"])

train_dataFrame.to_sql("train",engine, index=False,if_exists="replace")
test_dataFrame.to_sql("test",engine, index=False,if_exists="replace")
