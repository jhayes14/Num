# coding: utf-8
import pandas as pd
data = pd.read_csv('/home/loschen/Downloads/otto-group-product-classification-challenge_public_leaderboard.csv')
import matplotlib.pyplot as plt
data = data[data.Score<0.65]
data = data.sort(columns='Score')
data = data.drop_duplicates(cols='TeamName')
data['Score'].hist(bins=50)
plt.show()
