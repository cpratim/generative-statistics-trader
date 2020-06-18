import json
import pickle
import numpy as np
import os
from functions import *
from math import floor
from time import sleep
from indicators import rsi, macd

LOCATION = 'data/minute'

def read_data(f):
    with open(f, 'r') as df:
        return json.loads(df.read())

def dump_data(f, d):
    with open(f, 'w') as df:
        json.dump(d, df, indent=4)

def read_data_bin(f):
    with open(f, 'rb') as df:
        return pickle.load(df)

def dump_data_bin(f, d):
    with open(f, 'wb') as df:
        pickle.dump(d, df)

STATS = 'stats'

normal_funcs = [normal_lin, normal_exp]
sum_funcs = [sum_ari, sum_lin, sum_avg]

#[v, o, c, h, l]
files = sorted(os.listdir(LOCATION))[-1:]
dump = [read_data_bin(f'{LOCATION}/{file}') for file in files]
opt = read_data(f'data/{STATS}/optimized.json')

def backtest(dump, period, sym, freq, func, stop_loss, th, cond, short=False):
    thresh = opt[str(period)][str(freq)][sym] * th
    shares = 0
    holding = 0
    last_buy = 10000
    last_freq = []
    profit = 0
    pos = 0
    failed = 0
    funds = 1000
    tries = 0
    type_ = 0
    for data in dump:
        if sym in data:
            min_ = data[sym]
            last_freq = min_[:freq]
            slow, fast, macd_ = macd([p[1] for p in min_])
            rsi_ = rsi([p[1] for p in min_])
            for m in min_[freq:]:
                mac = [fast[pos:freq+pos], slow[pos:freq+pos], macd_[pos:freq+pos]]
                rs = rsi_[pos:freq+pos]
                t, sig = func(last_freq, rs, mac)
                if sig == 1 and type_ == 0 and m[1] < cond * last_buy:
                    tries += 1
                    open_ = m[1]
                    last_buy = open_ 
                    shares += floor(funds/open_)
                    holding += 1
                    type_ = 1
                if sig == 0 and type_ == 0 and short is True:
                    open_ = m[1] * 1.01
                    last_buy = open_ 
                    shares += floor(funds/open_)
                    holding += 1
                    type_ = -1
                if holding >= 1: 
                    high = m[3]
                    low = m[4]
                    holding += 1
                    if type_ == 1:
                        goal = last_buy * (1 + thresh/100)
                        if high > goal:
                            p = shares * (goal - last_buy)
                            profit += p
                            holding = 0
                            shares = 0
                            type_ = 0
                    if type_ == -1:
                        goal = last_buy * (1 - thresh/100)
                        if low < goal:
                            p = shares * (last_buy - goal)
                            profit += p
                            holding = 0
                            shares = 0
                            type_ = 0
                if holding == freq * stop_loss:
                    price = m[1]
                    if type_ == -1:
                        p = shares * (last_buy - price)
                        profit += p
                    if type_ == 1:
                        p = shares * (price - last_buy)
                        profit += p
                    return profit
                last_freq.pop(0)
                last_freq.append(m)
                pos += 1
            last_close = min_[-1][2]
            if type_ == -1:  profit += shares * (last_buy - last_close)
            if type_ == 1: profit += shares * (last_close - last_buy)

    return profit


d = read_data(f'data/{STATS}/params.json')

period = 1
freq = 30
profit = 0
sym = [s for s in d]
ind = {}
t = []
cond = 2
sl = 5
th = 1
for s in sym:
	thresh = read_data(f'data/{STATS}/optimized.json')[str(period)][str(freq)][s]
	t.append(f'[{s}]: {thresh}%')
	print(d[s])
	params, g = d[s]['params'], d[s]['g']
	def func(inp, rsi, macd, params=params, g=g):
		p, f, n, s, m, fn = params
		vs, os, cs, hs, ls = [[c[i] for c in inp][p[i]:] for i in range(5)]
		pc = len(os)
		hps, lps = [(hs[i] - os[i])/os[i] * 100 for i in range(len(os))], [(os[i] - ls[i])/os[i] * 100 for i in range(len(os))]
		ft = [vs, os, cs, hs, ls, hps, lps, rsi] + list(macd)
		features = [ft[i] for i in f]
		features = [normal_funcs[n[i]](features[i]) for i in range(len(features))]
		sums = [sum_funcs[s[i]](features[i]) for i in range(len(features))]
		sums = [(m[i] * sums[i]) for i in range(len(features))]
		t = sum_funcs[fn[0]](sums)
		return t, int(t < g)
		#return 1, 1

	prof = backtest(dump=dump, sym=s, period=1, freq=freq, func=func, cond=cond, th=th, stop_loss=sl, short=False)
	profit += prof
	ind[s] = prof

print(t)
print(sym)
print(ind)
print(profit)
