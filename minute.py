import json
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
from random import shuffle
from scipy.optimize import minimize
from polygon import PolygonRest
from config import KEY_LIVE
import pickle
from math import floor
#import auto_push

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

def raw_dump(s=None, e=None):
    if s is None and e is None:
        files = sorted(os.listdir(LOCATION))
        dump = [read_data_bin(f'{LOCATION}/{file}') for file in files]
        return dump
    files = sorted(os.listdir(LOCATION))[-e:-s]
    if s == 0: files = sorted(os.listdir(LOCATION))[-e:]
    dump = [read_data_bin(f'{LOCATION}/{file}') for file in files]
    return dump


def get_data(dump, sym, freq, thresh):
    inp, signals = [], []
    for data in dump:
        if sym in data:
            min_ = data[sym]
            for i in range(len(min_)-2*freq+1):
                _in = [c[0:] for c in min_[i:freq+i]]
                inp.append(_in)
                close = _in[-1][2]
                future = (min_[freq+i:2*freq+i])
                high, low = max([max(s[1:]) for s in future]), min([min(s[1:]) for s in future])
                if high > close * (1 + thresh/100): signals.append(1)
                else: signals.append(0)
    X, Y = np.array(inp), np.array(signals)
    return X, Y

def get_data_reg(dump, sym, freq):
    inp, highs = [], []
    for data in dump:
        for s in sym:
            if s in data:
                min_ = data[s]
                for i in range(len(min_)-2*freq+1):
                    _in = [c[0:] for c in min_[i:freq+i]]
                    inp.append(_in)
                    close = _in[-1][2]
                    future = (min_[freq+i:2*freq+i])
                    high, low = max([max(s[1:]) for s in future]), min([min(s[1:]) for s in future])
                    highp, lowp = (high - close)/close * 100, (close - low)/close * 100
                    highs.append(low)
    X, Y = np.array(inp), np.array(highs)
    return X, Y

def ratio(data, freq=26, thresh=.4, tr=False, sym=None):
    signals = []
    pos = 0
    if sym in data:    
        min_ = data[sym]
        for i in range(len(min_)-2*freq+1):
            _in = (min_[i:freq+i])
            close = _in[-1][2]
            future = (min_[freq+i:2*freq+i])
            high, low = max([max(s[1:]) for s in future]), min([min(s[1:]) for s in future])
            if high > close * (1 + thresh/100): signals.append(1)
            else: signals.append(0)
        pos = len([s for s in signals if s == 1])
    if tr:
        return len(signals), pos
    return pos/len(signals) * 100

def total_ratio(dump, t, sr=26, sym=None):
    ct = 0
    buys = 0
    for data in dump:
       c, b = ratio(data, freq=sr, thresh=t, tr=True, sym=sym)
       ct += c
       buys += b
    if ct == 0: return None
    return buys/ct * 100

def optimize(dump, dev=2, i=50, sym=None, freq=30):
    optimized = lambda r, d, i: (r > i - d) and (r < i + d)
    ret = {}
    if type(sym) == str:
        sym = [sym]
    for s in sym:
        tries = 0
        t = .5
        r = 0
        while not optimized(r, dev, i):
            r = total_ratio(dump, t=t, sym=s, sr=freq)
            if r is None:
                t = None
                break
            error = (r - i)
            if optimized(r, dev, i): break
            if tries == 100:
                t = None
                break
            t += error/500
            tries += 1   
        ret[s] = t
    return ret

def common(dump):
    symbols = []
    for data in dump:
        symbols.append([s for s in data])
    r = symbols[0]
    for sym in symbols:
        r = list(set(r) & set(sym))
    return r


def backtest_reg(data, sym, freq, model):
    shares, holding, profit, pos, type_ = [0 for i in range(5)]
    last_freq = []
    funds = 1000
    perc = 0
    if sym in data:
        min_ = data[sym]
        last_freq = min_[:freq]
        for m in min_[30:]:
            d = [c[0:] for c in last_freq]
            f = [[c[i] for c in d][0:] for i in range(5)]
            inp = np.array([sum(ft)/len(ft) for ft in f])
            perc = model.predict([inp])[0]
            #perc = 1.5
            if perc > .5 and type_ == 0:
                last_buy = m[1] 
                shares += floor(funds/last_buy)
                holding += 1
                type_ = 1
            if holding >= 1: 
                high = m[3]
                low = m[4]
                holding += 1
                if type_ == 1:
                    goal = last_buy * (1 + perc/100)
                    if high > goal:
                        p = shares * (goal - last_buy)
                        profit += p
                        holding, shares, type_ = 0, 0, 0
            if holding == freq:
                price = m[1]
                if type_ == 1:
                    profit += shares * (price - last_buy)
                #return profit/funds * 100
            last_freq.pop(0)
            last_freq.append(m)
            pos += 1
        last_close = min_[-1][2]
        if type_ == 1: profit += shares * (last_close - last_buy)
    return profit


def backtest_model(data, sym, freq, model, stop_loss, thresh, raw=False):
    shares, holding, profit, pos, type_ = [0 for i in range(5)]
    last_freq = []
    last_buy, last_sell = 1000, 1000
    funds = 2500
    if sym in data:
        min_ = data[sym]
        last_freq = min_[:freq]
        for m in min_[30:]:
            sig = model.predict(last_freq)
            #sig = 1
            if sig == 1 and type_ == 0:
                last_buy = m[1] 
                shares += floor(funds/last_buy)
                holding += 1
                type_ = 1
            if holding >= 1: 
                high = m[3]
                low = m[4]
                holding += 1
                if type_ == 1:
                    goal = last_buy * (1 + thresh/100)
                    if high > goal:
                        last_sell = goal
                        p = shares * (goal - last_buy)
                        profit += p
                        holding, shares, type_ = 0, 0, 0
            if holding == freq * stop_loss:
                price = m[1]
                if type_ == 1:
                    profit += shares * (price - last_buy)
                if raw is True:
                    return profit
                return profit/funds * 100
            last_freq.pop(0)
            last_freq.append(m)
            pos += 1
        last_close = min_[-1][2]
        if type_ == 1: profit += shares * (last_close - last_buy)
    if raw is True:
        return profit
    return profit/funds * 100


def prevelance(dump, sym):
    return len([1 for d in dump if sym in d]), len(dump)






#sym = common(dump)

'''
th = []
vol = []
for s in sym:
    print(s)
    thresh = optimize(dump, sym=s, freq=26)[s]
    if len(vol) < 5:
        vol.append(s)
        th.append(thresh)
    else:
        if s not in vol:
            for i in range(len(th)):
                if thresh > th[i]:
                    vol[i] = s
                    th[i] = thresh
                    break

print(vol)
print(th)
'''