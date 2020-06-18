from controls import *
from functions import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics
import os
import numpy as np
from minute import optimize, common, backtest_model, raw_dump, get_data
from scipy.optimize import minimize, differential_evolution
from math import e, pi
from threading import Thread
from datetime import datetime, timedelta
import sys


BEST = {'params': [], 'fp': 10, 'profit': 0}

class GenerativeStatistics(object):

    def __init__(self, generative=True):

        if generative is True: self.dump = raw_dump(0, 30)
        self.nf = [normal, normal_ari, normal_lin]
        self.sf = [sum_avg, sum_squared, sum_ari]

    def _min_later(self, min_):
        seconds = 60 * min_
        now = datetime.now()
        target = now + timedelta(seconds=seconds)
        return target

    def _timed_out(self, target):
        now = datetime.now()
        return (target - now).seconds > 80000


    def generate_model(self, sym, max_time):


        self.target = self._min_later(max_time)
        bounds = ((1, 2.5), (.5, 2), (.1, .8), (0, .2), (0, .2), (.5, 1.5))
        def _minimize(s):
            result = differential_evolution(self._profit, 
                                            bounds=bounds, 
                                            args=(s,), 
                                            disp=True, 
                                            workers=1)
            print(result.success)
            x = result.x
            print('Optimized Parameters: ')
            print(x)
        for s in sym:
            _minimize(s)

    def _transform(self, X, Y, sm, nm):
        transformed = []
        for _in in X:
            features = [[c[i] for c in _in][0:] for i in range(5)]
            trans = [self.sf[sm](self.nf[nm](f)) for f in features]
            transformed.append(trans)
        x_train, x_test, y_train, y_test = train_test_split(transformed, Y, test_size=0.3, random_state=1)
        pos = [s for s in Y if s == 1]
        split = len(pos)/len(Y)
        return x_train, x_test, y_train, y_test, split

    def _parse(self, params):
        return [int(round(i * 10)) for i in params[:-1]] + [params[-1]]

    def _profit(self, params, sym):
        if self._timed_out(self.target):
            sys.exit(BEST)
            os._exit()
        try:
            profits = []
            f = lambda p: (e/pi) ** p 
            parsed = self._parse(params)
            freq, period, stop_loss, nm, sm, th = parsed
            for i in range(len(self.dump)-period):
                train_dump = self.dump[i:i+period]
                test_data = self.dump[i+period]
                model = Model(sym, parsed, dump=train_dump)
                thresh = model.environment()['thresh']
                profits.append(backtest_model(test_data, sym, freq, model, stop_loss, thresh))
            profit = sum(profits)/len(profits)
            significance = len(profits)/10
            fp = f(profit * significance)
            if fp < BEST['fp']:
                BEST['fp'] = fp
                BEST['params'] = parsed
                BEST['profit'] = profit
            return fp
        except Exception as error:
            print(error)
            return 2

    def _backtest(self, sym, model):
        test_data = raw_dump(0, 1)[0]
        env = model.environment()
        freq, stop_loss, thresh = env['freq'], env['stop_loss'], env['thresh']
        profit = backtest_model(test_data, sym, freq, model, stop_loss, thresh, raw=True)
        print(profit)
        return profit

class Model(object):

    def __init__(self, sym, params, dump=None):

        self.dump = dump
        self.params = self._parse(params)
        self.sym = sym
        self.nf = [normal, normal_ari, normal_lin]
        self.sf = [sum_avg, sum_squared, sum_ari]
        self.classifier = DecisionTreeClassifier(criterion="entropy")
        self._train()

    def _parse(self, params):
        return [int(round(i)) for i in params[:-1]] + [params[-1]]

    def _transform(self, X, Y, sm, nm):
        transformed = []
        for _in in X:
            features = [[c[i] for c in _in][0:] for i in range(5)]
            trans = [self.sf[sm](self.nf[nm](f)) for f in features]
            transformed.append(trans)
        x_train, x_test, y_train, y_test = train_test_split(transformed, Y, test_size=0.3, random_state=1)
        return x_train, x_test, y_train, y_test

    def _train(self):
        freq, period, stop_loss, nm, sm, th = self.params
        if self.dump is None: dump = raw_dump(1, period+1)
        else: dump = self.dump
        self.thresh = optimize(dump, sym=self.sym, freq=freq)[self.sym] * th
        if self.thresh is None: self.thresh = .5
        X, Y = get_data(dump, self.sym, freq, self.thresh)
        x_train, x_test, y_train, y_test = self._transform(X, Y, sm, nm)
        self.classifier = self.classifier.fit(x_train, y_train)

    def predict(self, inp):
        freq, period, stop_loss, nm, sm, th = self.params   
        features = [[c[i] for c in inp][0:] for i in range(5)]
        trans = [self.sf[sm](self.nf[nm](f)) for f in features]
        sig = self.classifier.predict([trans])[0]
        return sig

    def environment(self):
        freq, period, stop_loss, nm, sm, th = self.params
        return {'freq': freq, 'stop_loss': stop_loss, 'thresh': self.thresh}



#sym = ['PLAY', 'OAS', 'NCLH', 'AAL', 'CCL'] ['NCLH', 'ERI', 'SAVE', 'RCL', 'KSS']

#OAS [23, 7, 7, 0, 2, 1.2059970598431586]
#PLAY [22, 8, 4, 1, 2, 1.2480840394849788]

sym = ['ERI']
genstat = GenerativeStatistics()
genstat.generate_model(sym, max_time=30)
