from controls import *
from functions import *
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import os
import numpy as np
from minute import optimize, common, backtest_reg, backtest_model, raw_dump, get_data, get_data_reg
from scipy.optimize import minimize, differential_evolution
from math import e, pi
from threading import Thread


class GenerativeStatistics(object):

	def __init__(self):

		self.dump = raw_dump(1, 31)
		self.classifier = RandomForestRegressor()
		self._normalize = lambda features: [sum_squared(normal_ari(f)) for f in features]

	def _transform(self, X, Y):
		transformed = []
		for _in in X:
		    features = [[c[i] for c in _in][0:] for i in range(5)]
		    transformed.append(self._normalize(features))
		x_train, x_test, y_train, y_test = train_test_split(transformed, Y, test_size=0.3, random_state=1)
		return x_train, x_test, y_train, y_test

	def generate_model(self, sym):
		X, Y = get_data_reg(self.dump, ['OAS'], 30)
		x_train, x_test, y_train, y_test = self._transform(X, Y)
		self.classifier = self.classifier.fit(x_train, y_train)
		pred = self.classifier.predict(x_test)
		for i in range(len(pred)):
			print(y_test[i], pred[i])
		test_data = raw_dump(0, 1)[0]
		print('MSE', self.classifier.score(x_test, y_test))
		profit = backtest_reg(test_data, sym, 30, self.classifier)
		print(profit)



genstat = GenerativeStatistics()
genstat.generate_model('OAS')