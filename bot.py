import alpaca_trade_api as alpaca
from time import sleep
from datetime import datetime, timedelta
import numpy as np
from polygon import AlpacaSocket, PolygonRest
from math import floor
import sys
import re
import json
from threading import Thread
from functions import *
from config import KEY_LIVE
from indicators import rsi, macd
from genstat import Model
#import auto_push

KEY = 'PKFS9WSW07TA35WIERXC'
SECRET = 'JQ3CfYRx4wNpVY3qVxdhSG0SjqhEG1QKKRyZnOVt'

print('cpratim')

date = lambda: str(datetime.now())[:10]
timestamp = lambda: str(datetime.now())[11:19]
nano = 1000000000

def read_data(f):
    with open(f, 'r') as df:
        return json.loads(df.read())

def until_open():
	now = datetime.now()
	y, m, d = [int(s) for s in str(now)[:10].split('-')]
	market_open = datetime(y, m, d, 10, 5)
	return ((market_open - now).seconds)

def market_close(unix):
	t = datetime.fromtimestamp(unix)
	y, m, d = [int(s) for s in date().split('-')]
	return((datetime(y, m, d, 16, 0, 0) - t).seconds)

def market_open():
	now = datetime.now()
	y, m, d = [int(s) for s in str(now)[:10].split('-')]
	mo = datetime(y, m, d, 9, 30)
	return mo

def until_close(now):
	y, m, d = [int(s) for s in str(now)[:10].split('-')]
	return((datetime(y, m, d, 16, 0, 0) - now).seconds)

normal_funcs = [normal_lin, normal_exp]
sum_funcs = [sum_ari, sum_lin, sum_avg, sum_min, sum_max, sum_squared]


class AlgoBot(object):

	def __init__(self, funds=5000, wait=True, sandbox=True):

		base = 'https://api.alpaca.markets'
		mins = 6.5 * 60
		if sandbox is True: base = 'https://paper-api.alpaca.markets'
		self.client = alpaca.REST(KEY, SECRET, base)
		self.params = read_data('data/stats/params.json')
		self.symbols = [s for s in self.params]
		self.env = {}
		self.funds = funds
		self.active, self.max_period, self.orders, self.margin, self.models = [{} for i in range(5)]
		self.pending, self.time_series = [], []
		for s in self.symbols:
			params = self.params[s]
			model = Model(s, params)
			self.models[s] = model
			self.env[s] = model.environment()
		print('All Models Trained')
		self.polygon = PolygonRest(KEY_LIVE)
		if wait is True: self._wait()
		self.start()

	def _wait(self):
		time = until_open()
		print(f'Sleeping {time} seconds until Market Open')
		sleep(time)
		now = str(datetime.now())
		print(f'Starting Bot at {now}')

	def _handle(self, bars):
		out = []
		for b in bars:
			out.append([b.v, b.o, b.c, b.h, b.l])
		return out

	def _log(self, error):
		print('Error:', error)
		return

	def ticker(self, s):
		f = 0
		while True:
			if s in self.symbols:
				now = datetime.now()
				try:
					p = self.polygon.get_last_price(s)
				except Exception as error: 
					self._log(error)
					p = None
				if f % 20 == 0:
					if s in self.active:
						order_id = self.active[s]['id']
						fp = self._fill(order_id)
						if fp is not None and s in self.active:
							qty = self.active[s]['s']
							alert = f'({timestamp()}) [-] Sold {qty} shares of {s} at {fp} per share \n'
							print(alert)
							self.funds += (float(qty) * float(fp))
							del self.active[s]
						if (self.max_period[s] - now).seconds > 80000:
							qty = self.active[s]['s']
							type_ = self.active[s]['t']
							if type_ == 'long':
								self.pending.append(s)
								self.client.cancel_order(order_id)
								Thread(target=self.sell, args=(s, qty, None)).start()
							if s in self.symbols: self.symbols.remove(s)
							return 
				if s not in self.active and f % 60 == 0 and p is not None:
					try:
						bars = self.client.get_barset(s, 'minute', limit=self.env[s]['freq'])[s]
						inp = self._handle(bars)
						signal = self.models[s].predict(inp)
						if signal == 1 and s not in self.pending:
							available = self.funds/(len(self.symbols) - len(self.active))
							qty = floor(available/p)
							if qty > 0:
								self.pending.append(s)
								Thread(target=self.buy, args=(s, qty, None)).start()
					except Exception as error:
						self._log(error)
			f += 5
			sleep(5)

	def start(self):
		for s in self.symbols:
			Thread(target=self.ticker, args=(s,)).start()

	def _remove(self, s):
		self.symbols.remove(s)

	def buy(self, symbol, qty, price=None):
		order = str()
		try:
			if price is None: order = self.client.submit_order(symbol=symbol, side='buy', type='market', qty=abs(qty), time_in_force='day')
			else: order = self.client.submit_order(symbol=symbol, side='buy', type='limit', limit_price=price, qty=abs(qty), time_in_force='day')
			_id = order.id
			tries = 0
			while self._fill(_id) is None:
				sleep(2)
				tries += 1
				if tries == 5:
					self.client.cancel_order(_id)
					self.pending.remove(symbol)
					return
		except Exception as error:
			self._log(error)
			self.pending.remove(symbol)
			return 
		lp = float(self._fill(_id))
		self.funds -= (qty * lp)
		g = lp * (1 + self.env[symbol]['thresh']/100)
		order = self.client.submit_order(symbol=symbol, side='sell', type='limit', limit_price=g, qty=abs(qty), time_in_force='day')
		self.active[symbol] = {'t': 'long', 's': abs(qty), 'p': lp, 'id': order.id}
		self.max_period[symbol] = datetime.now() + timedelta(minutes=self.env[symbol]['freq'] * self.env[symbol]['stop_loss'])
		self.pending.remove(symbol)
		alert = f'({timestamp()}) [+] Bought {qty} shares of {symbol} at {lp} per share \n'
		print(alert)
		return

	def sell(self, symbol, qty, price=None):
		order = str()
		try:
			if price is None: order = self.client.submit_order(symbol=symbol, side='sell', type='market', qty=abs(qty), time_in_force='day')
			else: order = self.client.submit_order(symbol=symbol, side='sell', type='limit', limit_price=price, qty=abs(qty), time_in_force='day')
			_id = order.id
			tries = 0
			while self._fill(_id) is None:
				sleep(2)
				tries += 1
				if tries == 5:
					self.client.cancel_order(_id)
					self.pending.remove(symbol)
					return
		except Exception as error:
			self._log(error)
			self.pending.remove(symbol)
			return 
		lp = float(self._fill(_id))
		self.funds += lp * qty
		self.pending.remove(symbol)
		del self.active[symbol]
		alert = f'({timestamp()}) [+] Sold {qty} shares of {symbol} at {lp} per share \n'
		print(alert)
		return

	def _fill(self, _id):
		return self.client.get_order(_id).filled_avg_price

	def _liquidate(self):
		for position in self.client.list_positions():
			qty = int(position.qty)
			s = position.symbol
			self.pending.append(s)
			if qty > 0:

				Thread(target=self.sell, args=(s, qty, None)).start()
			else:
				Thread(target=self.buy, args=(s, qty, None)).start()

ab = AlgoBot(wait=False)
ab.start()
