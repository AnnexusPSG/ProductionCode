# --------------Full model run as of 10/20/2020 --------------------------------
import os
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdblp
import re
import statsmodels.api as sm
from datetime import datetime, timedelta
from scipy.optimize import minimize
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from xbbg import blp
from datetime import date
import getpass
from sklearn.linear_model import LinearRegression
from dateutil.relativedelta import relativedelta
from openpyxl import load_workbook
# -----------Fama French Imports------------------
import zipfile
import pandas_datareader.data as web
import urllib.request

username = getpass.getuser()

src = "C:/Users/yprasad/Dropbox (Annexus)/Portfolio Solutions Group/Portfolio Analysis/"
dest_research = "C:/Users/yprasad/Dropbox (Annexus)/Portfolio Solutions Group/Portfolio Analysis/Research/"
fia_src = "C:/Users/yprasad/Dropbox (Annexus)/Portfolio Solutions Group/FIA Time Series/py_fia_bav_time_series/"
dest_simulation = "C:/Users/yprasad/Dropbox (Annexus)/Portfolio Solutions Group/Portfolio Analysis/Simulation/"

start = '10/29/1996'
# end = '12/31/2020'
end = (date.today() - timedelta(days=1)).strftime("%m/%d/%Y")
user_start_date = '12/31/2010'
pd.set_option("display.max_rows", 10, "display.max_columns", 10)


def asset_expected_returns(sdate, edate):
	"""The function reads the portfolio assets and pulls the daily total return from bloomberg"""
	start = datetime.strptime(sdate, '%m/%d/%Y').strftime('%Y%m%d')
	end = datetime.strptime(edate, '%m/%d/%Y').strftime('%Y%m%d')
	
	inputs_df = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='asset_weights', index_col=[0],
							  parse_dates=True)
	from_income_sheet = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='income_model_inputs',
									  index_col=[0],
									  parse_dates=True)
	sp_estimated_ret = float(from_income_sheet.loc['sp500_forecast', 'inputs'])
	
	# ----------------Convert Annual S&P 500 Forecast to Monthly Forecast---------------------------
	sp_monthly_ret = (1 + sp_estimated_ret) ** (1 / 12) - 1
	
	# fia_bav_file = pd.read_csv(src + "py_fia_time_series.csv", index_col=[0], parse_dates=True)
	
	# ---read the fia used to determine the underlying index for expected returns
	fia_used = from_income_sheet.loc['fia_name', 'inputs']
	
	# ---Select the underlying indexfor expected returns------------
	if 'moz' in fia_used:
		fia_index = 'JMOZAIC2 Index'
	
	elif 'zebra' in fia_used:
		fia_index = 'ZEDGENY Index'
	else:
		fia_index = None
	
	fia_par_rate = float(from_income_sheet.loc['par_rate', 'inputs'])
	fia_spread = float(from_income_sheet.loc['spread', 'inputs'])
	
	# inputs_df = pd.read_csv(src + "asset_weights.csv", index_col=[0])
	tickers_ex_fia = list(inputs_df.index[:-2])
	tickers_ex_fia.append('SBMMTB3 Index')
	tickers_ex_fia.append(fia_index)
	
	# asset_tickers = [s + ' US EQUITY' if 'Index' not in s else s for s in tickers]
	asset_tickers = [s + ' US EQUITY' if 'Index' not in s else s for s in tickers_ex_fia]
	
	# asset_tickers = [s + ' US EQUITY' for s in list(inputs_df.index[:-2])]
	
	relative_index = 'SPXT Index'
	
	if relative_index in asset_tickers:
		asset_tickers = asset_tickers
	
	else:
		asset_tickers.append(relative_index)
	
	con = pdblp.BCon(debug=False, port=8194, timeout=100000)
	
	if not con.start():
		print("ERROR: ****Connection could not be established with the data source****")
	
	bbg_prices = con.bdh(asset_tickers, 'TOT_RETURN_INDEX_NET_DVDS', start, end,
						 elms=[("periodicityAdjustment", 'CALENDAR'),
							   ("periodicitySelection", 'DAILY'),
							   ('nonTradingDayFillMethod', 'PREVIOUS_VALUE'),
							   ('nonTradingDayFillOption', 'NON_TRADING_WEEKDAYS')])
	con.stop()
	
	bbg_prices = bbg_prices.unstack().droplevel(1).unstack().T
	bbg_prices.columns.name = None
	bbg_prices.index.name = 'Date'
	
	# ----------Assets Alpha_Beta vs SPX --------added 9/29/2020-------------------
	regr_data = bbg_prices.copy()
	# regr_data.loc[:, fia_used] = fia_bav_file.loc[:, fia_used]
	regr_data = regr_data.resample('M', closed='right').last()
	regr_data_change = regr_data.pct_change().fillna(0)
	# asset_tickers.append(fia_used)
	alpha = []
	beta = []
	for names in asset_tickers:
		if names == relative_index:
			alpha.append(0.0)
			beta.append(1.0)
		else:
			data = regr_data_change.copy()
			data = data.loc[:, [names, relative_index]]
			# --- last 60 months of data
			data = data.iloc[-60:]
			data = data.dropna(axis=0)
			Y = data.loc[:, names]
			X = data.loc[:, relative_index]
			X = sm.add_constant(X)
			model = sm.OLS(Y, X).fit()
			alpha.append(model.params[0])
			beta.append(model.params[1])
	
	regression_stats = pd.DataFrame(index=asset_tickers, columns=['alpha', 'beta', 'adj_beta', 'est_ret'])
	regression_stats.loc[:, 'alpha'] = alpha
	regression_stats.loc[:, 'beta'] = beta
	regression_stats.loc[:, 'adj_beta'] = 0.67 * regression_stats.loc[:, 'beta'] + 0.33 * 1.0
	regression_stats.loc[:, 'est_ret'] = regression_stats.loc[:, 'alpha'] + \
										 regression_stats.loc[:, 'adj_beta'] * sp_monthly_ret
	
	# ----------------------Convert monthly estimated returns to annual-------------------------------
	regression_stats.loc[:, 'Annualized Returns'] = (1 + regression_stats.loc[:, 'est_ret']) ** 12 - 1
	regression_stats.loc[fia_index, 'Annualized Returns'] = (regression_stats.loc[fia_index, 'Annualized Returns'] *
															 fia_par_rate) - fia_spread
	
	# ------------------------Annualized Risk based on the last 60 months of returns-------------------------
	std_df = regr_data_change.iloc[-60:].std() * np.sqrt(12)
	regression_stats.loc[:, 'Annualized Risk'] = std_df
	regression_stats.to_csv(src + 'assets_forecast_returns.csv')
	
	print(regression_stats)


def qualitative_from_bloomberg(sdate, edate):
	"""The function pulls the qualitative data from Bloomberg"""
	
	start = datetime.strptime(sdate, '%m/%d/%Y').strftime('%Y%m%d')
	end = datetime.strptime(edate, '%m/%d/%Y').strftime('%Y%m%d')
	
	# inputs_df = pd.read_csv(src + "asset_weights.csv", index_col='Asset')
	
	inputs_df = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='asset_weights', index_col=[0],
							  parse_dates=True)
	
	bbg_tickers = [s + ' US EQUITY' for s in inputs_df.index[:-2]]
	
	fields = ['Security_Name', 'FUND_EXPENSE_RATIO', 'FUND_ASSET_CLASS_FOCUS', 'FUND_MKT_CAP_FOCUS', 'FUND_STRATEGY',
			  'FUND_BENCHMARK_PRIM', 'SECURITY_TYP', 'CUR_MKT_CAP', 'FUND_INCEPT_DT', 'EQY_INIT_PO_DT']
	
	con = pdblp.BCon(debug=False, port=8194, timeout=100000)
	if not con.start():
		print("ERROR: ****Connection could not be established with the data source****")
	
	# bbg_df = con.bdh(equity, field, start, end, elms=[("periodicityAdjustment", 'CALENDAR'),
	#                                                   ("periodicitySelection", pselect),
	#                                                   ('nonTradingDayFillMethod', 'PREVIOUS_VALUE'),
	#                                                   ('nonTradingDayFillOption', nontrading)])
	bbg_df = blp.bdp(bbg_tickers, fields)
	con.stop()
	
	bbg_df.loc[:, 'ticker'] = inputs_df.index[:-2]
	bbg_df.sort_values(by=['fund_asset_class_focus', 'fund_mkt_cap_focus', 'security_name'], inplace=True)
	# ---research and fix the error---
	bbg_df.loc[:, 'eqy_init_po_dt'] = '1/1/1900'
	bbg_df.loc[:, 'cur_mkt_cap'] = np.nan
	# ---delete after solution
	
	mcap = list(bbg_df['cur_mkt_cap'])
	mcap_clsf = []
	mcap = [float(mc) for mc in mcap]
	
	for mktcap in mcap:
		
		if str(mktcap) == 'nan':
			mcap_clsf.append('Unclassified')
		elif mktcap < 300000000:
			mcap_clsf.append('Small-Cap')
		elif (mktcap >= 300000000) and (mktcap < 10000000000):
			mcap_clsf.append('Mid-Cap')
		elif (mktcap >= 10000000000) and (mktcap < 80000000000):
			mcap_clsf.append('Large-Cap')
		elif mktcap >= 80000000000:
			mcap_clsf.append('Mega-Cap')
	
	bbg_df.loc[:, 'size_classification'] = mcap_clsf
	
	for count in np.arange(len(bbg_df.index)):
		if (bbg_df.loc[bbg_df.index[count], 'size_classification'] == 'Unclassified') and \
				(bbg_df.loc[bbg_df.index[count], 'security_typ'] in ['Open-End Fund', 'ETP']) and \
				(bbg_df.loc[bbg_df.index[count], 'fund_asset_class_focus'] == 'Equity'):
			bbg_df.loc[bbg_df.index[count], 'size_classification'] = bbg_df.loc[
				bbg_df.index[count], 'fund_mkt_cap_focus']
		
		elif (bbg_df.loc[bbg_df.index[count], 'size_classification'] == 'Unclassified') and \
				(bbg_df.loc[bbg_df.index[count], 'security_typ'] in ['Open-End Fund', 'ETP']) and \
				(bbg_df.loc[bbg_df.index[count], 'fund_asset_class_focus'] == 'Fixed Income'):
			bbg_df.loc[bbg_df.index[count], 'size_classification'] = bbg_df.loc[
				bbg_df.index[count], 'fund_asset_class_focus']
	
	bbg_df.loc[:, 'size_classification'] = bbg_df.loc[:, 'size_classification'].apply(lambda x: x.upper())
	bbg_df.to_csv(src + 'bbg_qualitative_data.csv')


def data_from_bloomberg2(equity, field, start, end, pselect, nontrading):
	start = datetime.strptime(start, '%m/%d/%Y').strftime('%Y%m%d')
	end = datetime.strptime(end, '%m/%d/%Y').strftime('%Y%m%d')
	con = pdblp.BCon(debug=False, port=8194, timeout=100000)
	if not con.start():
		print("ERROR: ****Connection could not be established with the data source****")
	
	bbg_df = con.bdh(equity, field, start, end, elms=[("periodicityAdjustment", 'CALENDAR'),
													  ("periodicitySelection", pselect),
													  ('nonTradingDayFillMethod', 'PREVIOUS_VALUE'),
													  ('nonTradingDayFillOption', nontrading)])
	con.stop()
	if bbg_df.empty:
		print("ERROR: ****Couldn't fetch data. Dataframe is empty****")
	else:
		
		return bbg_df


def get_asset_prices_bloomberg(sdate, edate):
	"""The function reads the portfolio assets and pulls the daily total return from bloomberg"""
	start = datetime.strptime(sdate, '%m/%d/%Y').strftime('%Y%m%d')
	end = datetime.strptime(edate, '%m/%d/%Y').strftime('%Y%m%d')
	
	inputs_df = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='asset_weights', index_col=[0],
							  parse_dates=True)
	sp_500_forecast = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='income_model_inputs', index_col=[0],
									parse_dates=True)
	sp_estimated_ret = float(sp_500_forecast.loc['sp500_forecast', 'inputs'])
	
	# inputs_df = pd.read_csv(src + "asset_weights.csv", index_col=[0])
	tickers = list(inputs_df.index[:-2])
	
	# -------------TODO: Build Logic to identify assets as Index or Equity and pull prices accordingly--------------
	# -----------------If the assets are Index---------------------------------
	if 'Index' in tickers[0]:
		asset_tickers = [s + ' US EQUITY' if 'Index' not in s else s for s in tickers]
	else:
		# ------------------If the assets are not index -----------------------------
		asset_tickers = [s + ' US EQUITY' for s in tickers]
	
	qual_fields = ['Security_Name', 'FUND_EXPENSE_RATIO', 'FUND_ASSET_CLASS_FOCUS', 'FUND_MKT_CAP_FOCUS',
				   'FUND_STRATEGY', 'FUND_BENCHMARK_PRIM', 'SECURITY_TYP', 'CUR_MKT_CAP',
				   'FUND_INCEPT_DT', 'EQY_INIT_PO_DT']
	
	# bbg_prices = pd.read_csv(src + 'daily_prices_bbg.csv', index_col=[0], parse_dates=True)
	
	con = pdblp.BCon(debug=False, port=8194, timeout=100000)
	#
	if not con.start():
		print("ERROR: ****Connection could not be established with the data source****")

	bbg_prices = con.bdh(asset_tickers, 'TOT_RETURN_INDEX_NET_DVDS', start, end,
						 elms=[("periodicityAdjustment", 'CALENDAR'),
							   ("periodicitySelection", 'DAILY'),
							   ('nonTradingDayFillMethod', 'PREVIOUS_VALUE'),
							   ('nonTradingDayFillOption', 'NON_TRADING_WEEKDAYS')])

	bbg_df = blp.bdp(asset_tickers, qual_fields)
	con.stop()
	
	bbg_prices = bbg_prices.unstack().droplevel(1).unstack().T
	bbg_prices.columns.name = None
	bbg_prices.index.name = 'Date'
	bbg_prices = bbg_prices[asset_tickers]
	bbg_prices.to_csv(src + 'daily_prices_bbg.csv')
	print(bbg_prices)
	
	# --Pull the qualitative data----------
	bbg_df_cols = list(bbg_df.columns)
	req_cols = [c.lower() for c in qual_fields]
	missing_cols = [c for c in req_cols if c not in bbg_df_cols]
	
	if len(missing_cols) > 0:
		for col in missing_cols:
			bbg_df.loc[:, col] = np.nan
	else:
		pass
	
	# bbg_df.loc[:, 'ticker'] = inputs_df.index[:-2]
	bbg_df.sort_values(by=['fund_asset_class_focus', 'fund_mkt_cap_focus', 'security_name'], inplace=True)
	ls_tickers = [t.split(' ')[0] for t in bbg_df.index]
	bbg_df.loc[:, 'ticker'] = ls_tickers
	
	mcap = list(bbg_df['cur_mkt_cap'])
	mcap_clsf = []
	mcap = [float(mc) for mc in mcap]
	
	for mktcap in mcap:
		
		if str(mktcap) == 'nan':
			mcap_clsf.append('Unclassified')
		elif mktcap < 300000000:
			mcap_clsf.append('Small-Cap')
		elif (mktcap >= 300000000) and (mktcap < 10000000000):
			mcap_clsf.append('Mid-Cap')
		elif (mktcap >= 10000000000) and (mktcap < 80000000000):
			mcap_clsf.append('Large-Cap')
		elif mktcap >= 80000000000:
			mcap_clsf.append('Mega-Cap')
	
	bbg_df.loc[:, 'size_classification'] = mcap_clsf
	
	for count in np.arange(len(bbg_df.index)):
		if (bbg_df.loc[bbg_df.index[count], 'size_classification'] == 'Unclassified') and \
				(bbg_df.loc[bbg_df.index[count], 'security_typ'] in ['Open-End Fund', 'ETP']) and \
				(bbg_df.loc[bbg_df.index[count], 'fund_asset_class_focus'] == 'Equity'):
			bbg_df.loc[bbg_df.index[count], 'size_classification'] = bbg_df.loc[
				bbg_df.index[count], 'fund_mkt_cap_focus']
		
		elif (bbg_df.loc[bbg_df.index[count], 'size_classification'] == 'Unclassified') and \
				(bbg_df.loc[bbg_df.index[count], 'security_typ'] in ['Open-End Fund', 'ETP']) and \
				(bbg_df.loc[bbg_df.index[count], 'fund_asset_class_focus'] == 'Fixed Income'):
			bbg_df.loc[bbg_df.index[count], 'size_classification'] = bbg_df.loc[
				bbg_df.index[count], 'fund_asset_class_focus']
		
		elif (bbg_df.loc[bbg_df.index[count], 'size_classification'] == 'Unclassified') and \
				(bbg_df.loc[bbg_df.index[count], 'security_typ'] in ['Open-End Fund', 'ETP']) and \
				(bbg_df.loc[bbg_df.index[count], 'fund_asset_class_focus'] == 'Money Market'):
			bbg_df.loc[bbg_df.index[count], 'size_classification'] = bbg_df.loc[
				bbg_df.index[count], 'fund_asset_class_focus']
		
		elif (bbg_df.loc[bbg_df.index[count], 'size_classification'] != 'Unclassified') and \
				(bbg_df.loc[bbg_df.index[count], 'security_typ'] in ['Open-End Fund', 'ETP']) and \
				(bbg_df.loc[bbg_df.index[count], 'fund_asset_class_focus'] == 'Equity'):
			bbg_df.loc[bbg_df.index[count], 'size_classification'] = bbg_df.loc[
				bbg_df.index[count], 'fund_mkt_cap_focus']
		
		elif bbg_df.loc[bbg_df.index[count], 'fund_asset_class_focus'] in ['Alternative', 'Fixed Income']:
			bbg_df.loc[bbg_df.index[count], 'size_classification'] = bbg_df.loc[bbg_df.index[count],
																				'fund_asset_class_focus']
	
	copy_bbg_prices = bbg_prices.copy()
	copy_bbg_prices = copy_bbg_prices[bbg_df.index.tolist()]
	inception_date = list(copy_bbg_prices.isnull().idxmin())
	bbg_df.loc[:, 'inception_date'] = inception_date
	days = (pd.to_datetime(edate) - pd.to_datetime(sdate)).days
	bbg_df.loc[:, "% History available"] = bbg_df.loc[:, 'inception_date'].apply(
		lambda x: (pd.to_datetime(edate) - pd.to_datetime(x)).days) / days
	
	bbg_df.loc[:, 'size_classification'].fillna(bbg_df.loc[:, 'fund_strategy'], inplace=True)
	bbg_df.loc[:, 'fund_asset_class_focus'].fillna('Equity', inplace=True)
	bbg_df.loc[:, 'size_classification'] = bbg_df.loc[:, 'size_classification'].apply(lambda x: str(x).upper())
	bbg_df.to_csv(src + 'bbg_qualitative_data.csv')


def feature_selection_model(sdate, edate):
	"""This funtion creates the price files for the portfolio assets. Also chooses proxy assets from the BM library"""
	
	universe_file = pd.read_csv(src + 'benchmark_prices.csv', index_col='Date', parse_dates=True)
	assets_file = pd.read_csv(src + 'daily_prices_bbg.csv', index_col='Date', parse_dates=True)
	qual_data = pd.read_csv(src + 'bbg_qualitative_data.csv', index_col=[0], parse_dates=True)
	
	# ----------------Add Cash and Portfolio Benchmark -------------------------------
	
	bm_data = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='portfolio_benchmark', index_col=[0])
	bm_tickers = list(bm_data.loc[:, 'BBG Ticker'].values)
	bm_weights = list(bm_data.loc[:, 'wts_bm'].values)
	bm_tickers.append('SBMMTB3 Index')
	
	start = datetime.strptime(sdate, '%m/%d/%Y').strftime('%Y%m%d')
	end = datetime.strptime(edate, '%m/%d/%Y').strftime('%Y%m%d')
	fields = ['FUND_BENCHMARK_PRIM']
	
	con = pdblp.BCon(debug=False, port=8194, timeout=100000)
	if not con.start():
		print("ERROR: ****Connection could not be established with the data source****")
	
	bbg_bm = con.bdh(bm_tickers, 'TOT_RETURN_INDEX_NET_DVDS', start, end, elms=[("periodicityAdjustment", 'CALENDAR'),
																				("periodicitySelection", 'DAILY'),
																				('nonTradingDayFillMethod',
																				 'PREVIOUS_VALUE'),
																				('nonTradingDayFillOption',
																				 'NON_TRADING_WEEKDAYS')])
	con.stop()
	
	bbg_bm = bbg_bm.unstack().droplevel(1).unstack().T
	bbg_bm = bbg_bm[bm_tickers]
	bbg_bm.columns.name = None
	
	combined_data = assets_file.resample('BM', closed='right').last()
	combined_data = combined_data.pct_change().fillna(0)
	combined_data = 100 * combined_data.add(1).cumprod()
	bbg_bm_monthly = bbg_bm.resample('BM', closed='right').last()
	bm_monthly_ret = bbg_bm_monthly.pct_change().fillna(0)
	cash = bm_monthly_ret.loc[:, 'SBMMTB3 Index']
	cash = 100 * cash.add(1).cumprod()
	bbg_bm_monthly = bm_monthly_ret.drop('SBMMTB3 Index', axis=1)
	bm_combined = bbg_bm_monthly.dot(bm_weights)
	bm_nav = 100 * bm_combined.add(1).cumprod()
	
	# --Append cash and Benchmark time series to the dataframe of asset prices----
	combined_data.loc[:, 'Cash'] = cash
	combined_data.loc[:, 'BM'] = bm_nav
	
	combined_data.to_csv(src + 'asset_price.csv')


def rebase_dataframe(fia_name, ts_fia):
	"""Call the function to rebase the raw data file based on the starting date of the FIA, input the name of the FIA
      for which the base portfolio needs to be created. Also change the fees of the assets list if required """
	
	read_frame = pd.read_csv(src + "asset_price.csv", index_col="Date", parse_dates=True)
	
	# fia_frame = pd.read_csv(src + "py_fia_time_series.csv", index_col="Date", parse_dates=True)
	fia_frame = ts_fia
	original_names = read_frame.columns.tolist()
	fia_names = fia_frame.columns.tolist()
	fia_list = [word for word in original_names if "_" in word]
	
	if fia_name in fia_names:
		file_path = src + fia_name.lower()
		s_date = pd.to_datetime(fia_frame[fia_name].dropna().index[0])
		read_frame.loc[:, fia_name] = fia_frame.loc[:, fia_name]
		read_frame = read_frame[s_date:]
		new_dir = src + fia_name.lower() + "/"
	else:
		new_dir = None
	
	resampled_frame = read_frame.resample('BM', closed='right').last()
	
	# ---------------------Drop BM from the dataframe------------------------------------
	frame_ret = resampled_frame.copy()
	# drop_cash = drop_cash.drop(['Cash'], axis=1)
	frame_ret = frame_ret.pct_change().fillna(0)
	period = len(frame_ret)
	cret = frame_ret.add(1).prod()
	ann_std = frame_ret.std() * np.sqrt(12)
	cov_mat = frame_ret.cov()
	annual_returns = cret ** (12 / period) - 1
	
	def prices(returns, base):
		# Converts returns into prices
		s = [base]
		for i in range(len(returns)):
			s.append(base * (1 + returns[i]))
		return np.array(s)
	
	def dd(returns, tau):
		# Returns the draw-down given time period tau
		values = prices(returns, 100)
		pos = len(values) - 1
		pre = pos - tau
		drawdown = float('+inf')
		# Find the maximum drawdown given tau
		while pre >= 0:
			dd_i = (values[pos] / values[pre]) - 1
			if dd_i < drawdown:
				drawdown = dd_i
			pos, pre = pos - 1, pre - 1
		# Drawdown should be positive
		return abs(drawdown)
	
	def max_dd(returns):
		# Returns the maximum draw-down for any tau in (0, T) where T is the length of the return series
		max_drawdown = float('-inf')
		for i in range(0, len(returns)):
			drawdown_i = dd(returns, i)
			if drawdown_i > max_drawdown:
				max_drawdown = drawdown_i
		# Max draw-down should be positive
		return abs(max_drawdown)
	
	cols = frame_ret.columns
	asset_max_dd = [max_dd(frame_ret[c]) for c in cols]
	ann_ret_df = pd.DataFrame(index=cols, columns=['Annualized Returns', 'Annualized Risk', 'Max DD'])
	ann_ret_df.loc[:, 'Annualized Returns'] = annual_returns
	ann_ret_df.loc[:, 'Annualized Risk'] = ann_std
	ann_ret_df.loc[:, 'Max DD'] = asset_max_dd
	ann_ret_df.index.name = "Symbol"
	
	resampled_frame = resampled_frame.pct_change()
	
	# Gross nav frame
	gross_nav = resampled_frame.add(1).cumprod()
	gross_nav.iloc[0] = 1.0
	gross_nav = 100 * gross_nav
	
	if not os.path.exists(new_dir):
		os.mkdir(new_dir)
		print("Directory ", new_dir, " Created ")
	else:
		print("Directory ", new_dir, " already exists")
	
	gross_nav.to_csv(new_dir + "gross_nav.csv")
	
	resampled_frame = resampled_frame.add(1).cumprod()
	resampled_frame.iloc[0] = 1.0
	resampled_frame = 100 * resampled_frame
	perf = round(100 * resampled_frame.pct_change(), 2).groupby(resampled_frame.index.year).sum()
	
	perf.to_csv(new_dir + 'asset_yearly_performance.csv')
	# round(100 * resampled_frame.pct_change(), 2).groupby(resampled_frame.index.year).sum().plot(kind='bar',
	#                                                                                             figsize=(14, 7))
	# plt.grid()
	# plt.title('Asset Returns by Year')
	# plt.ylabel("% return")
	# plt.xlabel(' ')
	# plt.xticks(rotation=45)
	# plt.savefig(new_dir + "yearly_performance.png", bbox_inches='tight', dpi=500, transparent=False)
	resampled_frame.to_csv(new_dir + "net_nav.csv")
	resampled_frame.to_csv(src + "net_nav.csv")
	
	# ----------------------------Save annual returns and cov mat for slide-------------------------------------
	ann_ret_df.to_csv(new_dir + 'assets_annual_return.csv')


def create_portfolio_rebalance(ts_fia, fia_name, term, base, mgmt_fees, fee_per):
	"""--Original Version 1--- Call the function to generate the portfolio returns with no FIA using either static
	Mozaic or ZEBRAEDGE and base=True Base portfolio is saved under the FIA (static Moz or Zebra) named folder. The
	assets are rebalanced monthly .Call the function to generate the portfolio returns with FIA and any type of par.
	Input the FIA name and base=False The weights csv file and the order of its row must match the column order or the
	nav_net.csv file. """
	
	# create the list of asset weights
	# wts_frame = pd.read_csv(src + "asset_weights.csv", index_col='Asset')
	
	wts_frame = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='asset_weights', index_col=[0],
							  parse_dates=True)
	
	income_info = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='income_model_inputs', index_col=[0],
								parse_dates=True)
	
	# fia_frame = pd.read_csv(src + "py_fia_time_series.csv", index_col="Date", parse_dates=True)
	fia_frame = ts_fia
	fia_names = fia_frame.columns.tolist()
	
	# if base is False means portfolio with an FIA
	if fia_name in fia_names and base is False:
		file_path = src + fia_name.lower() + "/"
	
	elif fia_name == 'static_MOZAIC' and base is True:
		file_path = src + fia_name.lower() + "/"
	
	else:
		file_path = src + fia_name.lower() + "/"
	
	# if the base is True means base portfolio with no FIA
	if base is True:
		fia_wt = wts_frame['base'].tolist()
		file_path = src + fia_name.lower() + "/"
	else:
		fia_wt = wts_frame['fia'].tolist()
		file_path = src + fia_name.lower() + "/"
	
	asset_names = wts_frame.index.tolist()
	asset_names = [fia_name if name == 'FIA' else name for name in asset_names]
	dollar_names = ['dollar_{}'.format(s) for s in asset_names]
	dollar_list = []
	
	# --------------------Hypothetical Starting Investment---------------
	start_amount = 1000000
	
	for i in range(len(dollar_names)):
		dollar_list.append(start_amount * fia_wt[i])
	
	dollar_list = [0 if math.isnan(x) else x for x in dollar_list]
	
	nav_net = pd.read_csv(file_path + "net_nav.csv", index_col='Date', parse_dates=True)
	base_universe = nav_net.copy()
	
	base_universe_returns = base_universe.pct_change().fillna(0)
	base_cols = base_universe_returns.columns.tolist()
	wts_names = ['wts_{}'.format(s) for s in asset_names]
	
	# create dataframe for advisor fee, resample the dataframe for quarter end
	fee_frame = pd.DataFrame(index=base_universe_returns.index, columns=['qtr_fees'])
	fee_frame.qtr_fees = mgmt_fees / (fee_per * 100)
	fee_frame = fee_frame.resample('BQ', closed='right').last()
	combined_fee_frame = pd.concat([base_universe, fee_frame], axis=1)
	combined_fee_frame.loc[:, 'qtr_fees'] = combined_fee_frame.qtr_fees.fillna(0)
	
	# ----Logic for not deducting fees if the portfolio is starting at the end of any quarter else charge fee----
	
	# prorate the first qtr fees if asset managed for less than a qtr
	# date_delta = (pd.to_datetime(actual_sdate) - base_universe_returns.index.to_list()[0]).days
	
	# fia_frame = pd.read_csv(src + "py_fia_time_series.csv", index_col="Date", parse_dates=True)
	# fia_names = fia_frame.columns.tolist()
	
	if fia_name in fia_names:
		# file_path = src + fia_name.lower() + "/"
		d2 = pd.to_datetime(fia_frame[fia_name].dropna().index[0])
	
	# date_delta = (fee_frame.index.to_list()[0] - base_universe_returns.index.to_list()[0]).days
	
	date_delta = (base_universe_returns.index.to_list()[0] - d2).days
	prorate_days = date_delta / 90
	first_qtr_fee = prorate_days * ((mgmt_fees * .01) / fee_per)
	
	# d1 = datetime.datetime.strptime(siegel_sd, "%m/%d/%Y").month
	d1 = base_universe_returns.index.to_list()[0].month
	
	# d2 = datetime.datetime.strptime(actual_sdate, "%m/%d/%Y").month
	
	month_dff = d1 - d2.month
	
	if d1 == 12:
		combined_fee_frame.iloc[month_dff]['qtr_fees'] = 0
	else:
		combined_fee_frame.iloc[month_dff]['qtr_fees'] = first_qtr_fee
	adv_fees = []
	
	for i in range(len(dollar_names)):
		base_universe_returns[dollar_names[i]] = dollar_list[i]
	
	counter = 1
	asset_dollars = []
	asset_wts = []
	term = term * 12
	
	# -----For Income Model - to calculate rider fees--------------
	fia_premium = start_amount * fia_wt[-1]
	income_bonus = float(income_info.loc['income_bonus', 'inputs'])
	bonus_term = int(income_info.loc['bonus_term', 'inputs'])
	income_growth = float(income_info.loc['income_growth', 'inputs'])
	rider_fee = float(income_info.loc['rider_fee', 'inputs'])
	
	# -------convert annual income growth and rider fee to monthly and quarterly---------
	income_growth = income_growth / 12.0
	rider_fee = rider_fee / 4.0
	fia_income_base = fia_premium * (1 + income_bonus)
	rider_fee_deduction = []
	months = 0
	rider_amt = 0
	
	for idx, row in base_universe_returns.iterrows():
		if 0 < months <= (bonus_term * 12):
			fia_income_base = fia_income_base * (1 + income_growth)
		else:
			fia_income_base = fia_income_base
		
		months += 1
		rows_filtered = base_universe_returns.reindex(columns=asset_names)
		row_returns = rows_filtered.loc[idx].tolist()
		# row_returns = base_universe_returns.loc[idx, asset_names].tolist()
		# row_returns = list(base_universe_returns.reindex(columns=asset_names).loc[idx])
		returns = [1 + r for r in row_returns]
		dollar_list = [r * dollars for r, dollars in zip(returns, dollar_list)]
		with_fia_asset = sum(dollar_list)
		closing_wts = [(d / with_fia_asset) for d in dollar_list]
		asset_dollars.append(dollar_list)
		asset_wts.append(closing_wts)
		
		# ---------------------Advisor fees deduction-----------------------
		fia = base_universe_returns.loc[idx, 'dollar_' + fia_name]
		fee = combined_fee_frame.loc[idx, 'qtr_fees']
		
		# Logic for portfolio rebalance
		# fia = base_universe_returns.loc[idx, 'dollar_FIA']
		# fee = combined_fee_frame.loc[idx, 'qtr_fees']
		# deduct_fees = (sum_total - fia) * fee
		# total_value = sum_total - deduct_fees
		# adv_fees.append(deduct_fees)
		
		# ----------------------Convert yearly product life to monthly--------------------
		
		if (counter - 1) % term == 0:
			
			opening_wts = dollar_list
			opening_sum = sum(opening_wts)
			new_wts = [wt / opening_sum for wt in opening_wts]
			fia_dollar = sum(dollar_list)
			deduct_fees = (fia_dollar - fia) * fee
			fia_dollar = fia_dollar - deduct_fees
			adv_fees.append(deduct_fees)
			rider_fee_deduction.append(rider_amt)
			
			# Rebalancing all the assets back to their original weight on the day of FIA rebalance net of advisor fees
			dollar_list = [wts * fia_dollar for wts in fia_wt]
			print("Portfolio rebalanced in month {}".format(counter))
		
		else:
			# Excluding the FIA from calculating the monthly rebalance weights for other assets when FIA cannot be \
			# rebalanced
			
			fia_dollar = dollar_list[-1]
			opening_wts = dollar_list[:-1]
			opening_sum = sum(opening_wts)
			
			# new weights of the assets are calculated based on to their previous closing value relative to the total
			# portfolio value excluding the FIA. Trending assets gets more allocation for the next month
			# new_wts = [wt / opening_sum for wt in opening_wts]
			
			# new weights of tha non fia assets scaled back to its original wts. Assets are brought back to its target
			# weights. Kind of taking profits from trending assets and dollar cost averaging for lagging assets
			without_fia_wt = fia_wt[:-1]
			
			# ---Condition check if the portfolio has only one assets
			if np.sum(without_fia_wt) == 0.0:
				new_wts = without_fia_wt
			else:
				new_wts = [wt / sum(without_fia_wt) for wt in without_fia_wt]
			
			non_fia_dollar = sum(dollar_list) - dollar_list[-1]
			max_av = max(dollar_list[-1], fia_income_base)
			
			# ------Check to assign rider fee 0 for base portfolio----------
			if idx.month % 3 == 0 and not base:
				rider_amt = max_av * rider_fee
			else:
				rider_amt = 0.0
			
			deduct_fees = non_fia_dollar * fee
			
			# ----------Advisor fees is dedcuted--------------------------
			non_fia_dollar = non_fia_dollar - deduct_fees
			adv_fees.append(deduct_fees)
			
			dollar_list = [wts * non_fia_dollar for wts in new_wts]
			# ------------Deducting Rider Fee Quarterly-------------
			fia_dollar = fia_dollar - rider_amt
			rider_fee_deduction.append(rider_amt)
			dollar_list.append(fia_dollar)
		
		counter += 1
	
	asset_wt_df = pd.DataFrame(asset_wts, index=base_universe_returns.index, columns=wts_names)
	asset_wt_df['sum_wts'] = asset_wt_df.sum(axis=1)
	asset_dollar_df = pd.DataFrame(asset_dollars, index=base_universe_returns.index, columns=dollar_names)
	asset_dollar_df.loc[:, 'rider_fee_from_fia'] = rider_fee_deduction
	asset_dollar_df['Total'] = asset_dollar_df.sum(axis=1)
	base_universe_returns.drop(dollar_names, axis=1, inplace=True)
	joined_df = pd.concat([base_universe_returns, asset_dollar_df, asset_wt_df], axis=1, ignore_index=False)
	if not base:
		joined_df[fia_name + '_portfolio'] = joined_df.Total.pct_change().fillna(0)
	else:
		joined_df['base_portfolio_returns'] = joined_df.Total.pct_change().fillna(0)
	joined_df['advisor_fees'] = adv_fees
	if base:
		joined_df.to_csv(file_path + "base_portfolio.csv")
	else:
		joined_df.to_csv(file_path + fia_name + "_portfolio.csv")


def portfolio_analytics(file_path, fia_name, fia_wt):
	read_nav = pd.read_csv(file_path + "net_nav.csv", index_col='Date', parse_dates=True)
	# loading all the data files ----------------------------------------------------------
	base_df = pd.read_csv(file_path + "base_portfolio.csv", index_col='Date', parse_dates=True)
	fia_df = pd.read_csv(file_path + fia_name + "_portfolio.csv", index_col='Date',
						 parse_dates=True)
	bm_df = read_nav.loc[:, 'BM']
	# loading files completed--------------------------------------------------------------
	
	# dynamically change the names on the chart tiles--------------------------------------
	# chart_name = fia_name.split('_')[1].upper()
	chart_name = fia_name
	# Ends----------------------------------------------------------------------------------
	
	# Calculate the risk free rate for period-annualized retruns------------------------
	rfr_df = read_nav.loc[:, 'Cash']
	
	rfr = []
	period = [12, 36, 60, 120, len(rfr_df)]
	for p in period:
		data = rfr_df.loc[rfr_df.index[-(p + 1):]]
		if len(data) >= 12:
			per_ret = data.pct_change().dropna()
			cret = per_ret.add(1).cumprod()
			ann_return = cret.iloc[-1] ** (12 / (len(data) - 1)) - 1
			rfr.append(ann_return)
		else:
			ann_return = (data.iloc[-1] / data.iloc[0]) - 1
			rfr.append(ann_return)
	
	# period = [12, 36, 60, 120, len(rfr_df)]
	# rfr = [rfr_df.loc[rfr_df.index[-(p + 1):]].apply(ann_rfr_rate).values[0] for p in period]
	
	# Risk Free Return block ends---------------------------------------------------------
	
	# ----------------------------weights base portfolio------------------------------------
	wts_df = base_df.filter(regex='wt')
	wts_df.drop('sum_wts', axis=1, inplace=True)
	# round(100 * wts_df, 2).plot(figsize=(15, 8))
	# plt.grid()
	# plt.title('Asset Weights (Base Portfolio (No FIA))')
	# plt.ylabel('% weight')
	# plt.legend(loc='lower left')
	# plt.savefig(file_path + "wts_base.png", transparent=False)
	# # plt.clf()
	
	# --------------------------weights with FIA-------------------------------------------------
	wts_df = fia_df.filter(regex='wt')
	wts_df.drop('sum_wts', axis=1, inplace=True)
	# round(100 * wts_df, 2).plot(figsize=(15, 8))
	# plt.grid()
	# plt.title('Asset Weights - Portfolio with - {}'.format(chart_name))
	# plt.ylabel('% weight')
	# plt.savefig(file_path + "wts_fia.png", transparent=False)
	# # plt.clf()
	
	frame = [base_df['base_portfolio_returns'], fia_df['{}_portfolio'.format(fia_name)]]
	
	# ------------------------create a combined dataframe of returns-----------------------------------
	concat_returns = pd.concat(frame, axis=1)
	frame2 = concat_returns.copy()
	frame2 = frame2.add(1).cumprod()
	frame2.iloc[0] = 1
	frame2 = 100 * frame2
	# temp = frame2[['base_portfolio_returns', 'mozaic_portfolio_returns_rebalance']]
	# temp = temp.rename(
	#     columns={'base_portfolio_returns': '60/40',
	#              'mozaic_portfolio_returns_rebalance': '60/40 with JP Morgan Mozaic II'})
	
	frame2.rename(columns={'base_portfolio_returns': 'Base_Portfolio',
						   '{}_portfolio'.format(fia_name): 'Portfolio_with_{}'.format(chart_name)},
				  inplace=True)
	
	# combined2 = pd.concat([combined1, frame2], axis=1)
	frame2.to_csv(file_path + "combined_nav.csv")
	
	# Table for Monthly return calculations
	month_cols = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct',
				  11: 'Nov', 12: 'Dec'}
	
	copy_frame = frame2.copy()
	base_portfolio = copy_frame.asfreq('BM').set_index([copy_frame.index.year, copy_frame.index.month])[
		copy_frame.columns[0]].pct_change()
	base_frame = pd.DataFrame(base_portfolio).unstack()
	base_frame.columns = base_frame.columns.droplevel()
	base_frame.rename(columns=month_cols, inplace=True)
	base_frame.to_csv(file_path + "base_portfolio_returns_table.csv")
	
	fia_portfolio = copy_frame.asfreq('BM').set_index([copy_frame.index.year, copy_frame.index.month])[
		copy_frame.columns[1]].pct_change()
	fia_frame = pd.DataFrame(fia_portfolio).unstack()
	fia_frame.columns = fia_frame.columns.droplevel()
	fia_frame.rename(columns=month_cols, inplace=True)
	fia_frame.to_csv(file_path + fia_name + "_portfolio_returns_table.csv")
	
	try:
		# YTD
		stats_df = pd.DataFrame(columns=frame2.columns)
		yearly = frame2.resample('Y', closed='right').last().pct_change()
		
		# ------------Remove the current partial month to calculate N period statistics----------
		portfolio_nav = frame2.copy()
		portfolio_nav = portfolio_nav[:-1]
		# 1 Year
		N = 12
		one_yr = portfolio_nav.iloc[-(N + 1):]
		
		r1 = 1 + (one_yr.pct_change().dropna())
		r1 = r1.cumprod().iloc[-1] ** (12 / N) - 1
		risk1 = one_yr.iloc[1:].pct_change().std() * np.sqrt(12)
		sharpe1 = (r1 - .01 * rfr[0]) / risk1
		
		# 3 Year
		N = 36
		if len(portfolio_nav) < N:
			r3 = 0
			risk3 = 0
			sharpe3 = 0
		
		else:
			three_yr = portfolio_nav.iloc[-(N + 1):]
			r3 = 1 + (three_yr.pct_change().dropna())
			r3 = r3.cumprod().iloc[-1] ** (12 / N) - 1
			risk3 = three_yr.iloc[1:].pct_change().std() * np.sqrt(12)
			sharpe3 = (r3 - .01 * rfr[1]) / risk3
		
		# 5 Year
		N = 60
		if len(portfolio_nav) < N:
			r5 = 0
			risk5 = 0
			sharpe5 = 0
		else:
			five_yr = portfolio_nav.iloc[-(N + 1):]
			r5 = 1 + (five_yr.pct_change().dropna())
			r5 = r5.cumprod().iloc[-1] ** (12 / N) - 1
			risk5 = five_yr.iloc[1:].pct_change().std() * np.sqrt(12)
			sharpe5 = (r5 - .01 * rfr[2]) / risk5
		
		# 10 Year
		N = 120
		if len(portfolio_nav) < N:
			r10 = 0
			risk10 = 0
			sharpe10 = 0
		else:
			ten_yr = portfolio_nav.iloc[-(N + 1):]
			r10 = 1 + (ten_yr.pct_change().dropna())
			r10 = r10.cumprod().iloc[-1] ** (12 / N) - 1
			risk10 = ten_yr.iloc[1:].pct_change().std() * np.sqrt(12)
			sharpe10 = (r10 - .01 * rfr[3]) / risk10
		
		# Inception
		N = len(frame2)
		ri = 1 + (frame2.pct_change().dropna())
		ri = ri.cumprod().iloc[-1] ** (12 / N) - 1
		riski = frame2.pct_change().std() * np.sqrt(12)
		sharpei = (ri - .01 * rfr[4]) / riski
		
		stats_df.loc['YTD', :] = yearly.loc[yearly.index[-1], :].values
		
		stats_df.loc['Annualized Return (% 1Y)', :] = round(r1, 4)
		stats_df.loc['Annualized Risk (% 1Y)', :] = round(risk1, 4)
		stats_df.loc['Sharpe Ratio (1Y)', :] = round(sharpe1, 4)
		
		stats_df.loc['Annualized Return (% 3Y)', :] = round(r3, 4)
		stats_df.loc['Annualized Risk (% 3Y)', :] = round(risk3, 4)
		stats_df.loc['Sharpe Ratio (3Y)', :] = round(sharpe3, 4)
		
		stats_df.loc['Annualized Return (% 5Y)', :] = round(r5, 4)
		stats_df.loc['Annualized Risk (% 5Y)', :] = round(risk5, 4)
		stats_df.loc['Sharpe Ratio (5Y)', :] = round(sharpe5, 4)
		
		stats_df.loc['Annualized Return (% 10Y)', :] = round(r10, 4)
		stats_df.loc['Annualized Risk (% 10Y)', :] = round(risk10, 4)
		stats_df.loc['Sharpe Ratio (10Y)', :] = round(sharpe10, 4)
		
		stats_df.loc['Annualized Return (% inception)', :] = round(ri, 4)
		stats_df.loc['Annualized Risk (% inception)', :] = round(riski, 4)
		stats_df.loc['Sharpe Ratio (inception)', :] = round(sharpei, 4)
		
		stats_df.loc['% CAGR', :] = round((frame2.iloc[-1] / frame2.iloc[0] - 1), 4)
		stats_df.loc['$ Growth', :] = round(frame2.iloc[-1], 4)
		
		# --------------------Drawdown----------------------------------
		# Calculate the maximum value of returns using rolling().max()
		roll_max = frame2.rolling(min_periods=1, window=36).max()
		
		# Calculate daily draw-down for rolling max
		monthly_drawdown = frame2 / roll_max - 1.0
		
		# Calculate maximum daily draw-down
		max_monthly_drawdown = monthly_drawdown.rolling(min_periods=1, window=36).min()
		
		# -----------Determine the start and end date of the drawdowns------------------
		# ----------------------------Current Portfolio---------------------------
		asset_1 = frame2.loc[:, frame2.columns[0]]
		dd_end1 = np.argmax(np.maximum.accumulate(asset_1) - asset_1)
		dd_start1 = np.argmax(asset_1[:dd_end1])
		sdate1 = frame2.index[dd_start1]
		endate1 = frame2.index[dd_end1]
		
		# -------------------------------Proposed Portfolio-------------------------
		asset_2 = frame2.loc[:, frame2.columns[1]]
		dd_end2 = np.argmax(np.maximum.accumulate(asset_2) - asset_2)
		dd_start2 = np.argmax(asset_2[:dd_end2])
		sdate2 = frame2.index[dd_start2]
		endate2 = frame2.index[dd_end2]
		
		stats_df.loc['% AvgDD', :] = abs(round(monthly_drawdown.mean(), 2))
		stats_df.loc['% MaxDD', :] = abs(round(max_monthly_drawdown.min(), 2))
		stats_df.loc['Skew', :] = round(frame2.pct_change().skew(), 2)
		stats_df.loc['Kurtosis', :] = round(frame2.pct_change().kurtosis(), 2)
		
		stats_df.to_csv(file_path + fia_name + "_stats.csv")
	
	# ----------------------plot monthly drawdowns-----------------------------------
	# round(100 * monthly_drawdown, 2).plot(figsize=(12, 8))
	# plt.grid()
	# plt.title('Rolling Monthly Drawdown')
	# plt.ylabel("% Drawdown")
	# plt.savefig(file_path + "monthly_drawdown.png", transparent=False)
	# # plt.clf()
	
	# -----------------------Plot MaxDrawdown plot----------------------------------
	# round(100 * max_monthly_drawdown, 2).plot(figsize=(12, 8))
	# plt.grid()
	# plt.title('Rolling MaxDrawdon')
	# plt.ylabel("% Drawdown")
	# plt.savefig(file_path + "max_drawdown.png", transparent=False)
	# # plt.clf()
	
	except Exception as e:
		print("Error Occured in generating stats_df dataframe", e)
	
	# -------------------------------Plot returns-----------------------------
	# frame2.plot(figsize=(12, 8))
	# plt.grid()
	# plt.title('Growth of $100')
	# plt.ylabel("$ Growth")
	# plt.savefig(file_path + "cagr_plot.png", transparent=False)
	# # plt.clf()
	
	# ---------------------correlation plot----------------------------------------------
	
	ret_df = frame2.pct_change().fillna(0)
	# corr_df = frame2.pct_change().rolling(min_periods=1, window=36).corr().dropna()
	corr_df = ret_df[ret_df.columns[0]].rolling(36).corr(ret_df[ret_df.columns[1]])
	corr_df = np.round(corr_df, 2)
	# corr_df.plot(kind='line', figsize=(10, 8))
	# plt.grid()
	# plt.title('Portfolio Rolling Correlation - 36 months')
	# plt.xticks([])
	# plt.xlabel("corr with base model")
	# plt.ylabel("% correlation")
	# plt.margins(0, 0)
	# plt.savefig(file_path + "correlation.png", transparent=False, bbox_inches='tight', pad_inches=0)
	# # plt.clf()
	
	# Asset correlation
	asset_df = pd.read_csv(file_path + "net_nav.csv", index_col='Date', parse_dates=True)
	
	# renaming the static_FIA names to just FIA for charts
	old = asset_df.filter(regex='_').columns.values[0]
	new = old.split("_")[1]
	asset_df.rename(columns={old: new}, inplace=True)
	
	# -----------------------Asset Correlation plot---------------------------------------
	corrMatrix = asset_df.pct_change().fillna(0).corr()
	# fig, ax = plt.subplots(figsize=(6, 6))
	# mask = np.zeros_like(corrMatrix.corr())
	# mask[np.triu_indices_from(mask)] = 1
	# sns.heatmap(corrMatrix, mask=mask, ax=ax, annot=True)
	# plt.title("Asset Correlation Plot")
	# plt.savefig(file_path + "correlation_matrix.png", bbox_inches='tight', dpi=500, transparent=False)
	# # plt.clf()
	
	# ------------------------Asset Sharpe Ratio----------------------------------------------------
	asset_ann_ret = asset_df.pct_change().fillna(0).mean() * 12
	asset_ann_risk = asset_df.pct_change().std() * np.sqrt(12)
	asset_sharpe = (asset_ann_ret - rfr[-1]) / asset_ann_risk
	asset_sharpe = pd.DataFrame(asset_sharpe, columns=['Sharpe Ratio'])
	asset_sharpe = asset_sharpe.replace([np.inf, -np.inf], np.nan)
	asset_sharpe.to_csv(file_path + "asset_sharpe_data.csv")
	# plt.scatter(x=asset_df.columns, y=asset_sharpe, marker='*')
	# plt.title("Assets Sharpe Ratio")
	# plt.grid()
	# plt.xticks(rotation=45)
	# plt.savefig(file_path + "asset_sharpe_ratio.png", bbox_inches='tight', dpi=500, transparent=False)
	# # plt.clf()
	
	# --------------------------------------Plot yearly returns------------------------------------------
	offset_date = frame2.index[0] - pd.DateOffset(years=1)
	d = pd.to_datetime(offset_date)
	new_df = frame2.copy()
	new_df.reset_index(inplace=True)
	ndf = new_df.append({'Date': d, new_df.columns[1]: 100, new_df.columns[2]: 100}, ignore_index=True)
	ndf = ndf.set_index('Date')
	ndf.sort_index(inplace=True)
	yearly_ret = ndf.resample('Y', closed='right').last().pct_change()
	yearly_ret.index = yearly_ret.index.year
	yearly_ret = round(100 * yearly_ret[1:], 2)
	# cols = {'base': '60/40', 'moz_rebalance': '60/40 with JP Morgan Mozaic II'}
	# yearly_ret = yearly_ret.rename(columns=cols)
	# yearly_ret.plot(kind='bar', grid=True, figsize=(15, 7))
	# plt.title('Returns by Year')
	# plt.xlabel(' ')
	# plt.ylabel('% return')
	# plt.savefig(file_path + "return_by_year.png", bbox_inches='tight', dpi=500, transparent=False)
	# # plt.clf()
	yearly_ret.T.to_csv(file_path + "yearly_returns_table.csv")
	
	# --------------------------Excess returns for regression statistics--------------------------------------------
	reg_df = frame2.copy()
	reg_df.loc[:, 'Benchmark'] = bm_df
	reg_df = reg_df.pct_change().fillna(0)
	# reg_df['Benchmark'] = bm_df
	reg_df.dropna(inplace=True)
	
	cols = reg_df.columns.tolist()
	rfr_change = rfr_df.pct_change().fillna(0)
	reg_df.loc[:, 'rfr'] = rfr_change
	reg_df.dropna(inplace=True)
	excess_ret = pd.DataFrame(index=reg_df.index)
	for c in cols:
		excess_ret[c] = reg_df[c].subtract(reg_df['rfr'])
	
	# ------------------------Regression for base Model - CAPM-------------------------
	bY = excess_ret[cols[0]]
	bX = excess_ret[cols[-1]]
	bX = sm.add_constant(bX)
	base_mod = sm.OLS(bY, bX).fit()
	base_mod = sm.OLS(bY, bX).fit()
	base_mod.summary()
	result_base = base_mod.params.to_list()
	base_rsq = base_mod.rsquared
	result_base.append(base_rsq)
	
	# --------------------Regression for portfolio - CAPM -----------------------------
	Y = excess_ret[cols[1]]
	X = excess_ret[cols[-1]]
	X = sm.add_constant(X)
	port_mod = sm.OLS(Y, X).fit()
	result_port = port_mod.params.to_list()
	port_rsq = port_mod.rsquared
	result_port.append(port_rsq)
	regdf = pd.DataFrame([result_base, result_port], columns=['Alpha', 'Beta', 'Rsq'])
	regdf['idx'] = ['Base', fia_name]
	regdf.set_index('idx', inplace=True)
	regdf.loc[:, 'Alpha'] = regdf.Alpha * 12
	regdf.to_csv(file_path + "regression_table.csv")
	
	try:
		def vol(returns):
			# Return the standard deviation of returns
			return np.std(returns)
		
		def beta(rtn, mkt):
			m = np.array([rtn, mkt])
			statistic_beta = np.cov(m)[0][1] / np.var(mkt)
			# Return the covariance of m divided by the standard deviation of the market returns
			return statistic_beta
		
		def lpm(returns, threshold, order):
			# This method returns a lower partial moment of the returns
			# Create an array he same length as returns containing the minimum return threshold
			threshold_array = np.empty(len(returns))
			threshold_array.fill(threshold)
			# Calculate the difference between the threshold and the returns
			diff = threshold_array - returns
			# Set the minimum of each to 0
			diff = np.clip(diff, 0, None)
			# Return the sum of the different to the power of order
			return np.sum(diff ** order) / len(returns)
		
		def hpm(returns, threshold, order):
			# This method returns a higher partial moment of the returns
			# Create an array he same length as returns containing the minimum return threshold
			threshold_array = np.empty(len(returns))
			threshold_array.fill(threshold)
			# Calculate the difference between the returns and the threshold
			diff = returns - threshold_array
			# Set the minimum of each to 0
			diff = np.clip(diff, 0, None)
			# Return the sum of the different to the power of order
			return np.sum(diff ** order) / len(returns)
		
		def var(returns, alpha):
			# This method calculates the historical simulation var of the returns
			sorted_returns = np.sort(returns)
			# Calculate the index associated with alpha
			index = int(alpha * len(sorted_returns))
			# VaR should be positive
			return abs(sorted_returns[index])
		
		def cvar(returns, alpha):
			# This method calculates the condition VaR of the returns
			sorted_returns = np.sort(returns)
			# Calculate the index associated with alpha
			index = int(alpha * len(sorted_returns))
			# Calculate the total VaR beyond alpha
			sum_var = sorted_returns[0]
			for i in range(1, index):
				sum_var += sorted_returns[i]
			# Return the average VaR
			# CVaR should be positive
			return abs(sum_var / index)
		
		def prices(returns, base):
			# Converts returns into prices
			s = [base]
			for i in range(len(returns)):
				s.append(base * (1 + returns[i]))
			return np.array(s)
		
		def dd(returns, tau):
			# Returns the draw-down given time period tau
			values = prices(returns, 100)
			pos = len(values) - 1
			pre = pos - tau
			drawdown = float('+inf')
			# Find the maximum drawdown given tau
			while pre >= 0:
				dd_i = (values[pos] / values[pre]) - 1
				if dd_i < drawdown:
					drawdown = dd_i
				pos, pre = pos - 1, pre - 1
			# Drawdown should be positive
			return abs(drawdown)
		
		def max_dd(returns):
			# Returns the maximum draw-down for any tau in (0, T) where T is the length of the return series
			max_drawdown = float('-inf')
			for i in range(0, len(returns)):
				drawdown_i = dd(returns, i)
				if drawdown_i > max_drawdown:
					max_drawdown = drawdown_i
			# Max draw-down should be positive
			return abs(max_drawdown)
		
		def average_dd(returns, periods):
			# Returns the average maximum drawdown over n periods
			drawdowns = []
			for i in range(0, len(returns)):
				drawdown_i = dd(returns, i)
				drawdowns.append(drawdown_i)
			drawdowns = sorted(drawdowns)
			total_dd = abs(drawdowns[0])
			for i in range(1, periods):
				total_dd += abs(drawdowns[i])
			return total_dd / periods
		
		def average_dd_squared(returns, periods):
			# Returns the average maximum drawdown squared over n periods
			drawdowns = []
			for i in range(0, len(returns)):
				drawdown_i = math.pow(dd(returns, i), 2.0)
				drawdowns.append(drawdown_i)
			drawdowns = sorted(drawdowns)
			total_dd = abs(drawdowns[0])
			for i in range(1, periods):
				total_dd += abs(drawdowns[i])
			return total_dd / periods
		
		def annualized_return(data):
			cret = data.add(1).cumprod()
			if len(data) >= 12:
				# per_ret = data.pct_change().dropna()
				# cret = cret.add(1).cumprod()
				ann_return = cret.iloc[-1] ** (12 / (len(data) - 1)) - 1
				return ann_return
			else:
				return (cret.iloc[-1] / cret.iloc[0]) - 1
		
		def treynor_ratio(returns, market, rf):
			# excess return vs portfolio beta
			er = annualized_return(returns)
			# rf = ann_rfr_rate(rf)
			return (er - rf) / beta(returns, market)
		
		def sharpe_ratio(returns, rf):
			# er = np.mean(returns)
			er = annualized_return(returns)
			return (er - rf) / (vol(returns) * np.sqrt(12))
		
		def information_ratio(returns, benchmark):
			diff = returns - benchmark
			return np.mean(diff) / vol(diff)
		
		def modigliani_ratio(returns, benchmark, rf):
			er = annualized_return(returns)
			np_rf = np.empty(len(returns))
			np_rf.fill(rf)
			rdiff = returns - np_rf
			bdiff = benchmark - np_rf
			return (er - rf) * (vol(rdiff) / vol(bdiff)) + rf
		
		def excess_var(returns, rf, alpha):
			er = annualized_return(returns)
			return (er - rf) / var(returns, alpha)
		
		def conditional_sharpe_ratio(returns, rf, alpha):
			er = annualized_return(returns)
			return (er - rf) / cvar(returns, alpha)
		
		def omega_ratio(returns, rf, target=0):
			er = annualized_return(returns)
			return (er - rf) / lpm(returns, target, 1)
		
		def sortino_ratio(returns, rf, target=0):
			# Excess returns below threshold returns
			er = annualized_return(returns)
			return (er - rf) / math.sqrt(lpm(returns, target, 2))
		
		def kappa_three_ratio(returns, rf, target=0):
			er = annualized_return(returns)
			return (er - rf) / math.pow(lpm(returns, target, 3), float(1 / 3))
		
		def gain_loss_ratio(returns, target=0):
			# $ gains per $ risk
			return hpm(returns, target, 1) / lpm(returns, target, 1)
		
		def upside_potential_ratio(returns, target=0):
			return hpm(returns, target, 1) / math.sqrt(lpm(returns, target, 2))
		
		def calmar_ratio(returns, rf):
			# excess return vs max DD
			er = annualized_return(returns)
			return (er - rf) / max_dd(returns)
		
		def sterling_ratio(returns, rf, periods):
			# excess returns vs avg DD
			er = annualized_return(returns)
			return (er - rf) / average_dd(returns, periods)
		
		def burke_ratio(returns, rf, periods):
			er = annualized_return(returns)
			return (er - rf) / math.sqrt(average_dd_squared(returns, periods))
		
		rf = annualized_return(reg_df.rfr)
		mkt = reg_df.Benchmark
		cols = reg_df.columns[:2]
		
		ls_beta = [beta(reg_df[c], mkt) for c in cols]
		ls_var = [var(reg_df[c], .05) for c in cols]
		ls_cvar = [cvar(reg_df[c], .05) for c in cols]
		ls_dd = [dd(reg_df[c], 36) for c in cols]
		ls_max_dd = [max_dd(reg_df[c]) for c in cols]
		ls_avg_dd = [average_dd(reg_df[c], 36) for c in cols]
		ls_avg_dd_sq = [average_dd_squared(reg_df[c], 36) for c in cols]
		ls_treynor = [treynor_ratio(reg_df[c], mkt, rf) for c in cols]
		ls_sharpe = [sharpe_ratio(reg_df[c], rf) for c in cols]
		ls_info_ratio = [information_ratio(reg_df[c], mkt) for c in cols]
		ls_modigliani = [modigliani_ratio(reg_df[c], mkt, rf) for c in cols]
		ls_excess_var = [excess_var(reg_df[c], rf, .05) for c in cols]
		ls_cond_sharpe = [conditional_sharpe_ratio(reg_df[c], rf, .05) for c in cols]
		ls_omega = [omega_ratio(reg_df[c], rf, 0) for c in cols]
		ls_sortino = [sortino_ratio(reg_df[c], rf, 0) for c in cols]
		ls_kappa = [kappa_three_ratio(reg_df[c], rf, 0) for c in cols]
		ls_gain_to_loss = [gain_loss_ratio(reg_df[c], 0) for c in cols]
		ls_upside_potential = [upside_potential_ratio(reg_df[c], 0) for c in cols]
		ls_calmar = [calmar_ratio(reg_df[c], rf) for c in cols]
		ls_sterling = [sterling_ratio(reg_df[c], rf, 12) for c in cols]
		ls_burke = [burke_ratio(reg_df[c], rf, 12) for c in cols]
		
		index = ['YTD', 'Annualized Return - 1Y', 'Annualized Return - 3Y', 'Annualized Return - 5Y',
				 'Annualized Return - 10Y'
			, 'Annualized Return - Inception', 'Annualized Risk 1Y', 'Annualized Risk 3Y', 'Annualized Risk 5Y',
				 'Annualized Risk 10Y', 'Annualized Risk Inception',
				 'Sharpe 1Y', 'Sharpe 3Y', 'Sharpe 5Y', 'Sharpe 10Y', 'Sharpe Inception', 'Beta', 'Regr-Beta',
				 'R Square', 'Annualized Alpha', 'VAR(5%)', 'CVAR(5%)', 'DD', 'MaxDD', 'AvgDD', 'AvgDD_SQ',
				 'Treynor Ratio', 'Sharpe Ratio', 'Information Ratio', 'Modigliani Ratio', 'Excess VAR (5%)',
				 'Conditional Sharpe Ratio (5%)', 'Omega Ratio', 'Sortino Ratio', 'Kappa Three', 'Gain to Loss',
				 'Upside Potential Ratio', 'Calmar Ratio', 'Sterling Ratio', 'Burke Ratio', '% CAGR', '$ Growth']
		
		cols = reg_df.columns
		cols = cols.drop(['Benchmark', 'rfr'])
		results = pd.DataFrame(index=index, columns=cols)
		
		# -----------------------------Returns--------------------------------------------------------
		results.loc['YTD'] = stats_df.loc['YTD', :].tolist()
		results.loc['Annualized Return - 1Y'] = stats_df.loc['Annualized Return (% 1Y)', :].tolist()
		results.loc['Annualized Return - 3Y'] = stats_df.loc['Annualized Return (% 3Y)', :].tolist()
		results.loc['Annualized Return - 5Y'] = stats_df.loc['Annualized Return (% 5Y)', :].tolist()
		results.loc['Annualized Return - 10Y'] = stats_df.loc['Annualized Return (% 10Y)', :].tolist()
		results.loc['Annualized Return - Inception'] = stats_df.loc['Annualized Return (% inception)', :].tolist()
		
		# -----------------------------Risk--------------------------------------------------------
		results.loc['Annualized Risk 1Y'] = stats_df.loc['Annualized Risk (% 1Y)', :].tolist()
		results.loc['Annualized Risk 3Y'] = stats_df.loc['Annualized Risk (% 3Y)', :].tolist()
		results.loc['Annualized Risk 5Y'] = stats_df.loc['Annualized Risk (% 5Y)', :].tolist()
		results.loc['Annualized Risk 10Y'] = stats_df.loc['Annualized Risk (% 10Y)', :].tolist()
		results.loc['Annualized Risk Inception'] = stats_df.loc['Annualized Risk (% inception)', :].tolist()
		
		# -----------------------------Sharpe--------------------------------------------------------
		results.loc['Sharpe 1Y'] = stats_df.loc['Sharpe Ratio (1Y)', :].tolist()
		results.loc['Sharpe 3Y'] = stats_df.loc['Sharpe Ratio (3Y)', :].tolist()
		results.loc['Sharpe 5Y'] = stats_df.loc['Sharpe Ratio (5Y)', :].tolist()
		results.loc['Sharpe 10Y'] = stats_df.loc['Sharpe Ratio (10Y)', :].tolist()
		results.loc['Sharpe Inception'] = stats_df.loc['Sharpe Ratio (inception)', :].tolist()
		
		# -----------------------------Beta-----------------------------------------------------------
		results.loc['Beta'] = ls_beta
		results.loc['Regr-Beta'] = regdf.Beta.values.tolist()
		
		# -----------------------------R Square-----------------------------------------------------------
		results.loc['R Square'] = regdf.Rsq.values.tolist()
		
		# -----------------------------Alpha-----------------------------------------------------------
		results.loc['Annualized Alpha'] = regdf.Alpha.values.tolist()
		
		# -----------------------------VAR-----------------------------------------------------------
		results.loc['VAR(5%)'] = ls_var
		
		# -----------------------------CVAR-----------------------------------------------------------
		results.loc['CVAR(5%)'] = ls_cvar
		
		# -----------------------------DD-----------------------------------------------------------
		# results.loc['DD'] = ls_dd
		results.loc['DD'] = abs(round(monthly_drawdown.mean(), 2))
		
		# -----------------------------MaxDD-----------------------------------------------------------
		# results.loc['MaxDD'] = ls_max_dd
		results.loc['MaxDD'] = abs(round(max_monthly_drawdown.min(), 2))
		
		# -----------------------------DD-----------------------------------------------------------
		results.loc['AvgDD'] = ls_avg_dd
		
		# -----------------------------DD-----------------------------------------------------------
		results.loc['AvgDD_SQ'] = ls_avg_dd_sq
		
		# -----------------------------Treynor-----------------------------------------------------------
		results.loc['Treynor Ratio'] = ls_treynor
		
		# -----------------------------Sharpe Ratio-----------------------------------------------------------
		results.loc['Sharpe Ratio'] = ls_sharpe
		
		# -----------------------------Information Ratio-----------------------------------------------------------
		results.loc['Information Ratio'] = ls_info_ratio
		
		# -----------------------------Modigliani Ratio-----------------------------------------------------------
		results.loc['Modigliani Ratio'] = ls_modigliani
		
		# -----------------------------Excess VAR-----------------------------------------------------------
		results.loc['Excess VAR (5%)'] = ls_excess_var
		
		# -----------------------------Conditional Sharpe-----------------------------------------------------------
		results.loc['Conditional Sharpe Ratio (5%)'] = ls_cond_sharpe
		
		# -----------------------------Omega-----------------------------------------------------------
		results.loc['Omega Ratio'] = ls_omega
		
		# -----------------------------Sortino-----------------------------------------------------------
		results.loc['Sortino Ratio'] = ls_sortino
		
		# -----------------------------Kappa Three-----------------------------------------------------------
		results.loc['Kappa Three'] = ls_kappa
		
		# -----------------------------Gain to Loss-----------------------------------------------------------
		results.loc['Gain to Loss'] = ls_gain_to_loss
		
		# -----------------------------Upside Potential Ratio---------------------------------------------------------
		results.loc['Upside Potential Ratio'] = ls_upside_potential
		
		# -----------------------------Calmar Ratio-----------------------------------------------------------
		results.loc['Calmar Ratio'] = ls_calmar
		
		# -----------------------------Sterling Ratio-----------------------------------------------------------
		results.loc['Sterling Ratio'] = ls_sterling
		
		# -----------------------------Burke Ratio-----------------------------------------------------------
		results.loc['Burke Ratio'] = ls_burke
		
		# -----------------------------CAGR-----------------------------------------------------------
		results.loc['% CAGR'] = stats_df.loc['% CAGR'].values.tolist()
		
		# -----------------------------$ Growth-----------------------------------------------------------
		results.loc['$ Growth'] = stats_df.loc['$ Growth'].values.tolist()
		
		# ------------------------------Correlation-----------------------------------------------------
		corrmat = reg_df.corr()
		ls_corr = corrmat.Benchmark
		ls_corr = ls_corr[:2].tolist()
		results.loc['Correlation'] = ls_corr
	
	except Exception as e:
		print("Exception occurred calculating the ratios: ", e)
	
	results.to_csv(file_path + fia_name + str(fia_wt) + '_detail statistics.csv')
	
	report_index = ['YTD', 'Annualized Return - 1Y', 'Annualized Return - 3Y', 'Annualized Return - 5Y',
					'Annualized Return - 10Y', 'Annualized Return - Inception', 'Beta', 'R Square', 'Correlation',
					'Annualized Alpha', 'Annualized Risk Inception', 'Sharpe Inception', 'AvgDD',
					'MaxDD', '% CAGR', '$ Growth']
	
	report_df = pd.DataFrame(index=report_index, columns=results.columns)
	for idx in report_index:
		report_df.loc[idx] = results.loc[idx]
	
	report_df.to_csv(file_path + fia_name + str(fia_wt) + '_mkt statistics.csv')
	
	print("Analytics complete")


def adhoc_metrics(fpath):
	data = pd.read_csv(fpath + "combined_nav.csv", index_col='Date', parse_dates=True)
	sdate = data.index[0].strftime('%m/%d/%Y')
	edate = data.index[-1:][0].strftime('%m/%d/%Y')
	
	broad_mkt = data_from_bloomberg2(['SPXT Index', 'SBMMTB3 Index'], 'TOT_RETURN_INDEX_NET_DVDS', sdate, edate,
									 'DAILY',
									 'NON_TRADING_WEEKDAYS')
	broad_mkt = broad_mkt.resample('BM', closed='right').last()
	data.loc[:, 'broad_mkt'] = broad_mkt[broad_mkt.columns[0]]
	data.loc[:, 'cash'] = broad_mkt[broad_mkt.columns[1]]
	
	monthly_returns = data.pct_change().fillna(0)
	excess_return = monthly_returns.sub(monthly_returns.cash, axis=0)
	
	cr_ret = monthly_returns.add(1).prod()
	ann_ret = cr_ret ** (12 / len(monthly_returns)) - 1
	ann_std = monthly_returns.std() * np.sqrt(12)
	broad_mkt_excret = ann_ret['broad_mkt'] - ann_ret['cash']
	broad_mkt_sharpe = broad_mkt_excret / ann_std['broad_mkt']
	sharpe_df = pd.DataFrame(broad_mkt_sharpe, index=['SP500'], columns=['Sharpe Ratio'])
	sharpe_df.to_csv(fpath + "broad_mkt_sharpe_ratio.csv")
	# ---Base------------
	X = excess_return[excess_return.columns[2]]
	X = sm.add_constant(X)
	Y = excess_return[excess_return.columns[0]]
	model = sm.OLS(Y, X).fit()
	
	base_alpha = model.params[0]
	base_beta = model.params[1]
	
	# ---FIA POrtfolio------------
	X = excess_return[excess_return.columns[2]]
	X = sm.add_constant(X)
	Y = excess_return[excess_return.columns[1]]
	model = sm.OLS(Y, X).fit()
	
	fia_alpha = model.params[0]
	fia_beta = model.params[1]
	
	# -----create dataframe---------------
	regr_data = pd.DataFrame([[base_alpha, base_beta], [fia_alpha, fia_beta]], columns=['Alpha (Monthly)', 'Beta'],
							 index=['Base Portfolio', 'FIA Portfolio'])
	
	regr_data.index.name = 'vsSP500'
	
	regr_data.to_csv(fpath + "portfolio_beta.csv")


def formatted_output(filepath):
	# --This method formats data to be used with the marketing slide------------
	
	# qualitative_from_bloomberg(start, end)
	
	read_qual_data = pd.read_csv(src + "bbg_qualitative_data.csv", index_col=[0])
	
	read_asset_weights = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='asset_weights', index_col=[0],
									   parse_dates=True)
	# read_asset_weights = pd.read_csv(src + "asset_weights.csv", index_col='Asset')
	
	read_info = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='info', index_col=[0],
							  parse_dates=True)
	portfolio_value = read_info.loc['total_portfolio_value', 'Value']
	
	perc_wts = read_asset_weights.copy()
	
	perc_wts = (100 * perc_wts).round(2).astype(str) + '%'
	read_qual_data.reset_index(inplace=True)
	
	read_qual_data.set_index('ticker', inplace=True)
	
	dollar_assets = (portfolio_value * read_asset_weights).round(0)
	
	dollar_assets.rename(columns={'base': 'dollar_base', 'fia': 'dollar_fia'}, inplace=True)
	joined_df = pd.concat([read_qual_data, read_asset_weights, dollar_assets], axis=1)
	
	joined_df.loc['Cash'] = joined_df.loc['Cash'].fillna('Cash')
	joined_df.loc['FIA'] = joined_df.loc['FIA'].fillna('FIA')
	
	joined_df.loc['Cash', 'fund_expense_ratio'] = 0.0
	joined_df.loc['FIA', 'fund_expense_ratio'] = 0.0
	
	allocation_slide = joined_df.loc[:, ['base', 'fia', 'fund_mkt_cap_focus']].groupby('fund_mkt_cap_focus').sum()
	allocation_slide.loc[:, 'Asset Reallocation (%)'] = (allocation_slide.base - allocation_slide.fia) * -1
	# allocation_slide = allocation_slide.apply(lambda x: round(x * 100, 2))
	allocation_slide = allocation_slide.applymap(lambda x: "{:.2%}".format(x))
	allo_rename = {'base': 'Current Portfolio', 'fia': 'Proposed Portfolio'}
	allocation_slide.rename(columns=allo_rename, inplace=True)
	allocation_slide = allocation_slide[['Current Portfolio', 'Asset Reallocation (%)', 'Proposed Portfolio']]
	allocation_slide.index.name = 'Asset Classes'
	allocation_slide.rename(index={'FIA': 'Principal Protected'}, inplace=True)
	
	allocation_slide.to_csv(filepath + 'allocation_slide.csv')
	
	joined_df.loc[:, 'fund_expense_ratio'] = joined_df.loc[:, 'fund_expense_ratio'].astype(str) + '%'
	copy_joined = joined_df.copy()
	
	copy_joined.reset_index(inplace=True)
	
	base_cols = ['fund_asset_class_focus', 'security_name', 'level_0', 'base', 'dollar_base', 'fund_expense_ratio']
	fia_cols = ['fund_asset_class_focus', 'security_name', 'level_0', 'fia', 'dollar_fia', 'fund_expense_ratio']
	
	base_data = copy_joined[base_cols]
	base_data = base_data[base_data.dollar_base > 0]
	
	fia_data = copy_joined[fia_cols]
	fia_data = fia_data[fia_data.dollar_fia > 0]
	
	rename_base = {'security_name': 'Asset Class/Investment Product', 'level_0': 'Symbol',
				   'base': 'Current % of Portfolio', 'dollar_base': 'Current Investment Value',
				   'fund_expense_ratio': 'Manager Fee/Expense Ratio'}
	
	rename_fia = {'security_name': 'Asset Class/Investment Product', 'level_0': 'Symbol',
				  'fia': 'Proposed % of Portfolio', 'dollar_fia': 'Proposed Investment Value',
				  'fund_expense_ratio': 'Manager Fee/Expense Ratio'}
	
	base_data.rename(columns=rename_base, inplace=True)
	
	fia_data.rename(columns=rename_fia, inplace=True)
	
	base_data.loc[:, 'Current % of Portfolio'] = round(100 * base_data.loc[:, 'Current % of Portfolio'], 2).astype(
		str) + '%'
	
	fia_data.loc[:, 'Proposed % of Portfolio'] = round(100 * fia_data.loc[:, 'Proposed % of Portfolio'], 2).astype(
		str) + '%'
	
	base_data.to_csv(filepath + 'base_slide.csv')
	
	fia_data.to_csv(filepath + 'fia_slide.csv')


def summary_output_file(advfee, p_life, rebalfreq, fia_alloc, filepath, fia):
	df1 = pd.read_csv(filepath + "combined_nav.csv")
	sd = df1.Date.to_list()[0]
	ed = df1.Date.to_list()[-1]
	df2 = pd.DataFrame([fia, fia_alloc, advfee, p_life, rebalfreq, sd, ed, 'D:G'],
					   index=['FIA', 'Fia_alloc', 'advfee', 'cdsc', 'rebalance freq', 'start', 'end', 'EF Data'])
	
	df3 = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='asset_weights', index_col=[0],
						parse_dates=True)
	
	# df3 = pd.read_csv(src + "asset_weights.csv", index_col=['Asset'])
	df3 = df3.loc[:, ['base', 'fia']]
	df3_temp = round(100 * df3, 2).astype(str) + '%'
	df3 = pd.concat([df3, df3_temp], axis=1)
	df4 = pd.read_csv(filepath + fia + str(fia_alloc) + "_mkt statistics.csv", index_col=[0])
	
	# df5 = pd.read_csv(filepath + fia + str(fia_alloc) + "_detail statistics.csv")
	
	df6 = pd.read_csv(filepath + "yearly_returns_table.csv", index_col=[0])
	df6.loc[df6.index[0], 'since_incpetion'] = 100 * df4.loc['Annualized Return - Inception', df4.columns[0]]
	df6.loc[df6.index[1], 'since_incpetion'] = 100 * df4.loc['Annualized Return - Inception', df4.columns[1]]
	df6.loc[:, 'Max'] = df6.max(axis=1)
	df6.loc[:, 'Min'] = df6.min(axis=1)
	
	df7 = pd.read_csv(filepath + "asset_sharpe_data.csv")
	# df8 = pd.read_csv(filepath + "uc_optimized_ef_filtered_data.csv")
	# df9 = pd.read_csv(file_path + "base_portfolio_returns_table.csv", index_col='Date')
	# df10 = pd.read_csv(file_path + fia + "_portfolio_returns_table.csv", index_col='Date')
	df11 = pd.read_csv(file_path + "assets_annual_return.csv", index_col='Symbol')
	df12 = pd.read_csv(file_path + "portfolio_beta.csv", index_col=['vsSP500'])
	df13 = pd.read_csv(file_path + "broad_mkt_sharpe_ratio.csv")
	df14 = pd.read_csv(file_path + "allocation_slide.csv", index_col='Asset Classes')
	df15 = pd.read_csv(file_path + "base_slide.csv", index_col=[0])
	df16 = pd.read_csv(file_path + "fia_slide.csv", index_col=[0])
	df17 = pd.read_csv(src + "bbg_qualitative_data.csv", index_col=[0])
	df18 = pd.read_csv(filepath + fia + str(fia_alloc) + "_detail statistics.csv", index_col=[0])
	
	writer = pd.ExcelWriter(filepath + fia + '_summary.xlsx', engine='xlsxwriter')
	df1.to_excel(writer, sheet_name='time_series')
	df2.to_excel(writer, sheet_name='summary')
	df3.to_excel(writer, sheet_name='Allocations (%)')
	df4.to_excel(writer, sheet_name='selective_statistics')
	# df5.to_excel(writer, sheet_name='detail_statistics')
	df6.T.to_excel(writer, sheet_name='yearly_returns')
	df7.to_excel(writer, sheet_name='asset_sharpe_ratio')
	# df8.to_excel(writer, sheet_name='efficient_frontier')
	# df9.to_excel(writer, sheet_name='base_portfolio_returns')
	# df10.to_excel(writer, sheet_name='fia_portfolio_returns')
	df11.to_excel(writer, sheet_name='assets_metrics')
	df12.to_excel(writer, sheet_name='beta_vs_sp500')
	df13.to_excel(writer, sheet_name='SP500_sharpe')
	df14.to_excel(writer, sheet_name='allocation_slide')
	df15.to_excel(writer, sheet_name='current_portfolio')
	df16.to_excel(writer, sheet_name='proposed_portfolio')
	
	# ----  Ledger for Allocation ------
	df15.set_index('Symbol', inplace=True)
	
	df15['fund_asset_class_focus'] = df15['fund_asset_class_focus'].fillna('Equity')
	df15['Manager Fee/Expense Ratio'] = 0.0
	df16['fund_asset_class_focus'] = df16['fund_asset_class_focus'].fillna('Equity')
	df16.set_index('Symbol', inplace=True)
	merged_df = pd.concat([df15, df16], axis=1)
	# merged_df = pd.merge(df16, df15, left_index=True, right_index=True)
	merged_df.fillna(0, inplace=True)
	merged_df.loc[:, 'Asset Reallocation $'] = merged_df['Proposed Investment Value'].sub(
		merged_df['Current Investment Value'])
	drop_col = ['Current % of Portfolio', 'Manager Fee/Expense Ratio', 'Proposed % of Portfolio']
	merged_df.drop(drop_col, axis=1, inplace=True)
	col_order = ['fund_asset_class_focus', 'Asset Class/Investment Product', 'Current Investment Value',
				 'Asset Reallocation $', 'Proposed Investment Value']
	merged_df = merged_df[col_order]
	merged_df.set_index('fund_asset_class_focus', inplace=True)
	# df19 = merged_df[:-1].sort_index()
	# df19 = df19.append(merged_df.loc['FIA'], ignore_index=False)
	
	df17.to_excel(writer, sheet_name='bbg_classification')
	df18.to_excel(writer, sheet_name='detail_statistics')
	# df19.to_excel(writer, sheet_name='portfolio_ledger')
	
	writer.save()


def format_marketing_slides(file_path, fia_cols):
	src_alloc = pd.read_excel(file_path + fia_cols + '_summary.xlsx', sheet_name='Allocations (%)', index_col=[0])
	src_stats = pd.read_excel(file_path + fia_cols + '_summary.xlsx', sheet_name='selective_statistics', index_col=[0])
	
	# --------------------Page 2 for Base Portfolio (3 Pages)-----------------------
	src_page2 = pd.read_excel(file_path + fia_cols + '_summary.xlsx', sheet_name='current_portfolio', index_col=[1])
	src_page2.drop(src_page2.columns[0], axis=1, inplace=True)
	
	page_2_a = src_alloc.loc[:, 'base']
	page_2_a = page_2_a[page_2_a != 0]
	page_2_a.name = 'Current Portfolio'
	page_2_b = src_stats.loc[['Annualized Return - Inception', 'Annualized Risk Inception'], 'Base_Portfolio']
	page_2_b.name = 'Current Portfolio'
	
	src_page2['Current % of Portfolio'] = src_page2['Current % of Portfolio'].apply(lambda x: float(x.split('%')[0]))
	src_page2['Manager Fee/Expense Ratio'] = src_page2['Manager Fee/Expense Ratio'].apply(
		lambda x: float(x.split('%')[0]))
	src_page2['wtd_exp_ratio'] = 0.01 * src_page2['Current % of Portfolio'].multiply(
		0.01 * src_page2['Manager Fee/Expense Ratio'])
	val = ['Current % of Portfolio', 'Current Investment Value', 'Manager Fee/Expense Ratio', 'wtd_exp_ratio']
	page_2_c = pd.pivot_table(src_page2, values=val,
							  index=[src_page2.index, src_page2['Asset Class/Investment Product'], src_page2.Symbol],
							  aggfunc=np.sum, margins=True)
	
	# ---------------------------PAGE 2 ENDS-------------------------------------------------------
	
	# --------------------Page 3 for Proposed Portfolio (3 Pages)-----------------------
	src_page3 = pd.read_excel(file_path + fia_cols + '_summary.xlsx', sheet_name='proposed_portfolio', index_col=[1])
	src_page3.drop(src_page3.columns[0], axis=1, inplace=True)
	
	page_3_a = src_alloc.loc[:, 'fia']
	page_3_a = page_3_a[page_3_a != 0]
	page_3_a.name = 'Proposed Portfolio'
	proposed_fia = 'Portfolio_with_' + fia_cols
	page_3_b = src_stats.loc[['Annualized Return - Inception', 'Annualized Risk Inception'], proposed_fia]
	page_3_b.name = 'Proposed Portfolio'
	src_page3['Proposed % of Portfolio'] = src_page3['Proposed % of Portfolio'].apply(lambda x: float(x.split('%')[0]))
	src_page3['Manager Fee/Expense Ratio'] = src_page3['Manager Fee/Expense Ratio'].apply(
		lambda x: float(x.split('%')[0]))
	src_page3['wtd_exp_ratio'] = 0.01 * src_page3['Proposed % of Portfolio'].multiply(
		0.01 * src_page3['Manager Fee/Expense Ratio'])
	val = ['Proposed % of Portfolio', 'Proposed Investment Value', 'Manager Fee/Expense Ratio', 'wtd_exp_ratio']
	page_3_c = pd.pivot_table(src_page3, values=val,
							  index=[src_page3.index, src_page3['Asset Class/Investment Product'], src_page3.Symbol],
							  aggfunc=np.sum, margins=True)
	
	# ---------------------------PAGE 3 ENDS-------------------------------------------------------
	
	# ---------------------Page 4 Portfolio Asset Allocation Targets ---------------------------
	src_bbg_alloc = pd.read_excel(file_path + fia_cols + '_summary.xlsx', sheet_name='bbg_classification',
								  index_col=[0])
	copy_alloc = src_alloc.copy()
	src_bbg_alloc = src_bbg_alloc.loc[:, ['fund_asset_class_focus', 'ticker', 'size_classification']]
	
	# --------------------Re-classification of the assets -------------------
	src_bbg_alloc['clean_classification'] = np.where(src_bbg_alloc['fund_asset_class_focus'] == 'Fixed Income',
													 'Fixed Income', src_bbg_alloc['size_classification'])
	
	src_bbg_alloc.set_index('ticker', inplace=True)
	src_bbg_alloc.drop(['fund_asset_class_focus', 'size_classification'], axis=1, inplace=True)
	copy_alloc['classification'] = src_bbg_alloc
	copy_alloc.loc['Cash', 'classification'] = 'Cash'
	copy_alloc.loc['FIA', 'classification'] = 'Principal Protected'
	copy_alloc.drop(['base.1', 'fia.1'], axis=1, inplace=True)
	grouped_by_class = copy_alloc.groupby(copy_alloc.classification).apply(sum)
	grouped_by_class.drop('classification', axis=1, inplace=True)
	grouped_by_class['Asset Reclassification (%)'] = round(
		grouped_by_class.loc[:, 'fia'].sub(grouped_by_class.loc[:, 'base']), 3)
	grouped_by_class.index.name = 'Asset Classes'
	grouped_by_class = grouped_by_class[['base', 'Asset Reclassification (%)', 'fia']]
	grouped_by_class.rename(columns={'base': 'Current Portfolio', 'fia': 'Proposed Portfolio'}, inplace=True)
	page_4 = grouped_by_class.apply(lambda x: round(x, 3))
	
	# -------------------------------PAGE 4 ENDS---------------------------------------------------
	
	# -------------------------------Page 5 - Portfolio Ledger-------------------------------------
	# page_5 = pd.read_excel(file_path + fia_cols + '_summary.xlsx', sheet_name='portfolio_ledger', index_col=[0])
	
	# ---------------------------PAGE 5 ENDS ------------------------------------------------------
	
	# --------------------------Page 6 - NAV growth and portfolio Statistics
	page_6a = pd.read_excel(file_path + fia_cols + '_summary.xlsx', sheet_name='time_series', index_col=[1],
							parse_Dates=True)
	page_6a.drop(page_6a.columns[0], axis=1, inplace=True)
	page_6a = page_6a.rename(columns={page_6a.columns[0]: 'Current Portfolio',
									  page_6a.columns[1]: 'Proposed Portfolio'})
	
	# --------------------------------Stats Table-------------
	src_yearly_ret = pd.read_excel(file_path + fia_cols + '_summary.xlsx', sheet_name='yearly_returns', index_col=[0])
	select_index = ['Annualized Return - Inception', 'Annualized Risk Inception', 'MaxDD']
	page_6b = src_stats.loc[select_index, :]
	page_6b.loc['Maximum Return'] = 0.01 * src_yearly_ret.loc['Max', :]
	page_6b.loc['Minimum Return'] = 0.01 * src_yearly_ret.loc['Min', :]
	page_6b = page_6b.rename(columns={page_6b.columns[0]: 'Current Portfolio',
									  page_6b.columns[1]: 'Proposed Portfolio'})
	page_6b.reset_index(inplace=True)
	new_names = ['Annualized Average Return', 'Annualized Average Risk', 'Maximum Drawdown', 'Maximum Return',
				 'Minimum Return']
	page_6b['Names'] = new_names
	page_6b.set_index('Names', inplace=True)
	page_6b.drop('index', axis=1, inplace=True)
	page_6b.index.name = ''
	
	# -----------------------------Page 6 ENDs--------------------------------------------
	
	# -------------------------------------Page 7 and 8 - Annual Returns Table----------------------------
	page_7 = src_yearly_ret.copy()
	page_7 = page_7[:-2]
	page_7 = page_7.rename(columns={page_7.columns[0]: 'Current Portfolio', page_7.columns[1]: 'Proposed Portfolio'})
	
	# --------------------------------------Page 7 and 8 Ends----------------------------------------------
	
	# ----------------------------Page 9 - Risk and Returns --------------------------------------------
	page_9 = page_6b.copy()
	page_9 = page_9[:-3]
	
	# ---------------------------Page 9 ENDS------------------------------------------------------------
	
	# --------------------------Page 10 Sharpe ratios and Stress Analysis -------------------------------
	
	sp_500_prices = pd.read_csv(src + 'benchmark_prices.csv', index_col='Date', parse_dates=True)
	sp_500_prices = sp_500_prices['SPXT Index']
	sp_500_prices = sp_500_prices.resample('BM', closed='right').last()
	
	page_10a = pd.read_excel(file_path + fia_cols + '_summary.xlsx', sheet_name='SP500_sharpe', index_col=[1])
	page_10a.drop(page_10a.columns[0], axis=1, inplace=True)
	page_10a.index.name = ''
	page_10a.loc['Current Portfolio'] = src_stats.loc['Sharpe Inception', src_stats.columns[0]]
	page_10a.loc['Proposed Portfolio'] = src_stats.loc['Sharpe Inception', src_stats.columns[1]]
	
	nav_df = page_6a.copy()
	data_beg = pd.to_datetime(nav_df.index[0])
	start1 = pd.to_datetime('2/28/2000')
	
	# -- adjust the start data for cases like Zebra index as it started in the middle of the tech bubble in 2000------
	adj_start = max(data_beg, start1)
	end1 = pd.to_datetime('10/31/2002')
	
	start2 = pd.to_datetime('09/30/2007')
	end2 = pd.to_datetime('02/28/2009')
	
	nav_df.index = pd.to_datetime(nav_df.index)
	
	# ---Tech Bubble DataFrame ---------------------
	tech_bubble1 = nav_df.loc[adj_start:end1, :]
	# tech_sp = sp_500_prices.copy()
	tech_bubble = tech_bubble1.copy()
	
	if (end1 - adj_start).days > 0:
		tech_bubble.loc[:, 'S&P 500 TR'] = sp_500_prices.loc[adj_start: end1]
		cols = ['S&P 500 TR', 'Current Portfolio', 'Proposed Portfolio']
		tech_bubble = tech_bubble[cols]
		page_10b = pd.DataFrame(index=['Tech Bubble (2000-2002)', 'Financial Crisis (2007-2009)'
			, 'Annualized Return - Since Inception'],
								columns=tech_bubble.columns)
		tech_ret = list(tech_bubble.iloc[-1] / tech_bubble.iloc[0] - 1)
		page_10b.iloc[0] = tech_ret
	
	else:
		cols = ['S&P 500 TR', 'Current Portfolio', 'Proposed Portfolio']
		page_10b = pd.DataFrame(index=['Tech Bubble (2000-2002)', 'Financial Crisis (2007-2009)'
			, 'Annualized Return - Since Inception'],
								columns=cols)
		page_10b.iloc[0] = 'Not Available'
	
	# ----------Financial crisis ----------------------
	if (start2 - adj_start).days > 0:
		fin_bubble1 = nav_df.loc[start2: end2, :]
		fin_bubble = fin_bubble1.copy()
		fin_bubble.loc[:, 'S&P 500 TR'] = sp_500_prices.loc[start2: end2]
		fin_bubble = fin_bubble[cols]
		fin_ret = list(fin_bubble.iloc[-1] / fin_bubble.iloc[0] - 1)
		page_10b.iloc[1] = fin_ret
	else:
		page_10b.iloc[1] = 'Not Available'
	
	sp_inception = sp_500_prices.loc[pd.to_datetime(nav_df.index[0]):]
	gr_ret = sp_inception.iloc[-1] / sp_inception.iloc[0]
	inv_yrs = len(sp_inception) / 12
	sp_ann_ret = gr_ret ** (1 / inv_yrs) - 1
	
	page_10a.loc['Current Portfolio'] = src_stats.loc['Sharpe Inception', src_stats.columns[0]]
	
	page_10b.loc[page_10b.index[2], page_10b.columns[0]] = sp_ann_ret
	page_10b.loc[page_10b.index[2], page_10b.columns[1]] = src_stats.loc['Annualized Return - Inception',
																		 src_stats.columns[0]]
	page_10b.loc[page_10b.index[2], page_10b.columns[2]] = src_stats.loc['Annualized Return - Inception',
																		 src_stats.columns[1]]
	
	# -------------Page 11 - Detail Portfolio Statistics-----------------------------------
	page_11 = src_stats.copy()
	page_11.rename(columns={page_11.columns[0]: 'Current Portfolio',
							page_11.columns[1]: 'Proposed Portfolio'}, inplace=True)
	
	writer = pd.ExcelWriter(file_path + fia_cols + '_formatted_summary.xlsx', engine='xlsxwriter')
	
	# -------------Base Portfolio Page 2
	page_2_a.to_excel(writer, sheet_name='current_pie')
	page_2_b.to_excel(writer, sheet_name='current_matrix')
	page_2_c.to_excel(writer, sheet_name='current_holdings')
	
	# ----Proposed Portfolio Page 3
	page_3_a.to_excel(writer, sheet_name='proposed_pie')
	page_3_b.to_excel(writer, sheet_name='proposed_matrix')
	page_3_c.to_excel(writer, sheet_name='proposed_holdings')
	
	# ---Asset Reallocation between Current and Proposed Portfolio
	page_4.to_excel(writer, sheet_name='asset_reallocation')
	
	# -----Portfolio Ledger --------------------------
	# page_5.to_excel(writer, sheet_name='portfolio_ledger')
	
	# -------Portfolio NAV and short Stats----------------------
	page_6a.to_excel(writer, sheet_name='nav_timeseries')
	page_6b.to_excel(writer, sheet_name='stats_table')
	
	# ------------Yearly returns for 2 slides 7 & 8--------------
	page_7.to_excel(writer, sheet_name='yearly_returns')
	
	# ------------Risk Return Matrix--------------
	page_9.to_excel(writer, sheet_name='risk_return_matrix')
	
	# ----------------Sharpe Ratio and Stress test table--------------
	page_10a.to_excel(writer, sheet_name='comparative_sharpe')
	page_10b.to_excel(writer, sheet_name='scenario_matrix')
	
	# ----------------------Selective Statistics Table-----------------
	page_11.to_excel(writer, sheet_name='detail_stats')
	
	# -------Product Information-----------------------
	# page_12 = pd.read_excel(file_path + fia_cols + '_summary.xlsx', sheet_name='summary', index_col=[0])
	page_12 = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='info', index_col=[0])
	page_12.to_excel(writer, sheet_name='product_info')
	
	# --------------Page 13 - BBG Classification Data----------------------
	page_13 = pd.read_csv(src + "bbg_qualitative_data.csv", index_col=[0], parse_dates=True)
	page_13.to_excel(writer, sheet_name='bbg_classifications')
	
	writer.save()


def model_for_bav_time_series(raw_index_name, par_rate, spread, term, prodname, start_date, rdate, livedate,
							  iname, optimize=False):
	# optimize parameter is True if the dynamic par rate is used from the reverse_bsm_model at each rebalance date else
	# is False to use the par rates from Paul's spreadsheet with option margin, initial margin optimization which is
	# a faster method. Ensure the "input_dynamic_par_rate_from_excel.csv" is updated using Paul's excel model, dynamic
	# par rate from the BAV sheets. Once update ensure the par rate csv file is updated to include these par rates for
	# the rebalance dates.
	
	# --------------Read the raw FIA index price file----------------------------------
	df = pd.read_csv(fia_src + "fia_index_data_from_bbg.csv", usecols=['ticker', raw_index_name],
					 parse_dates=True, skiprows=[1, 2])
	# df.ticker = pd.to_datetime(df.ticker, format='%Y-%m-%d').dt.strftime('%m/%d/%Y')
	df.set_index('ticker', inplace=True, drop=True)
	df.dropna(inplace=True)
	if pd.to_datetime(df.index[0]) != start_date:
		start_date = start_date.strftime('%Y-%m-%d')
		df = df.loc[start_date:]
		pct_chg = df.pct_change().fillna(0)
		df = 100 * pct_chg.add(1).cumprod()
	else:
		df = df
	
	df.index = pd.to_datetime(df.index)
	# --------------------------Read the par rates for the time series------------------------------------
	
	# TODO: Build a csv file with the par rates for all the fia indices and read the file
	new_dir = fia_src + prodname + "/"
	# par_rate_df = pd.read_csv(src + "par_rate_model.csv", usecols=['Date', 'par_rate'], parse_dates=True)
	par_rate_df = pd.read_csv(new_dir + prodname + "_par_rate_model.csv", usecols=['Date', 'par_rate'],
							  parse_dates=True)
	
	par_rate_df.set_index('Date', inplace=True, drop=True)
	par_rate_df = par_rate_df.apply(lambda x: round(x, 2))
	s1 = par_rate_df.index[0]
	s2 = par_rate_df[-1:].index[0]
	dates = pd.date_range(s1, s2, freq='1M') - pd.offsets.MonthBegin(1)
	par_rate_df['bom_index'] = dates
	par_rate_df.reset_index(inplace=True, drop=False)
	# par_rate_df.bom_index = pd.to_datetime(par_rate_df.bom_index, format='%Y-%m-%d').dt.strftime('%m/%d/%Y')
	par_rate_df.set_index('bom_index', inplace=True, drop=True)
	par_rate_df.index = pd.to_datetime(par_rate_df.index)
	
	if iname == 'Mozaic':
		
		dyn_par_from_excel = pd.read_excel(fia_src + 'py_daily_par_spread.xlsx', sheet_name='daily_par_mozaic',
										   index_col=[0], parse_dates=True)
		dyn_par_from_excel = dyn_par_from_excel.loc[start_date:, :]
	
	# dyn_par_from_excel = pd.read_excel(fia_src + 'input_mozaic_par_spread.xlsx', sheet_name='par_rates',
	# index_col=[0], parse_dates=True)
	
	else:
		dyn_par_from_excel = pd.read_excel(fia_src + 'py_daily_par_spread.xlsx', sheet_name='daily_par_zebra',
										   index_col=[0], parse_dates=True)
		dyn_par_from_excel = dyn_par_from_excel.loc[start_date:, :]
	
	# dyn_par_from_excel = pd.read_excel(fia_src + 'input_zebra_par_spread.xlsx', sheet_name='par_rates',
	# index_col=[0], parse_dates=True)
	
	if not optimize:
		# dyn_par_from_excel = pd.read_csv(src + "input_dynamic_par_rate_from_excel.csv", index_col='Date',
		#                                  parse_dates=True)
		dyn_par_from_excel = dyn_par_from_excel.loc[:, prodname]
		idx = dyn_par_from_excel.index.to_list()
		for i in idx:
			par_rate_df.loc[par_rate_df.index == i, 'par_rate'] = dyn_par_from_excel[i]
	
	start_rebal_date = livedate
	# par_rate_df = pd.read_excel(src + "fia_par_rates.xlsx", index_col='Date', parse_dates=True)
	# fia_par_rate = par_rate_df.copy()
	# fia_par_rate = fia_par_rate.loc[:, par_rate_model]
	
	# ------------------------to adjust the par rate if the starting date is not the BOM, ffill the par rates--------
	psuedo_fia = pd.DataFrame(par_rate_df, index=df.index)
	psuedo_fia.ffill(inplace=True)
	fia_par_rate = psuedo_fia.copy()
	fia_par_rate = fia_par_rate.loc[:, 'par_rate']
	
	# ----------------select any custom date to run the analysis---------------------------------
	df = df[start_date:]
	ls_dates = []
	sdate = df.index[0]
	edate = date.today()
	check_date = df.index[0]
	start_date = df.index[0]
	
	# -----------------------generate list of daily dates--------------------------------------------
	while check_date <= edate:
		ls_date_range = pd.date_range(start=sdate, end=sdate + pd.DateOffset(years=term))
		ls_dates.extend(ls_date_range)
		sdate = ls_date_range[-1]
		check_date = ls_dates[-1]
	
	ls_dates = list(dict.fromkeys(ls_dates))
	
	# ------------------generate list of rebalance dates-------------------------------
	rebal_dates = [start_date]
	while start_date < edate:
		new_date = start_date + relativedelta(years=term)
		rebal_dates.append(new_date)
		start_date = new_date
	
	raw_dates = list(df.index)
	
	# clean_rebal_date = list(dyn_par_from_excel.index)
	clean_rebal_date = rebal_dates
	# ---------------------------------Math and formula's to calculate the variables---------------------------
	for i in range(len(clean_rebal_date) - 1):
		if i < len(clean_rebal_date) - 1:
			
			# TODO check for the .loc assignment to address the warning signs
			raw_index_pos = df.index.to_list().index(clean_rebal_date[i])
			new_start = df.index.to_list()[raw_index_pos + 1]
			df.loc[clean_rebal_date[i]: clean_rebal_date[i + 1], 'Term'] = i + 1
			df.loc[clean_rebal_date[i]: clean_rebal_date[i + 1], 'start_date'] = clean_rebal_date[i]
			df.loc[clean_rebal_date[i]: clean_rebal_date[i + 1], 'end_date'] = clean_rebal_date[i + 1]
			df.loc[new_start: clean_rebal_date[i + 1], 'index_base_during_term'] = \
				df.loc[clean_rebal_date[i], raw_index_name]
		
		else:
			break
	for i in range(len(clean_rebal_date) - 1):  # was rebal_dates
		raw_index_pos = df.index.to_list().index(clean_rebal_date[i])
		new_start = df.index.to_list()[raw_index_pos + 1]
		
		if optimize:
			
			df.loc[new_start:clean_rebal_date[i + 1], 'term_par_rate'] = fia_par_rate.loc[
				clean_rebal_date[i]]
		else:
			df.loc[new_start:clean_rebal_date[i + 1], 'term_par_rate'] = fia_par_rate.loc[
				clean_rebal_date[i]]
	
	bav_df = df.copy()
	
	# --read the merged csv file with  dynamic model par from Paul's file and live rates. Using the data create a
	# a dataframe with daily dates and fill forward the par rate.  This is needed for the new model.
	
	# read_model_live_par_file = pd.read_csv(src + "input_dynamic_par_rate_from_excel_Mozaic.csv", index_col='Date',
	#                                        parse_dates=True)
	
	read_model_live_par_file = dyn_par_from_excel
	newidx = pd.date_range(read_model_live_par_file.index[0], read_model_live_par_file.index[-1])
	new_df = pd.DataFrame(index=newidx)
	idx = read_model_live_par_file.index
	
	for i in range(len(idx) - 1):
		# beg_par_rate = read_model_live_par_file.loc[idx[i], prodname]
		beg_par_rate = read_model_live_par_file.loc[idx[i]]
		start = idx[i]
		end = idx[i + 1] - pd.DateOffset(days=1)
		new_df.loc[start:end, prodname] = beg_par_rate
	new_df.ffill(inplace=True)
	
	bav_df.loc[:, 'term_par_rate'] = new_df.loc[:, prodname]
	bav_df.loc[:, 'term_par_rate'] = bav_df.loc[:, 'term_par_rate'].ffill()
	
	# ----read the merged live and model spread data and generate a dataframe on a daily basis with fill forward spread-
	if iname == 'Mozaic':
		read_model_live_spread_file = pd.read_excel(fia_src + 'input_mozaic_par_spread.xlsx', sheet_name='spread',
													index_col=[0], parse_dates=True)
	# read_model_live_spread_file = pd.read_csv(src + "input_spreads_mozaic.csv", index_col='Date',
	# parse_dates=True)
	else:
		read_model_live_spread_file = pd.read_excel(fia_src + 'input_zebra_par_spread.xlsx', sheet_name='spread',
													index_col=[0], parse_dates=True)
	
	# read_model_live_spread_file = pd.read_csv(src + "input_spreads_zebra.csv", index_col='Date', parse_dates=True)
	
	newidx = pd.date_range(read_model_live_spread_file.index[0], read_model_live_spread_file.index[-1])
	new_df = pd.DataFrame(index=newidx)
	idx = read_model_live_spread_file.index
	
	for i in range(len(idx) - 1):
		beg_spread_rate = read_model_live_spread_file.loc[idx[i], prodname]
		start = idx[i]
		end = idx[i + 1] - pd.DateOffset(days=1)
		new_df.loc[start:end, prodname] = beg_spread_rate
	new_df.ffill(inplace=True)
	
	bav_df.loc[:, 'spread'] = new_df.loc[:, prodname]
	bav_df.loc[:, 'spread'] = bav_df.loc[:, 'spread'].ffill()
	
	bav_df.loc[:, 'term_par_rate'] = bav_df.loc[:, 'term_par_rate'].apply(lambda x: min(par_rate, x))
	# bav_df.loc[:, 'term_par_rate'] = bav_df.loc[:, 'term_par_rate'].apply(lambda x: round(x, 2))
	bav_df.loc[:, 'index_cumm_return_time_par'] = bav_df['term_par_rate'] * (
			(bav_df[raw_index_name] - bav_df['index_base_during_term']) / bav_df['index_base_during_term'])
	bav_df.loc[:, 'index_cumm_return_time_par'].fillna(0, inplace=True)
	
	bav_df['dummy'] = pd.to_datetime(bav_df.index)
	bav_df['day_count'] = (bav_df['dummy'] - pd.to_datetime(bav_df['start_date'])).dt.days
	
	bav_df['term_day_count'] = (bav_df['end_date'] - bav_df['start_date']).dt.days
	bav_df.loc[:, 'term_starts'] = [1 if d in idx else 0 for d in raw_dates]  # replace idx by clean_rebal_date
	
	for index, row in bav_df.iterrows():
		if bav_df.loc[index, 'term_starts'] != 1:
			bav_df.loc[index, 'elapsed_time'] = (
					(bav_df.loc[index, 'dummy'] - bav_df.loc[index, 'start_date']).days / 365)
		else:
			bav_df.loc[index, 'elapsed_time'] = 3
	
	bav_df['elapsed_time'] = bav_df['elapsed_time']
	bav_df.loc[:, 'AR'] = ((1 + bav_df.loc[:, 'index_cumm_return_time_par']) ** (
			1 / bav_df.loc[:, 'elapsed_time'])) - 1 - bav_df.loc[:, 'spread']
	bav_df.loc[:, 'annualized_return'] = bav_df.loc[:, 'AR'].apply(lambda x: max(0, x))
	bav_df.loc[:, 'cumm_product_return'] = (1 + bav_df.loc[:, 'annualized_return']) ** bav_df.loc[:, 'elapsed_time']
	
	for i in range(len(raw_dates)):
		if i == 0:
			bav_df.loc[bav_df.index[i], 'floor'] = 100
			bav_df.loc[bav_df.index[i], 'bav'] = 100
		
		elif bav_df.loc[bav_df.index[i - 1], 'term_starts'] != 1:
			bav_df.loc[bav_df.index[i], 'floor'] = bav_df.loc[bav_df.index[i - 1], 'floor']
			a = bav_df.loc[bav_df.index[i], 'floor']
			b = bav_df.loc[bav_df.index[i], 'cumm_product_return']
			c = a * b
			bav_df.loc[bav_df.index[i], 'bav'] = c
		
		else:
			bav_df.loc[bav_df.index[i], 'floor'] = bav_df.loc[bav_df.index[i - 1], 'bav']
			a = bav_df.loc[bav_df.index[i], 'floor']
			b = bav_df.loc[bav_df.index[i], 'cumm_product_return']
			c = a * b
			bav_df.loc[bav_df.index[i], 'bav'] = c
	
	cols = ['term_starts', 'Term', raw_index_name, 'start_date', 'end_date', 'index_base_during_term', 'term_par_rate',
			'index_cumm_return_time_par', 'day_count', 'term_day_count', 'elapsed_time', 'annualized_return',
			'cumm_product_return', 'floor', 'bav']
	
	bav_df = bav_df[cols]
	
	# ---------------Check if the folder is available, if not then create one----------------------------
	new_dir = src + prodname.lower() + "/"
	
	if not os.path.exists(new_dir):
		os.mkdir(new_dir)
		print("Directory ", new_dir, " Created ")
	else:
		print("Directory ", new_dir, " already exists")
	
	# -------------save a copy to the FIA directory-------------------------------
	# bav_df.to_csv(new_dir + prodname + "_bav.csv", float_format="%.4f")
	#
	# bav_df.to_csv(src + "bav.csv")
	#
	# bav_df = bav_df[cols]
	# new_dir = src + "dynamic_bavs" + "/"
	# bav_df.to_csv(new_dir + prodname + "_bav.csv")
	return bav_df


def get_fama_french():
	"""Fully Functional Method to pull and parse FAMA FRENCH Factors. Can be added to daily batch file"""
	# Web url
	file_name_5F = 'F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'
	file_name_mom = 'F-F_Momentum_Factor_daily_CSV.zip'
	
	ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/" + file_name_5F
	
	# Download the file and save it
	# We will name it fama_french.zip file
	
	urllib.request.urlretrieve(ff_url, 'fama_french.zip')
	zip_file = zipfile.ZipFile('fama_french.zip', 'r')
	
	# Next we extact the file data
	zip_file.extractall()
	
	# Make sure you close the file after extraction
	zip_file.close()
	
	# Now open the CSV file
	ff_factors_5F = pd.read_csv('F-F_Research_Data_5_Factors_2x3_daily.csv', skiprows=3, index_col=0)
	# We want to find out the row with NULL value
	# We will skip these rows
	
	# ff_row = ff_factors.isnull().any(1).nonzero()[0][0]
	
	# Read the csv file again with skipped rows
	# ff_factors = pd.read_csv('F-F_Research_Data_Factors.csv', skiprows=3, nrows=ff_row, index_col=0)
	
	# Format the date index
	ff_factors_5F.index = pd.to_datetime(ff_factors_5F.index, format='%Y%m%d')
	
	# --------------------Momentum Factors----------------------------------------------
	ff_url_mom = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/" + file_name_mom
	
	# Download the file and save it
	# We will name it fama_french.zip file
	
	urllib.request.urlretrieve(ff_url_mom, 'fama_french_momo.zip')
	zip_file = zipfile.ZipFile('fama_french_momo.zip', 'r')
	
	# Next we extact the file data
	zip_file.extractall()
	
	# Make sure you close the file after extraction
	zip_file.close()
	
	# Now open the CSV file
	ff_factors_mom = pd.read_csv('F-F_Momentum_Factor_daily.csv', skiprows=13, skipfooter=1, index_col=0,
								 parse_dates=True, engine='python')
	# ff_factors_mom = pd.read_csv('F-F_Momentum_Factor_daily.csv', skiprows=3, index_col=0)
	# We want to find out the row with NULL value
	# We will skip these rows
	
	# ff_row = ff_factors.isnull().any(1).nonzero()[0][0]
	
	# Read the csv file again with skipped rows
	# ff_factors = pd.read_csv('F-F_Research_Data_Factors.csv', skiprows=3, nrows=ff_row, index_col=0)
	
	# Format the date index
	ff_factors_mom.index = pd.to_datetime(ff_factors_mom.index, format='%Y%m%d')
	ff_factors_mom.rename(columns={ff_factors_mom.columns[0]: 'MOM'}, inplace=True)
	
	# Format dates to end of month
	# ff_factors.index = ff_factors.index + pd.offsets.MonthEnd()
	
	# -----------------merge 5 factors and momentum factors---------------
	merged_factors = pd.merge(ff_factors_5F, ff_factors_mom, left_index=True, right_index=True)
	# Convert from percent to decimal
	merged_factors = merged_factors.apply(lambda x: x / 100)
	cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'RF']
	merged_factors = merged_factors[cols]
	merged_factors = merged_factors.loc['1996-10-01':, :]
	merged_factors.to_csv(src + "fama_french_factors_daily.csv")


def asset_prices_using_fama_french_only(sdate, edate):
	"""Fully Function Fama French Only regression model"""
	regr_score = {}
	regr_coef = {}
	regr_alpha = {}
	dict_regr_ret = {}
	merged_ret = {}
	dict_asset_prices = {}
	
	comp_ann = {}
	comp_reg = {}
	comp_beg_data = {}
	
	read_ff_factors = pd.read_csv(src + 'fama_french_factors_daily.csv', index_col=[0], parse_dates=True)
	read_ff_factors = read_ff_factors.loc[sdate:edate, :]
	asset_prices = pd.read_csv(src + 'daily_prices_bbg.csv', index_col=[0], parse_dates=True)
	asset_prices = asset_prices.loc[sdate:edate, :]
	excess_X = read_ff_factors.copy()
	excess_X = excess_X.subtract(excess_X.RF, axis=0)
	excess_X.loc[:, excess_X.columns[0]] = read_ff_factors.loc[:, read_ff_factors.columns[0]]
	excess_X.drop('RF', axis=1, inplace=True)
	
	for i in np.arange(len(asset_prices.columns)):
		y_orig = asset_prices.loc[:, asset_prices.columns[i]]
		
		# -------Comparison Block Begins for Orignal vs Regression returns-----------------
		comp_data = y_orig.copy()
		comp_data.dropna(inplace=True)
		comp_data = comp_data.pct_change().fillna(0)
		beg_date = comp_data.index[0]
		end_date = comp_data.index[-1]
		# --------------Block Ends-----------------------------------------
		
		y_change = y_orig.pct_change().ffill()
		rf = read_ff_factors.RF
		excess_y = y_change - rf
		excess_y.name = asset_prices.columns[i]
		regr_df = pd.merge(excess_y, excess_X, left_index=True, right_index=True)
		regr_df.dropna(axis=0, inplace=True)
		y = regr_df.loc[:, regr_df.columns[0]]
		X = regr_df.loc[:, regr_df.columns[1:]]
		model = LinearRegression().fit(X, y)
		
		# --------------Save alpha, beta and R2 in dictionary---------------------
		regr_score.update({regr_df.columns[0]: model.score(X, y)})
		regr_coef.update({regr_df.columns[0]: model.coef_})
		regr_alpha.update({regr_df.columns[0]: model.intercept_})
		
		# ----------Calculate regression returns and merge will original returns-------------------
		regressed_returns = model.intercept_ + excess_X.dot(regr_coef[regr_df.columns[0]]) + rf
		
		# ---------------Compare Original Returns vs Regressed Returns---------------------------
		comp_regr = regressed_returns.copy()
		comp_regr = comp_regr.loc[beg_date: end_date]
		comp_regr.name = 'regressed_returns'
		joined_comp = pd.merge(comp_regr, comp_data, left_index=True, right_index=True)
		joined_comp.iloc[0] = 0.0
		cp = joined_comp.add(1).cumprod()
		cp = cp.iloc[-1]
		fact = 252 / len(joined_comp)
		ann_ret_comp = cp ** fact - 1
		comp_reg.update({regr_df.columns[0]: ann_ret_comp[0]})
		comp_ann.update({regr_df.columns[0]: ann_ret_comp[1]})
		comp_beg_data.update({regr_df.columns[0]: beg_date})
		# -----------------Comparision Block Ends------------------------------------------------
		
		orig_regr_merged = y_change.fillna(regressed_returns)
		dict_regr_ret.update({regr_df.columns[0]: regressed_returns})
		merged_ret.update({regr_df.columns[0]: orig_regr_merged})
		
		# -------------------Convert daily returns to daily NAV's------------------------------
		orig_regr_merged.iloc[0] = 0.0
		net_asset_value = orig_regr_merged.add(1).cumprod() * 100
		dict_asset_prices.update({regr_df.columns[0]: net_asset_value})
	
	# ---------------Compare Original Returns vs Regressed Returns---------------------------
	df_ann = pd.DataFrame.from_dict(comp_ann, orient='index', columns=['OrigRet'])
	df_reg = pd.DataFrame.from_dict(comp_reg, orient='index', columns=['Regr'])
	merged_comp = pd.merge(df_ann, df_reg, left_index=True, right_index=True)
	merged_comp.loc[:, 'date'] = comp_beg_data.values()
	merged_comp.to_csv(dest_research + 'ff_unadj_comp.csv')
	# -------------------------Block Ends------------------------------------------
	
	regression_based_navs = pd.DataFrame.from_dict(dict_asset_prices)
	regression_based_navs.ffill(inplace=True)
	regression_based_navs.to_csv(src + 'asset_price.csv')
	regression_r2 = pd.DataFrame.from_dict(regr_score, orient='index', columns=['R2'])
	regression_r2.to_csv(dest_research + 'ff_only_r2.csv')
	
	# -----------------Annual Metrics---------------------
	month_samp = regression_based_navs.resample('M', closed='right').last()
	ann_fact = 12 / len(month_samp)
	ann_ret = (month_samp.iloc[-1] / month_samp.iloc[0]) ** ann_fact - 1
	ann_risk = month_samp.pct_change().fillna(0).std() * np.sqrt(12)
	ann_df = pd.DataFrame({'ret_unadj': ann_ret, 'risk_unadj': ann_risk})
	ann_df.to_csv(dest_research + 'unadj_metrics.csv')
	
	# ---------------------Annual Metrics - Last 5 Years---------------------
	last_five = month_samp.iloc[-60:]
	ann_fact = 12 / len(last_five)
	ann_ret = (last_five.iloc[-1] / last_five.iloc[0]) ** ann_fact - 1
	ann_risk = last_five.pct_change().fillna(0).std() * np.sqrt(12)
	ann_df = pd.DataFrame({'ret_ff_unadj': ann_ret, 'risk_ff_unadj': ann_risk})
	ann_df.to_csv(dest_research + 'unadj_ff_metrics_last5.csv')


def asset_prices_using_fama_french_adjusted(sdate, edate):
	"""Fully Function Fama French Only regression model based on t-values greater than 1.96"""
	regr_score = {}
	regr_coef = {}
	regr_alpha = {}
	dict_regr_ret = {}
	merged_ret = {}
	dict_asset_prices = {}
	
	comp_ann = {}
	comp_reg = {}
	comp_beg_data = {}
	
	read_ff_factors = pd.read_csv(src + 'fama_french_factors_daily.csv', index_col=[0], parse_dates=True)
	read_ff_factors = read_ff_factors.loc[sdate:edate, :]
	asset_prices = pd.read_csv(src + 'daily_prices_bbg.csv', index_col=[0], parse_dates=True)
	asset_prices = asset_prices.loc[sdate:edate, :]
	excess_X = read_ff_factors.copy()
	excess_X = excess_X.subtract(excess_X.RF, axis=0)
	excess_X.loc[:, excess_X.columns[0]] = read_ff_factors.loc[:, read_ff_factors.columns[0]]
	excess_X.drop('RF', axis=1, inplace=True)
	
	for i in np.arange(len(asset_prices.columns)):
		y_orig = asset_prices.loc[:, asset_prices.columns[i]]
		
		# -------------------Comp Data----------------------------------
		comp_data = y_orig.copy()
		comp_data.dropna(inplace=True)
		comp_data = comp_data.pct_change().fillna(0)
		beg_date = comp_data.index[0]
		end_date = comp_data.index[-1]
		# -------------------------Comp Block Ends---------------------
		
		y_change = y_orig.pct_change().ffill()
		rf = read_ff_factors.RF
		excess_y = y_change - rf
		excess_y.name = asset_prices.columns[i]
		regr_df = pd.merge(excess_y, excess_X, left_index=True, right_index=True)
		regr_df.dropna(axis=0, inplace=True)
		y = regr_df.loc[:, regr_df.columns[0]]
		X = regr_df.loc[:, regr_df.columns[1:]]
		X = sm.add_constant(X)
		model = sm.OLS(y, X).fit()
		
		# --------------Filter Regressors using abs t-values greater than 1.96-----------
		abs_t = abs(model.tvalues)
		abs_t = abs_t.drop('const', axis=0)
		t_val_list = list(abs_t[abs_t >= 1.96].index)
		
		excess_X = excess_X[t_val_list]
		regr_df = pd.merge(excess_y, excess_X, left_index=True, right_index=True)
		regr_df.dropna(axis=0, inplace=True)
		y = regr_df.loc[:, regr_df.columns[0]]
		X = regr_df.loc[:, regr_df.columns[1:]]
		X = sm.add_constant(X)
		model = sm.OLS(y, X).fit()
		
		# --------------Save alpha, beta and R2 in dictionary---------------------
		regr_score.update({regr_df.columns[0]: model.rsquared})
		regr_coef.update({regr_df.columns[0]: model.params.values[1:]})
		regr_alpha.update({regr_df.columns[0]: model.params.values[:1][0]})
		
		# ----------Calculate regression returns and merge will original returns-------------------
		regressed_returns = model.params.values[:1][0] + excess_X.dot(regr_coef[regr_df.columns[0]]) + rf
		
		# ---------------Compare Original Returns vs Regressed Returns---------------------------
		comp_regr = regressed_returns.copy()
		comp_regr = comp_regr.loc[beg_date: end_date]
		comp_regr.name = 'regressed_returns'
		joined_comp = pd.merge(comp_regr, comp_data, left_index=True, right_index=True)
		joined_comp.iloc[0] = 0.0
		cp = joined_comp.add(1).cumprod()
		cp = cp.iloc[-1]
		fact = 252 / len(joined_comp)
		ann_ret_comp = cp ** fact - 1
		comp_reg.update({regr_df.columns[0]: ann_ret_comp[0]})
		comp_ann.update({regr_df.columns[0]: ann_ret_comp[1]})
		comp_beg_data.update({regr_df.columns[0]: beg_date})
		# -----------------Comparision Block Ends------------------------------------------------
		
		orig_regr_merged = y_change.fillna(regressed_returns)
		dict_regr_ret.update({regr_df.columns[0]: regressed_returns})
		merged_ret.update({regr_df.columns[0]: orig_regr_merged})
		
		# -------------------Convert daily returns to daily NAV's------------------------------
		orig_regr_merged.iloc[0] = 0.0
		net_asset_value = orig_regr_merged.add(1).cumprod() * 100
		dict_asset_prices.update({regr_df.columns[0]: net_asset_value})
	
	# ---------------Compare Original Returns vs Regressed Returns---------------------------
	df_ann = pd.DataFrame.from_dict(comp_ann, orient='index', columns=['OrigRet'])
	df_reg = pd.DataFrame.from_dict(comp_reg, orient='index', columns=['Regr'])
	merged_comp = pd.merge(df_ann, df_reg, left_index=True, right_index=True)
	merged_comp.loc[:, 'date'] = comp_beg_data.values()
	merged_comp.to_csv(dest_research + 'ff_adj_comp.csv')
	
	# -----------------Compare Block Ends------------------------------------------------
	regression_based_navs = pd.DataFrame.from_dict(dict_asset_prices)
	regression_based_navs.ffill(inplace=True)
	regression_based_navs.to_csv(dest_research + 'ff_only_adjusted.csv')
	regression_r2 = pd.DataFrame.from_dict(regr_score, orient='index', columns=['R2'])
	regression_r2.to_csv(dest_research + 'ff_only_r2_adjusted.csv')
	
	# -----------------Annual Metrics---------------------
	month_samp = regression_based_navs.resample('M', closed='right').last()
	ann_fact = 12 / len(month_samp)
	ann_ret = (month_samp.iloc[-1] / month_samp.iloc[0]) ** ann_fact - 1
	ann_risk = month_samp.pct_change().fillna(0).std() * np.sqrt(12)
	ann_df = pd.DataFrame({'ret_adj': ann_ret, 'risk_adj': ann_risk})
	ann_df.to_csv(dest_research + 'adj_metrics.csv')
	
	# ---------------------Annual Metrics - Last 5 Years---------------------
	last_five = month_samp.iloc[-60:]
	ann_fact = 12 / len(last_five)
	ann_ret = (last_five.iloc[-1] / last_five.iloc[0]) ** ann_fact - 1
	ann_risk = last_five.pct_change().fillna(0).std() * np.sqrt(12)
	ann_df = pd.DataFrame({'ret_ff_adj': ann_ret, 'risk_ff_adj': ann_risk})
	ann_df.to_csv(dest_research + 'adj_ff_metrics_last5.csv')


def asset_prices_using_fama_french_lasso(sdate, edate):
	from sklearn.model_selection import GridSearchCV
	from sklearn.model_selection import RepeatedKFold
	from sklearn.linear_model import Lasso
	
	"""Fully Function Fama French Only regression model"""
	regr_score = {}
	regr_coef = {}
	regr_alpha = {}
	dict_regr_ret = {}
	merged_ret = {}
	dict_asset_prices = {}
	comp_ann = {}
	comp_reg = {}
	comp_beg_data = {}
	read_ff_factors = pd.read_csv(src + 'fama_french_factors_daily.csv', index_col=[0], parse_dates=True)
	read_ff_factors = read_ff_factors.loc[sdate:edate, :]
	read_ff_factors.loc[:, 'Mkt-RF'] = read_ff_factors.loc[:, 'Mkt-RF'] + read_ff_factors.loc[:, 'RF']
	read_ff_factors.rename(columns={'Mkt-RF': 'Fama_Mkt'}, inplace=True)
	read_ff_factors.drop('RF', axis=1, inplace=True)
	
	asset_prices = pd.read_csv(src + 'daily_prices_bbg.csv', index_col=[0], parse_dates=True)
	asset_prices_orig = asset_prices.copy()
	
	asset_prices = asset_prices.loc[sdate:edate, :]
	
	bm_prices = pd.read_csv(src + 'benchmark_prices.csv', index_col=[0], parse_dates=True)
	bm_prices = bm_prices.loc[sdate:edate, :]
	bm_prices_change = bm_prices.copy()
	bm_prices_change = bm_prices_change.pct_change().fillna(0)
	
	merged_indep = pd.merge(bm_prices_change, read_ff_factors, left_index=True, right_index=True)
	
	X = merged_indep.copy()
	
	for i in np.arange(len(asset_prices.columns)):
		y_orig = asset_prices.loc[:, asset_prices.columns[i]]
		
		comp_data = y_orig.copy()
		comp_data.dropna(inplace=True)
		comp_data = comp_data.pct_change().fillna(0)
		beg_date = comp_data.index[0]
		end_date = comp_data.index[-1]
		
		# ---------Data Set up for Regression---------------------
		y_change = y_orig.pct_change().ffill()
		y_change.name = asset_prices.columns[i]
		regr_df = pd.merge(y_change, X, left_index=True, right_index=True)
		regr_df.dropna(axis=0, inplace=True)
		y = regr_df.loc[:, regr_df.columns[0]]
		X = regr_df.loc[:, regr_df.columns[1:]]
		X = X.replace([np.inf, -np.inf], np.nan).ffill()
		model = Lasso()
		cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
		model = LassoCV(alphas=np.arange(0, 1, 0.01), cv=cv, n_jobs=-1, max_iter=100000, tol=1e-10)
		model.fit(X, y)
		print('alpha: %f' % model.alpha_)
		print(model.coef_)
		print(len(model.coef_))
		
		# --------------Save alpha, beta and R2 in dictionary---------------------
		regr_score.update({regr_df.columns[0]: model.score(X, y)})
		regr_coef.update({regr_df.columns[0]: model.coef_})
		regr_alpha.update({regr_df.columns[0]: model.intercept_})
		
		# ----------Calculate regression returns and merge will original returns-------------------
		regressed_returns = model.intercept_ + merged_indep.dot(regr_coef[regr_df.columns[0]])
		
		# ---------------Compare Original Returns vs Regressed Returns---------------------------
		comp_regr = regressed_returns.copy()
		comp_regr = comp_regr.loc[beg_date: end_date]
		comp_regr.name = 'regressed_returns'
		joined_comp = pd.merge(comp_regr, comp_data, left_index=True, right_index=True)
		joined_comp.iloc[0] = 0.0
		cp = joined_comp.add(1).cumprod()
		cp = cp.iloc[-1]
		fact = 252 / len(joined_comp)
		ann_ret_comp = cp ** fact - 1
		comp_reg.update({regr_df.columns[0]: ann_ret_comp[0]})
		comp_ann.update({regr_df.columns[0]: ann_ret_comp[1]})
		comp_beg_data.update({regr_df.columns[0]: beg_date})
		# -----------------Comparision Block Ends------------------------------------------------
		
		orig_regr_merged = y_change.fillna(regressed_returns)
		dict_regr_ret.update({regr_df.columns[0]: regressed_returns})
		merged_ret.update({regr_df.columns[0]: orig_regr_merged})
		
		# -------------------Convert daily returns to daily NAV's------------------------------
		orig_regr_merged.iloc[0] = 0.0
		net_asset_value = orig_regr_merged.add(1).cumprod() * 100
		dict_asset_prices.update({regr_df.columns[0]: net_asset_value})
	
	regression_based_navs = pd.DataFrame.from_dict(dict_asset_prices)
	regression_based_navs.ffill(inplace=True)
	# ------Save the NAV File---------------------------------------
	regression_based_navs.to_csv(dest_research + 'ff_lasso.csv')
	
	regression_r2 = pd.DataFrame.from_dict(regr_score, orient='index', columns=['R2'])
	regression_r2.to_csv(dest_research + 'ff_lasso_r2.csv')
	
	# ---------------Compare Original Returns vs Regressed Returns---------------------------
	df_ann = pd.DataFrame.from_dict(comp_ann, orient='index', columns=['OrigRet'])
	df_reg = pd.DataFrame.from_dict(comp_reg, orient='index', columns=['Regr'])
	merged_comp = pd.merge(df_ann, df_reg, left_index=True, right_index=True)
	merged_comp.loc[:, 'date'] = comp_beg_data.values()
	merged_comp.to_csv(dest_research + 'ff_lasso_comp.csv')
	
	# ---------------------Annual Metrics---------------------
	month_samp = regression_based_navs.resample('M', closed='right').last()
	ann_fact = 12 / len(month_samp)
	ann_ret = (month_samp.iloc[-1] / month_samp.iloc[0]) ** ann_fact - 1
	ann_risk = month_samp.pct_change().fillna(0).std() * np.sqrt(12)
	ann_df = pd.DataFrame({'ret_ff_lasso': ann_ret, 'risk_ff_lasso': ann_risk})
	ann_df.to_csv(dest_research + 'ff_lasso_metrics.csv')
	
	# ---------------------Annual Metrics - Last 5 Years---------------------
	last_five = month_samp.iloc[-60:]
	ann_fact = 12 / len(last_five)
	ann_ret = (last_five.iloc[-1] / last_five.iloc[0]) ** ann_fact - 1
	ann_risk = last_five.pct_change().fillna(0).std() * np.sqrt(12)
	ann_df = pd.DataFrame({'ret_ff_lasso': ann_ret, 'risk_ff_lasso': ann_risk})
	ann_df.to_csv(dest_research + 'ff_lasso_metrics_last5.csv')
	
	# ----------------Last 60 for original prices------------------
	last5_orig = asset_prices_orig.resample('M', closed='right').last()
	month_samp = last5_orig.iloc[-60:]
	last_five = month_samp.iloc[-60:]
	ann_fact = 12 / len(last_five)
	ann_ret = (last_five.iloc[-1] / last_five.iloc[0]) ** ann_fact - 1
	ann_risk = last_five.pct_change().fillna(0).std() * np.sqrt(12)
	ann_df = pd.DataFrame({'ret_orig': ann_ret, 'risk_orig': ann_risk})
	ann_df.to_csv(dest_research + 'orig_metrics_last5.csv')
	print('test')


def portfolio_regression_using_fama_french_only(sdate, edate):
	"""Fully Function Fama French Only regression model"""
	regr_score = {}
	regr_coef = {}
	regr_alpha = {}
	dict_regr_ret = {}
	merged_ret = {}
	dict_asset_prices = {}
	
	comp_ann = {}
	comp_reg = {}
	comp_beg_data = {}
	
	read_ff_factors = pd.read_csv(src + 'fama_french_factors_daily.csv', index_col=[0], parse_dates=True)
	read_ff_factors = read_ff_factors.loc[sdate:edate, :]
	asset_prices = pd.read_csv(src + 'daily_prices_bbg.csv', index_col=[0], parse_dates=True)
	asset_prices = asset_prices.loc[sdate:edate, :]
	clean_prices = asset_prices.dropna()
	
	excess_X = read_ff_factors.copy()
	excess_X = excess_X.subtract(excess_X.RF, axis=0)
	excess_X.loc[:, excess_X.columns[0]] = read_ff_factors.loc[:, read_ff_factors.columns[0]]
	excess_X.drop('RF', axis=1, inplace=True)
	
	for i in np.arange(len(asset_prices.columns)):
		y_orig = asset_prices.loc[:, asset_prices.columns[i]]
		
		# -------Comparison Block Begins for Orignal vs Regression returns-----------------
		comp_data = y_orig.copy()
		comp_data.dropna(inplace=True)
		comp_data = comp_data.pct_change().fillna(0)
		beg_date = comp_data.index[0]
		end_date = comp_data.index[-1]
		# --------------Block Ends-----------------------------------------
		
		y_change = y_orig.pct_change().ffill()
		rf = read_ff_factors.RF
		excess_y = y_change - rf
		excess_y.name = asset_prices.columns[i]
		regr_df = pd.merge(excess_y, excess_X, left_index=True, right_index=True)
		regr_df.dropna(axis=0, inplace=True)
		y = regr_df.loc[:, regr_df.columns[0]]
		X = regr_df.loc[:, regr_df.columns[1:]]
		model = LinearRegression().fit(X, y)
		
		# --------------Save alpha, beta and R2 in dictionary---------------------
		regr_score.update({regr_df.columns[0]: model.score(X, y)})
		regr_coef.update({regr_df.columns[0]: model.coef_})
		regr_alpha.update({regr_df.columns[0]: model.intercept_})
		
		# ----------Calculate regression returns and merge will original returns-------------------
		regressed_returns = model.intercept_ + excess_X.dot(regr_coef[regr_df.columns[0]]) + rf
		
		# ---------------Compare Original Returns vs Regressed Returns---------------------------
		comp_regr = regressed_returns.copy()
		comp_regr = comp_regr.loc[beg_date: end_date]
		comp_regr.name = 'regressed_returns'
		joined_comp = pd.merge(comp_regr, comp_data, left_index=True, right_index=True)
		joined_comp.iloc[0] = 0.0
		cp = joined_comp.add(1).cumprod()
		cp = cp.iloc[-1]
		fact = 252 / len(joined_comp)
		ann_ret_comp = cp ** fact - 1
		comp_reg.update({regr_df.columns[0]: ann_ret_comp[0]})
		comp_ann.update({regr_df.columns[0]: ann_ret_comp[1]})
		comp_beg_data.update({regr_df.columns[0]: beg_date})
		# -----------------Comparision Block Ends------------------------------------------------
		
		orig_regr_merged = y_change.fillna(regressed_returns)
		dict_regr_ret.update({regr_df.columns[0]: regressed_returns})
		merged_ret.update({regr_df.columns[0]: orig_regr_merged})
		
		# -------------------Convert daily returns to daily NAV's------------------------------
		orig_regr_merged.iloc[0] = 0.0
		net_asset_value = orig_regr_merged.add(1).cumprod() * 100
		dict_asset_prices.update({regr_df.columns[0]: net_asset_value})
	
	# ---------------Compare Original Returns vs Regressed Returns---------------------------
	df_ann = pd.DataFrame.from_dict(comp_ann, orient='index', columns=['OrigRet'])
	df_reg = pd.DataFrame.from_dict(comp_reg, orient='index', columns=['Regr'])
	merged_comp = pd.merge(df_ann, df_reg, left_index=True, right_index=True)
	merged_comp.loc[:, 'date'] = comp_beg_data.values()
	merged_comp.to_csv(dest_research + 'ff_unadj_comp.csv')
	# -------------------------Block Ends------------------------------------------
	
	regression_based_navs = pd.DataFrame.from_dict(dict_asset_prices)
	regression_based_navs.ffill(inplace=True)
	regression_based_navs.to_csv(dest_research + 'ff_only.csv')
	regression_r2 = pd.DataFrame.from_dict(regr_score, orient='index', columns=['R2'])
	regression_r2.to_csv(dest_research + 'ff_only_r2.csv')
	
	# -----------------Annual Metrics---------------------
	month_samp = regression_based_navs.resample('M', closed='right').last()
	ann_fact = 12 / len(month_samp)
	ann_ret = (month_samp.iloc[-1] / month_samp.iloc[0]) ** ann_fact - 1
	ann_risk = month_samp.pct_change().fillna(0).std() * np.sqrt(12)
	ann_df = pd.DataFrame({'ret_unadj': ann_ret, 'risk_unadj': ann_risk})
	ann_df.to_csv(dest_research + 'unadj_metrics.csv')
	
	# ---------------------Annual Metrics - Last 5 Years---------------------
	last_five = month_samp.iloc[-60:]
	ann_fact = 12 / len(last_five)
	ann_ret = (last_five.iloc[-1] / last_five.iloc[0]) ** ann_fact - 1
	ann_risk = last_five.pct_change().fillna(0).std() * np.sqrt(12)
	ann_df = pd.DataFrame({'ret_ff_unadj': ann_ret, 'risk_ff_unadj': ann_risk})
	ann_df.to_csv(dest_research + 'unadj_ff_metrics_last5.csv')


def portfolio_level_regression_using_fama_french_only(sdate, edate):
	"""Fully Function Fama French Only regression model"""
	regr_score = {}
	regr_coef = {}
	regr_alpha = {}
	dict_regr_ret = {}
	merged_ret = {}
	dict_asset_prices = {}
	
	comp_ann = {}
	comp_reg = {}
	comp_beg_data = {}
	
	# -------read portfolio weights---------------
	wts_df = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='asset_weights', index_col=[0])
	wts_arr = np.array(wts_df.base)
	
	read_ff_factors = pd.read_csv(src + 'fama_french_factors_daily.csv', index_col=[0], parse_dates=True)
	read_ff_factors = read_ff_factors.loc[sdate:edate, :]
	# ---Monthly-----------
	# read_ff_factors = read_ff_factors.resample('W', closed='right').last()
	asset_prices = pd.read_csv(src + 'daily_prices_bbg.csv', index_col=[0], parse_dates=True)
	asset_prices = asset_prices.loc[sdate:edate, :]
	# -Monthly-------------
	# asset_prices = asset_prices.resample('W', closed='right').last()
	clean_prices = asset_prices.dropna()
	daily_change = clean_prices.pct_change().fillna(0)
	no_of_assets = len(daily_change.columns)
	wts_arr = wts_arr[:no_of_assets]
	
	# ---------Daily Portfolio Returns-------GROSS---------------------
	daily_port_ret = daily_change.dot(wts_arr)
	daily_port_ret.name = 'port_return'
	
	excess_X = read_ff_factors.copy()
	excess_X = excess_X.merge(daily_port_ret, left_index=True, right_index=True)
	excess_X = excess_X.subtract(excess_X.RF, axis=0)
	excess_X.loc[:, excess_X.columns[0]] = read_ff_factors.loc[:, read_ff_factors.columns[0]]
	excess_X.drop('RF', axis=1, inplace=True)
	y_orig = excess_X.loc[:, 'port_return']
	excess_X.drop('port_return', axis=1, inplace=True)
	excess_X = sm.add_constant(excess_X)
	
	# ----------------Fitting the OLS model---------------------------
	model = sm.OLS(y_orig, excess_X).fit()
	coef_arr = model.params
	alpha = coef_arr[0]
	beta = coef_arr[1:]
	hist = read_ff_factors.subtract(read_ff_factors.RF, axis=0)
	hist.drop('RF', axis=1, inplace=True)
	ret = hist.dot(beta) + read_ff_factors.RF + alpha
	orig_regr_df = pd.DataFrame({'regr': ret, 'orig': daily_port_ret})
	orig_regr_df.dropna(inplace=True)
	
	# --------------Portfolio Construction---------------------
	cr = ret.add(1).cumprod()
	cr.resample('BM', closed='right').last()
	monthly_ret = cr.resample('BM', closed='right').last()
	monthly_nav = cr.resample('BM', closed='right').last()
	monthly_ret = monthly_nav.pct_change().fillna(0)
	port_register = pd.DataFrame(monthly_ret)
	port_register = pd.DataFrame(monthly_ret, columns=['Returns'])
	adv_fess = 0.75 / 400
	fees = np.zeros(len(port_register))
	fees[::3] = adv_fess
	port_register.loc[:, 'fees'] = fees
	port_register.loc[:, 'ret_net_fees'] = port_register['Returns'] - port_register['fees']
	port_register.loc[port_register.index[0], 'ret_net_fees'] = 0.0
	port_register.loc[:, 'NAV'] = 1000000 * port_register['ret_net_fees'].add(1).cumprod()
	
	# TODO calculate portfolio returns with FIA using proportion of returns from above and FIA BAV
	
	# -------Comparison Block Begins for Orignal vs Regression returns-----------------
	comp_data = y_orig.copy()
	comp_data.dropna(inplace=True)
	comp_data = comp_data.pct_change().fillna(0)
	beg_date = comp_data.index[0]
	end_date = comp_data.index[-1]
	# --------------Block Ends-----------------------------------------
	
	y_change = y_orig.pct_change().ffill()
	rf = read_ff_factors.RF
	excess_y = y_change - rf
	excess_y.name = asset_prices.columns[0]
	regr_df = pd.merge(excess_y, excess_X, left_index=True, right_index=True)
	regr_df.dropna(axis=0, inplace=True)
	y = regr_df.loc[:, regr_df.columns[0]]
	X = regr_df.loc[:, regr_df.columns[1:]]
	model = LinearRegression().fit(X, y)
	
	# --------------Save alpha, beta and R2 in dictionary---------------------
	regr_score.update({regr_df.columns[0]: model.score(X, y)})
	regr_coef.update({regr_df.columns[0]: model.coef_})
	regr_alpha.update({regr_df.columns[0]: model.intercept_})
	
	# ----------Calculate regression returns and merge will original returns-------------------
	regressed_returns = model.intercept_ + excess_X.dot(regr_coef[regr_df.columns[0]]) + rf
	
	# ---------------Compare Original Returns vs Regressed Returns---------------------------
	comp_regr = regressed_returns.copy()
	comp_regr = comp_regr.loc[beg_date: end_date]
	comp_regr.name = 'regressed_returns'
	joined_comp = pd.merge(comp_regr, comp_data, left_index=True, right_index=True)
	joined_comp.iloc[0] = 0.0
	cp = joined_comp.add(1).cumprod()
	cp = cp.iloc[-1]
	fact = 252 / len(joined_comp)
	ann_ret_comp = cp ** fact - 1
	comp_reg.update({regr_df.columns[0]: ann_ret_comp[0]})
	comp_ann.update({regr_df.columns[0]: ann_ret_comp[1]})
	comp_beg_data.update({regr_df.columns[0]: beg_date})
	# -----------------Comparision Block Ends------------------------------------------------
	
	orig_regr_merged = y_change.fillna(regressed_returns)
	dict_regr_ret.update({regr_df.columns[0]: regressed_returns})
	merged_ret.update({regr_df.columns[0]: orig_regr_merged})
	
	# -------------------Convert daily returns to daily NAV's------------------------------
	orig_regr_merged.iloc[0] = 0.0
	net_asset_value = orig_regr_merged.add(1).cumprod() * 100
	dict_asset_prices.update({regr_df.columns[0]: net_asset_value})
	
	# ---------------Compare Original Returns vs Regressed Returns---------------------------
	df_ann = pd.DataFrame.from_dict(comp_ann, orient='index', columns=['OrigRet'])
	df_reg = pd.DataFrame.from_dict(comp_reg, orient='index', columns=['Regr'])
	merged_comp = pd.merge(df_ann, df_reg, left_index=True, right_index=True)
	merged_comp.loc[:, 'date'] = comp_beg_data.values()
	merged_comp.to_csv(dest_research + 'ff_unadj_comp.csv')
	# -------------------------Block Ends------------------------------------------
	
	regression_based_navs = pd.DataFrame.from_dict(dict_asset_prices)
	regression_based_navs.ffill(inplace=True)
	regression_based_navs.to_csv(dest_research + 'ff_only.csv')
	regression_r2 = pd.DataFrame.from_dict(regr_score, orient='index', columns=['R2'])
	regression_r2.to_csv(dest_research + 'ff_only_r2.csv')
	
	# -----------------Annual Metrics---------------------
	month_samp = regression_based_navs.resample('M', closed='right').last()
	ann_fact = 12 / len(month_samp)
	ann_ret = (month_samp.iloc[-1] / month_samp.iloc[0]) ** ann_fact - 1
	ann_risk = month_samp.pct_change().fillna(0).std() * np.sqrt(12)
	ann_df = pd.DataFrame({'ret_unadj': ann_ret, 'risk_unadj': ann_risk})
	ann_df.to_csv(dest_research + 'unadj_metrics.csv')
	
	# ---------------------Annual Metrics - Last 5 Years---------------------
	last_five = month_samp.iloc[-60:]
	ann_fact = 12 / len(last_five)
	ann_ret = (last_five.iloc[-1] / last_five.iloc[0]) ** ann_fact - 1
	ann_risk = last_five.pct_change().fillna(0).std() * np.sqrt(12)
	ann_df = pd.DataFrame({'ret_ff_unadj': ann_ret, 'risk_ff_unadj': ann_risk})
	ann_df.to_csv(dest_research + 'unadj_ff_metrics_last5.csv')


def get_historical_data_xignite(sdate, edate):
	import requests
	import json
	symbol = 'MSFT'
	url = "https://globalhistorical.xignite.com/v3/xGlobalHistorical.json/GetGlobalHistoricalQuotesRange?IdentifierType=Symbol&Identifier=" + symbol + "&IdentifierAsOfDate=&AdjustmentMethod=All&StartDate=" + sdate + "&EndDate=" + edate + "&_token=48722196BA7B495CB2C50694B78ABB69"
	
	x = requests.get(url)
	parsed = json.loads(x.text)
	df = pd.DataFrame(parsed['HistoricalQuotes'])
	df.set_index('Date', inplace=True)
	df.sort_index(axis=0, inplace=True)
	df = df[['Close']]
	print(x)


def create_portfolio_rebalance_with_income(req_inc, ts_fia, fia_name, term, base, mgmt_fees, fee_per):
	"""Simulation Model for Slides 2- 9. Simulation based on actual historical returns - Version 1"""
	# ------------------read the BAV file to calculate the annual returns------------------------
	fia_bav_df = pd.read_csv(src + "py_fia_time_series.csv", index_col=[0], parse_dates=True)
	fia_bav_df = fia_bav_df.loc[:, fia_name]
	fia_bav_df.loc[pd.to_datetime('1995-12-31')] = 100
	fia_bav_df.sort_index(inplace=True)
	fia_ann_ret = np.array(fia_bav_df.resample('Y', closed='right').last().pct_change().dropna())
	# read the raw FIA index and calculate yearly returns to run the income model and calculate the yearly FIA income
	fia_index = pd.read_csv(fia_src + "fia_index_data_from_bbg.csv", index_col=[0], parse_dates=True, skiprows=[1])
	fia_index.index.name = ''
	fia_index = fia_index.loc[fia_index.index[1]:, ]
	fia_index.index = pd.to_datetime(fia_index.index)
	if 'moz' in fia_name:
		fia_index = fia_index[['JMOZAIC2 Index']]
	else:
		fia_index = fia_index[['ZEDGENY Index']]
	
	# yearly_fia_data = fia_index.resample('BY', closed='right').last()
	# yearly_fia_ret = yearly_fia_data.pct_change().fillna(0)
	# fia_ann_ret = yearly_fia_ret.values.flatten()
	num_of_years = len(fia_ann_ret)
	
	# -----------Build the Income Model using the returns above--------------------
	# ---------------INCOME MODEL--------------------------------------------
	"""Random assets returns are generated for N trials and Income and accumulation is simulated. The quantile analysis
	is run using the simulated N portofolios. Version 1 - Original Standard Simulation"""
	
	sim_fia_cv = pd.DataFrame(index=range(num_of_years + 1))
	
	sim_base_total = pd.DataFrame(index=range(num_of_years + 1))
	sim_base_income = pd.DataFrame(index=range(num_of_years + 1))
	
	sim_port_total = pd.DataFrame(index=range(num_of_years + 1))
	sim_port_income = pd.DataFrame(index=range(num_of_years + 1))
	
	sim_base_total_pre_income = pd.DataFrame(index=range(num_of_years + 1))
	sim_port_total_pre_income = pd.DataFrame(index=range(num_of_years + 1))
	
	sim_base_total_preincome = pd.DataFrame(index=range(num_of_years + 1))
	sim_port_total_preincome = pd.DataFrame(index=range(num_of_years + 1))
	
	# read_income_inputs = pd.read_csv(src + "income_model_inputs.csv", index_col='Items')
	read_income_inputs = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_model_inputs',
									   index_col=[0])
	
	read_returns_est = pd.read_csv(src + "assets_forecast_returns.csv", index_col=[0])
	read_returns_est = read_returns_est.loc[:, ['Annualized Returns', 'Annualized Risk']]
	# read_returns_est = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_assets_returns_estimates',
	#                                  index_col=[0])
	#
	# read_returns_est.drop(['BM', read_returns_est.index[-1]], axis=0, inplace=True)
	# read_portfolio_inputs = pd.read_csv(src + "income_portfolio_inputs.csv", index_col='Items')
	
	# read_asset_weights = pd.read_csv(src + "asset_weights.csv", index_col='Asset')
	read_asset_weights = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='asset_weights',
									   index_col=[0])
	
	read_asset_weights.drop(read_asset_weights.index[-1], axis=0, inplace=True)
	
	# read random returns for simulation
	read_normal = pd.read_csv(src + 'sort_normal.csv', index_col=[0], parse_dates=True)
	read_small = pd.read_csv(src + 'sort_small_to_large.csv', index_col=[0], parse_dates=True)
	read_large = pd.read_csv(src + 'sort_large_to_small.csv', index_col=[0], parse_dates=True)
	assets_col_names = list(read_normal.columns)
	
	tickers = list(read_asset_weights.index)
	wts = np.array(read_asset_weights.loc[:, 'base'])
	
	def asset_median_returns(data, ticker):
		return data.filter(regex=ticker).median(axis=1)
	
	# dataframe for unsorted returns (normal)
	median_returns_normal = pd.DataFrame({t: asset_median_returns(read_normal, t) for t in tickers})
	median_returns_normal.loc[:, 'portfolio_return'] = median_returns_normal.dot(wts)
	median_normal_fia = pd.DataFrame({'FIA': asset_median_returns(read_normal, 'r_FIA')})
	
	# dataframe for smallest to largest returns
	median_returns_smallest = pd.DataFrame({t: asset_median_returns(read_small, t) for t in tickers})
	median_returns_smallest.loc[:, 'portfolio_return'] = median_returns_smallest.dot(wts)
	median_smallest_fia = pd.DataFrame({'FIA': asset_median_returns(read_small, 'r_FIA')})
	
	# dataframe for unsorted returns (normal)
	median_returns_largest = pd.DataFrame({t: asset_median_returns(read_large, t) for t in tickers})
	median_returns_largest.loc[:, 'portfolio_return'] = median_returns_largest.dot(wts)
	median_largest_fia = pd.DataFrame({'FIA': asset_median_returns(read_large, 'r_FIA')})
	
	years = list(range(0, num_of_years))
	income_cols = ['year', 'strategy_term', 'index_returns', 'term_ret', 'term_ret_with_par', 'term_annualize',
				   'ann_net_spread', 'term_ret_netspr', 'high_inc_benefit_base', 'rider_fee', 'eoy_income',
				   'contract_value']
	
	term = int(read_income_inputs.loc['term', 'inputs'])
	fia_ret = read_returns_est.loc[read_returns_est.index[-1], 'Annualized Returns']
	fia_risk = read_returns_est.loc[read_returns_est.index[-1], 'Annualized Risk']
	par_rate = float(read_income_inputs.loc['par_rate', 'inputs'])
	spread = float(read_income_inputs.loc['spread', 'inputs'])
	bonus_term = int(read_income_inputs.loc['bonus_term', 'inputs'])
	premium = float(read_income_inputs.loc['premium', 'inputs'])
	income_bonus = float(read_income_inputs.loc['income_bonus', 'inputs'])
	income_starts = int(read_income_inputs.loc['start_income_years', 'inputs'])
	income_growth = float(read_income_inputs.loc['income_growth', 'inputs'])
	rider_fee = float(read_income_inputs.loc['rider_fee', 'inputs'])
	inc_payout_factor = float(read_income_inputs.loc['income_payout_factor', 'inputs'])
	contract_bonus = float(read_income_inputs.loc['contract_bonus', 'inputs'])
	social = float(read_income_inputs.loc['social', 'inputs'])
	inflation = float(read_income_inputs.loc['inflation', 'inputs'])
	wtd_cpn_yield = float(read_income_inputs.loc['wtd_coupon_yld', 'inputs'])
	life_expectancy = int(read_income_inputs.loc['life_expectancy_age', 'inputs'])
	clients_age = int(read_income_inputs.loc['clients_age', 'inputs'])
	ann_income = req_inc
	
	runs = 0
	returns_dict = {}
	asset_dict = {}
	fia_dict = {}
	
	# ------------------------------INCOME MODEL-------------------------------------------------
	income_df = pd.DataFrame(index=years, columns=income_cols)
	income_df.loc[:, 'year'] = years
	income_df.loc[:, 'strategy_term'] = income_df.loc[:, 'year'] % term
	income_df.loc[:, 'strategy_term'] = income_df['strategy_term'].apply(lambda x: 1 if x == 0 else 0)
	# shift by 1 as the index is forced to start in 1995 a year earlier than original 1996 to capture the returns for
	# 1996 and avoid N/A's
	income_df.loc[:, 'strategy_term'] = income_df.loc[:, 'strategy_term'].shift(1).fillna(0)
	income_df.loc[:, 'index_returns'] = fia_ann_ret
	income_df.loc[:, 'index_returns'] = income_df.loc[:, 'index_returns'].shift(1).fillna(0)
	
	cumprod = (1. + income_df['index_returns']).rolling(window=term).agg(lambda x: x.prod()) - 1
	cumprod.fillna(0, inplace=True)
	income_df.loc[:, 'term_ret'] = np.where(income_df.loc[:, 'strategy_term'] == 1, cumprod, 0)
	income_df.loc[:, 'term_ret_with_par'] = income_df.loc[:, 'term_ret'] * par_rate
	income_df.loc[:, 'term_annualize'] = income_df.loc[:, 'term_ret_with_par'].apply(
		lambda x: (1 + x) ** (1 / term) - 1)
	income_df.loc[:, 'ann_net_spread'] = income_df.loc[:, 'term_annualize'] - spread
	income_df.loc[:, 'ann_net_spread'] = np.where(income_df.loc[:, 'strategy_term'] == 1,
												  income_df.loc[:, 'ann_net_spread'], 0)
	income_df.loc[:, 'term_ret_netspr'] = income_df.loc[:, 'ann_net_spread'].apply(lambda x: (1 + x) ** term - 1)
	
	for counter in years:
		if counter == 0:
			income_df.loc[counter, 'high_inc_benefit_base'] = premium * (1 + income_bonus)
		
		elif counter <= min(bonus_term, income_starts):
			income_df.loc[counter, 'high_inc_benefit_base'] = income_df.loc[counter - 1, 'high_inc_benefit_base'] * \
															  (1 + income_growth)
		else:
			income_df.loc[counter, 'high_inc_benefit_base'] = income_df.loc[counter - 1, 'high_inc_benefit_base']
	
	# income_df.loc[:, 'rider_fee'] = income_df.loc[:, 'high_inc_benefit_base'] * rider_fee
	income_df.loc[:, 'eoy_income'] = np.where(income_df.loc[:, 'year'] > income_starts,
											  income_df.loc[:, 'high_inc_benefit_base'] * inc_payout_factor, 0)
	
	# for counter in years:
	#
	#     if counter == 0:
	#         income_df.loc[counter, 'contract_value'] = premium * (1 + contract_bonus)
	#
	#     elif income_df.loc[counter, 'strategy_term'] == 1:
	#         # ----------rider fee calculated off the contract value--------------------------
	#         income_df.loc[:, 'rider_fee'] = income_df.loc[counter - 1, 'contract_value'] * rider_fee
	#         x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee']
	#         # x2 = (x1 * (1 + income_df.loc[counter, 'term_ret_netspr'])) - income_df.loc[counter, 'eoy_income']
	#         x2 = (x1 - income_df.loc[counter, 'eoy_income']) * (1 + income_df.loc[counter, 'term_ret_netspr'])
	#         income_df.loc[counter, 'contract_value'] = x2
	#
	#     else:
	#         # ----------rider fee calculated off the contract value--------------------------
	#         income_df.loc[:, 'rider_fee'] = income_df.loc[counter - 1, 'contract_value'] * rider_fee
	#         x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee'] - \
	#              income_df.loc[counter, 'eoy_income']
	#
	#         income_df.loc[counter, 'contract_value'] = x1
	
	# --------------------------Based on the Blake's spreadsheet on 11/24/2020------------------
	for counter in years:
		
		if counter == 0:
			income_df.loc[counter, 'high_inc_benefit_base'] = premium * (1 + income_bonus)  # N
			income_df.loc[counter, 'contract_value'] = premium * (1 + contract_bonus)  # O
		
		elif income_df.loc[counter, 'strategy_term'] == 1:
			
			# ----------rider fee calculated off the contract value--------------------------
			income_df.loc[:, 'rider_fee'] = income_df.loc[counter - 1, 'contract_value'] * rider_fee
			x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee']
			# x2 = (x1 * (1 + income_df.loc[counter, 'term_ret_netspr'])) - income_df.loc[counter, 'eoy_income']
			x2 = (x1 - income_df.loc[counter, 'eoy_income']) * (1 + income_df.loc[counter, 'term_ret_netspr'])
			income_df.loc[counter, 'contract_value'] = x2
		
		else:
			# ----------rider fee calculated off the contract value--------------------------
			income_df.loc[:, 'rider_fee'] = income_df.loc[counter - 1, 'contract_value'] * rider_fee
			x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee'] - \
				 income_df.loc[counter, 'eoy_income']
			
			income_df.loc[counter, 'contract_value'] = x1
	
	# variable stores the income number that is used in the base and fia portfolio calcs.
	if base:
		income_from_fia = 0.0
	else:
		income_from_fia = income_df.loc[income_df.index[-1], 'eoy_income']
	
	# income_df.loc[:, 'contract_value'] = income_df.loc[:, 'contract_value'].apply(lambda x: 0 if x <= 0 else x)
	#
	# sim_fia_cv.loc[:, str(runs)] = income_df.loc[:, 'contract_value']
	# ---------------------------Income Model Ends-------------------------------------------------
	
	# create the list of asset weights
	# wts_frame = pd.read_csv(src + "asset_weights.csv", index_col='Asset')
	# ---------------------------Output Income Model-----------------------------------
	income_df.to_csv(dest_simulation + 'income_model.csv')
	wts_frame = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='asset_weights', index_col=[0],
							  parse_dates=True)
	# fia_frame = pd.read_csv(src + "py_fia_time_series.csv", index_col="Date", parse_dates=True)
	fia_frame = ts_fia
	fia_names = fia_frame.columns.tolist()
	
	# if base is False means portfolio with an FIA
	if fia_name in fia_names and base is False:
		file_path = src + fia_name.lower() + "/"
	
	elif fia_name == 'static_MOZAIC' and base is True:
		file_path = src + fia_name.lower() + "/"
	
	else:
		file_path = src + fia_name.lower() + "/"
	
	# if the base is True means base portfolio with no FIA
	if base is True:
		fia_wt = wts_frame['base'].tolist()
		file_path = src + fia_name.lower() + "/"
	else:
		# fia_wt = wts_frame['fia'].tolist()
		fia_wt = wts_frame['base'].tolist()
		file_path = src + fia_name.lower() + "/"
	
	asset_names = wts_frame.index.tolist()
	asset_names = [fia_name if name == 'FIA' else name for name in asset_names]
	dollar_names = ['dollar_{}'.format(s) for s in asset_names]
	dollar_list = []
	if base:
		start_amount = 1000000
	# fia_wt = 0.0
	else:
		start_amount = 1000000 - premium
	# fia_wt = 0.0
	
	for i in range(len(dollar_names)):
		dollar_list.append(start_amount * fia_wt[i])
	
	dollar_list = [0 if math.isnan(x) else x for x in dollar_list]
	pre_inc_dollar_list = dollar_list
	nav_net = pd.read_csv(file_path + "net_nav.csv", index_col='Date', parse_dates=True)
	base_universe = nav_net.copy()
	base_universe_returns = base_universe.pct_change().fillna(0)
	base_cols = base_universe_returns.columns.tolist()
	wts_names = ['wts_{}'.format(s) for s in asset_names]
	
	# ----------create dataframe for advisor fee, resample the dataframe for quarter end--------------
	fee_frame = pd.DataFrame(index=base_universe_returns.index, columns=['qtr_fees'])
	fee_frame.qtr_fees = mgmt_fees / (fee_per * 100)
	fee_frame = fee_frame.resample('BQ', closed='right').last()
	combined_fee_frame = pd.concat([base_universe, fee_frame], axis=1)
	combined_fee_frame.loc[:, 'qtr_fees'] = combined_fee_frame.qtr_fees.fillna(0)
	
	# ---------------Logic for not deducting fees if the portfolio is starting at the end of any quarter else charge fee
	
	# prorate the first qtr fees if asset managed for less than a qtr
	# date_delta = (pd.to_datetime(actual_sdate) - base_universe_returns.index.to_list()[0]).days
	
	# fia_frame = pd.read_csv(src + "py_fia_time_series.csv", index_col="Date", parse_dates=True)
	# fia_names = fia_frame.columns.tolist()
	
	if fia_name in fia_names:
		# file_path = src + fia_name.lower() + "/"
		d2 = pd.to_datetime(fia_frame[fia_name].dropna().index[0])
	
	# date_delta = (fee_frame.index.to_list()[0] - base_universe_returns.index.to_list()[0]).days
	
	date_delta = (base_universe_returns.index.to_list()[0] - d2).days
	prorate_days = date_delta / 90
	first_qtr_fee = prorate_days * ((mgmt_fees * .01) / fee_per)
	
	# d1 = datetime.datetime.strptime(siegel_sd, "%m/%d/%Y").month
	d1 = base_universe_returns.index.to_list()[0].month
	
	# d2 = datetime.datetime.strptime(actual_sdate, "%m/%d/%Y").month
	
	month_dff = d1 - d2.month
	
	if d1 == 12:
		combined_fee_frame.iloc[month_dff]['qtr_fees'] = 0
	else:
		combined_fee_frame.iloc[month_dff]['qtr_fees'] = first_qtr_fee
	adv_fees = []
	
	for i in range(len(dollar_names)):
		base_universe_returns[dollar_names[i]] = dollar_list[i]
	
	counter = 1
	asset_dollars = []
	pre_asset_dollars = []
	post_asset_dollars = []
	asset_wts = []
	term = term * 12
	infl_factor = 0
	# cols = {'SPXT Index': 'US Large Cap Core', 'SPTRMDCP Index': 'US Mid Cap', 'SPTRSMCP Index': 'US Small Cap',
	#         'M2EA Index': 'Developed Ex-US', 'LUATTRUU Index': 'US Treasury', 'LUACTRUU Index': 'US High Yield Corp',
	#         'Cash':'Cash', 'FIA':fia_name}
	check_col = base_universe_returns.columns[0]
	if 'US Large Cap Core' in check_col:
		asset_names = ['US Large Cap Core', 'US Mid Cap', 'US Small Cap', 'Developed Ex-US', 'US Treasury',
					   'US High Yield Corp', 'Cash', fia_name]
	else:
		asset_names = asset_names
	
	for idx, row in base_universe_returns.iterrows():
		rows_filtered = base_universe_returns.reindex(columns=asset_names)
		row_returns = rows_filtered.loc[idx].tolist()
		# row_returns = base_universe_returns.loc[idx, asset_names].tolist()
		returns = [1 + r for r in row_returns]
		# -------Assets grows monthly at the rate r---------------
		dollar_list = [r * dollars for r, dollars in zip(returns, dollar_list)]
		# pre_inc_dollar_list = [r * dollars for r, dollars in zip(returns, pre_inc_dollar_list)]
		# --------EOM portfolio value-------------------------
		with_fia_asset = sum(dollar_list)
		closing_wts = [(d / with_fia_asset) for d in dollar_list]
		asset_dollars.append(dollar_list)
		asset_wts.append(closing_wts)
		
		# ---------------------Advisor fees deduction-----------------------
		fia = base_universe_returns.loc[idx, 'dollar_' + fia_name]
		fee = combined_fee_frame.loc[idx, 'qtr_fees']
		
		# Logic for portfolio rebalance
		# fia = base_universe_returns.loc[idx, 'dollar_FIA']
		# fee = combined_fee_frame.loc[idx, 'qtr_fees']
		# deduct_fees = (sum_total - fia) * fee
		# total_value = sum_total - deduct_fees
		# adv_fees.append(deduct_fees)
		
		# ----------------------Convert yearly product life to monthly--------------------
		
		if (counter - 1) % term == 0:
			opening_wts = dollar_list
			opening_sum = sum(opening_wts)
			new_wts = [wt / opening_sum for wt in opening_wts]
			fia_dollar = sum(dollar_list)
			deduct_fees = (fia_dollar - fia) * fee
			# pre_income_fees = deduct_fees
			
			# ------------EOM ending portfolio value after fees deducted from the non FIA assets
			fia_dollar = fia_dollar - deduct_fees
			pre_asset_dollars.append(fia_dollar)
			post_asset_dollars.append(fia_dollar)
			adv_fees.append(deduct_fees)
			
			# Rebalancing all the assets back to their original weight on the day of FIA rebalance net of advisor fees
			dollar_list = [wts * fia_dollar for wts in fia_wt]
			print("Portfolio rebalanced in month {}".format(counter))
		
		else:
			# Excluding the FIA from calculating the monthly rebalance weights for other assets when FIA cannot be \
			# rebalanced
			fia_dollar = dollar_list[-1]
			opening_wts = dollar_list[:-1]
			opening_sum = sum(opening_wts)
			
			# new weights of the assets are calculated based on to their previous closing value relative to the total
			# portfolio value excluding the FIA. Trending assets gets more allocation for the next month
			# new_wts = [wt / opening_sum for wt in opening_wts]
			
			# new weights of tha non fia assets scaled back to its original wts. Assets are brought back to its target
			# weights. Kind of taking profits from trending assets and dollar cost averaging for lagging assets
			without_fia_wt = fia_wt[:-1]
			
			# ---Condition check if the portfolio has only one assets
			if np.sum(without_fia_wt) == 0.0:
				new_wts = without_fia_wt
			else:
				new_wts = [wt / sum(without_fia_wt) for wt in without_fia_wt]
			
			non_fia_dollar = sum(dollar_list) - dollar_list[-1]
			income_cond1 = (counter >= income_starts * 12)
			income_cond2 = (idx.month == 12)
			
			if income_cond1 & income_cond2:
				pre_income_fees = (non_fia_dollar * fee)
				deduct_fees = (non_fia_dollar * fee) + (ann_income * ((1 + inflation) ** infl_factor) - income_from_fia)
				infl_factor += 1
			else:
				pre_income_fees = (non_fia_dollar * fee)
				deduct_fees = non_fia_dollar * fee
			
			# ----------Advisor fees is dedcuted--------------------------
			pre_inc_non_fia_dollar = non_fia_dollar - pre_income_fees
			non_fia_dollar = non_fia_dollar - deduct_fees
			adv_fees.append(deduct_fees)
			
			dollar_list = [wts * non_fia_dollar for wts in new_wts]
			# pre_inc_dollar_list = [wts * pre_inc_non_fia_dollar for wts in new_wts]
			dollar_list.append(fia_dollar)
			
			# ----------List of pre-income ending portfolio dollar--------------------
			pre_asset_dollars.append(pre_inc_non_fia_dollar)
			# ----------List of post-income ending portfolio dollar--------------------
			post_asset_dollars.append(non_fia_dollar)
		
		counter += 1
	
	# -------------Create a dataframe for PRE - INCOME ending portfolio value---------------------
	df_preinc = pd.DataFrame(pre_asset_dollars, index=base_universe_returns.index,
							 columns=['base_{}'.format(str(ann_income))])
	df_preinc = df_preinc.resample('Y', closed='right').last()
	# -----------------------------Block ends--------------------------------
	
	# -------------Create a dataframe for POST - INCOME ending portfolio value---------------------
	df_postinc = pd.DataFrame(post_asset_dollars, index=base_universe_returns.index,
							  columns=['base_{}'.format(str(ann_income))])
	df_postinc = df_postinc.resample('Y', closed='right').last()
	
	# -----------------Adding EoY contract Value-------------------
	if not base:
		df_postinc.reset_index(inplace=True)
		df_postinc.loc[:, df_postinc.columns[1]] = df_postinc.loc[:, df_postinc.columns[1]] + income_df.contract_value
		df_postinc.set_index('Date', drop=True, inplace=True)
	else:
		df_postinc = df_postinc
	# --------------------------Block ends-------------------
	
	# # ---------------------Block to save the detail asset level portfolio rebalances and simulation---------
	# asset_wt_df = pd.DataFrame(asset_wts, index=base_universe_returns.index, columns=wts_names)
	# asset_wt_df['sum_wts'] = asset_wt_df.sum(axis=1)
	# asset_dollar_df = pd.DataFrame(asset_dollars, index=base_universe_returns.index, columns=dollar_names)
	# asset_dollar_df['Total'] = asset_dollar_df.sum(axis=1)
	# base_universe_returns.drop(dollar_names, axis=1, inplace=True)
	# joined_df = pd.concat([base_universe_returns, asset_dollar_df, asset_wt_df], axis=1, ignore_index=False)
	#
	# if not base:
	#     joined_df[fia_name + '_portfolio'] = joined_df.Total.pct_change().fillna(0)
	# else:
	#     joined_df['base_portfolio_returns'] = joined_df.Total.pct_change().fillna(0)
	# joined_df['advisor_fees'] = adv_fees
	#
	# if base:
	#     accumulation_total = joined_df[['Total']]
	#     accumulation_yearly = accumulation_total.resample('Y', closed='right').last()
	#     accumulation_yearly.rename(columns={'Total':'total_net_income'}, inplace=True)
	#     accumulation_yearly.total_net_income.clip(0, inplace=True)
	#     joined_df = joined_df.resample('Y', closed='right').last()
	#     joined_df.clip(0, inplace=True)
	#     joined_df.to_csv(dest_simulation + "base_portfolio_" + str(ann_income) + ".csv")
	# else:
	#     accumulation_total = joined_df[['Total']]
	#     accumulation_yearly = accumulation_total.resample('Y', closed='right').last()
	#     accumulation_yearly.reset_index(drop=False, inplace=True)
	#     accumulation_yearly.Total.clip(0, inplace=True)
	#     income_df.contract_value.clip(0, inplace=True)
	#
	#     # -----------------Adding Contract value for FIA---------------------------
	#     accumulation_yearly.loc[:, 'total_net_income'] = accumulation_yearly.Total + income_df.contract_value
	#     accumulation_yearly.set_index('Date', drop=True, inplace=True)
	#     joined_df = joined_df.resample('Y', closed='right').last()
	#     joined_df.clip(0, inplace=True)
	#     joined_df.to_csv(dest_simulation + "fia_portfolio_" + str(ann_income) + ".csv")
	#
	# # ------------------------------Detail Block Ends-----------------------------------------
	
	# return accumulation_yearly.total_net_income
	# ---------------------return series----------------------------
	return df_preinc.loc[:, df_preinc.columns[0]], df_postinc.loc[:, df_postinc.columns[0]]


def portfolio_model_with_income_historical(req_inc, ts_fia, fia_name, term, base, mgmt_fees, fee_per):
	# """Simulation Model for Slides 2- 9. Simulation based on actual historical returns - VERSION 2. The model is
	
	# variable stores the income number that is used in the base and fia portfolio calcs.
	
	# ------------------read the BAV file to calculate the annual returns------------------------
	fia_bav_df = pd.read_csv(src + "py_fia_time_series.csv", index_col=[0], parse_dates=True)
	fia_bav_df = fia_bav_df.loc[:, fia_name]
	fia_bav_df.loc[pd.to_datetime('1995-12-31')] = 100
	fia_bav_df.sort_index(inplace=True)
	fia_ann_ret = np.array(fia_bav_df.resample('Y', closed='right').last().pct_change().dropna())
	# read the raw FIA index and calculate yearly returns to run the income model and calculate the yearly FIA income
	fia_index = pd.read_csv(fia_src + "fia_index_data_from_bbg.csv", index_col=[0], parse_dates=True, skiprows=[1])
	fia_index.index.name = ''
	fia_index = fia_index.loc[list(fia_index.index)[1]:, ]
	fia_index.index = pd.to_datetime(fia_index.index)
	
	if 'moz' in fia_name:
		fia_index = fia_index[['JMOZAIC2 Index']]
	else:
		fia_index = fia_index[['ZEDGENY Index']]
	
	num_of_years = len(fia_ann_ret)
	
	read_income_inputs = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_model_inputs',
									   index_col=[0])
	
	read_returns_est = pd.read_csv(src + "assets_forecast_returns.csv", index_col=[0])
	read_returns_est = read_returns_est.loc[:, ['Annualized Returns', 'Annualized Risk']]
	# read_returns_est = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_assets_returns_estimates',
	#                                  index_col=[0])
	
	read_asset_weights = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='asset_weights',
									   index_col=[0])
	
	read_asset_weights.drop(read_asset_weights.index[-1], axis=0, inplace=True)
	
	fia_name = read_income_inputs.loc['fia_name', 'inputs']
	
	years = list(range(0, num_of_years))
	income_cols = ['year', 'strategy_term', 'index_returns', 'term_ret', 'term_ret_with_par', 'term_annualize',
				   'ann_net_spread', 'term_ret_netspr', 'high_inc_benefit_base', 'rider_fee', 'eoy_income',
				   'contract_value']
	
	term = int(read_income_inputs.loc['term', 'inputs'])
	fia_ret = read_returns_est.loc[read_returns_est.index[-1], 'Annualized Returns']
	fia_risk = read_returns_est.loc[read_returns_est.index[-1], 'Annualized Risk']
	par_rate = float(read_income_inputs.loc['par_rate', 'inputs'])
	spread = float(read_income_inputs.loc['spread', 'inputs'])
	bonus_term = int(read_income_inputs.loc['bonus_term', 'inputs'])
	premium = float(read_income_inputs.loc['premium', 'inputs'])
	income_bonus = float(read_income_inputs.loc['income_bonus', 'inputs'])
	income_starts = int(read_income_inputs.loc['start_income_years', 'inputs'])
	income_growth = float(read_income_inputs.loc['income_growth', 'inputs'])
	rider_fee = float(read_income_inputs.loc['rider_fee', 'inputs'])
	inc_payout_factor = float(read_income_inputs.loc['income_payout_factor', 'inputs'])
	contract_bonus = float(read_income_inputs.loc['contract_bonus', 'inputs'])
	social = float(read_income_inputs.loc['social', 'inputs'])
	inflation = float(read_income_inputs.loc['inflation', 'inputs'])
	wtd_cpn_yield = float(read_income_inputs.loc['wtd_coupon_yld', 'inputs'])
	life_expectancy = int(read_income_inputs.loc['life_expectancy_age', 'inputs'])
	clients_age = int(read_income_inputs.loc['clients_age', 'inputs'])
	ann_income = req_inc
	
	guaranteed_inc = []
	non_guaranteed_inc = []
	
	# ------------------------------INCOME MODEL-------------------------------------------------
	income_df = pd.DataFrame(index=years, columns=income_cols)
	income_df.loc[:, 'year'] = years
	# income_df.loc[:, 'strategy_term'] = income_df.loc[:, 'year'] % term
	# income_df.loc[:, 'strategy_term'] = income_df['strategy_term'].apply(lambda x: 1 if x == 0 else 0)
	#
	# # shift by 1 as the index is forced to start in 1995 a year earlier than original 1996 to capture the returns for
	# # 1996 and avoid N/A's
	# income_df.loc[:, 'strategy_term'] = income_df.loc[:, 'strategy_term'].shift(1).fillna(0)
	# income_df.loc[:, 'index_returns'] = fia_ann_ret
	# income_df.loc[:, 'index_returns'] = income_df.loc[:, 'index_returns'].shift(1).fillna(0)
	#
	# cumprod = (1. + income_df['index_returns']).rolling(window=term).agg(lambda x: x.prod()) - 1
	# cumprod.fillna(0, inplace=True)
	# income_df.loc[:, 'term_ret'] = np.where(income_df.loc[:, 'strategy_term'] == 1, cumprod, 0)
	# income_df.loc[:, 'term_ret_with_par'] = income_df.loc[:, 'term_ret'] * par_rate
	# income_df.loc[:, 'term_annualize'] = income_df.loc[:, 'term_ret_with_par'].apply(
	#     lambda x: (1 + x) ** (1 / term) - 1)
	# income_df.loc[:, 'ann_net_spread'] = income_df.loc[:, 'term_annualize'] - spread
	# income_df.loc[:, 'ann_net_spread'] = np.where(income_df.loc[:, 'strategy_term'] == 1,
	#                                               income_df.loc[:, 'ann_net_spread'], 0)
	
	
	# -------------YoY FIA BAV Growth and not Index Growth----------
	income_df.loc[:, 'term_ret_netspr'] = fia_ann_ret
	income_df['term_ret_netspr'] = income_df['term_ret_netspr'].shift(1).fillna(0)
	
	# for counter in years:
	#     if counter == 0:
	#         income_df.loc[counter, 'high_inc_benefit_base'] = premium * (1 + income_bonus)
	#
	#     elif counter <= min(bonus_term, income_starts):
	#         income_df.loc[counter, 'high_inc_benefit_base'] = income_df.loc[counter - 1, 'high_inc_benefit_base'] * \
	#                                                           (1 + income_growth)
	#     else:
	#         income_df.loc[counter, 'high_inc_benefit_base'] = income_df.loc[counter - 1, 'high_inc_benefit_base']
	#
	# income_df.loc[:, 'eoy_income'] = np.where(income_df.loc[:, 'year'] > income_starts,
	#                                           income_df.loc[:, 'high_inc_benefit_base'] * inc_payout_factor, 0)
	income_df.loc[:, 'eoy_income'] = 0.0
	
	# --------------------------Based on the Blake's spreadsheet on 11/24/2020------------------
	# Only applicable to Accumulation plus Income model as the historical returns from the accumulation model used for
	# simulation is post rider fee. To avoid double counting of rider fee, we ignore the rider fee on income side.
	rider_fee = 0.0
	
	income_starts += 1
	for counter in years:
		
		if counter == 0:
			income_df.loc[counter, 'high_inc_benefit_base'] = premium * (1 + income_bonus)  # N
			income_df.loc[counter, 'contract_value'] = premium * (1 + contract_bonus)  # O
			income_df.loc[counter, 'rider_fee'] = 0.0
		# else income_df.loc[counter, 'strategy_term'] == 1:
		
		else:
			# -----------------rider fee calculated off the contract value--------------------------
			income_df.loc[counter, 'rider_fee'] = income_df.loc[counter - 1, 'contract_value'] * rider_fee
			
			if 1 < counter < income_starts:
				x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee']
				x2 = (x1 - income_df.loc[counter, 'eoy_income']) * (1 + income_df.loc[counter, 'term_ret_netspr'])
				income_df.loc[counter, 'contract_value'] = x2  # O
				income_df.loc[counter, 'high_inc_benefit_base'] = income_df.loc[counter, 'contract_value']
			
			elif counter == income_starts:
				income_df.loc[counter, 'eoy_income'] = income_df.loc[
														   counter - 1, 'high_inc_benefit_base'] * inc_payout_factor
				income_df.loc[counter, 'high_inc_benefit_base'] = income_df.loc[counter - 1, 'high_inc_benefit_base']
				x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee']
				x2 = (x1 - income_df.loc[counter, 'eoy_income']) * (1 + income_df.loc[counter, 'term_ret_netspr'])
				income_df.loc[counter, 'contract_value'] = x2  # O
			else:
				income_df.loc[counter, 'eoy_income'] = income_df.loc[counter - 1, 'eoy_income']
				income_df.loc[counter, 'high_inc_benefit_base'] = income_df.loc[counter - 1, 'high_inc_benefit_base']
				x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee']
				x2 = (x1 - income_df.loc[counter, 'eoy_income']) * (1 + income_df.loc[counter, 'term_ret_netspr'])
				income_df.loc[counter, 'contract_value'] = x2  # O
	
	if base:
		income_from_fia = 0.0
	else:
		income_from_fia = income_df.loc[income_df.index[-1], 'eoy_income']
	
	# ---------------------------Income Model Ends-------------------------------------------------
	
	wts_frame = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='asset_weights', index_col=[0],
							  parse_dates=True)
	# fia_frame = pd.read_csv(src + "py_fia_time_series.csv", index_col="Date", parse_dates=True)
	fia_frame = ts_fia
	fia_names = fia_frame.columns.tolist()
	
	# if base is False means portfolio with an FIA
	if fia_name in fia_names and base is False:
		file_path = src + fia_name.lower() + "/"
	
	elif fia_name == 'static_MOZAIC' and base is True:
		file_path = src + fia_name.lower() + "/"
	
	else:
		file_path = src + fia_name.lower() + "/"
	
	# if the base is True means base portfolio with no FIA
	if base is True:
		fia_wt = wts_frame['base'].tolist()
		file_path = src + fia_name.lower() + "/"
	else:
		# fia_wt = wts_frame['fia'].tolist()
		fia_wt = wts_frame['base'].tolist()
		file_path = src + fia_name.lower() + "/"
	
	# -------------Read Accumulation Analysis file to get portfolio returns on annual basis-----------------
	file_name = fia_name + "_summary.xlsx"
	hist_port_returns = pd.read_excel(file_path + file_name, sheet_name='time_series',
									  index_col=[0], parse_dates=True)
	hist_port_returns = hist_port_returns[hist_port_returns.columns[0:3]]
	hist_port_returns.set_index('Date', drop=True, inplace=True)
	hist_port_returns.loc[pd.to_datetime('1995-12-31')] = 100
	hist_port_returns.index = pd.to_datetime(hist_port_returns.index)
	hist_port_returns.sort_index(inplace=True)
	last_month = hist_port_returns.index[-1:].month
	
	# ------------Upsample to Annual Time Series----------------------------
	port_ann_ret = hist_port_returns.resample('Y', closed='right').last().pct_change().fillna(0)
	returns_arr = np.array(port_ann_ret[port_ann_ret.columns[0]])
	
	if last_month == 12:
		returns_arr = returns_arr
	else:
		returns_arr = returns_arr[:-1]
	# ----------------Block ends-------------------------------------------
	
	if base:
		start_amount = 1000000
		contract_value = 0.0
		pre_inc_cv = 0.0
	
	else:
		start_amount = 1000000 - premium
		contract_value = income_df.contract_value
		pre_inc_cv = income_df.contract_value.shift(1).fillna(premium)
	
	port_constructions = pd.DataFrame(returns_arr, index=np.arange(len(returns_arr)), columns=['ann_return'])
	port_constructions['required_income'] = np.array([income * (1 + inflation) ** count
													  for count in np.arange(len(returns_arr))])
	port_constructions['income_from_fia'] = income_from_fia
	port_constructions['req_income_net_fia'] = port_constructions['required_income'] - income_from_fia
	port_constructions['req_income_net_fia'].clip(0, inplace=True)
	port_constructions['required_income'] = port_constructions['required_income'].shift(income_starts).fillna(0)
	port_constructions['req_income_net_fia'] = port_constructions['req_income_net_fia'].shift(income_starts).fillna(0)
	
	for age in years:
		if age == 0:
			port_constructions.loc[age, 'pre_income_value'] = start_amount * (1 + returns_arr[age])
		else:
			port_constructions.loc[age, 'pre_income_value'] = port_constructions.loc[age - 1, 'post_income_value'] * \
															  (1 + returns_arr[age])
		
		port_constructions.loc[age, 'post_income_value'] = port_constructions.loc[age, 'pre_income_value'] - \
														   port_constructions.loc[age, 'req_income_net_fia']
	
	return port_constructions['pre_income_value'] + pre_inc_cv, port_constructions['post_income_value'] + contract_value


def portfolio_construction_version3(ts_fia, fia_name, term, base, mgmt_fees, fee_per):
	"""Original Version 3 Based on Blake and EW conversation on 12/17/2020. See email from Blake.
	Rider Fee is deducted on a quarterly basis in the accumulation model and is not dedcuted in the income model to
	avoid double counting. Call the function to generate the portfolio returns with no FIA using either static Mozaic
	or ZEBRAEDGE and base=True Base portfolio is saved under the FIA (static Moz or Zebra) named folder. The assets are
	rebalanced monthly .Call the function to generate the portfolio returns with FIA and any type of par. Input the FIA
	name and base=False The weights csv file and the order of its row must match the column order or the nav_net.csv
	file. """
	
	wts_frame = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='asset_weights', index_col=[0],
							  parse_dates=True)
	
	income_info = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='income_model_inputs', index_col=[0],
								parse_dates=True)
	
	fia_frame = ts_fia
	fia_names = fia_frame.columns.tolist()
	
	# if base is False means portfolio with an FIA
	if fia_name in fia_names and base is False:
		file_path = src + fia_name.lower() + "/"
	
	elif fia_name == 'static_MOZAIC' and base is True:
		file_path = src + fia_name.lower() + "/"
	
	else:
		file_path = src + fia_name.lower() + "/"
	
	# if the base is True means base portfolio with no FIA
	if base is True:
		fia_wt = wts_frame['base'].tolist()
		file_path = src + fia_name.lower() + "/"
	else:
		fia_wt = wts_frame['fia'].tolist()
		file_path = src + fia_name.lower() + "/"
	
	asset_names = wts_frame.index.tolist()
	asset_names = [fia_name if name == 'FIA' else name for name in asset_names]
	dollar_names = ['dollar_{}'.format(s) for s in asset_names]
	dollar_list = []
	
	# --------------------Hypothetical Starting Investment---------------
	start_amount = 1000000
	
	for i in range(len(dollar_names)):
		dollar_list.append(start_amount * fia_wt[i])
	
	dollar_list = [0 if math.isnan(x) else x for x in dollar_list]
	
	nav_net = pd.read_csv(file_path + "net_nav.csv", index_col='Date', parse_dates=True)
	base_universe = nav_net.copy()
	
	base_universe_returns = base_universe.pct_change().fillna(0)
	base_cols = base_universe_returns.columns.tolist()
	wts_names = ['wts_{}'.format(s) for s in asset_names]
	
	# create dataframe for advisor fee, resample the dataframe for quarter end
	fee_frame = pd.DataFrame(index=base_universe_returns.index, columns=['qtr_fees'])
	fee_frame.qtr_fees = mgmt_fees / (fee_per * 100)
	fee_frame = fee_frame.resample('BQ', closed='right').last()
	combined_fee_frame = pd.concat([base_universe, fee_frame], axis=1)
	combined_fee_frame.loc[:, 'qtr_fees'] = combined_fee_frame.qtr_fees.fillna(0)
	
	if fia_name in fia_names:
		d2 = pd.to_datetime(fia_frame[fia_name].dropna().index[0])
	
	date_delta = (base_universe_returns.index.to_list()[0] - d2).days
	prorate_days = date_delta / 90
	first_qtr_fee = prorate_days * ((mgmt_fees * .01) / fee_per)
	
	d1 = base_universe_returns.index.to_list()[0].month
	month_dff = d1 - d2.month
	
	if d1 == 12:
		combined_fee_frame.iloc[month_dff]['qtr_fees'] = 0
	else:
		combined_fee_frame.iloc[month_dff]['qtr_fees'] = first_qtr_fee
	adv_fees = []
	
	for i in range(len(dollar_names)):
		base_universe_returns[dollar_names[i]] = dollar_list[i]
	
	counter = 1
	asset_dollars = []
	asset_wts = []
	term = term * 12
	
	# -----For Income Model - to calculate rider fees--------------
	fia_premium = start_amount * fia_wt[-1]
	income_bonus = float(income_info.loc['income_bonus', 'inputs'])
	bonus_term = int(income_info.loc['bonus_term', 'inputs'])
	income_growth = float(income_info.loc['income_growth', 'inputs'])
	rider_fee = float(income_info.loc['rider_fee', 'inputs'])
	
	# -------convert annual income growth and rider fee to monthly and quarterly---------
	income_growth = income_growth / 12.0
	rider_fee = rider_fee / 4.0
	fia_income_base = fia_premium * (1 + income_bonus)
	rider_fee_deduction = []
	months = 0
	rider_amt = 0
	
	# Income Base growth and rider fee calculations
	for idx, row in base_universe_returns.iterrows():
		if 0 < months <= (bonus_term * 12):
			fia_income_base = fia_income_base * (1 + income_growth)
		else:
			fia_income_base = fia_income_base
		
		months += 1
		rows_filtered = base_universe_returns.reindex(columns=asset_names)
		row_returns = rows_filtered.loc[idx].tolist()
		# row_returns = base_universe_returns.loc[idx, asset_names].tolist()
		returns = [1 + r for r in row_returns]
		dollar_list = [r * dollars for r, dollars in zip(returns, dollar_list)]
		with_fia_asset = sum(dollar_list)
		closing_wts = [(d / with_fia_asset) for d in dollar_list]
		asset_dollars.append(dollar_list)
		asset_wts.append(closing_wts)
		
		# ---------------------Advisor fees deduction-----------------------
		fia = base_universe_returns.loc[idx, 'dollar_' + fia_name]
		fee = combined_fee_frame.loc[idx, 'qtr_fees']
		
		# ----------------------Convert yearly product life to monthly--------------------
		if (counter - 1) % term == 0:
			opening_wts = dollar_list
			opening_sum = sum(opening_wts)
			new_wts = [wt / opening_sum for wt in opening_wts]
			fia_dollar = sum(dollar_list)
			deduct_fees = (fia_dollar - fia) * fee
			fia_dollar = fia_dollar - deduct_fees
			adv_fees.append(deduct_fees)
			rider_fee_deduction.append(rider_amt)
			
			# Rebalancing all the assets back to their original weight on the day of FIA rebalance net of advisor fees
			dollar_list = [wts * fia_dollar for wts in fia_wt]
			print("Portfolio rebalanced in month {}".format(counter))
		
		else:
			# Excluding the FIA from calculating the monthly rebalance weights for other assets when FIA cannot be \
			# rebalanced
			
			fia_dollar = dollar_list[-1]
			opening_wts = dollar_list[:-1]
			opening_sum = sum(opening_wts)
			
			# new weights of the assets are calculated based on to their previous closing value relative to the total
			# portfolio value excluding the FIA. Trending assets gets more allocation for the next month
			# new_wts = [wt / opening_sum for wt in opening_wts]
			
			# new weights of tha non fia assets scaled back to its original wts. Assets are brought back to its target
			# weights. Kind of taking profits from trending assets and dollar cost averaging for lagging assets
			without_fia_wt = fia_wt[:-1]
			
			# ---Condition check if the portfolio has only one assets
			if np.sum(without_fia_wt) == 0.0:
				new_wts = without_fia_wt
			else:
				new_wts = [wt / sum(without_fia_wt) for wt in without_fia_wt]
			
			non_fia_dollar = sum(dollar_list) - dollar_list[-1]
			# max_av = max(dollar_list[-1], fia_income_base)
			
			# ------Check to assign rider fee 0 for base portfolio----------
			if idx.month % 3 == 0 and not base:
				rider_amt = fia_income_base * rider_fee
			# rider_amt = max_av * rider_fee
			else:
				rider_amt = 0.0
			
			deduct_fees = non_fia_dollar * fee
			
			# ----------Advisor fees is dedcuted--------------------------
			non_fia_dollar = non_fia_dollar - deduct_fees
			adv_fees.append(deduct_fees)
			
			dollar_list = [wts * non_fia_dollar for wts in new_wts]
			# ------------Deducting Rider Fee Quarterly-------------
			fia_dollar = fia_dollar - rider_amt
			rider_fee_deduction.append(rider_amt)
			dollar_list.append(fia_dollar)
		
		counter += 1
	
	asset_wt_df = pd.DataFrame(asset_wts, index=base_universe_returns.index, columns=wts_names)
	asset_wt_df['sum_wts'] = asset_wt_df.sum(axis=1)
	asset_dollar_df = pd.DataFrame(asset_dollars, index=base_universe_returns.index, columns=dollar_names)
	asset_dollar_df.loc[:, 'rider_fee_from_fia'] = rider_fee_deduction
	asset_dollar_df['Total'] = asset_dollar_df.sum(axis=1)
	base_universe_returns.drop(dollar_names, axis=1, inplace=True)
	joined_df = pd.concat([base_universe_returns, asset_dollar_df, asset_wt_df], axis=1, ignore_index=False)
	if not base:
		joined_df[fia_name + '_portfolio'] = joined_df.Total.pct_change().fillna(0)
	else:
		joined_df['base_portfolio_returns'] = joined_df.Total.pct_change().fillna(0)
	joined_df['advisor_fees'] = adv_fees
	if base:
		joined_df.to_csv(file_path + "base_portfolio.csv")
	else:
		joined_df.to_csv(file_path + fia_name + "_portfolio.csv")


def portfolio_construction_static_version3(ts_fia, fia_name, term, base, mgmt_fees, fee_per):
	"""------------------Original Version 3 Based on Blake and EW conversation on 12/17/2020 but with Static BAV Model.
	See email from Blake. Rider Fee is deducted on a quarterly basis in the accumulation model and is not dedcuted in
	the income model to avoid double counting. Call the function to generate the portfolio returns with no FIA using
	either static Mozaic or ZEBRAEDGE and base=True Base portfolio is saved under the FIA (static Moz or Zebra) named
	folder. The assets are rebalanced monthly .Call the function to generate the portfolio returns with FIA and any
	type of par. Input the FIA name and base=False The weights csv file and the order of its row must match the column
	order or the nav_net.csv file."""
	
	wts_frame = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='asset_weights', index_col=[0],
							  parse_dates=True)
	
	income_info = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='income_model_inputs', index_col=[0],
								parse_dates=True)
	
	fia_frame = ts_fia
	fia_names = fia_frame.columns.tolist()
	
	# if base is False means portfolio with an FIA
	if fia_name in fia_names and base is False:
		file_path = src + fia_name.lower() + "/"
	
	elif fia_name == 'static_MOZAIC' and base is True:
		file_path = src + fia_name.lower() + "/"
	
	else:
		file_path = src + fia_name.lower() + "/"
	
	# if the base is True means base portfolio with no FIA
	if base is True:
		fia_wt = wts_frame['base'].tolist()
		file_path = src + fia_name.lower() + "/"
	else:
		fia_wt = wts_frame['fia'].tolist()
		file_path = src + fia_name.lower() + "/"
	
	asset_names = wts_frame.index.tolist()
	asset_names = [fia_name if name == 'FIA' else name for name in asset_names]
	dollar_names = ['dollar_{}'.format(s) for s in asset_names]
	dollar_list = []
	
	# --------------------Hypothetical Starting Investment---------------
	start_amount = 1000000
	
	for i in range(len(dollar_names)):
		dollar_list.append(start_amount * fia_wt[i])
	
	dollar_list = [0 if math.isnan(x) else x for x in dollar_list]
	
	nav_net = pd.read_csv(file_path + "net_nav.csv", index_col='Date', parse_dates=True)
	base_universe = nav_net.copy()
	
	base_universe_returns = base_universe.pct_change().fillna(0)
	base_cols = base_universe_returns.columns.tolist()
	wts_names = ['wts_{}'.format(s) for s in asset_names]
	
	# create dataframe for advisor fee, resample the dataframe for quarter end
	fee_frame = pd.DataFrame(index=base_universe_returns.index, columns=['qtr_fees'])
	fee_frame.qtr_fees = mgmt_fees / (fee_per * 100)
	fee_frame = fee_frame.resample('BQ', closed='right').last()
	combined_fee_frame = pd.concat([base_universe, fee_frame], axis=1)
	combined_fee_frame.loc[:, 'qtr_fees'] = combined_fee_frame.qtr_fees.fillna(0)
	
	if fia_name in fia_names:
		d2 = pd.to_datetime(fia_frame[fia_name].dropna().index[0])
	
	date_delta = (base_universe_returns.index.to_list()[0] - d2).days
	prorate_days = date_delta / 90
	first_qtr_fee = prorate_days * ((mgmt_fees * .01) / fee_per)
	
	d1 = base_universe_returns.index.to_list()[0].month
	month_dff = d1 - d2.month
	
	if d1 == 12:
		combined_fee_frame.iloc[month_dff]['qtr_fees'] = 0
	else:
		combined_fee_frame.iloc[month_dff]['qtr_fees'] = first_qtr_fee
	adv_fees = []
	
	for i in range(len(dollar_names)):
		base_universe_returns[dollar_names[i]] = dollar_list[i]
	
	counter = 1
	asset_dollars = []
	asset_wts = []
	term = term * 12
	
	# -----For Income Model - to calculate rider fees--------------
	fia_premium = start_amount * fia_wt[-1]
	income_bonus = float(income_info.loc['income_bonus', 'inputs'])
	bonus_term = int(income_info.loc['bonus_term', 'inputs'])
	income_growth = float(income_info.loc['income_growth', 'inputs'])
	rider_fee = float(income_info.loc['rider_fee', 'inputs'])
	income_start_year = int(income_info.loc['start_income_years', 'inputs'])
	
	# -------convert annual income growth and rider fee to monthly and quarterly---------
	income_growth = income_growth / 12.0
	rider_fee = rider_fee / 4.0
	fia_income_base = fia_premium * (1 + income_bonus)
	rider_fee_deduction = []
	months = 0
	rider_amt = 0
	
	# Income Base growth and rider fee calculations
	for idx, row in base_universe_returns.iterrows():
		if 0 < months <= (bonus_term * 12):
			fia_income_base = fia_income_base * (1 + income_growth)
		else:
			fia_income_base = fia_income_base
		
		months += 1
		rows_filtered = base_universe_returns.reindex(columns=asset_names)
		row_returns = rows_filtered.loc[idx].tolist()
		# row_returns = base_universe_returns.loc[idx, asset_names].tolist()
		returns = [1 + r for r in row_returns]
		dollar_list = [r * dollars for r, dollars in zip(returns, dollar_list)]
		with_fia_asset = sum(dollar_list)
		closing_wts = [(d / with_fia_asset) for d in dollar_list]
		asset_dollars.append(dollar_list)
		asset_wts.append(closing_wts)
		
		# ---------------------Advisor fees deduction-----------------------
		fia = base_universe_returns.loc[idx, 'dollar_' + fia_name]
		fee = combined_fee_frame.loc[idx, 'qtr_fees']
		
		# ----------------------Convert yearly product life to monthly. FIA rebalance stops the year income  starts----
		if ((counter - 1) % term == 0) and (counter < income_start_year):
			opening_wts = dollar_list
			opening_sum = sum(opening_wts)
			new_wts = [wt / opening_sum for wt in opening_wts]
			fia_dollar = sum(dollar_list)
			deduct_fees = (fia_dollar - fia) * fee
			fia_dollar = fia_dollar - deduct_fees
			adv_fees.append(deduct_fees)
			rider_fee_deduction.append(rider_amt)
			
			# Rebalancing all the assets back to their original weight on the day of FIA rebalance net of advisor fees
			dollar_list = [wts * fia_dollar for wts in fia_wt]
			print("Portfolio rebalanced in month {}".format(counter))
		
		else:
			# Excluding the FIA from calculating the monthly rebalance weights for other assets when FIA cannot be \
			# rebalanced
			
			fia_dollar = dollar_list[-1]
			opening_wts = dollar_list[:-1]
			opening_sum = sum(opening_wts)
			
			# new weights of the assets are calculated based on to their previous closing value relative to the total
			# portfolio value excluding the FIA. Trending assets gets more allocation for the next month
			# new_wts = [wt / opening_sum for wt in opening_wts]
			
			# new weights of the non fia assets scaled back to its original wts. Assets are brought back to its target
			# weights. Kind of taking profits from trending assets and dollar cost averaging for lagging assets
			without_fia_wt = fia_wt[:-1]
			
			# ---Condition check if the portfolio has only one assets
			if np.sum(without_fia_wt) == 0.0:
				new_wts = without_fia_wt
			else:
				new_wts = [wt / sum(without_fia_wt) for wt in without_fia_wt]
			
			non_fia_dollar = sum(dollar_list) - dollar_list[-1]
			max_av = max(dollar_list[-1], fia_income_base)
			
			# ------Check to assign rider fee 0 for base portfolio----------
			if idx.month % 3 == 0 and not base:
				# rider_amt = fia_income_base * rider_fee
				rider_amt = max_av * rider_fee
			else:
				rider_amt = 0.0
			
			deduct_fees = non_fia_dollar * fee
			
			# ----------Advisor fees is dedcuted--------------------------
			non_fia_dollar = non_fia_dollar - deduct_fees
			adv_fees.append(deduct_fees)
			
			dollar_list = [wts * non_fia_dollar for wts in new_wts]
			# ------------Deducting Rider Fee Quarterly-------------
			fia_dollar = fia_dollar - rider_amt
			rider_fee_deduction.append(rider_amt)
			dollar_list.append(fia_dollar)
		
		counter += 1
	
	asset_wt_df = pd.DataFrame(asset_wts, index=base_universe_returns.index, columns=wts_names)
	asset_wt_df['sum_wts'] = asset_wt_df.sum(axis=1)
	asset_dollar_df = pd.DataFrame(asset_dollars, index=base_universe_returns.index, columns=dollar_names)
	asset_dollar_df.loc[:, 'rider_fee_from_fia'] = rider_fee_deduction
	asset_dollar_df['Total'] = asset_dollar_df.sum(axis=1)
	base_universe_returns.drop(dollar_names, axis=1, inplace=True)
	joined_df = pd.concat([base_universe_returns, asset_dollar_df, asset_wt_df], axis=1, ignore_index=False)
	if not base:
		joined_df[fia_name + '_portfolio'] = joined_df.Total.pct_change().fillna(0)
	else:
		joined_df['base_portfolio_returns'] = joined_df.Total.pct_change().fillna(0)
	joined_df['advisor_fees'] = adv_fees
	if base:
		joined_df.to_csv(file_path + "base_portfolio.csv")
	else:
		joined_df.to_csv(file_path + fia_name + "_portfolio.csv")


def accumulation_plus_income_historical_version3(ts_fia, fia_name, term, base, mgmt_fees, fee_per, starting_income=0.0):
	"""Simulation Model for Slides 2- 9. Simulation based on actual historical returns - VERSION 3. The model is based
	on the review call by BL and EW on 12/17/2020. See email from 12/17/2020 and 12/18/2020 from BL"""
	
	# ------------------read the BAV file to calculate the annual returns------------------------
	fia_bav_df = pd.read_csv(src + "py_fia_time_series.csv", index_col=[0], parse_dates=True)
	fia_bav_df = fia_bav_df.loc[:, fia_name]
	fia_bav_df.dropna(inplace=True)
	first_date = list(fia_bav_df.index)[0]
	new_start_date = str(first_date.year - 1) + "/12/31"
	fia_bav_df.loc[pd.to_datetime(new_start_date)] = 100
	# fia_bav_df.loc[pd.to_datetime('1995-12-31')] = 100
	fia_bav_df.sort_index(inplace=True)
	fia_ann_ret = np.array(fia_bav_df.resample('Y', closed='right').last().pct_change().dropna())
	# read the raw FIA index and calculate yearly returns to run the income model and calculate the yearly FIA income
	fia_index = pd.read_csv(fia_src + "fia_index_data_from_bbg.csv", index_col=[0], parse_dates=True, skiprows=[1])
	fia_index.index.name = ''
	fia_index = fia_index.loc[list(fia_index.index)[1]:, ]
	fia_index.index = pd.to_datetime(fia_index.index)
	
	if 'moz' in fia_name:
		fia_index = fia_index[['JMOZAIC2 Index']]
	else:
		fia_index = fia_index[['ZEDGENY Index']]
	
	# ------------Read the Accumulation File from Accumulation Model-------------
	
	port_file_name = src + "/" + fia_name + "/" + fia_name
	col_name = "dollar_" + fia_name
	hist_fia_ts = pd.read_csv(port_file_name + "_portfolio.csv", index_col=[0], parse_dates=True)
	# ---------------------------------------------------
	year = pd.to_datetime(hist_fia_ts.index[0]).year - 1
	dates = str(year) + "-12-31"
	
	read_income_inputs = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_model_inputs',
									   index_col=[0])
	
	hist_fia_ts.loc[pd.to_datetime(dates), :] = read_income_inputs.loc['premium', 'inputs']
	hist_fia_ts.sort_index(inplace=True)
	# ---------------------------
	ann_ret_from_fia_accum = hist_fia_ts.loc[:, col_name].resample('Y', closed='right').last().pct_change().fillna(0)
	ann_ret_from_fia_accum = np.array(ann_ret_from_fia_accum)
	ann_ret_from_fia_accum[:2] = 0.0
	
	read_income_inputs = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_model_inputs',
									   index_col=[0])
	
	read_returns_est = pd.read_csv(src + "assets_forecast_returns.csv", index_col=[0])
	
	read_returns_est = read_returns_est.loc[:, ['Annualized Returns', 'Annualized Risk']]
	
	read_asset_weights = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='asset_weights',
									   index_col=[0])
	
	read_asset_weights.drop(read_asset_weights.index[-1], axis=0, inplace=True)
	
	fia_name = read_income_inputs.loc['fia_name', 'inputs']
	
	num_of_years = len(ann_ret_from_fia_accum)
	years = list(range(0, num_of_years))
	
	income_cols = ['year', 'strategy_term', 'index_returns', 'term_ret', 'term_ret_with_par', 'term_annualize',
				   'ann_net_spread', 'term_ret_netspr', 'high_inc_benefit_base', 'rider_fee', 'eoy_income',
				   'contract_value']
	
	portfolio_size = float(read_income_inputs.loc['risky_assets', 'Base'])
	premium = float(read_income_inputs.loc['premium', 'inputs'])
	income_bonus = float(read_income_inputs.loc['income_bonus', 'inputs'])
	income_starts = int(read_income_inputs.loc['start_income_years', 'inputs'])
	income_growth = float(read_income_inputs.loc['income_growth', 'inputs'])
	inc_payout_factor = float(read_income_inputs.loc['income_payout_factor', 'inputs'])
	contract_bonus = float(read_income_inputs.loc['contract_bonus', 'inputs'])
	inflation = float(read_income_inputs.loc['inflation', 'inputs'])
	ann_income = float(read_income_inputs.loc['annual_income', 'inputs'])
	
	# ------------------------------INCOME MODEL-------------------------------------------------
	income_df = pd.DataFrame(index=years, columns=income_cols)
	income_df.loc[:, 'year'] = years
	
	# -------------YoY FIA BAV Growth Net of Spread and Rider Fee and not Index Growth----------
	income_df.loc[:, 'term_ret_netspr'] = ann_ret_from_fia_accum
	
	# --------------------------Based on the Blake's spreadsheet on 11/24/2020------------------
	# Only applicable to Accumulation plus Income model as the historical returns from the accumulation model used for
	# simulation is post rider fee. To avoid double counting of rider fee, we ignore the rider fee on income side.
	income_df.loc[:, 'eoy_income'] = 0.0
	income_df.loc[:, 'rider_fee'] = 0.0
	counter_end = income_starts
	income_starts += 1
	
	for counter in years:
		
		if counter == 0:
			income_df.loc[counter, 'high_inc_benefit_base'] = premium * (1 + income_bonus)  # N
			income_df.loc[counter, 'contract_value'] = premium * (1 + contract_bonus)  # O
			income_df.loc[counter, 'eoy_income'] = 0.0
		# else income_df.loc[counter, 'strategy_term'] == 1:
		
		else:
			
			if 1 <= counter < income_starts:
				# ---term_ret_netspr =  annual fia growth rate net of spread and rider fees--
				income_df.loc[counter, 'contract_value'] = income_df.loc[counter - 1, 'contract_value'] * \
														   (1 + income_df.loc[counter, 'term_ret_netspr'])
				
				income_df.loc[counter, 'high_inc_benefit_base'] = income_df.loc[counter - 1, 'high_inc_benefit_base'] * \
																  (1 + income_growth)
				income_df.loc[counter, 'eoy_income'] = 0.0
			
			elif counter == income_starts:
				income_df.loc[counter, 'eoy_income'] = income_df.loc[
														   counter - 1, 'high_inc_benefit_base'] * inc_payout_factor
				
				income_df.loc[counter, 'high_inc_benefit_base'] = income_df.loc[counter - 1, 'high_inc_benefit_base']
				
				x2 = (income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'eoy_income']) * \
					 (1 + income_df.loc[counter, 'term_ret_netspr'])
				
				income_df.loc[counter, 'contract_value'] = x2
			else:
				income_df.loc[counter, 'eoy_income'] = income_df.loc[counter - 1, 'eoy_income']
				income_df.loc[counter, 'high_inc_benefit_base'] = income_df.loc[counter - 1, 'high_inc_benefit_base']
				x2 = (income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'eoy_income']) * \
					 (1 + income_df.loc[counter, 'term_ret_netspr'])
				income_df.loc[counter, 'contract_value'] = x2  # O
	
	if base:
		income_from_fia = 0.0
	else:
		income_from_fia = income_df.loc[income_df.index[-1], 'eoy_income']
	
	# ---------------------------Income Model Ends-------------------------------------------------
	
	wts_frame = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='asset_weights', index_col=[0],
							  parse_dates=True)
	# fia_frame = pd.read_csv(src + "py_fia_time_series.csv", index_col="Date", parse_dates=True)
	fia_frame = ts_fia
	fia_names = fia_frame.columns.tolist()
	
	# if base is False means portfolio with an FIA
	if fia_name in fia_names and base is False:
		file_path = src + fia_name.lower() + "/"
	
	elif fia_name == 'static_MOZAIC' and base is True:
		file_path = src + fia_name.lower() + "/"
	
	else:
		file_path = src + fia_name.lower() + "/"
	
	# if the base is True means base portfolio with no FIA
	if base is True:
		fia_wt = wts_frame['base'].tolist()
		file_path = src + fia_name.lower() + "/"
	else:
		# fia_wt = wts_frame['fia'].tolist()
		fia_wt = wts_frame['base'].tolist()
		file_path = src + fia_name.lower() + "/"
	
	# -------------Read Accumulation Analysis file to get portfolio returns on annual basis-----------------
	file_name = fia_name + "_summary.xlsx"
	hist_port_returns = pd.read_excel(file_path + file_name, sheet_name='time_series', index_col=[0], parse_dates=True)
	hist_port_returns = hist_port_returns[hist_port_returns.columns[0:3]]
	hist_port_returns.set_index('Date', drop=True, inplace=True)
	
	# ---Logic to assing dummy year a value to properly account for the yearly returns-------
	hist_port_returns.loc[pd.to_datetime(dates)] = 100
	hist_port_returns.index = pd.to_datetime(hist_port_returns.index)
	hist_port_returns.sort_index(inplace=True)
	last_month = hist_port_returns.index[-1:].month
	
	# ---------------Upsample to Annual Time Series - Returns is the Total Portfolio Returns of the Base Portfolio----
	port_ann_ret = hist_port_returns.resample('Y', closed='right').last().pct_change().fillna(0)
	returns_arr = np.array(port_ann_ret[port_ann_ret.columns[0]])
	
	# ----------------Block ends-------------------------------------------
	
	if base:
		start_amount = portfolio_size
		contract_value = 0.0
		pre_inc_cv = 0.0
	
	else:
		start_amount = portfolio_size - premium
		contract_value = income_df.contract_value
		pre_inc_cv = income_df.contract_value.shift(1).fillna(premium)
	
	port_constructions = pd.DataFrame(returns_arr, index=list(np.arange(len(returns_arr))), columns=['ann_return'])
	port_constructions['required_income'] = np.array([ann_income * (1 + inflation) ** count
													  for count in np.arange(len(returns_arr))])
	port_constructions['income_from_fia'] = income_from_fia
	port_constructions['req_income_net_fia'] = port_constructions['required_income'] - income_from_fia
	port_constructions['req_income_net_fia'].clip(0, inplace=True)
	port_constructions.loc[:port_constructions.index[income_starts - 1], 'required_income'] = 0.0
	port_constructions.loc[:port_constructions.index[income_starts - 1], 'req_income_net_fia'] = 0.0
	port_constructions.loc[:, 'contract_value'] = income_df.loc[:, 'contract_value']
	inflation_factor = 0
	req_annual_income = float(read_income_inputs.loc['annual_income', 'inputs'])
	income_starts = counter_end
	if not base:
		port_constructions['guaranteed_income'] = income_from_fia
	else:
		port_constructions['guaranteed_income'] = 0.0
	
	for age in range(len(years)):
		print(age)
		if age == 0:
			port_constructions.loc[age, 'pre_income_value'] = start_amount * (1 + returns_arr[age])
			port_constructions['required_income'] = 0.0
			port_constructions['req_income_net_fia'] = 0.0
			port_constructions.loc[age, 'post_income_value'] = port_constructions.loc[age, 'pre_income_value']
		
		# -----------------------------------EDIT MODE ON----------------------------------------------------
		elif 1 <= age <= income_starts:
			port_constructions.loc[age, 'pre_income_value'] = port_constructions.loc[age - 1, 'post_income_value'] * \
															  (1 + returns_arr[age])
			port_constructions.loc[age, 'required_income'] = 0.0
			port_constructions.loc[age, 'req_income_net_fia'] = 0
			port_constructions.loc[age, 'non_guaranteed_income'] = 0.0
			port_constructions.loc[age, 'post_income_value'] = port_constructions.loc[age, 'pre_income_value']
		
		else:
			if age == income_starts + 1 and base:
				# ----Calculate the first required income $ amount.
				port_constructions.loc[age, 'pre_income_value'] = port_constructions.loc[age - 1, 'post_income_value'] * \
																  (1 + returns_arr[age])
				x1 = port_constructions.loc[age, 'pre_income_value']
				infl_adj_income = x1 * (req_annual_income * (1 + inflation) ** inflation_factor)
				port_constructions.loc[age, 'required_income'] = infl_adj_income
				port_constructions.loc[age, 'non_guaranteed_income'] = max(0, port_constructions.loc[
					age, 'required_income'] \
																		   - port_constructions.loc[
																			   age, 'guaranteed_income'])
				starting_income = infl_adj_income
				port_constructions.loc[age, 'post_income_value'] = port_constructions.loc[age, 'pre_income_value']
			
			elif age == income_starts + 1 and not base:
				port_constructions.loc[age, 'pre_income_value'] = port_constructions.loc[age - 1, 'post_income_value'] * \
																  (1 + returns_arr[age])
				# ----Calculate the first required income $ amount.
				infl_adj_income = starting_income * ((1 + inflation) ** inflation_factor)
				port_constructions.loc[age, 'required_income'] = infl_adj_income
				port_constructions.loc[age, 'non_guaranteed_income'] = max(0, port_constructions.loc[
					age, 'required_income']
																		   - port_constructions.loc[
																			   age, 'guaranteed_income'])
				starting_income = 0.0
				port_constructions.loc[age, 'post_income_value'] = port_constructions.loc[age, 'pre_income_value']
			else:
				port_constructions.loc[age, 'pre_income_value'] = port_constructions.loc[age - 1, 'post_income_value'] * \
																  (1 + returns_arr[age])
				# ----Adjusted the previous income $ amount by inflation.
				port_constructions.loc[age, 'required_income'] = port_constructions.loc[age - 1, 'required_income'] \
																 * (1 + inflation)
				port_constructions.loc[age, 'non_guaranteed_income'] = max(0, port_constructions.loc[
					age, 'required_income'] \
																		   - port_constructions.loc[
																			   age, 'guaranteed_income'])
				
				port_constructions.loc[age, 'post_income_value'] = port_constructions.loc[age, 'pre_income_value'] - \
																   port_constructions.loc[age, 'non_guaranteed_income']
			inflation_factor += 1
	
	# ----------------------------EDIT MODE ENDS--------------------------------------------------
	#  else:
	#      port_constructions.loc[age, 'pre_income_value'] = port_constructions.loc[age - 1, 'post_income_value'] * \
	#                                                        (1 + returns_arr[age])
	#
	#  port_constructions.loc[age, 'post_income_value'] = port_constructions.loc[age, 'pre_income_value'] - \
	#                                                     port_constructions.loc[age, 'req_income_net_fia']
	
	port_constructions['post_income_value'] = port_constructions['post_income_value'] + contract_value
	
	if base:
		
		port_constructions.drop(['contract_value', 'income_from_fia', 'req_income_net_fia', 'pre_income_value'], axis=1,
								inplace=True)
		
		cols = ['ann_return', 'required_income', 'non_guaranteed_income', 'guaranteed_income', 'post_income_value']
		port_constructions = port_constructions[cols]
		port_constructions.to_csv(src + fia_name + "/" + "base_growth_plus_income.csv")
	else:
		port_constructions.drop(['contract_value', 'income_from_fia', 'req_income_net_fia', 'pre_income_value'], axis=1,
								inplace=True)
		cols = ['ann_return', 'required_income', 'non_guaranteed_income', 'guaranteed_income', 'post_income_value']
		port_constructions = port_constructions[cols]
		port_constructions.to_csv(src + fia_name + "/" + "fia_growth_plus_income.csv")
	return starting_income


def income_model_constant_portfolio_returnv3(base, starting_income=0.0, num_of_years=30, method='normal'):
	"""Simulation based on the CONSTANT (Leveled) growth rate provided by the users for the risky portfolio
	and the FIA"""
	
	read_income_inputs = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_model_inputs',
									   index_col=[0])
	
	read_returns_est = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_assets_returns_estimates',
									 index_col=[0])
	
	read_asset_weights = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='asset_weights',
									   index_col=[0])
	
	read_asset_weights.drop(read_asset_weights.index[-1], axis=0, inplace=True)
	
	# read random returns for simulation
	read_normal = pd.read_csv(src + 'sort_normal.csv', index_col=[0], parse_dates=True)
	
	years = list(range(0, num_of_years + 1))
	income_cols = ['year', 'strategy_term', 'index_returns', 'term_ret', 'term_ret_with_par', 'term_annualize',
				   'ann_net_spread', 'term_ret_netspr', 'high_inc_benefit_base', 'rider_fee', 'eoy_income',
				   'contract_value']
	
	term = int(read_income_inputs.loc['term', 'inputs'])
	fia_ret = read_returns_est.loc[read_returns_est.index[-1], 'Annualized Returns']
	fia_risk = read_returns_est.loc[read_returns_est.index[-1], 'Annualized Risk']
	par_rate = float(read_income_inputs.loc['par_rate', 'inputs'])
	spread = float(read_income_inputs.loc['spread', 'inputs'])
	bonus_term = int(read_income_inputs.loc['bonus_term', 'inputs'])
	premium = float(read_income_inputs.loc['premium', 'inputs'])
	income_bonus = float(read_income_inputs.loc['income_bonus', 'inputs'])
	fia_name = read_income_inputs.loc['fia_name', 'inputs']
	
	income_starts = int(read_income_inputs.loc['start_income_years', 'inputs'])
	income_growth = float(read_income_inputs.loc['income_growth', 'inputs'])
	rider_fee = float(read_income_inputs.loc['rider_fee', 'inputs'])
	inc_payout_factor = float(read_income_inputs.loc['income_payout_factor', 'inputs'])
	contract_bonus = float(read_income_inputs.loc['contract_bonus', 'inputs'])
	social = float(read_income_inputs.loc['social', 'inputs'])
	inflation = float(read_income_inputs.loc['inflation', 'inputs'])
	wtd_cpn_yield = float(read_income_inputs.loc['wtd_coupon_yld', 'inputs'])
	life_expectancy = int(read_income_inputs.loc['life_expectancy_age', 'inputs'])
	clients_age = int(read_income_inputs.loc['clients_age', 'inputs'])
	
	# -------------For Constant Growth Rates------------------------
	const_fia_index_ret = float(read_income_inputs.loc['const_fia_index_ret', 'inputs'])
	const_risky_port_ret = float(read_income_inputs.loc['const_risky_port_ret', 'inputs'])
	
	# ---------------INCOME MODEL--------------------------------------------
	runs = 0
	returns_dict = {}
	asset_dict = {}
	fia_dict = {}
	
	income_df = pd.DataFrame(index=years, columns=income_cols)
	income_df.loc[:, 'year'] = years
	income_df.loc[:, 'strategy_term'] = income_df.loc[:, 'year'] % term
	income_df.loc[:, 'strategy_term'] = income_df['strategy_term'].apply(lambda x: 1 if x == 0 else 0)
	
	# ----------------CONSTANT FIA INDEX GROWTH RATE-------------------
	income_df.loc[:, 'term_ret_netspr'] = const_fia_index_ret
	counter_end = income_starts
	income_starts = income_starts + 1
	
	for counter in years:
		if counter == 0:
			income_df.loc[counter, 'high_inc_benefit_base'] = premium * (1 + income_bonus)  # N
			income_df.loc[counter, 'contract_value'] = premium * (1 + contract_bonus)  # O
			income_df.loc[counter, 'eoy_income'] = 0.0
		else:
			
			if 1 <= counter < income_starts:
				# ---term_ret_netspr =  annual fia growth rate net of spread and rider fees--
				income_df.loc[counter, 'contract_value'] = income_df.loc[counter - 1, 'contract_value'] * \
														   (1 + income_df.loc[counter, 'term_ret_netspr'])
				
				income_df.loc[counter, 'high_inc_benefit_base'] = income_df.loc[counter - 1, 'high_inc_benefit_base'] * \
																  (1 + income_growth)
				income_df.loc[counter, 'eoy_income'] = 0.0
			
			elif counter == income_starts:
				income_df.loc[counter, 'eoy_income'] = income_df.loc[
														   counter - 1, 'high_inc_benefit_base'] * inc_payout_factor
				
				income_df.loc[counter, 'high_inc_benefit_base'] = income_df.loc[counter - 1, 'high_inc_benefit_base']
				
				x2 = (income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'eoy_income']) * \
					 (1 + income_df.loc[counter, 'term_ret_netspr'])
				
				income_df.loc[counter, 'contract_value'] = x2
			else:
				income_df.loc[counter, 'eoy_income'] = income_df.loc[counter - 1, 'eoy_income']
				income_df.loc[counter, 'high_inc_benefit_base'] = income_df.loc[counter - 1, 'high_inc_benefit_base']
				x2 = (income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'eoy_income']) * \
					 (1 + income_df.loc[counter, 'term_ret_netspr'])
				income_df.loc[counter, 'contract_value'] = x2  # O
	
	# variable stores the income number that is used in the base and fia portfolio calcs.
	
	income_from_fia = income_df.loc[income_df.index[-1], 'eoy_income']
	
	income_df.loc[:, 'contract_value'] = income_df.loc[:, 'contract_value'].apply(lambda x: 0 if x <= 0 else x)
	
	# -------------------------------------BASE/FIA MODEL---------------------------------------------
	income_portfolio_df = pd.DataFrame(index=income_df.index)
	req_annual_income = float(read_income_inputs.loc['annual_income', 'inputs'])
	adv_fees = float(read_income_inputs.loc['advisor_fees', 'inputs'])
	inflation_factor = 0
	if base:
		initial_investment = float(read_income_inputs.loc['risky_assets', 'Base'])
		income_portfolio_df.loc[:, 'guaranteed_income'] = 0.0
		income_portfolio_df.loc[:, 'contract_value'] = 0.0
	
	# infl_adj_income = [req_annual_income * (1 + inflation) ** count for count in range(len(income_df))]
	# income_net_fia_income = infl_adj_income
	
	# --------------Portfolio Accumulation at Constant Rate-----------------
	#
	# income_portfolio_df.loc[:, 'infl_adj_income'] = infl_adj_income
	
	# income_portfolio_df.loc[:, 'non_guaranteed_income'] = income_net_fia_income
	
	else:
		initial_investment = float(read_income_inputs.loc['risky_assets', 'FIA'])
		income_portfolio_df.loc[:, 'guaranteed_income'] = income_from_fia
		income_portfolio_df.loc[:, 'contract_value'] = income_df.loc[:, 'contract_value']
	# adv_fees = float(read_income_inputs.loc['advisor_fees', 'inputs'])
	# req_annual_income = float(read_income_inputs.loc['annual_income', 'inputs'])
	# infl_adj_income = [req_annual_income * (1 + inflation) ** count for count in range(len(income_df))]
	# --------------Portfolio Accumulation at Constant Rate-----------------
	# income_portfolio_df = pd.DataFrame(index=income_df.index)
	# income_portfolio_df.loc[:, 'infl_adj_income'] = infl_adj_income
	
	# x = income_portfolio_df.loc[:, 'infl_adj_income'] - income_portfolio_df.loc[:, 'guaranteed_income']
	# income_portfolio_df.loc[:, 'non_guaranteed_income'] = x.apply(lambda x: max(0, x))
	
	income_starts = counter_end
	for year in range(len(income_portfolio_df)):
		if year == 0:
			income_portfolio_df.loc[year, 'eoy_gross_mv'] = initial_investment
			income_portfolio_df.loc[year, 'req_annual_income'] = 0.0
			income_portfolio_df.loc[year, 'guaranteed_income'] = 0.0
			income_portfolio_df.loc[year, 'non_guaranteed_income'] = 0.0
			income_portfolio_df.loc[year, 'advisor_fee'] = 0.0
			income_portfolio_df.loc[year, 'eoy_net_mv'] = initial_investment
		
		elif 1 <= year <= income_starts:
			income_portfolio_df.loc[year, 'eoy_gross_mv'] = income_portfolio_df.loc[year - 1, 'eoy_net_mv'] * \
															(1 + const_risky_port_ret)
			
			income_portfolio_df.loc[year, 'req_annual_income'] = 0.0
			income_portfolio_df.loc[year, 'guaranteed_income'] = 0.0
			income_portfolio_df.loc[year, 'non_guaranteed_income'] = 0.0
			
			income_portfolio_df.loc[year, 'advisor_fee'] = income_portfolio_df.loc[year, 'eoy_gross_mv'] * adv_fees
			
			income_portfolio_df.loc[year, 'eoy_net_mv'] = income_portfolio_df.loc[year, 'eoy_gross_mv'] - \
														  income_portfolio_df.loc[year, 'advisor_fee']
		
		else:
			income_portfolio_df.loc[year, 'eoy_gross_mv'] = income_portfolio_df.loc[year - 1, 'eoy_net_mv'] * \
															(1 + const_risky_port_ret)
			
			# ----Calculate the first required income $ amount.
			if year == income_starts + 1 and base:
				# ----Calculate the first required income $ amount.
				x1 = income_portfolio_df.loc[year, 'eoy_gross_mv'] + income_portfolio_df.loc[year, 'contract_value']
				infl_adj_income = x1 * (req_annual_income * (1 + inflation) ** inflation_factor)
				income_portfolio_df.loc[year, 'req_annual_income'] = infl_adj_income
				starting_income = infl_adj_income
			elif year == income_starts + 1 and not base:
				# ----Calculate the first required income $ amount.
				# x1 = income_portfolio_df.loc[year, 'eoy_gross_mv'] + income_portfolio_df.loc[year, 'contract_value']
				infl_adj_income = starting_income * ((1 + inflation) ** inflation_factor)
				income_portfolio_df.loc[year, 'req_annual_income'] = infl_adj_income
				starting_income = 0.0
			else:
				# ----Adjusted the previous income $ amount by inflation.
				income_portfolio_df.loc[year, 'req_annual_income'] = income_portfolio_df.loc[
																		 year - 1, 'req_annual_income'] \
																	 * (1 + inflation)
			
			income_portfolio_df.loc[year, 'non_guaranteed_income'] = max(0, income_portfolio_df.loc[
				year, 'req_annual_income'] \
																		 - income_portfolio_df.loc[
																			 year, 'guaranteed_income'])
			
			income_portfolio_df.loc[year, 'advisor_fee'] = income_portfolio_df.loc[year, 'eoy_gross_mv'] * adv_fees
			
			income_portfolio_df.loc[year, 'eoy_net_mv'] = income_portfolio_df.loc[year, 'eoy_gross_mv'] - \
														  income_portfolio_df.loc[year, 'advisor_fee'] - \
														  income_portfolio_df.loc[year, 'non_guaranteed_income']
			inflation_factor += 1
	
	if base:
		income_portfolio_df.loc[:, 'eoy_net_mv'] = income_portfolio_df.loc[:, 'eoy_net_mv']
		cols = ['eoy_gross_mv', 'req_annual_income', 'guaranteed_income', 'non_guaranteed_income', 'eoy_net_mv']
		income_portfolio_df = income_portfolio_df[cols]
		income_portfolio_df.to_csv(src + fia_name + "/" + "base_constant_return.csv")
	
	else:
		income_portfolio_df.loc[:, 'eoy_net_mv'] = income_portfolio_df.loc[:, 'eoy_net_mv'] + \
												   income_df.loc[:, 'contract_value']
		cols = ['eoy_gross_mv', 'req_annual_income', 'guaranteed_income', 'non_guaranteed_income', 'eoy_net_mv']
		income_portfolio_df = income_portfolio_df[cols]
		income_portfolio_df.to_csv(src + fia_name + "/" + "fia_constant_return.csv")
	return starting_income


def merge_income_analysis_files():
	read_income_inputs = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_model_inputs',
									   index_col=[0])
	
	fia_name = read_income_inputs.loc['fia_name', 'inputs']
	base_file = pd.read_csv(src + "/" + fia_name + "/" + "base_growth_plus_income.csv", index_col=[0])
	port_file = pd.read_csv(src + "/" + fia_name + "/" + "fia_growth_plus_income.csv", index_col=[0])
	
	# ------------Constant Growth Portfolio Analysis
	base_constant = pd.read_csv(src + fia_name + "/" + "base_constant_return.csv", index_col=[0])
	fia_constant = pd.read_csv(src + fia_name + "/" + "fia_constant_return.csv", index_col=[0])
	
	file_path = src + fia_name + "/" + fia_name + "_formatted_summary.xlsx"
	book = load_workbook(file_path)
	writer = pd.ExcelWriter(file_path, engine='openpyxl')
	writer.book = book
	base_file.to_excel(writer, sheet_name='base_growth_inc')
	port_file.to_excel(writer, sheet_name='port_growth_inc')
	
	base_constant.to_excel(writer, sheet_name='base_constant_growth')
	fia_constant.to_excel(writer, sheet_name='port_constant_growth')
	
	writer.save()
	writer.close()


if __name__ == "__main__":
	
	# ---Functional Methods:
	# # Asset Forecast returns based on regression alpha, adj beta and consensus S&P forecast, read for Income simulation
	asset_expected_returns(start, end)
	
	# # -----Call function to get the TR time series for portfolio assets and asset classifications ----------------
	get_asset_prices_bloomberg(start, end)
	
	# # -------Call function to get the proxy assets and nav for the portfolio assets -----------
	feature_selection_model(start, end)
	
	# -------------Other user inputs-------------------------------------------------------------------
	
	wts_frame = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='asset_weights', index_col=[0],
							  parse_dates=True)
	
	fia_allocation = wts_frame.loc['FIA', 'fia']
	
	# -----------------Read the FIA time series file and create a list of FIA to run the sample models-----------------
	# fia_products = pd.read_csv(src + "py_fia_time_series.csv", index_col="Date", parse_dates=True)
	
	fia_info = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='info', index_col=[0],
							 parse_dates=True)
	income_info = pd.read_excel(src + 'portfolio_information.xlsx', sheet_name='income_model_inputs', index_col=[0],
								parse_dates=True)
	# --------Create the time series for the FIA - IF blended, create a blended BAV time series
	fia_frame = pd.read_csv(src + "py_fia_time_series.csv", index_col="Date", parse_dates=True)
	fia_names = list(fia_info.loc['fia', :])
	fia_names = [x for x in fia_names if pd.notnull(x)]
	wts_fia = list(fia_info.loc['weights', :])
	wts_fia = [float(x) for x in wts_fia if pd.notnull(x)]
	index_start_date = list(fia_info.loc['index_start_date', :])
	index_start_date = [x for x in index_start_date if x != 0]
	index_start_date = [pd.to_datetime(x) for x in index_start_date]
	check_start = max(index_start_date)
	
	# # ----------------------read custom start date---------------------
	offset_date = fia_info.loc['custom_start_date', 'Value']
	offset_date = pd.to_datetime(offset_date)
	
	print("Running Analysis for FIA and weights..{}, {}".format(fia_names, wts_fia))
	print('*' * 100)
	wts_fia = [float(w) for w in wts_fia]
	fia_dataframe = fia_frame.loc[:, fia_names]
	
	if len(fia_names) > 1:
		cols = 'Blended_FIA'
	else:
		cols = fia_info.loc['fia', 'Value']
	
	# ------------------------FLEXIBLE DATES ENDS -----------------------------
	
	# ----------------------------FIA Product Life-------------------------------------------
	fia_cols = cols
	product_life_in_years = int(fia_info.loc['product_life', 'Value'])
	
	# advisor management fees
	perc_advisor_fee = float(fia_info.loc['adv_fees', 'Value'])
	
	# 1 for annual, 2 for semi annual, 4 for quarterly, 12 for monthly
	fee_frequency = int(fia_info.loc['fee_schedule', 'Value'])
	
	file_path = src + fia_cols.lower() + "/"
	print("Running portfolio model for {} and rebalance every {}".format(fia_cols, product_life_in_years))
	
	# ------------------------------------------------------------------------------------
	
	# ---------------------------Function Calls--------------------------------------------------------
	# Call the function to rebase the raw data file based on the starting date of the FIA, input the name of the FIA
	# for which the base portfolio needs to be created. Also change the fees of the assets list if required.
	blended_fia_ts = fia_frame.loc[:, fia_names]
	rebase_dataframe(fia_cols, blended_fia_ts)
	
	# Call the function to generate the portfolio returns with no FIA using either static Mozaic or ZEBRAEDGE and
	# base=True Base portfolio is save under the FIA (static Moz or Zebra) named folder. Call the function to
	# generate the portfolio returns with FIA and any type of par. Input the FIA name and base=False The weights csv
	# file and the order of its row must match the column order or the nav_net.csv file.
	
	# Version 1 - Original for base portfolio
	# create_portfolio_rebalance(blended_fia_ts, fia_cols, product_life_in_years, True, perc_advisor_fee, fee_frequency)
	
	# Version 1 - Original for FIA portfolio
	# create_portfolio_rebalance(blended_fia_ts, fia_cols, product_life_in_years, False, perc_advisor_fee, fee_frequency)
	
	# # # Version 3 - for base portfolio with the rider fee included with dynamic BAV
	# portfolio_construction_version3(blended_fia_ts, fia_cols, product_life_in_years, True, perc_advisor_fee, fee_frequency)
	# #
	# # # # Version 3 - for FIA portfolio with the rider fee included with Dynamic BAV
	# portfolio_construction_version3(blended_fia_ts, fia_cols, product_life_in_years, False, perc_advisor_fee, fee_frequency)
	
	# # Version 3 - for base portfolio with the rider fee included with Static BAV
	portfolio_construction_static_version3(blended_fia_ts, fia_cols, product_life_in_years, True, perc_advisor_fee,
										   fee_frequency)
	#
	# # # Version 3 - for FIA portfolio with the rider fee included with Static BAV
	portfolio_construction_static_version3(blended_fia_ts, fia_cols, product_life_in_years, False, perc_advisor_fee,
										   fee_frequency)
	
	# # Run portfolio analytics base on the FIA selected and weight of the FIA
	portfolio_analytics(file_path, fia_cols, fia_allocation)
	
	# # ----------------Beta vs S&P500-----------------------------
	adhoc_metrics(file_path)
	
	# # -------------presentation slides---------------------------------
	formatted_output(file_path)
	
	# --------------------Generate the consolidated file for marketing----------------------
	summary_output_file(perc_advisor_fee, product_life_in_years, fee_frequency, fia_allocation, file_path, fia_cols)
	
	# -------------New output with marketing slide----------------------------
	format_marketing_slides(file_path, fia_cols)
	
	# -------MODEL 2 ------Historical Income Plus Growth Model using the Required Income from the Base Model-------
	
	base_inc = accumulation_plus_income_historical_version3(blended_fia_ts, fia_cols, product_life_in_years, True,
															perc_advisor_fee, fee_frequency, 0.0)
	
	accumulation_plus_income_historical_version3(blended_fia_ts, fia_cols, product_life_in_years, False,
												 perc_advisor_fee, fee_frequency, base_inc)
	# ---------------ENDS - Growth plus Income Model-------------------
	
	# -----------------------MODEL 3 ---------Base - Constant Return Portfolio----------------
	# store the value of starting income number from base model to be used in the FIA model to keep the starting income
	# consistent for both.
	inc_base = income_model_constant_portfolio_returnv3(True, 0.0, num_of_years=30, method='normal')
	
	# -----------------MODEL 3 ----FIA - Constant Return Portfolio----------------
	# ----passing the starting income returned from base model to fia income model (var: inc_base)--------
	
	inc_fia = income_model_constant_portfolio_returnv3(False, inc_base, num_of_years=30, method='normal')
	
	# ---------------ENDS MODEL 3 Constant Growth Model-------------------
	
	merge_income_analysis_files()
