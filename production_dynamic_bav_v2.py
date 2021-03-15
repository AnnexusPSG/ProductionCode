# --------Nationwide Dynamic BAV Model -------
""" Notes: All the rebalance dates and par rate dates are aligned with the dates in Paul's Model. The margins, budget
and other parameters are refreshed yearly from Paul's spread sheet. The rbalance date was little different than Paul's
model until 11/2/2020. Starting 11/03/2020, Excel Model and Python Dates and Par Rates are aligned. Few differences in
models BAV values could be because of the number of days used in both the models i.e business day vs calendar days and
some rounding numbers. """

import pandas as pd
import numpy as np
import math
import datetime
import pdblp
from scipy.stats import norm
from datetime import date
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import statsmodels.api as sm
import os
import re

start = "10/1/1996"
# end = "10/22/2020"
end = date.today().strftime("%m/%d/%Y")
tdate = date.today()
tdate = tdate.strftime("%m%d%Y")
# start_rebal_date = '1/9/2017'


src = "C:/Users/yprasad/Dropbox (Annexus)/Portfolio Solutions Group/FIA Time Series/py_fia_bav_time_series/"
dropbox = "C:/Users/yprasad/Dropbox (Annexus)/RIA Time Series/Dynamic Time Series/"
fia_src = "C:/Users/yprasad\Dropbox (Annexus)/Portfolio Solutions Group/FIA Time Series/"
destination = "C:/Users/yprasad/Dropbox (Annexus)/Portfolio Solutions Group/Portfolio Analysis/"
shared_dest = "C:/Users/yprasad/Dropbox (Annexus)/Portfolio Solutions/Portfolio Analysis/"
gui_folder = "C:/Users/yprasad/Dropbox (Annexus)/Portfolio Solutions Group/Model Portfolios/py_output/GUI/"


def data_from_bloomberg(equity, field, start, end, pselect, nontrading, fia=False):
	start = datetime.strptime(start, '%m/%d/%Y').strftime('%Y%m%d')
	end = datetime.strptime(end, '%m/%d/%Y').strftime('%Y%m%d')
	con = pdblp.BCon(debug=False, port=8194, timeout=5000)
	if not con.start():
		print("ERROR: ****Connection could not be established with the data source****")
	
	bbg_df = con.bdh(equity, field, start, end, elms=[("periodicityAdjustment", 'CALENDAR'),
													  ("periodicitySelection", pselect),
													  ('nonTradingDayFillMethod', 'PREVIOUS_VALUE'),
													  ('nonTradingDayFillOption', nontrading)])
	con.stop()
	if bbg_df.empty:
		print("ERROR: ****Couldn't fetch data. Dataframe is empty****")
	
	elif not fia:
		bbg_df = bbg_df.resample('M', closed='right').last()
		bbg_df.ffill(inplace=True)
		bbg_df.to_csv(src + 'data_from_bbg.csv')
		read_bbg_file = pd.read_csv(src + 'data_from_bbg.csv', header=[0])
		read_bbg_file.iloc[0:1] = np.nan
		read_bbg_file.dropna(axis=0, inplace=True)
		read_bbg_file.set_index('ticker', inplace=True)
		read_bbg_file.index.name = 'Date'
		ticker_columns = [c.split(' ')[0] for c in read_bbg_file.columns.tolist()]
		read_bbg_file.columns = ticker_columns
		read_bbg_file.index = pd.to_datetime(read_bbg_file.index)
		read_bbg_file[ticker_columns] = read_bbg_file[ticker_columns].apply(pd.to_numeric, errors='coerce', axis=1)
		return read_bbg_file
	else:
		bbg_df.to_csv(src + 'fia_index_data_from_bbg.csv')


def model_for_bav_time_series(raw_index_name, par_rate, spread, term, prodname, start_date, rdate, livedate,
							  iname, optimize=False):
	# optimize parameter is True if the dynamic par rate is used from the reverse_bsm_model at each rebalance date else
	# is False to use the par rates from Paul's spreadsheet with option margin, initial margin optimization which is
	# a faster method. Ensure the "input_dynamic_par_rate_from_excel.csv" is updated using Paul's excel model, dynamic
	# par rate from the BAV sheets. Once update ensure the par rate csv file is updated to include these par rates for
	# the rebalance dates.
	
	# --------------Read the raw FIA index price file----------------------------------
	df = pd.read_csv(src + "fia_index_data_from_bbg.csv", usecols=['ticker', raw_index_name],
					 parse_dates=True, skiprows=[1, 2])
	# df.ticker = pd.to_datetime(df.ticker, format='%Y-%m-%d').dt.strftime('%m/%d/%Y')
	df.set_index('ticker', inplace=True, drop=True)
	df.dropna(inplace=True)
	df.index = pd.to_datetime(df.index)
	
	# --------------------------Read the par rates for the time series------------------------------------
	
	# TODO: Build a csv file with the par rates for all the fia indices and read the file
	new_dir = src + prodname + "/"
	# par_rate_df = pd.read_csv(src + "par_rate_model.csv", usecols=['Date', 'par_rate'], parse_dates=True)
	
	# par_rate_df = pd.read_csv(new_dir + prodname + "_par_rate_model.csv", usecols=['Date', 'par_rate'],
	# 						  parse_dates=True)
	#
	# par_rate_df.set_index('Date', inplace=True, drop=True)
	# par_rate_df = par_rate_df.apply(lambda x: round(x, 2))
	# s1 = par_rate_df.index[0]
	# s2 = par_rate_df[-1:].index[0]
	s1 = df.index[0]
	s2 = df[-1:].index[0]
	dates = pd.date_range(s1, s2, freq='1M') - pd.offsets.MonthBegin(1)
	# par_rate_df['bom_index'] = dates
	# par_rate_df.reset_index(inplace=True, drop=False)
	# # par_rate_df.bom_index = pd.to_datetime(par_rate_df.bom_index, format='%Y-%m-%d').dt.strftime('%m/%d/%Y')
	# par_rate_df.set_index('bom_index', inplace=True, drop=True)
	# par_rate_df.index = pd.to_datetime(par_rate_df.index)
	
	if iname == 'Mozaic':
		# dyn_par_from_excel = pd.read_excel(src + 'input_mozaic_par_spread.xlsx', sheet_name='par_rates',
		# index_col=[0], parse_dates=True)
		
		dyn_par_from_excel = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='mozaic_par_rates',
										   index_col=[0], parse_dates=True)
		
	elif iname == 'Zebra2':
		
		dyn_par_from_excel = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='zebra2_par_rates',
										   index_col=[0], parse_dates=True)
	
	elif iname == 'SGM':
		
		dyn_par_from_excel = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='sgm_par_rates',
										   index_col=[0], parse_dates=True)
	else:
		print("Index not Valid: Only Mozaic , Zebra2 and SGM allowed")
		dyn_par_from_excel = None
		# dyn_par_from_excel = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='zebra_par_rates',
		# 								   index_col=[0], parse_dates=True)
	
	if not optimize:
		# dyn_par_from_excel = pd.read_csv(src + "input_dynamic_par_rate_from_excel.csv", index_col='Date',
		#                                  parse_dates=True)
		dyn_par_from_excel = dyn_par_from_excel.loc[:, prodname]
		idx = dyn_par_from_excel.index.to_list()
		par_rate_df = pd.DataFrame(index=dates)
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
	
	# method to align the rebalance date with the first live data available for the product. This cuts off the last
	# live data from the proxy model and adjust the going forward rebalance date to begin with the first available live
	# rates
	
	# start_index = rebal_dates.index(pd.to_datetime(rdate))
	# ls_size = len(rebal_dates)
	# start_pos = start_index
	# # start_rebal_date = '1/9/2017'
	#
	# while start_pos < ls_size:
	#     if start_pos == start_index:
	#         rebal_dates[start_pos] = pd.to_datetime(start_rebal_date)
	#     else:
	#         rebal_dates[start_pos] = pd.to_datetime(start_rebal_date) + relativedelta(years=term)
	#         start_rebal_date = rebal_dates[start_pos]
	#     start_pos += 1
	#
	# # --------------------check if the rebalance date is on Sunday or Saturday----------------
	# clean_rebal_date = []
	# for day in rebal_dates:
	#     if day.weekday() == 5:
	#         if day - timedelta(days=1) in df.index.to_list():
	#             day_date = day - timedelta(days=1)
	#         else:
	#             day_date = day + timedelta(days=1)
	#
	#     elif day.weekday() == 6:
	#         day_date = day + timedelta(days=1)
	#
	#     else:
	#         day_date = day
	#
	#     clean_rebal_date.append(day_date)
	#
	# dummy_date = clean_rebal_date[-1] + relativedelta(years=term)
	# clean_rebal_date.append(dummy_date)
	raw_dates = list(df.index)
	
	# trial and error, can delete
	# read_model_live_par_file = pd.read_csv(src + "input_dynamic_par_rate_from_excel_Mozaic.csv", index_col='Date',
	#                                        parse_dates=True)
	#
	# read_model_live_par_file = pd.read_excel(src + 'input_mozaic_par_spread.xlsx', sheet_name='par_rates',
	#                                          index_col=[0],
	#                                          parse_dates=True)
	
	# clean_rebal_date = list(read_model_live_par_file.index)
	clean_rebal_date = list(dyn_par_from_excel.index)
	
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
	
	# ----To match Pauls Rebalance Date Par Rates - added on 11/3/2020--------------------
	bav_df.loc[:, 'term_par_rate'] = bav_df.loc[:, 'term_par_rate'].shift(1).bfill()
	
	# ----read the merged live and model spread data and generate a dataframe on a daily basis with fill forward spread-
	if iname == 'Mozaic':
		# read_model_live_spread_file = pd.read_excel(src + 'input_mozaic_par_spread.xlsx', sheet_name='spread',
		#                                             index_col=[0], parse_dates=True)
		
		read_model_live_spread_file = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='mozaic_spreads',
													index_col=[0], parse_dates=True)
	
	elif iname == 'Zebra2':
		# read_model_live_spread_file = pd.read_excel(src + 'input_mozaic_par_spread.xlsx', sheet_name='spread',
		#                                             index_col=[0], parse_dates=True)
		
		read_model_live_spread_file = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='zebra2_spreads',
													index_col=[0], parse_dates=True)
	
	elif iname == 'SGM':
		# read_model_live_spread_file = pd.read_excel(src + 'input_mozaic_par_spread.xlsx', sheet_name='spread',
		#                                             index_col=[0], parse_dates=True)
		
		read_model_live_spread_file = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='sgm_spreads',
													index_col=[0], parse_dates=True)
	
	else:
		# read_model_live_spread_file = pd.read_excel(src + 'input_zebra_par_spread.xlsx', sheet_name='spread',
		#                                             index_col=[0], parse_dates=True)
		print("Index not Valid: Only Mozaic , Zebra2 and SGM allowed")
		read_model_live_spread_file = None
		# read_model_live_spread_file = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='zebra_spreads',
		# 											index_col=[0], parse_dates=True)
	
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
	bav_df.to_csv(new_dir + prodname + "_bav.csv", float_format="%.4f")
	
	bav_df.to_csv(src + "bav.csv")
	
	bav_df = bav_df[cols]
	new_dir = src + "dynamic_bavs" + "/"
	bav_df.to_csv(new_dir + prodname + "_bav.csv")


def compile_bav_files(prodname):
	prod_dir = src + "dynamic_bavs" + '/'
	bav_file = pd.read_csv(prod_dir + prodname + '_bav.csv', index_col='ticker',
						   parse_dates=True)
	return bav_file.bav


def merge_bav_files():
	static_folder = src + "historical_bav/Static BAVs/"
	dynamic_folder = src + "historical_bav/Dynamic BAVs/"
	
	dy_mcols = {'Mozaic 9-Year Spread': 'dynamic_moz_9_spread',
				'Mozaic 9-Year Spread 70-10': 'dynamic_moz_9_spread_7010',
				'Mozaic 9-Year No Spread': 'dynamic_moz_9_no_spread',
				'Mozaic 9-Year No Spread 70-10': 'dynamic_moz_9_no_spread_7010',
				'Mozaic 12-Year Spread': 'dynamic_moz_12_spread',
				'Mozaic 12-Year No Spread': 'dynamic_moz_12_no_spread'}
	
	static_mcols = {'Mozaic 9-Year Spread': 'static_moz_9_spread',
					'Mozaic 9-Year Spread 70-10': 'static_moz_9_spread_7010',
					'Mozaic 9-Year No Spread': 'static_moz_9_no_spread',
					'Mozaic 9-Year No Spread 70-10': 'static_moz_9_no_spread_7010',
					'Mozaic 12-Year Spread': 'static_moz_12_spread',
					'Mozaic 12-Year No Spread': 'static_moz_12_no_spread'}
	
	# dy_zcols = {'Zebra 9-Year Spread': 'dynamic_zebra_9_spread',
	# 			'Zebra 9-Year Spread 70-10': 'dynamic_zebra_9_spread_7010',
	# 			'Zebra 9-Year No Spread': 'dynamic_zebra_9_no_spread',
	# 			'Zebra 9-Year No Spread 70-10': 'dynamic_zebra_9_no_spread_7010',
	# 			'Zebra 12-Year Spread': 'dynamic_zebra_12_spread',
	# 			'Zebra 12-Year No Spread': 'dynamic_zebra_12_no_spread'}
	#
	# static_zcols = {'Zebra 9-Year Spread': 'static_zebra_9_spread',
	# 				'Zebra 9-Year Spread 70-10': 'static_zebra_9_spread_7010',
	# 				'Zebra 9-Year No Spread': 'static_zebra_9_no_spread',
	# 				'Zebra 9-Year No Spread 70-10': 'static_zebra_9_no_spread_7010',
	# 				'Zebra 12-Year Spread': 'static_zebra_12_spread',
	# 				'Zebra 12-Year No Spread': 'static_zebra_12_no_spread'}
	
	dy_z2cols = {'Zebra II 9-Year Spread': 'dynamic_zebra2_9_spread',
				'Zebra II 9-Year Spread 70-10': 'dynamic_zebra2_9_spread_7010',
				'Zebra II 9-Year No Spread': 'dynamic_zebra2_9_no_spread',
				'Zebra II 9-Year No Spread 70-10': 'dynamic_zebra2_9_no_spread_7010',
				'Zebra II 12-Year Spread': 'dynamic_zebra2_12_spread',
				'Zebra II 12-Year No Spread': 'dynamic_zebra2_12_no_spread'}
	
	static_z2cols = {'Zebra II 9-Year Spread': 'static_zebra2_9_spread',
					'Zebra II 9-Year Spread 70-10': 'static_zebra2_9_spread_7010',
					'Zebra II 9-Year No Spread': 'static_zebra2_9_no_spread',
					'Zebra II 9-Year No Spread 70-10': 'static_zebra2_9_no_spread_7010',
					'Zebra II 12-Year Spread': 'static_zebra2_12_spread',
					'Zebra II 12-Year No Spread': 'static_zebra2_12_no_spread'}
	
	dy_sgm = {'SG Macro 9-Year Spread': 'dynamic_sgm_9_spread',
				 'SG Macro 9-Year Spread 70-10': 'dynamic_sgm_9_spread_7010',
				 'SG Macro 9-Year No Spread': 'dynamic_sgm_9_no_spread',
				 'SG Macro 9-Year No Spread 70-10': 'dynamic_sgm_9_no_spread_7010',
				 'SG Macro 12-Year Spread': 'dynamic_sgm_12_spread',
				 'SG Macro 12-Year No Spread': 'dynamic_sgm_12_no_spread'}
	
	static_sgm = {'SG Macro 9-Year Spread': 'static_sgm_9_spread',
					 'SG Macro 9-Year Spread 70-10': 'static_sgm_9_spread_7010',
					 'SG Macro 9-Year No Spread': 'static_sgm_9_no_spread',
					 'SG Macro 9-Year No Spread 70-10': 'static_sgm_9_no_spread_7010',
					 'SG Macro 12-Year Spread': 'static_sgm_12_spread',
					 'SG Macro 12-Year No Spread': 'static_sgm_12_no_spread'}
	
	# static_zebra_df = pd.read_csv(static_folder + tdate + "_static_zebra_combined_bav.csv", index_col='ticker',
	# 							  parse_dates=True)
	# static_zebra_df.loc[pd.to_datetime('2000-07-30'), :] = 100
	# static_zebra_df.loc[pd.to_datetime('2000-07-31'), :] = 100
	# static_zebra_df.sort_index(axis=0, inplace=True)
	# static_zebra_df.rename(columns=static_zcols, inplace=True)
	#
	# dynamic_zebra_df = pd.read_csv(dynamic_folder + tdate + "_dynamic_zebra_combined_bav.csv", index_col='ticker',
	# 							   parse_dates=True)
	# dynamic_zebra_df.loc[pd.to_datetime('2000-07-30'), :] = 100
	# dynamic_zebra_df.loc[pd.to_datetime('2000-07-31'), :] = 100
	# dynamic_zebra_df.sort_index(axis=0, inplace=True)
	# dynamic_zebra_df.rename(columns=dy_zcols, inplace=True)
	
	static_mozaic_df = pd.read_csv(static_folder + tdate + "_static_mozaic_combined_bav.csv", index_col='ticker',
								   parse_dates=True)
	static_mozaic_df.loc[pd.to_datetime('1996-10-30'), :] = 100
	static_mozaic_df.loc[pd.to_datetime('1996-10-31'), :] = 100
	static_mozaic_df.sort_index(axis=0, inplace=True)
	static_mozaic_df.rename(columns=static_mcols, inplace=True)
	
	dynamic_mozaic_df = pd.read_csv(dynamic_folder + tdate + "_dynamic_mozaic_combined_bav.csv", index_col='ticker',
									parse_dates=True)
	dynamic_mozaic_df.loc[pd.to_datetime('1996-10-30'), :] = 100
	dynamic_mozaic_df.loc[pd.to_datetime('1996-10-31'), :] = 100
	dynamic_mozaic_df.sort_index(axis=0, inplace=True)
	dynamic_mozaic_df.rename(columns=dy_mcols, inplace=True)
	
	static_zebra2_df = pd.read_csv(static_folder + tdate + "_static_zebra2_combined_bav.csv", index_col='ticker',
								  parse_dates=True)
	static_zebra2_df.loc[pd.to_datetime('2000-03-30'), :] = 100
	static_zebra2_df.loc[pd.to_datetime('2000-03-31'), :] = 100
	static_zebra2_df.sort_index(axis=0, inplace=True)
	static_zebra2_df.rename(columns=static_z2cols, inplace=True)
	
	dynamic_zebra2_df = pd.read_csv(dynamic_folder + tdate + "_dynamic_zebra2_combined_bav.csv", index_col='ticker',
								   parse_dates=True)
	dynamic_zebra2_df.loc[pd.to_datetime('2000-03-30'), :] = 100
	dynamic_zebra2_df.loc[pd.to_datetime('2000-03-31'), :] = 100
	dynamic_zebra2_df.sort_index(axis=0, inplace=True)
	dynamic_zebra2_df.rename(columns=dy_z2cols, inplace=True)
	
	static_sgm_df = pd.read_csv(static_folder + tdate + "_static_sgm_combined_bav.csv", index_col='ticker',
								   parse_dates=True)
	static_sgm_df.loc[pd.to_datetime('2002-07-30'), :] = 100
	static_sgm_df.loc[pd.to_datetime('2002-07-31'), :] = 100
	static_sgm_df.sort_index(axis=0, inplace=True)
	static_sgm_df.rename(columns=static_sgm, inplace=True)
	
	dynamic_sgm_df = pd.read_csv(dynamic_folder + tdate + "_dynamic_sgm_combined_bav.csv", index_col='ticker',
									parse_dates=True)
	dynamic_sgm_df.loc[pd.to_datetime('2002-07-30'), :] = 100
	dynamic_sgm_df.loc[pd.to_datetime('2002-07-31'), :] = 100
	dynamic_sgm_df.sort_index(axis=0, inplace=True)
	dynamic_sgm_df.rename(columns=dy_sgm, inplace=True)
	
	# frame = [static_mozaic_df, dynamic_mozaic_df, static_zebra_df, dynamic_zebra_df,
	# 		 static_zebra2_df, dynamic_zebra2_df, static_sgm_df, dynamic_sgm_df]
	frame = [static_mozaic_df, dynamic_mozaic_df, static_zebra2_df, dynamic_zebra2_df, static_sgm_df, dynamic_sgm_df]
	
	merged_df = pd.concat(frame, axis=1)
	merged_df.index.name = 'Date'
	merged_df.to_csv(fia_src + "py_fia_time_series.csv")
	merged_df.to_csv(destination + "py_fia_time_series.csv")
	merged_df.to_csv(shared_dest + "py_fia_time_series.csv")
	merged_df.to_csv(gui_folder + "py_fia_time_series.csv")


if __name__ == "__main__":
	
	print("Dynamic Process started.....")
	# ----------Pull Index Data From BBG and Save it in as a CSV File------------------------------
	
	# ----------read index file for swap rates, AAA and 10 year yield indices symbols and make symbol list-----------
	
	read_index_file = pd.read_csv(src + "bbg_index_symbols.csv", index_col='raw_index')
	bbg_index_symbols = read_index_file.index.to_list()
	
	# # ---------------Request BBG daily data for the swap rates, AAA, 10Yr yield index and save it as csv---------------
	# # ----Logic : If FIA = False, function returns the file with data from BBG, else it saves the file in the folder--
	#
	# bbg_dataframe = data_from_bloomberg(bbg_index_symbols, 'PX_LAST', start, end, 'MONTHLY', 'NON_TRADING_WEEKDAYS'
	# 									, False)
	# bbg_dataframe.ffill(inplace=True)
	# bbg_dataframe.to_csv(src + "bbg_indices_eom_prices.csv", float_format="%.20f")
	#
	# # ---------------BBG Function call for underlying FIA Raw Index Price-----------------------------
	# # ----Logic : If FIA = False, function returns the file with data from BBG, else it saves the file in the folder--
	#
	# read_raw_index_names = pd.read_csv(src + 'raw_index_fia_bbg.csv', index_col='raw_fia_index')
	# raw_fia_symbol = read_raw_index_names.index.to_list()
	# bbg_dataframe = data_from_bloomberg(raw_fia_symbol, 'PX_LAST', start, end, 'DAILY', 'ALL_CALENDAR_DAYS', True)
	
	prod_mozaic = ['dynamic_moz_9_spread',
				   'dynamic_moz_9_spread_7010',
				   'dynamic_moz_9_no_spread',
				   'dynamic_moz_9_no_spread_7010',
				   'dynamic_moz_12_spread',
				   'dynamic_moz_12_no_spread']
	
	# prod_zebra = ['dynamic_zebra_9_spread',
	# 			  'dynamic_zebra_9_spread_7010',
	# 			  'dynamic_zebra_9_no_spread',
	# 			  'dynamic_zebra_9_no_spread_7010',
	# 			  'dynamic_zebra_12_spread',
	# 			  'dynamic_zebra_12_no_spread']
	
	prod_zebra2 = ['dynamic_zebra2_9_spread',
				   'dynamic_zebra2_9_spread_7010',
				   'dynamic_zebra2_9_no_spread',
				   'dynamic_zebra2_9_no_spread_7010',
				   'dynamic_zebra2_12_spread',
				   'dynamic_zebra2_12_no_spread']
	
	prod_sgm = ['dynamic_sgm_9_spread',
				'dynamic_sgm_9_spread_7010',
				'dynamic_sgm_9_no_spread',
				'dynamic_sgm_9_no_spread_7010',
				'dynamic_sgm_12_spread',
				'dynamic_sgm_12_no_spread']
	
	for product in prod_mozaic:

		bav_inputs = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='parameters',
								   index_col=[0], parse_dates=True)

		prod_raw_index = bav_inputs.loc['bbg_ticker', product]
		prod_par_cap = float(bav_inputs.loc['par_rate_cap', product])
		prod_spread = float(bav_inputs.loc['spread', product]) * .01
		prod_term = int(bav_inputs.loc['term', product])
		prod_start = pd.to_datetime(bav_inputs.loc['start_date', product])
		start_live = bav_inputs.loc['live_start_date', product]

		# ---Identify the product Mozaic or Zebra Index and then find the date to cut off the last proxy contract term
		# to start the new product on the day of the first live rate available for that Index.

		if len(re.findall('moz', product)) >= 1:
			adj_rebal_date = "11/1/2017"
		elif len(re.findall('zebra2', product)) >= 1:
			adj_rebal_date = "4/1/2018"
		# elif len(re.findall('zebra', product)) >= 1:
		# 	adj_rebal_date = "7/6/2018"
		elif len(re.findall('sgm', product)) >= 1:
			adj_rebal_date = "8/1/2020"
		else:
			adj_rebal_date = 0

		indexname = 'Mozaic'

		# -----------Run BAV Model---------------------------------------------
		model_for_bav_time_series(prod_raw_index, prod_par_cap, prod_spread, prod_term, product,
								  prod_start, adj_rebal_date, start_live, indexname, False)

	# for product in prod_zebra:
	#
	# 	# ----------Read inputs for BAV model--------------------------
	#
	# 	# bav_inputs = pd.read_csv(src + "inputs_bav_model.csv", index_col='field', parse_dates=True)
	#
	# 	bav_inputs = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='parameters',
	# 							   index_col=[0], parse_dates=True)
	#
	# 	prod_raw_index = bav_inputs.loc['bbg_ticker', product]
	# 	prod_par_cap = float(bav_inputs.loc['par_rate_cap', product])
	# 	prod_spread = float(bav_inputs.loc['spread', product]) * .01
	# 	prod_term = int(bav_inputs.loc['term', product])
	# 	prod_start = pd.to_datetime(bav_inputs.loc['start_date', product])
	# 	start_live = bav_inputs.loc['live_start_date', product]
	#
	# 	# ---Identify the product Mozaic or Zebra Index and then find the date to cut off the last proxy contract term
	# 	# to start the new product on the day of the first live rate available for that Index.
	#
	# 	if len(re.findall('moz', product)) >= 1:
	# 		adj_rebal_date = "11/1/2017"
	# 	elif len(re.findall('zebra2', product)) >= 1:
	# 		adj_rebal_date = "4/1/2018"
	# 	elif len(re.findall('zebra', product)) >= 1:
	# 		adj_rebal_date = "7/6/2018"
	# 	elif len(re.findall('sgm', product)) >= 1:
	# 		adj_rebal_date = "8/1/2020"
	# 	else:
	# 		adj_rebal_date = 0
	#
	# 	indexname = 'Zebra'
	#
	# 	# -----------Run BAV Model---------------------------------------------
	# 	model_for_bav_time_series(prod_raw_index, prod_par_cap, prod_spread, prod_term, product,
	# 							  prod_start, adj_rebal_date, start_live, indexname, False)
	
	for product in prod_zebra2:

		# ----------Read inputs for BAV model--------------------------
		bav_inputs = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='parameters',
								   index_col=[0], parse_dates=True)
		prod_raw_index = bav_inputs.loc['bbg_ticker', product]
		prod_par_cap = float(bav_inputs.loc['par_rate_cap', product])
		prod_spread = float(bav_inputs.loc['spread', product]) * .01
		prod_term = int(bav_inputs.loc['term', product])
		prod_start = pd.to_datetime(bav_inputs.loc['start_date', product])
		start_live = bav_inputs.loc['live_start_date', product]

		# ---Identify the product Mozaic or Zebra Index and then find the date to cut off the last proxy contract term
		# to start the new product on the day of the first live rate available for that Index.

		if len(re.findall('moz', product)) >= 1:
			adj_rebal_date = "11/1/2017"
		elif len(re.findall('zebra2', product)) >= 1:
			adj_rebal_date = "4/1/2018"
		# elif len(re.findall('zebra', product)) >= 1:
		# 	adj_rebal_date = "7/6/2018"
		elif len(re.findall('sgm', product)) >= 1:
			adj_rebal_date = "8/1/2020"
		else:
			adj_rebal_date = 0

		indexname = 'Zebra2'

		# -----------Run BAV Model---------------------------------------------
		model_for_bav_time_series(prod_raw_index, prod_par_cap, prod_spread, prod_term, product,
								  prod_start, adj_rebal_date, start_live, indexname, False)
	
	for product in prod_sgm:
		
		# ----------Read inputs for BAV model--------------------------
		bav_inputs = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='parameters',
								   index_col=[0], parse_dates=True)
		prod_raw_index = bav_inputs.loc['bbg_ticker', product]
		prod_par_cap = float(bav_inputs.loc['par_rate_cap', product])
		prod_spread = float(bav_inputs.loc['spread', product]) * .01
		prod_term = int(bav_inputs.loc['term', product])
		prod_start = pd.to_datetime(bav_inputs.loc['start_date', product])
		start_live = bav_inputs.loc['live_start_date', product]
		
		# ---Identify the product Mozaic or Zebra Index and then find the date to cut off the last proxy contract term
		# to start the new product on the day of the first live rate available for that Index.
		
		if len(re.findall('moz', product)) >= 1:
			adj_rebal_date = "11/1/2017"
		elif len(re.findall('zebra2', product)) >= 1:
			adj_rebal_date = "4/1/2018"
		# elif len(re.findall('zebra', product)) >= 1:
		# 	adj_rebal_date = "7/6/2018"
		elif len(re.findall('sgm', product)) >= 1:
			adj_rebal_date = "8/1/2020"
		else:
			adj_rebal_date = 0
		
		indexname = 'SGM'
		
		# -----------Run BAV Model---------------------------------------------
		model_for_bav_time_series(prod_raw_index, prod_par_cap, prod_spread, prod_term, product,
								  prod_start, adj_rebal_date, start_live, indexname, False)
	
	mcols = {'dynamic_moz_9_spread': 'Mozaic 9-Year Spread',
			 'dynamic_moz_9_spread_7010': 'Mozaic 9-Year Spread 70-10',
			 'dynamic_moz_9_no_spread': 'Mozaic 9-Year No Spread',
			 'dynamic_moz_9_no_spread_7010': 'Mozaic 9-Year No Spread 70-10',
			 'dynamic_moz_12_spread': 'Mozaic 12-Year Spread',
			 'dynamic_moz_12_no_spread': 'Mozaic 12-Year No Spread'}
	
	# zcols = {'dynamic_zebra_9_spread': 'Zebra 9-Year Spread',
	# 		 'dynamic_zebra_9_spread_7010': 'Zebra 9-Year Spread 70-10',
	# 		 'dynamic_zebra_9_no_spread': 'Zebra 9-Year No Spread',
	# 		 'dynamic_zebra_9_no_spread_7010': 'Zebra 9-Year No Spread 70-10',
	# 		 'dynamic_zebra_12_spread': 'Zebra 12-Year Spread',
	# 		 'dynamic_zebra_12_no_spread': 'Zebra 12-Year No Spread'}
	
	z2cols = {'dynamic_zebra2_9_spread': 'Zebra II 9-Year Spread',
			 'dynamic_zebra2_9_spread_7010': 'Zebra II 9-Year Spread 70-10',
			 'dynamic_zebra2_9_no_spread': 'Zebra II 9-Year No Spread',
			 'dynamic_zebra2_9_no_spread_7010': 'Zebra II 9-Year No Spread 70-10',
			 'dynamic_zebra2_12_spread': 'Zebra II 12-Year Spread',
			 'dynamic_zebra2_12_no_spread': 'Zebra II 12-Year No Spread'}
	
	sgmcols = {'dynamic_sgm_9_spread': 'SG Macro 9-Year Spread',
			 'dynamic_sgm_9_spread_7010': 'SG Macro 9-Year Spread 70-10',
			 'dynamic_sgm_9_no_spread': 'SG Macro 9-Year No Spread',
			 'dynamic_sgm_9_no_spread_7010': 'SG Macro 9-Year No Spread 70-10',
			 'dynamic_sgm_12_spread': 'SG Macro 12-Year Spread',
			 'dynamic_sgm_12_no_spread': 'SG Macro 12-Year No Spread'}
	
	# prod_list = ['dynamic_moz_9_spread', 'dynamic_moz_9_spread_7010', 'dynamic_moz_9_no_spread',
	# 			 'dynamic_moz_9_no_spread_7010', 'dynamic_moz_12_spread', 'dynamic_moz_12_no_spread',
	# 			 'dynamic_zebra_9_spread', 'dynamic_zebra_9_spread_7010', 'dynamic_zebra_9_no_spread',
	# 			 'dynamic_zebra_9_no_spread_7010', 'dynamic_zebra_12_spread', 'dynamic_zebra_12_no_spread',
	# 			 'dynamic_zebra2_9_spread', 'dynamic_zebra2_9_spread_7010', 'dynamic_zebra2_9_no_spread',
	# 			 'dynamic_zebra2_9_no_spread_7010', 'dynamic_zebra2_12_spread', 'dynamic_zebra2_12_no_spread',
	# 			 'dynamic_sgm_9_spread', 'dynamic_sgm_9_spread_7010', 'dynamic_sgm_9_no_spread',
	# 			 'dynamic_sgm_9_no_spread_7010', 'dynamic_sgm_12_spread', 'dynamic_sgm_12_no_spread']
	
	prod_list = ['dynamic_moz_9_spread', 'dynamic_moz_9_spread_7010', 'dynamic_moz_9_no_spread',
				 'dynamic_moz_9_no_spread_7010', 'dynamic_moz_12_spread', 'dynamic_moz_12_no_spread',
				 'dynamic_zebra2_9_spread', 'dynamic_zebra2_9_spread_7010', 'dynamic_zebra2_9_no_spread',
				 'dynamic_zebra2_9_no_spread_7010', 'dynamic_zebra2_12_spread', 'dynamic_zebra2_12_no_spread',
				 'dynamic_sgm_9_spread', 'dynamic_sgm_9_spread_7010', 'dynamic_sgm_9_no_spread',
				 'dynamic_sgm_9_no_spread_7010', 'dynamic_sgm_12_spread', 'dynamic_sgm_12_no_spread']
	
	compiled_bav = pd.DataFrame({p: compile_bav_files(p) for p in prod_list})
	mozaic = ['dynamic_moz_9_spread',
			  'dynamic_moz_9_spread_7010',
			  'dynamic_moz_9_no_spread',
			  'dynamic_moz_9_no_spread_7010',
			  'dynamic_moz_12_spread',
			  'dynamic_moz_12_no_spread']
	
	# zebra = ['dynamic_zebra_9_spread',
	# 		 'dynamic_zebra_9_spread_7010',
	# 		 'dynamic_zebra_9_no_spread',
	# 		 'dynamic_zebra_9_no_spread_7010',
	# 		 'dynamic_zebra_12_spread',
	# 		 'dynamic_zebra_12_no_spread']
	
	ts_folder = src + "historical_bav/Dynamic BAVs/"
	compiled_mozaic = compiled_bav[mozaic].copy()
	compiled_mozaic.rename(columns=mcols, inplace=True)
	compiled_mozaic.to_csv(dropbox + "Dynamic Nationwide Time Series_MOZAIC.csv")
	compiled_mozaic.to_csv(ts_folder + tdate + "_dynamic_mozaic_combined_bav.csv")
	
	# compiled_zebra = compiled_bav[zebra].copy()
	# compiled_zebra.rename(columns=zcols, inplace=True)
	# compiled_zebra.dropna(inplace=True)
	# compiled_zebra.to_csv(dropbox + "Dynamic Nationwide Time Series_ZEBRA.csv")
	# compiled_zebra.to_csv(ts_folder + tdate + "_dynamic_zebra_combined_bav.csv")
	
	compiled_zebra2 = compiled_bav[list(z2cols.keys())].copy()
	compiled_zebra2.rename(columns=z2cols, inplace=True)
	compiled_zebra2.dropna(inplace=True)
	compiled_zebra2.to_csv(dropbox + "Dynamic Nationwide Time Series_ZEBRA2.csv")
	compiled_zebra2.to_csv(ts_folder + tdate + "_dynamic_zebra2_combined_bav.csv")
	
	compiled_sgm = compiled_bav[list(sgmcols.keys())].copy()
	compiled_sgm.rename(columns=sgmcols, inplace=True)
	compiled_sgm.dropna(inplace=True)
	compiled_sgm.to_csv(dropbox + "Dynamic Nationwide Time Series_SGM.csv")
	compiled_sgm.to_csv(ts_folder + tdate + "_dynamic_sgm_combined_bav.csv")
	
	merge_bav_files()
	
	print("Dynamic Process ends.")
