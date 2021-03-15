# ------------------------------------Nationwide Static BAV Model--------------------------------------
""" Notes: All the rebalance dates and par rate dates are aligned with the dates in Paul's Model. The margins, budget
and other parameters are refreshed yearly from Paul's spread sheet. The rbalance date was little different than Paul's
model until 11/2/2020. Starting 11/03/2020, Excel Model and Python Dates and Par Rates are aligned. Few differences in
models BAV values could be because of the number of days used in both the models i.e business day vs calendar days and
some rounding numbers. """

import pandas as pd
import numpy as np
import math
import datetime
import blpapi
import pdblp
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
from datetime import date

src = "C:/Users/yprasad/Dropbox (Annexus)/Portfolio Solutions Group/FIA Time Series/py_fia_bav_time_series/"
dropbox = "C:/Users/yprasad/Dropbox (Annexus)/RIA Time Series/Static Time Series/"

moz_sd = '10/30/1996'
zeb_sd = '06/30/2000'
shiller_sd = '10/30/2002'
siegel_sd = '11/27/2002'
endate = '03/31/2020'
actual_sdate = '12/27/2002'
tdate = date.today()
tdate = tdate.strftime("%m%d%Y")
start = "10/1/1996"
# end = "11/2/2020"
end = date.today().strftime("%m/%d/%Y")


def data_from_bloomberg(equity, field, start, end, pselect, nontrading, fia=False):
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


def generate_static_bav_timeseries(par_rate_model, index_name):
	# Using the current par and spread for the history
	# par_rate_model = "dynamic_moz_12_spread"
	# bav_input_data = pd.read_csv(src + "inputs_bav_model.csv", index_col='field')
	
	bav_input_data = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='parameters', index_col=[0],
								   parse_dates=True)
	# live_par_data = pd.read_csv(src2 + "live_par_rates.csv", index_col='Date')
	
	if index_name == 'JMOZAIC2 Index':
		# live_par_data = pd.read_excel(src + 'input_mozaic_par_spread.xlsx', sheet_name='par_rates', index_col=[0],
		#                               parse_dates=True)
		
		live_par_data = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='mozaic_par_rates',
									  index_col=[0], parse_dates=True)
		
		# read_spread = pd.read_excel(src + 'input_mozaic_par_spread.xlsx', sheet_name='spread', index_col=[0],
		#                             parse_dates=True)
		
		read_spread = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='mozaic_spreads',
									index_col=[0], parse_dates=True)
	
	elif index_name == 'ZEDGENY2 Index':
		# live_par_data = pd.read_excel(src + 'input_mozaic_par_spread.xlsx', sheet_name='par_rates', index_col=[0],
		#                               parse_dates=True)
		
		live_par_data = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='zebra2_par_rates',
									  index_col=[0], parse_dates=True)
		
		read_spread = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='zebra2_spreads',
									index_col=[0], parse_dates=True)
	
	elif index_name == 'SGMACRO Index':
		
		live_par_data = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='sgm_par_rates',
									  index_col=[0], parse_dates=True)
		
		read_spread = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='sgm_spreads',
									index_col=[0], parse_dates=True)

	else:
		print("Invalid Index: Must be Mozaic, Zebra2 or SGM")
		live_par_data = None
		read_spread = None
		# live_par_data = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='zebra2_par_rates',
		# 							  index_col=[0], parse_dates=True)
		#
		# read_spread = pd.read_excel(src + "fia_inputs_master_file.xlsx", sheet_name='zebra2_spreads',
		# 							index_col=[0], parse_dates=True)

	par_rate = live_par_data.loc[live_par_data.index[-1], par_rate_model]
	# spread = 0.0
	# read_spread = pd.read_csv(src + "input_spreads.csv", index_col='Date', parse_dates=True)
	spread = read_spread.loc[read_spread.index[-1], par_rate_model]
	# spread = float(bav_input_data.loc['spread', par_rate_model])
	# term = 3
	spread = float(spread)
	term = int(bav_input_data.loc['term', par_rate_model])
	
	# start_date = '8/1/2000'
	start_date = bav_input_data.loc['start_date', par_rate_model]
	start_date = pd.to_datetime(start_date).strftime('%m/%d/%Y')
	
	# ------------------------Input the raw index time series from Bloomberg------------------------------------
	bbg_raw_index = pd.read_csv(src + "fia_index_data_from_bbg.csv", index_col='ticker', parse_dates=True
								, skiprows=[1, 2])
	raw_index_name = index_name
	
	# raw_df = pd.read_csv(src + "raw_indices_time_series.csv", index_col='Date', parse_dates=True)
	df = bbg_raw_index.copy()
	# --------------------------Read the par rates for the time series------------------------------------
	# par_rate_df = pd.read_excel(src2 + "fia_par_rates.xlsx", index_col='Date', parse_dates=True)
	# fia_par_rate = par_rate_df.copy()
	# fia_par_rate = fia_par_rate.loc[:, par_rate_model]
	par_rate_df = live_par_data.copy()
	
	# ------------------------to adjust the par rate if the starting date is not the BOM, ffill the par
	# rates-------------
	psuedo_fia = pd.DataFrame(par_rate_df, index=df.index)
	psuedo_fia.ffill(inplace=True)
	fia_par_rate = psuedo_fia.copy()
	fia_par_rate = fia_par_rate.loc[:, par_rate_model]
	
	# ----------------select any custom date to run the analysis---------------------------------
	df = df[start_date:]
	ls_dates = []
	start = df.index[0]
	end = date.today()
	check_date = df.index[0]
	start_date = df.index[0]
	
	# -----------------------generate list of daily dates--------------------------------------------
	while check_date <= end:
		ls_date_range = pd.date_range(start=start, end=start + pd.DateOffset(years=term))
		ls_dates.extend(ls_date_range)
		start = ls_date_range[-1]
		check_date = ls_dates[-1]
	
	ls_dates = list(dict.fromkeys(ls_dates))
	
	# ------------------generate list of rebalance dates-------------------------------
	rebal_dates = [start_date]
	while start_date < end:
		new_date = start_date + relativedelta(years=term)
		rebal_dates.append(new_date)
		start_date = new_date
	
	# --------------------check if the rebalance date is on Sunday or Saturday----------------
	clean_rebal_date = []
	for day in rebal_dates:
		if day.weekday() == 5:
			if day - timedelta(days=1) in df.index.to_list():
				day_date = day - timedelta(days=1)
			else:
				day_date = day + timedelta(days=1)
		
		elif day.weekday() == 6:
			day_date = day + timedelta(days=1)
		else:
			day_date = day
		
		clean_rebal_date.append(day_date)
	
	dummy_date = clean_rebal_date[-1] + relativedelta(years=term)
	clean_rebal_date.append(dummy_date)
	raw_dates = df.index.to_list()
	
	# ---------------------------------Math and formula's to calculate the variables---------------------------
	for i in range(len(clean_rebal_date) - 1):
		if i < len(clean_rebal_date) - 2:
			
			raw_index_pos = df.index.to_list().index(clean_rebal_date[i])
			new_start = df.index.to_list()[raw_index_pos + 1]
			df.loc[clean_rebal_date[i]: clean_rebal_date[i + 1], 'Term'] = i + 1
			df.loc[clean_rebal_date[i]: clean_rebal_date[i + 1], 'start_date'] = clean_rebal_date[i]
			df.loc[clean_rebal_date[i]: clean_rebal_date[i + 1], 'end_date'] = clean_rebal_date[i + 1]
			df.loc[new_start: clean_rebal_date[i + 1], 'index_base_during_term'] = \
				df.loc[clean_rebal_date[i], raw_index_name]
		
		else:
			break
	for i in range(len(rebal_dates) - 1):
		raw_index_pos = df.index.to_list().index(rebal_dates[i])
		new_start = df.index.to_list()[raw_index_pos + 1]
		df.loc[new_start:rebal_dates[i + 1], 'term_par_rate'] = fia_par_rate.loc[rebal_dates[i]]
	
	bav_df = df.copy()
	bav_df.loc[:, 'term_par_rate'] = bav_df.loc[:, 'term_par_rate'].apply(lambda x: min(par_rate, x))
	bav_df.loc[:, 'index_cumm_return_time_par'] = bav_df['term_par_rate'] * (
			(bav_df[raw_index_name] - bav_df['index_base_during_term']) / bav_df['index_base_during_term'])
	bav_df.loc[:, 'index_cumm_return_time_par'].fillna(0, inplace=True)
	
	bav_df['dummy'] = pd.to_datetime(bav_df.index)
	bav_df['day_count'] = (bav_df['dummy'] - pd.to_datetime(bav_df['start_date'])).dt.days
	
	bav_df['term_day_count'] = (bav_df['end_date'] - bav_df['start_date']).dt.days
	bav_df.loc[:, 'term_starts'] = [1 if d in clean_rebal_date else 0 for d in raw_dates]
	
	for index, row in bav_df.iterrows():
		if bav_df.loc[index, 'term_starts'] != 1:
			bav_df.loc[index, 'elapsed_time'] = (
					(bav_df.loc[index, 'dummy'] - bav_df.loc[index, 'start_date']).days / 365)
		else:
			bav_df.loc[index, 'elapsed_time'] = 3
	
	bav_df['elapsed_time'] = bav_df['elapsed_time']
	bav_df.loc[:, 'AR'] = ((1 + bav_df.loc[:, 'index_cumm_return_time_par']) ** (
			1 / bav_df.loc[:, 'elapsed_time'])) - 1 - spread
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
	new_dir = src + "static_bavs" + "/"
	bav_df.to_csv(new_dir + par_rate_model + "_bav.csv")


def compile_bav_files(prodname):
	prod_dir = src + "static_bavs" + '/'
	bav_file = pd.read_csv(prod_dir + prodname + '_bav.csv', index_col='ticker', parse_dates=True)
	return bav_file.bav


if __name__ == "__main__":
	
	print("Static Process started.....")
	# ----------read index file for swap rates, AAA and 10 year yield indices symbole and make symbol list-----------
	read_index_file = pd.read_csv(src + "bbg_index_symbols.csv", index_col='raw_index')
	bbg_index_symbols = read_index_file.index.to_list()
	
	# ---------------Request BBG daily data for the swap rates, AAA, 10Yr yield index and save it as csv---------------
	# ----Logic : If FIA = False, function returns the file with data from BBG, else it saves the file in the folder--
	
	bbg_dataframe = data_from_bloomberg(bbg_index_symbols, 'PX_LAST', start, end, 'MONTHLY',
										'NON_TRADING_WEEKDAYS', False)
	bbg_dataframe.ffill(inplace=True)
	bbg_dataframe.to_csv(src + "bbg_indices_eom_prices.csv", float_format="%.20f")
	
	# ---------------BBG Function call for FIA Raw Index Price-----------------------------
	# ----Logic : If FIA = False, function returns the file with data from BBG, else it saves the file in the folder--
	
	read_raw_index_names = pd.read_csv(src + 'raw_index_fia_bbg.csv', index_col='raw_fia_index')
	raw_fia_symbol = read_raw_index_names.index.to_list()
	bbg_dataframe = data_from_bloomberg(raw_fia_symbol, 'PX_LAST', start, end, 'DAILY', 'ALL_CALENDAR_DAYS', True)
	
	# # -----------------Nationwide JPM Mozaic II Index Begins----------------------------------------------
	indexname = 'JMOZAIC2 Index'
	
	mozaic_list = ['dynamic_moz_9_spread',
				   'dynamic_moz_9_spread_7010',
				   'dynamic_moz_9_no_spread',
				   'dynamic_moz_9_no_spread_7010',
				   'dynamic_moz_12_spread',
				   'dynamic_moz_12_no_spread']
	
	for product in mozaic_list:
		generate_static_bav_timeseries(product, indexname)
	
	compiled_bav = pd.DataFrame({p: compile_bav_files(p) for p in mozaic_list})
	
	mcols = {'dynamic_moz_9_spread': 'Mozaic 9-Year Spread',
			 'dynamic_moz_9_spread_7010': 'Mozaic 9-Year Spread 70-10',
			 'dynamic_moz_9_no_spread': 'Mozaic 9-Year No Spread',
			 'dynamic_moz_9_no_spread_7010': 'Mozaic 9-Year No Spread 70-10',
			 'dynamic_moz_12_spread': 'Mozaic 12-Year Spread',
			 'dynamic_moz_12_no_spread': 'Mozaic 12-Year No Spread'}
	
	compiled_bav.rename(columns=mcols, inplace=True)
	compiled_bav.to_csv(dropbox + "Static Nationwide Time Series_MOZAIC.csv")
	ts_folder = src + "historical_bav/Static BAVs/"
	compiled_bav.to_csv(ts_folder + tdate + "_static_mozaic_combined_bav.csv")
	
	# # -----------------Nationwide JPM Mozaic II Index Ends----------------------------------------------
	
	# -----------------NationWide NYSE ZEBRA Index Begins-----------------------------------------------
	# indexname = 'ZEDGENY Index'
	#
	# zebra_list = ['dynamic_zebra_9_spread',
	# 			  'dynamic_zebra_9_spread_7010',
	# 			  'dynamic_zebra_9_no_spread',
	# 			  'dynamic_zebra_9_no_spread_7010',
	# 			  'dynamic_zebra_12_spread',
	# 			  'dynamic_zebra_12_no_spread']
	#
	# for product in zebra_list:
	# 	generate_static_bav_timeseries(product, indexname)
	#
	# compiled_bav = pd.DataFrame({p: compile_bav_files(p) for p in zebra_list})
	#
	# zcols = {'dynamic_zebra_9_spread': 'Zebra 9-Year Spread',
	# 		 'dynamic_zebra_9_spread_7010': 'Zebra 9-Year Spread 70-10',
	# 		 'dynamic_zebra_9_no_spread': 'Zebra 9-Year No Spread',
	# 		 'dynamic_zebra_9_no_spread_7010': 'Zebra 9-Year No Spread 70-10',
	# 		 'dynamic_zebra_12_spread': 'Zebra 12-Year Spread',
	# 		 'dynamic_zebra_12_no_spread': 'Zebra 12-Year No Spread'}
	#
	# compiled_bav.rename(columns=zcols, inplace=True)
	# compiled_bav.to_csv(dropbox + "Static Nationwide Time Series_ZEBRA.csv")
	# ts_folder = src + "historical_bav/Static BAVs/"
	# compiled_bav.to_csv(ts_folder + tdate + "_static_zebra_combined_bav.csv")
	# print("Static Process ends.")
	
	indexname = 'ZEDGENY2 Index'
	
	zebra2_list = ['dynamic_zebra2_9_spread',
				   'dynamic_zebra2_9_spread_7010',
				   'dynamic_zebra2_9_no_spread',
				   'dynamic_zebra2_9_no_spread_7010',
				   'dynamic_zebra2_12_spread',
				   'dynamic_zebra2_12_no_spread']
	
	for product in zebra2_list:
		generate_static_bav_timeseries(product, indexname)
	
	compiled_bav = pd.DataFrame({p: compile_bav_files(p) for p in zebra2_list})
	
	z2cols = {'dynamic_zebra2_9_spread': 'Zebra II 9-Year Spread',
			  'dynamic_zebra2_9_spread_7010': 'Zebra II 9-Year Spread 70-10',
			  'dynamic_zebra2_9_no_spread': 'Zebra II 9-Year No Spread',
			  'dynamic_zebra2_9_no_spread_7010': 'Zebra II 9-Year No Spread 70-10',
			  'dynamic_zebra2_12_spread': 'Zebra II 12-Year Spread',
			  'dynamic_zebra2_12_no_spread': 'Zebra II 12-Year No Spread'}
	
	compiled_bav.rename(columns=z2cols, inplace=True)
	compiled_bav.to_csv(dropbox + "Static Nationwide Time Series_ZEBRA2.csv")
	ts_folder = src + "historical_bav/Static BAVs/"
	compiled_bav.to_csv(ts_folder + tdate + "_static_zebra2_combined_bav.csv")
	print("Static Process ends.")
	
	indexname = 'SGMACRO Index'
	
	sgm_list = ['dynamic_sgm_9_spread',
				'dynamic_sgm_9_spread_7010',
				'dynamic_sgm_9_no_spread',
				'dynamic_sgm_9_no_spread_7010',
				'dynamic_sgm_12_spread',
				'dynamic_sgm_12_no_spread']
	
	for product in sgm_list:
		generate_static_bav_timeseries(product, indexname)
	
	compiled_bav = pd.DataFrame({p: compile_bav_files(p) for p in sgm_list})
	
	sgmcols = {'dynamic_sgm_9_spread': 'SG Macro 9-Year Spread',
			  'dynamic_sgm_9_spread_7010': 'SG Macro 9-Year Spread 70-10',
			  'dynamic_sgm_9_no_spread': 'SG Macro 9-Year No Spread',
			  'dynamic_sgm_9_no_spread_7010': 'SG Macro 9-Year No Spread 70-10',
			  'dynamic_sgm_12_spread': 'SG Macro 12-Year Spread',
			  'dynamic_sgm_12_no_spread': 'SG Macro 12-Year No Spread'}
	
	compiled_bav.rename(columns=sgmcols, inplace=True)
	compiled_bav.to_csv(dropbox + "Static Nationwide Time Series_SGM.csv")
	ts_folder = src + "historical_bav/Static BAVs/"
	compiled_bav.to_csv(ts_folder + tdate + "_static_sgm_combined_bav.csv")
	print("Static Process ends.")
# -----------------NationWide NYSE ZEBRA Index Ends------------------------------------------
