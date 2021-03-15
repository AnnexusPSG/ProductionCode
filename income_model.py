import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import os
# import math
# import datetime
# from dateutil.relativedelta import relativedelta
# from datetime import datetime, timedelta
from datetime import date

pd.set_option('display.max_columns', None)

src = "C:/Users/yprasad/Dropbox (Annexus)/Portfolio Solutions Group/Portfolio Analysis/"

fia_src = "C:/Users/yprasad\Dropbox (Annexus)/Portfolio Solutions Group/FIA Time Series/"

dest_simulation = "C:/Users/yprasad/Dropbox (Annexus)/Portfolio Solutions Group/Portfolio Analysis/Simulation/"

start = '10/29/1996'
end = date.today().strftime("%m/%d/%Y")


def copy_generate_random_returns(num_of_years=30, trials=100, method='normal'):
    returns_dict = {}
    asset_dict = {}
    frame = []
    years = list(range(0, num_of_years + 1))
    # ----------------------RANDOM RETURNS--------------------------
    read_asset_weights = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='asset_weights',
                                       index_col=[0])

    # read_asset_weights.drop(read_asset_weights.index[-1], axis=0, inplace=True)

    read_returns_est = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_assets_returns_estimates',
                                     index_col=[0])

    base_wts = read_asset_weights.loc[:, 'base']
    base_assets = list(base_wts.index)
    base_weights = list(base_wts.values)
    base_returns = list(read_returns_est.loc[:, 'Annualized Returns'].values)
    base_std = list(read_returns_est.loc[:, 'Annualized Risk'].values)

    r_cols = ['r_{}'.format(name) for name in base_assets]
    boy_value = ['bv_{}'.format(name) for name in base_assets]
    eoy_value = ['ev_{}'.format(name) for name in base_assets]

    read_asset_prices = pd.read_csv(src + 'net_nav.csv', index_col=[0], parse_dates=True)
    asset_cov = read_asset_prices.pct_change().fillna(0).cov()
    asset_cov = asset_cov * 12  # annualize covariance matrix
    random_returns = pd.DataFrame(index=years, columns=r_cols)
    runs = 0

    while runs <= trials:
        ret = np.random.multivariate_normal(base_returns, asset_cov, size=(len(random_returns.index), 1))
        trial_df = pd.DataFrame(index=list(np.arange(len(random_returns.index))), columns=base_assets)
        for i in np.arange(len(random_returns.index)):
            trial_df.loc[trial_df.index[i]] = ret[i]
            old = list(trial_df.columns)
            new = ['r_{}_{}'.format(cname, str(runs)) for cname in base_assets]
            old_new = dict(zip(old, new))
            trial_df.rename(columns=old_new, inplace=True)

        # asset_dict.update({'{}'.format(str(runs)): trial_df})
        # asset_dict.update(trial_df)
        frame.append(trial_df)

        # for c in range(len(r_cols)): # ret = np.random.normal(base_returns[c], base_std[c], size=(len(
        # random_returns.index), 1)) ret = np.random.multivariate_normal(base_returns, asset_cov, size=(len(
        # random_returns.index), 1)) trial_df = pd.DataFrame(index=list(np.arange(len(random_returns.index))),
        # columns=read_asset_prices.columns) for i in np.arange(len(random_returns.index)): trial_df.loc[
        # trial_df.index[i]] = ret[i] asset_dict.update({'{}_{}'.format(r_cols[c], str(runs)): trial_df})

        # random_returns.loc[:, r_cols[c]] = ret.flatten()
        # asset_dict.update({'{}_{}'.format(r_cols[c], str(runs)): ret.flatten()})

        runs += 1

    # asset_df = pd.DataFrame(asset_dict)
    asset_df = pd.concat(frame, axis=1)
    # asset_names = list(asset_df.columns)
    #
    # small_to_large = pd.DataFrame(index=asset_df.index)
    # large_to_small = pd.DataFrame(index=asset_df.index)
    #
    # for name in asset_names:
    #     small_to_large.loc[:, name] = np.sort(asset_df[name])
    #     large_to_small.loc[:, name] = -np.sort(-asset_df[name])
    #
    # small_to_large.to_csv(src + 'sort_small_to_large.csv')
    # large_to_small.to_csv(src + 'sort_large_to_small.csv')
    asset_df.to_csv(src + 'sort_normal.csv')

    # asset_median_returns = pd.DataFrame({c: asset_df.filter(regex=c).quantile(0.50, axis=1) for c in r_cols})
    # # asset_worst_returns = pd.DataFrame({c: asset_df.filter(regex=c).quantile(0.20, axis=1) for c in r_cols})
    # # asset_best_returns = pd.DataFrame({c: asset_df.filter(regex=c).quantile(0.80, axis=1) for c in r_cols})
    #
    # median_small_to_large = pd.DataFrame(index=asset_median_returns.index)
    # median_large_to_small = pd.DataFrame(index=asset_median_returns.index)
    #
    # median_asset_names = list(asset_median_returns.columns)
    # for name in median_asset_names:
    #     median_small_to_large.loc[:, name] = np.sort(asset_median_returns[name])
    #
    #     median_large_to_small.loc[:, name] = -np.sort(-asset_median_returns[name])
    #
    # # Unsorted median returns
    # asset_median_returns.to_csv(src + 'median_returns_unsorted.csv')
    #
    # # Sorted smallest to largest
    # median_small_to_large.to_csv(src + 'median_returns_smallest.csv')
    #
    # # Sorted largest to smallest
    # median_large_to_small.to_csv(src + 'median_returns_largest.csv')

    # asset_worst_returns.to_csv(src + 'median_worst.csv')
    # asset_best_returns.to_csv(src + 'median_best.csv')


def copy_generate_median_portfolio_from_simulation(num_of_years=30, trials=100, method='normal'):
    """Quantile Analysis on the simulated returns"""
    sim_fia_cv = pd.DataFrame(index=range(num_of_years + 1))

    sim_base_total = pd.DataFrame(index=range(num_of_years + 1))
    sim_base_income = pd.DataFrame(index=range(num_of_years + 1))

    sim_port_total = pd.DataFrame(index=range(num_of_years + 1))
    sim_port_income = pd.DataFrame(index=range(num_of_years + 1))

    # read_income_inputs = pd.read_csv(src + "income_model_inputs.csv", index_col='Items')
    read_income_inputs = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_model_inputs',
                                       index_col=[0])

    # read_returns_est = pd.read_csv(src + "income_assets_returns_estimates.csv", index_col='Symbol')
    read_returns_est = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_assets_returns_estimates',
                                     index_col=[0])

    # read_returns_est.drop(['BM', read_returns_est.index[-1]], axis=0, inplace=True)
    # read_portfolio_inputs = pd.read_csv(src + "income_portfolio_inputs.csv", index_col='Items')

    # read_asset_weights = pd.read_csv(src + "asset_weights.csv", index_col='Asset')
    read_asset_weights = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='asset_weights',
                                       index_col=[0])

    all_assets = list(read_asset_weights.index)
    read_asset_weights.drop(read_asset_weights.index[-1], axis=0, inplace=True)

    # read random returns for simulation
    read_normal = pd.read_csv(src + 'sort_normal.csv', index_col=[0], parse_dates=True)

    assets_col_names = list(read_normal.columns)

    tickers = list(read_asset_weights.index)
    wts = np.array(read_asset_weights.loc[:, 'base'])

    def asset_median_returns(data, ticker):
        return data.filter(regex=ticker).median(axis=1)

    # dataframe for unsorted returns (normal)
    median_returns_normal = pd.DataFrame({t: asset_median_returns(read_normal, t) for t in tickers})
    median_returns_normal.loc[:, 'portfolio_return'] = median_returns_normal.dot(wts)
    median_normal_fia = pd.DataFrame({'FIA': asset_median_returns(read_normal, 'r_FIA')})

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

    # # ---------------INCOME MODEL--------------------------------------------
    runs = 0
    returns_dict = {}
    asset_dict = {}
    fia_dict = {}
    while runs <= trials:
        print(runs)

        # --------------------BASE MODEL---------------------------------------------

        base_wts = read_asset_weights.loc[:, 'base']
        base_assets = list(base_wts.index)
        base_weights = list(base_wts.values)
        base_returns = list(read_returns_est.loc[:, 'Annualized Returns'].values)
        base_std = list(read_returns_est.loc[:, 'Annualized Risk'].values)

        base_investment = float(read_income_inputs.loc['risky_assets', 'Base'])
        adv_fees = float(read_income_inputs.loc['advisor_fees', 'Base'])

        # ----------------------RANDOM RETURNS--------------------------
        r_cols = ['r_{}'.format(name) for name in base_assets]
        boy_value = ['bv_{}'.format(name) for name in base_assets]
        eoy_value = ['ev_{}'.format(name) for name in base_assets]

        random_returns = pd.DataFrame(index=years, columns=r_cols)
        #
        # for c in range(len(r_cols)):
        #     ret = np.random.normal(base_returns[c], base_std[c], size=(len(random_returns.index), 1))

        this_run_cols = ['{}_{}'.format(cname, str(runs)) for cname in r_cols]
        random_returns = read_normal.loc[:, this_run_cols]
        base_df = random_returns.copy()

        # -------------BASE PORTFOLIO----------------------------
        for name in boy_value:
            base_df.loc[:, name] = 0.0

        for counter in years:
            period_returns = list(random_returns.loc[counter, :])
            if counter == 0:

                base_df.loc[counter, boy_value] = [base_weights[c] * base_investment for c in range(len(boy_value))]

                base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()

                base_investment = base_df.loc[counter, boy_value].sum()

            elif (counter > 0) and (counter < income_starts):

                base_df.loc[counter, boy_value] = [base_weights[c] * base_investment * (1 + period_returns[c])
                                                   for c in range(len(boy_value))]
                base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()

                # base_df.loc[counter, 'total_net_fees'] = base_df.loc[counter, 'total']

                # --coupon payment is invested back into the risky portfolio until the income is withdrawn----
                base_investment = base_df.loc[counter, 'total']

            else:

                base_df.loc[counter, boy_value] = [base_weights[c] * base_investment * (1 + period_returns[c])
                                                   for c in range(len(boy_value))]
                base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()

                base_investment = base_df.loc[counter, 'total']

        base_df.loc[:, 'adj_total'] = base_df.loc[:, 'total'].apply(lambda x: x if x > 0 else 0)

        asset_dict.update({'{}_{}'.format('s', str(runs)): base_df.loc[:, 'total']})
        # asset_dict.update({'{}_{}'.format('s', str(runs)): ret.flatten()})
        runs += 1

    asset_values = pd.DataFrame(asset_dict)
    assets_ror = asset_values.pct_change().fillna(0)
    terminal_values = asset_values.iloc[-1]
    terminal_df = pd.DataFrame(columns=['Terminal Value'])
    terminal_df['Terminal Value'] = terminal_values

    median_index = np.where(asset_values == np.median(np.array(terminal_values)))
    median_index = median_index[1][0]
    m_cols = ['{}_{}_{}'.format('r', r, str(median_index)) for r in all_assets]
    median_portfolio_asset_returns = read_normal.loc[:, m_cols]
    median_portfolio_asset_returns.loc[:, 'port_ending_values'] = asset_values.loc[:, '{}_{}'.format('s', median_index)]
    median_portfolio_asset_returns.loc[:, 'ror'] = median_portfolio_asset_returns[
        'port_ending_values'].pct_change().fillna(0)
    desc_sort = median_portfolio_asset_returns.sort_values(by=median_portfolio_asset_returns.columns[-1],
                                                           ascending=False)
    asc_sort = median_portfolio_asset_returns.sort_values(by=median_portfolio_asset_returns.columns[-1],
                                                          ascending=True)
    unsorted_df = median_portfolio_asset_returns
    unsorted_df.drop(unsorted_df.columns[-2:], axis=1, inplace=True)
    desc_sort.drop(desc_sort.columns[-2:], axis=1, inplace=True)
    asc_sort.drop(asc_sort.columns[-2:], axis=1, inplace=True)

    # reset index and drop existing index
    desc_sort.reset_index(drop=True, inplace=True)
    asc_sort.reset_index(drop=True, inplace=True)

    unsorted_df.to_csv(src + 'median_returns_unsorted.csv')
    desc_sort.to_csv(src + 'median_returns_largest.csv')
    asc_sort.to_csv(src + 'median_returns_smallest.csv')
    asset_values.to_csv(src + 'ending_values.csv')
    assets_ror.to_csv(src + 'ending_values_ror.csv')
    terminal_df.to_csv(src + 'terminal_values.csv')
    print("simulation completed....")


def generate_random_returns(num_of_years=30, trials=100, method='normal'):
    print("Generate random returns using method generate_random_returns()")
    frame = []
    years = list(range(0, num_of_years + 1))
    # ----------------------RANDOM RETURNS--------------------------
    read_asset_weights = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='asset_weights',
                                       index_col=[0])

    read_returns_est = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_assets_returns_estimates',
                                     index_col=[0])

    base_wts = read_asset_weights.loc[:, 'base']
    base_assets = list(base_wts.index)
    base_returns = list(read_returns_est.loc[:, 'Annualized Returns'].values)

    # -------------Remove the SPXT Index Value at the end------------------------------
    base_returns = base_returns[:-1]
    r_cols = ['r_{}'.format(name) for name in base_assets]

    # ------------read asset prices NAV values---------------------
    read_asset_prices = pd.read_csv(src + 'net_nav.csv', index_col=[0], parse_dates=True)
    cov_df = read_asset_prices.copy()
    cov_df.drop('BM', axis=1, inplace=True)
    asset_cov = cov_df.pct_change().fillna(0).cov()

    # annualize covariance matrix
    asset_cov = asset_cov * 12
    random_returns = pd.DataFrame(index=years, columns=r_cols)
    runs = 0

    while runs <= trials:
        ret = np.random.multivariate_normal(base_returns, asset_cov, size=(len(random_returns.index), 1))
        trial_df = pd.DataFrame(index=list(np.arange(len(random_returns.index))), columns=base_assets)
        for i in np.arange(len(random_returns.index)):
            trial_df.loc[trial_df.index[i]] = ret[i]
            old = list(trial_df.columns)
            new = ['r_{}_{}'.format(cname, str(runs)) for cname in base_assets]
            old_new = dict(zip(old, new))
            trial_df.rename(columns=old_new, inplace=True)

        frame.append(trial_df)
        runs += 1

    asset_df = pd.concat(frame, axis=1)
    asset_df.to_csv(src + 'sort_normal.csv')


def generate_median_portfolio_from_simulation(num_of_years=30, trials=100):
    print('Simulating for median portfolio...')
    """Select the median portfolio along with its corresponding asset returns for simulation. random assets and fia
    index returns are generated from N trials and portfolio returns are calculated. Based on the N terminal values of
    the portfolios, median terminal value is selected and their corresponding asset returns are save in a file to
    be simulated"""

    # read_income_inputs = pd.read_csv(src + "income_model_inputs.csv", index_col='Items')
    read_income_inputs = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_model_inputs',
                                       index_col=[0])

    # read_returns_est = pd.read_csv(src + "income_assets_returns_estimates.csv", index_col='Symbol')
    read_returns_est = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_assets_returns_estimates',
                                     index_col=[0])

    # read_returns_est.drop(['BM', read_returns_est.index[-1]], axis=0, inplace=True)
    # read_portfolio_inputs = pd.read_csv(src + "income_portfolio_inputs.csv", index_col='Items')

    # read_asset_weights = pd.read_csv(src + "asset_weights.csv", index_col='Asset')
    read_asset_weights = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='asset_weights',
                                       index_col=[0])

    all_assets = list(read_asset_weights.index)
    read_asset_weights.drop(read_asset_weights.index[-1], axis=0, inplace=True)

    # read random returns for simulation
    read_normal = pd.read_csv(src + 'sort_normal.csv', index_col=[0], parse_dates=True)

    tickers = list(read_asset_weights.index)
    wts = np.array(read_asset_weights.loc[:, 'base'])

    def asset_median_returns(data, ticker):
        return data.filter(regex=ticker).median(axis=1)

    # dataframe for unsorted returns (normal)
    median_returns_normal = pd.DataFrame({t: asset_median_returns(read_normal, t) for t in tickers})
    median_returns_normal.loc[:, 'portfolio_return'] = median_returns_normal.dot(wts)
    median_normal_fia = pd.DataFrame({'FIA': asset_median_returns(read_normal, 'r_FIA')})

    years = list(range(0, num_of_years + 1))
    income_starts = int(read_income_inputs.loc['start_income_years', 'inputs'])

    # # ---------------INCOME MODEL--------------------------------------------
    runs = 0
    asset_dict = {}
    while runs <= trials:
        print('for simulated returns_' + str(runs))
        # --------------------BASE MODEL---------------------------------------------

        base_wts = read_asset_weights.loc[:, 'base']
        base_assets = list(base_wts.index)
        base_weights = list(base_wts.values)
        base_investment = float(read_income_inputs.loc['risky_assets', 'Base'])

        # ----------------------RANDOM RETURNS--------------------------
        r_cols = ['r_{}'.format(name) for name in base_assets]
        boy_value = ['bv_{}'.format(name) for name in base_assets]

        this_run_cols = ['{}_{}'.format(cname, str(runs)) for cname in r_cols]
        random_returns = read_normal.loc[:, this_run_cols]
        base_df = random_returns.copy()

        # -------------BASE PORTFOLIO----------------------------
        for name in boy_value:
            base_df.loc[:, name] = 0.0

        for counter in years:
            period_returns = list(random_returns.loc[counter, :])
            if counter == 0:

                base_df.loc[counter, boy_value] = [base_weights[c] * base_investment for c in range(len(boy_value))]

                base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()

                base_investment = base_df.loc[counter, boy_value].sum()

            elif (counter > 0) and (counter < income_starts):

                base_df.loc[counter, boy_value] = [base_weights[c] * base_investment * (1 + period_returns[c])
                                                   for c in range(len(boy_value))]
                base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()

                # base_df.loc[counter, 'total_net_fees'] = base_df.loc[counter, 'total']

                # --coupon payment is invested back into the risky portfolio until the income is withdrawn----
                base_investment = base_df.loc[counter, 'total']

            else:

                base_df.loc[counter, boy_value] = [base_weights[c] * base_investment * (1 + period_returns[c])
                                                   for c in range(len(boy_value))]
                base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()

                base_investment = base_df.loc[counter, 'total']

        base_df.loc[:, 'adj_total'] = base_df.loc[:, 'total'].apply(lambda x: x if x > 0 else 0)

        asset_dict.update({'{}_{}'.format('s', str(runs)): base_df.loc[:, 'total']})

        runs += 1

    asset_values = pd.DataFrame(asset_dict)
    assets_ror = asset_values.pct_change().fillna(0)
    terminal_values = asset_values.iloc[-1]
    terminal_df = pd.DataFrame(columns=['Terminal Value'])
    terminal_df['Terminal Value'] = terminal_values

    median_index = np.where(asset_values == np.median(np.array(terminal_values)))
    median_index = median_index[1][0]
    m_cols = ['{}_{}_{}'.format('r', r, str(median_index)) for r in all_assets]
    median_portfolio_asset_returns = read_normal.loc[:, m_cols]
    median_portfolio_asset_returns.loc[:, 'port_ending_values'] = asset_values.loc[:, '{}_{}'.format('s', median_index)]
    median_portfolio_asset_returns.loc[:, 'ror'] = median_portfolio_asset_returns[
        'port_ending_values'].pct_change().fillna(0)
    desc_sort = median_portfolio_asset_returns.sort_values(by=median_portfolio_asset_returns.columns[-1],
                                                           ascending=False)
    asc_sort = median_portfolio_asset_returns.sort_values(by=median_portfolio_asset_returns.columns[-1],
                                                          ascending=True)
    unsorted_df = median_portfolio_asset_returns
    unsorted_df.drop(unsorted_df.columns[-2:], axis=1, inplace=True)
    desc_sort.drop(desc_sort.columns[-2:], axis=1, inplace=True)
    asc_sort.drop(asc_sort.columns[-2:], axis=1, inplace=True)

    # reset index and drop existing index
    desc_sort.reset_index(drop=True, inplace=True)
    asc_sort.reset_index(drop=True, inplace=True)

    unsorted_df.to_csv(src + 'median_returns_unsorted.csv')
    desc_sort.to_csv(src + 'median_returns_largest.csv')
    asc_sort.to_csv(src + 'median_returns_smallest.csv')
    asset_values.to_csv(src + 'ending_values.csv')
    assets_ror.to_csv(src + 'ending_values_ror.csv')
    terminal_df.to_csv(src + 'terminal_values.csv')
    print("finished simulation for median portfolio..")


def target_portfolio_simulation(num_of_years=30, trials=100, method='normal'):
    """Simulation run based on the target median portfolio selected. Median target portfolio is based on the
    median value of the terminal values from N trials and using the corresponding assets returns to generate income and
    accumulation analysis"""
    print("Running method target_portfolio_simulation()")

    sim_fia_cv = pd.DataFrame(index=range(num_of_years + 1))

    sim_base_total = pd.DataFrame(index=range(num_of_years + 1))
    sim_base_income = pd.DataFrame(index=range(num_of_years + 1))

    sim_port_total = pd.DataFrame(index=range(num_of_years + 1))
    sim_port_income = pd.DataFrame(index=range(num_of_years + 1))

    read_income_inputs = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_model_inputs',
                                       index_col=[0])

    read_returns_est = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_assets_returns_estimates',
                                     index_col=[0])

    read_asset_weights = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='asset_weights',
                                       index_col=[0])

    # read_asset_weights.drop(read_asset_weights.index[-1], axis=0, inplace=True)

    # read random returns for simulation
    read_normal = pd.read_csv(src + 'median_returns_unsorted.csv', index_col=[0], parse_dates=True)
    cols = [read_normal.columns[c].split('_')[1] for c in np.arange(len(read_normal.columns))]
    read_normal.rename(columns=dict(zip(list(read_normal.columns), cols)), inplace=True)

    read_small = pd.read_csv(src + 'median_returns_smallest.csv', index_col=[0], parse_dates=True)
    read_small.rename(columns=dict(zip(list(read_small.columns), cols)), inplace=True)

    read_large = pd.read_csv(src + 'median_returns_largest.csv', index_col=[0], parse_dates=True)
    read_large.rename(columns=dict(zip(list(read_large.columns), cols)), inplace=True)

    assets_col_names = list(read_normal.columns)

    tickers = list(read_asset_weights.index)
    wts = np.array(read_asset_weights.loc[:, 'base'])

    def asset_median_returns(data, ticker):
        return data.filter(regex=ticker).median(axis=1)

    # dataframe for unsorted returns (normal)
    median_returns_normal = read_normal.copy()
    median_returns_normal.loc[:, 'portfolio_return'] = median_returns_normal.dot(wts)
    median_normal_fia = pd.DataFrame({'FIA': asset_median_returns(read_normal, 'FIA')})

    # dataframe for smallest to largest returns
    median_returns_smallest = read_small.copy()
    median_returns_smallest.loc[:, 'portfolio_return'] = median_returns_smallest.dot(wts)
    median_smallest_fia = pd.DataFrame({'FIA': asset_median_returns(read_small, 'FIA')})

    # dataframe for largest to  smallest returns
    median_returns_largest = read_large.copy()
    median_returns_largest.loc[:, 'portfolio_return'] = median_returns_largest.dot(wts)
    median_largest_fia = pd.DataFrame({'FIA': asset_median_returns(read_large, 'FIA')})

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

    # ---------------INCOME MODEL--------------------------------------------
    runs = 0
    returns_dict = {}
    asset_dict = {}
    fia_dict = {}

    income_df = pd.DataFrame(index=years, columns=income_cols)
    income_df.loc[:, 'year'] = years
    income_df.loc[:, 'strategy_term'] = income_df.loc[:, 'year'] % term
    income_df.loc[:, 'strategy_term'] = income_df['strategy_term'].apply(lambda x: 1 if x == 0 else 0)

    if method == 'normal':
        income_df.loc[:, 'index_returns'] = read_normal.loc[:, 'FIA']

    elif method == 'smallest':
        income_df.loc[:, 'index_returns'] = read_small.loc[:, 'FIA']

    else:
        income_df.loc[:, 'index_returns'] = read_large.loc[:, 'FIA']

    # income_df.loc[:, 'index_returns'] = np.random.normal(fia_ret, fia_risk, size=(len(years), 1))

    cumprod = (1. + income_df['index_returns']).rolling(window=term).agg(lambda x: x.prod()) - 1
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

    income_df.loc[:, 'rider_fee'] = income_df.loc[:, 'high_inc_benefit_base'] * rider_fee
    income_df.loc[:, 'eoy_income'] = np.where(income_df.loc[:, 'year'] > income_starts,
                                              income_df.loc[:, 'high_inc_benefit_base'] * inc_payout_factor, 0)

    for counter in years:
        if counter == 0:
            income_df.loc[counter, 'contract_value'] = premium * (1 + contract_bonus)

        elif income_df.loc[counter, 'strategy_term'] == 1:
            x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee']
            x2 = (x1 * (1 + income_df.loc[counter, 'term_ret_netspr'])) - income_df.loc[counter, 'eoy_income']
            income_df.loc[counter, 'contract_value'] = x2

        else:
            x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee'] - \
                 income_df.loc[counter, 'eoy_income']

            income_df.loc[counter, 'contract_value'] = x1

    # variable stores the income number that is used in the base and fia portfolio calcs.

    income_from_fia = income_df.loc[income_df.index[-1], 'eoy_income']

    income_df.loc[:, 'contract_value'] = income_df.loc[:, 'contract_value'].apply(lambda x: 0 if x <= 0 else x)

    sim_fia_cv.loc[:, str(runs)] = income_df.loc[:, 'contract_value']

    # --------------------BASE MODEL---------------------------------------------
    base_wts = read_asset_weights.loc[:, 'base']
    base_assets = list(base_wts.index)
    base_weights = list(base_wts.values)
    base_returns = list(read_returns_est.loc[:, 'Annualized Returns'].values)
    base_std = list(read_returns_est.loc[:, 'Annualized Risk'].values)

    base_investment = float(read_income_inputs.loc['risky_assets', 'Base'])
    adv_fees = float(read_income_inputs.loc['advisor_fees', 'Base'])

    # -------------------required income----------------------------------
    req_annual_income = float(read_income_inputs.loc['annual_income', 'inputs'])
    income_needed = req_annual_income - social
    income_net_fia_income = max(0, income_needed - income_from_fia)
    cpn_income_base = base_investment * wtd_cpn_yield

    # ----------------------RANDOM RETURNS--------------------------
    r_cols = base_assets
    boy_value = ['bv_{}'.format(name) for name in base_assets]
    eoy_value = ['ev_{}'.format(name) for name in base_assets]

    random_returns = pd.DataFrame(index=income_df.index, columns=r_cols)

    for c in range(len(r_cols)):
        ret = np.random.normal(base_returns[c], base_std[c], size=(len(random_returns.index), 1))

    if method == 'smallest':
        random_returns = read_small.copy()

    elif method == 'largest':
        random_returns = read_large.copy()

    else:
        random_returns = read_normal.copy()

    base_df = random_returns.copy()
    fia_portfolio_df = random_returns.copy()
    port_investment = float(read_income_inputs.loc['risky_assets', 'FIA'])
    cpn_income_port = port_investment * wtd_cpn_yield

    # -------------BASE PORTFOLIO----------------------------
    for name in boy_value:
        base_df.loc[:, name] = 0.0

    for counter in years:
        period_returns = list(random_returns.loc[counter, :])
        if counter == 0:

            base_df.loc[counter, boy_value] = [base_weights[c] * base_investment for c in range(len(boy_value))]

            base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
            base_df.loc[counter, 'total_net_fees'] = 0.0
            base_df.loc[counter, 'income'] = 0.0
            base_investment = base_df.loc[counter, boy_value].sum()

        elif (counter > 0) and (counter < income_starts):

            base_df.loc[counter, boy_value] = [base_weights[c] * base_investment * (1 + period_returns[c])
                                               for c in range(len(boy_value))]
            base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
            base_df.loc[counter, 'adv_fees'] = base_df.loc[counter, 'total'] * adv_fees
            base_df.loc[counter, 'total_net_fees'] = base_df.loc[counter, 'total'] - base_df.loc[
                counter, 'adv_fees']

            # --coupon payment is invested back into the risky portfolio until the income is withdrawn----
            base_investment = base_df.loc[counter, 'total_net_fees'] + cpn_income_base

        else:

            base_df.loc[counter, boy_value] = [base_weights[c] * base_investment * (1 + period_returns[c])
                                               for c in range(len(boy_value))]
            base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
            base_df.loc[counter, 'adv_fees'] = base_df.loc[counter, 'total'] * adv_fees

            # ---req. income is adjusted for inflation from the second year of withdrawal. Reinvestment of coupon
            # stops from the year income starts. Req. income is reduced by the coupon payments

            if counter == income_starts:

                income_needed = req_annual_income - social
                base_df.loc[counter, 'income'] = income_needed - cpn_income_base
                income_needed = req_annual_income

            else:
                income_needed = income_needed * (1 + inflation) - social
                base_df.loc[counter, 'income'] = income_needed - cpn_income_base
                income_needed = income_needed + social

            base_df.loc[counter, 'total_net_fees'] = base_df.loc[counter, 'total'] - \
                                                     base_df.loc[counter, 'adv_fees'] - \
                                                     base_df.loc[counter, 'income']

            base_investment = base_df.loc[counter, 'total_net_fees']

    base_df.loc[:, 'adj_total'] = base_df.loc[:, 'total_net_fees'].apply(lambda x: x if x > 0 else 0)
    sim_base_total.loc[:, 's_{}'.format(str(runs))] = base_df.loc[:, 'total_net_fees']
    sim_base_income.loc[:, 's_{}'.format(str(runs))] = base_df.loc[:, 'income']

    # ----------------------------FIA PORTFOLIO----------------------------------------------
    for name in boy_value:
        fia_portfolio_df.loc[:, name] = 0.0

    for counter in years:
        period_returns = list(random_returns.loc[counter, :])
        if counter == 0:

            fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment
                                                        for c in range(len(boy_value))]
            fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
            fia_portfolio_df.loc[counter, 'total_net_fees'] = 0.0
            fia_portfolio_df.loc[counter, 'income'] = 0.0
            port_investment = fia_portfolio_df.loc[counter, boy_value].sum()

        elif (counter > 0) and (counter < income_starts):

            fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment * (1 + period_returns[c])
                                                        for c in range(len(boy_value))]
            fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
            fia_portfolio_df.loc[counter, 'adv_fees'] = fia_portfolio_df.loc[counter, 'total'] * adv_fees
            fia_portfolio_df.loc[counter, 'total_net_fees'] = fia_portfolio_df.loc[counter, 'total'] - \
                                                              fia_portfolio_df.loc[counter, 'adv_fees']

            port_investment = fia_portfolio_df.loc[counter, 'total_net_fees'] + cpn_income_port

        else:
            fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment * (1 + period_returns[c])
                                                        for c in range(len(boy_value))]
            fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
            fia_portfolio_df.loc[counter, 'adv_fees'] = fia_portfolio_df.loc[counter, 'total'] * adv_fees

            # ---req. income is adjusted for inflation from the second year of withdrawal. Reinvestment of coupon
            # stops from the year income starts. Req. income is reduced by the coupon payments

            if counter == income_starts:

                income_needed = req_annual_income - social
                income_net_fia_income = max(0, income_needed - income_from_fia)
                fia_portfolio_df.loc[counter, 'income'] = max(0, income_net_fia_income - cpn_income_port)
                income_needed = req_annual_income

            else:
                income_needed = income_needed * (1 + inflation) - social
                income_net_fia_income = max(0, income_needed - income_from_fia)
                fia_portfolio_df.loc[counter, 'income'] = max(0, income_net_fia_income - cpn_income_port)
                income_needed = income_needed + social

            fia_portfolio_df.loc[counter, 'total_net_fees'] = fia_portfolio_df.loc[counter, 'total'] - \
                                                              fia_portfolio_df.loc[counter, 'adv_fees'] - \
                                                              fia_portfolio_df.loc[counter, 'income']

            port_investment = fia_portfolio_df.loc[counter, 'total_net_fees']

    sim_port_total.loc[:, 's_{}'.format(str(runs))] = fia_portfolio_df.loc[:, 'total_net_fees'] + \
                                                      income_df.loc[:, 'contract_value']

    sim_port_income.loc[:, 's_{}'.format(str(runs))] = fia_portfolio_df.loc[:, 'income']

    fia_portfolio_df.loc[:, 'adj_total'] = fia_portfolio_df.loc[:, 'total_net_fees'].apply(
        lambda x: x if x > 0 else 0)

    # ---------income breakdown for Base portfolio----------------------------------
    base_df.to_csv(src + 'base_port_detail.csv')
    sim_base_total.to_csv(src + 'base_ending_values.csv')
    income_breakdown_base = pd.DataFrame(sim_base_total.quantile(0.5, axis=1))
    income_breakdown_base.loc[:, 'income_from_portfolio'] = sim_base_income.quantile(0.5, axis=1)
    income_breakdown_base.loc[:, 'fia_income'] = 0.0
    income_breakdown_base.loc[:, 'social_security_income'] = social
    income_breakdown_base.loc[:, 'coupon_income'] = cpn_income_base

    income_breakdown_base.rename(columns={income_breakdown_base.columns[0]: 'portfolio_ending_value'}, inplace=True)
    income_breakdown_base.loc[:, 'income_from_portfolio'][
        income_breakdown_base.loc[:, 'portfolio_ending_value'] <= 0] = 0
    income_breakdown_base.loc[:, 'total_income'] = income_breakdown_base.loc[:, income_breakdown_base.columns[1:]].sum(
        axis=1)

    # ------------Block Ends-------------------------------------------------------------

    # ---------income breakdown for FIA portfolio----------------------------------
    fia_portfolio_df.to_csv(src + 'fia_port_detail.csv')
    sim_port_total.to_csv(src + 'fiaport_ending_values.csv')

    income_breakdown_port = pd.DataFrame(sim_port_total.quantile(0.5, axis=1))
    income_breakdown_port.loc[:, 'income_from_portfolio'] = sim_port_income.quantile(0.5, axis=1)
    income_breakdown_port.loc[:, 'fia_income'] = income_from_fia
    income_breakdown_port.loc[:, 'social_security_income'] = social
    income_breakdown_port.loc[:, 'coupon_income'] = cpn_income_port

    income_breakdown_port.rename(columns={income_breakdown_port.columns[0]: 'portfolio_ending_value'}, inplace=True)
    income_breakdown_port.loc[:, 'income_from_portfolio'][
        income_breakdown_port.loc[:, 'portfolio_ending_value'] <= 0] = 0
    income_breakdown_port.loc[:, 'total_income'] = income_breakdown_port.loc[:, income_breakdown_port.columns[1:]].sum(
        axis=1)

    # ------------Block Ends-------------------------------------------------------------
    q_cut = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    sim_base_income[sim_base_total < income_needed] = 0.0

    sim_port_income[sim_port_total < income_net_fia_income] = 0

    sim_port_income = sim_port_income + income_from_fia

    # base_quantile = sim_base_total.loc[sim_base_total.index[-1]].quantile([0.05, 0.25, 0.50, 0.75, 0.90])

    # port_quantile = sim_port_total.loc[sim_port_total.index[-1]].quantile([0.05, 0.25, 0.50, 0.75, 0.90])

    base_quantile = sim_base_total.loc[sim_base_total.index[-1]].quantile(q_cut)

    port_quantile = sim_port_total.loc[sim_port_total.index[-1]].quantile(q_cut)

    # q_cut = [0.0, .05, 0.25, 0.5, 0.75, 0.95, 1.0]
    cols = ['Min', '5th', '25th', '50th', '75th', '90th', 'Max']

    # ----drop year 0--------
    sim_base_total = sim_base_total[1:]
    sim_port_total = sim_port_total[1:]

    # ----------------quantile analysis for base terminal value--------------------------
    base_qcut = pd.DataFrame(index=sim_base_total.index, columns=cols)
    for c in range(len(cols)):
        base_qcut.loc[:, cols[c]] = sim_base_total.quantile(q_cut[c], axis=1)

    base_qcut.clip(lower=0, inplace=True)

    # ----------------------quantile analysis for base income----------------------------
    base_income_qcut = pd.DataFrame(index=sim_base_income.index, columns=cols)
    for c in range(len(cols)):
        base_income_qcut.loc[:, cols[c]] = sim_base_income.quantile(q_cut[c], axis=1)

    # ----Remove NaN's prior to the income start years------------
    # base_income_qcut = base_income_qcut.loc[income_starts:]

    # -------------quantile analysis for portfolio terminal value ----------------
    port_qcut = pd.DataFrame(index=sim_port_total.index, columns=cols)
    for c in range(len(cols)):
        port_qcut.loc[:, cols[c]] = sim_port_total.quantile(q_cut[c], axis=1)

    port_qcut.clip(lower=0, inplace=True)

    # ---------------quantile analysis for portfolio income----------------------------
    port_income_qcut = pd.DataFrame(index=sim_port_income.index, columns=cols)
    for c in range(len(cols)):
        port_income_qcut.loc[:, cols[c]] = sim_port_income.quantile(q_cut[c], axis=1)

    # ----Remove NaN's prior to the income start years------------
    # port_income_qcut = port_income_qcut.loc[income_starts:]

    # ----------probability ending value will be less than 0 at the end of the horizon -----------------------
    base_legacy_risk = (sim_base_total.loc[sim_base_total.index[life_expectancy - clients_age]] < 0).sum() / (
            trials + 1)
    port_legacy_risk = (sim_port_total.loc[sim_port_total.index[life_expectancy - clients_age]] < 0).sum() / (
            trials + 1)

    legacy_risk = pd.DataFrame([base_legacy_risk, port_legacy_risk,
                                'Prob. of portfolio value less than 0 at the end of the expected life'],
                               index=['base', 'fia_portfolio', 'Notes'],
                               columns=['Ruin Probability'])

    # -----------Year-wise probability of ending value greater than 0 -----------------
    base_psuccess = sim_base_total.apply(lambda x: x > 0).sum(axis=1) / (trials + 1)
    port_psuccess = sim_port_total.apply(lambda x: x > 0).sum(axis=1) / (trials + 1)

    # -----------------------WRITING FILES TO EXCEL ---------------------------
    col_names = ['50th', 'age', 'comment']
    writer = pd.ExcelWriter(src + method + '_simulated_income_summary.xlsx', engine='xlsxwriter')
    read_income_inputs.to_excel(writer, sheet_name='inputs_for_income')

    read_returns_est.to_excel(writer, sheet_name='asset_returns_estimates')

    age_index = list(range(clients_age + 1, clients_age + len(base_qcut) + 1))
    base_qcut.loc[:, 'age'] = age_index
    base_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    base_qcut.loc[income_starts:, col_names].to_excel(writer, sheet_name='base_ending_value_quantiles')

    base_income_qcut = base_income_qcut.loc[1:, :]
    base_income_qcut.loc[:, 'age'] = age_index
    base_income_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    base_income_qcut.loc[income_starts:, col_names].to_excel(writer, sheet_name='base_income_quantiles')

    port_qcut.loc[:, 'age'] = age_index
    port_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    port_qcut.loc[income_starts:, col_names].to_excel(writer, sheet_name='fia_port_ending_value_quantiles')

    port_income_qcut = port_income_qcut.loc[1:, :]
    port_income_qcut.loc[:, 'age'] = age_index
    port_income_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    # port_income_qcut.loc[:, 'ending_contract_value'] = sim_fia_cv
    port_income_qcut.loc[income_starts:, col_names].to_excel(writer, sheet_name='fia_port_income_quantiles')

    # prob_success_df = pd.concat([base_psuccess, port_psuccess], axis=1)
    # prob_success_df.rename(columns={prob_success_df.columns[0]: 'prob(ending_value>0)_base',
    #                                 prob_success_df.columns[1]: 'prob(ending_value>0)_port'}, inplace=True)

    # prob_success_df.loc[:, 'age'] = age_index
    # prob_success_df.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    # prob_success_df.to_excel(writer, sheet_name='success_probability')

    income_breakdown_base = income_breakdown_base.loc[1:, :]
    income_breakdown_base.loc[:, 'age'] = age_index
    income_breakdown_base.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    income_breakdown_base.loc[income_starts:, :].to_excel(writer, sheet_name='base_income_breakdown_median')

    income_breakdown_port = income_breakdown_port.loc[1:, :]
    income_breakdown_port.loc[:, 'age'] = age_index
    income_breakdown_port.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    income_breakdown_port.loc[:, 'ending_contract_value'] = income_df.loc[:, 'contract_value']
    income_breakdown_port.loc[income_starts:, :].to_excel(writer, sheet_name='fia_income_breakdown_median')

    if method == 'normal':
        # median_returns_normal.loc[:, 'fia_median_returns'] = median_normal_fia
        median_returns_normal.to_excel(writer, sheet_name='gr_port_median_normal')

    elif method == 'smallest':
        # median_returns_smallest.loc[:, 'fia_median_returns'] = median_smallest_fia
        median_returns_smallest.to_excel(writer, sheet_name='gr_port_median_asc')

    else:
        # median_returns_largest.loc[:, 'fia_median_returns'] = median_largest_fia
        median_returns_largest.to_excel(writer, sheet_name='gr_port_median_desc')

    terminal_val = pd.read_csv(src + 'terminal_values.csv', index_col=[0])
    ending_val = pd.read_csv(src + 'ending_values.csv', index_col=[0])
    ending_val_ror = pd.read_csv(src + 'ending_values_ror.csv', index_col=[0])

    terminal_val.to_excel(writer, sheet_name='terminal_values')
    ending_val.to_excel(writer, sheet_name='port_ending_values')
    ending_val_ror.to_excel(writer, sheet_name='port_annual_growth')

    writer.save()

    # -----------------Plotting charts--------------------------------------------
    base_qcut.loc[income_starts:].plot(grid=True, title='Quantile Terminal Value - Base Portfolio')
    plt.savefig(src + "quantile_terminal_base.png")
    plt.close('all')

    base_income_qcut.plot(grid=True, title='Quantile Income - Base Portfolio')
    plt.savefig(src + "quantile_income_base.png")
    plt.close('all')

    base_psuccess.plot(grid=True, title='Probability of Success (Portfolio Ending Value > 0) - Base Portfolio')
    plt.savefig(src + "success_probabilty_base.png")
    plt.close('all')

    (1 - base_psuccess).plot(grid=True, title='Probability of Ruin (Portfolio Ending Value < 0) - Base Portfolio')
    plt.savefig(src + "ruin_probability_base.png")
    plt.close('all')

    port_qcut.loc[income_starts:].plot(grid=True, title='Quantile Terminal Value - FIA Portfolio')
    plt.savefig(src + "quantile_terminal_fia.png")
    plt.close('all')

    port_income_qcut.plot(grid=True, title='Quantile Income - FIA Portfolio')
    plt.savefig(src + "quantile_income_fia.png")
    plt.close('all')

    port_psuccess.plot(grid=True, title='Probability of Success (Portfolio Ending Value > 0) - FIA Portfolio')
    plt.savefig(src + "success_probabilty_fia.png")
    plt.close('all')

    (1 - port_psuccess).plot(grid=True, title='Probability of Ruin (Portfolio Ending Value < 0) - FIA Portfolio')
    plt.savefig(src + "ruin_probability_fia.png")
    plt.close('all')

    print("simulation completed for {}".format(method))


def portfolio_simulations_using_target_returns(num_of_years=30, trials=100):
    """Generate N portfolios randomizing the order of the annual returns from the median portfolio. Based on these
    portfolios calculate the probabilty of success and failure and other metrics"""
    print("Running portfolio_simulations_using_target_returns() method")
    sim_fia_cv = pd.DataFrame(index=range(num_of_years + 1))

    sim_base_total = pd.DataFrame(index=range(num_of_years + 1))
    sim_base_income = pd.DataFrame(index=range(num_of_years + 1))

    sim_port_total = pd.DataFrame(index=range(num_of_years + 1))
    sim_port_income = pd.DataFrame(index=range(num_of_years + 1))

    read_income_inputs = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_model_inputs',
                                       index_col=[0])

    read_returns_est = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_assets_returns_estimates',
                                     index_col=[0])

    read_asset_weights = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='asset_weights',
                                       index_col=[0])

    # read random returns for simulation
    read_normal = pd.read_csv(src + 'median_returns_unsorted.csv', index_col=[0], parse_dates=True)
    cols = [read_normal.columns[c].split('_')[1] for c in np.arange(len(read_normal.columns))]
    read_normal.rename(columns=dict(zip(list(read_normal.columns), cols)), inplace=True)
    idx = list(read_normal.index)

    runs = 0
    while runs <= trials:
        # --------Shuffling the path of random returns from the median portfolio
        np.random.shuffle(idx)
        read_normal = read_normal.set_index([idx])
        read_normal = read_normal.sort_index()

        # assets_col_names = list(read_normal.columns)
        # tickers = list(read_asset_weights.index)
        wts = np.array(read_asset_weights.loc[:, 'base'])

        def asset_median_returns(data, ticker):
            return data.filter(regex=ticker).median(axis=1)

        # dataframe for unsorted returns (normal)
        median_returns_normal = read_normal.copy()
        median_returns_normal.loc[:, 'portfolio_return'] = median_returns_normal.dot(wts)
        median_normal_fia = pd.DataFrame({'FIA': asset_median_returns(read_normal, 'FIA')})

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

        # ---------------INCOME MODEL--------------------------------------------
        # runs = 0
        returns_dict = {}
        asset_dict = {}
        fia_dict = {}

        income_df = pd.DataFrame(index=years, columns=income_cols)
        income_df.loc[:, 'year'] = years
        income_df.loc[:, 'strategy_term'] = income_df.loc[:, 'year'] % term
        income_df.loc[:, 'strategy_term'] = income_df['strategy_term'].apply(lambda x: 1 if x == 0 else 0)

        income_df.loc[:, 'index_returns'] = read_normal.loc[:, 'FIA']

        cumprod = (1. + income_df['index_returns']).rolling(window=term).agg(lambda x: x.prod()) - 1
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

        income_df.loc[:, 'rider_fee'] = income_df.loc[:, 'high_inc_benefit_base'] * rider_fee
        income_df.loc[:, 'eoy_income'] = np.where(income_df.loc[:, 'year'] > income_starts,
                                                  income_df.loc[:, 'high_inc_benefit_base'] * inc_payout_factor, 0)

        for counter in years:
            if counter == 0:
                income_df.loc[counter, 'contract_value'] = premium * (1 + contract_bonus)

            elif income_df.loc[counter, 'strategy_term'] == 1:
                x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee']
                x2 = (x1 * (1 + income_df.loc[counter, 'term_ret_netspr'])) - income_df.loc[counter, 'eoy_income']
                income_df.loc[counter, 'contract_value'] = x2

            else:
                x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee'] - \
                     income_df.loc[counter, 'eoy_income']

                income_df.loc[counter, 'contract_value'] = x1

        # variable stores the income number that is used in the base and fia portfolio calcs.

        income_from_fia = income_df.loc[income_df.index[-1], 'eoy_income']

        income_df.loc[:, 'contract_value'] = income_df.loc[:, 'contract_value'].apply(lambda x: 0 if x <= 0 else x)

        sim_fia_cv.loc[:, str(runs)] = income_df.loc[:, 'contract_value']

        # --------------------BASE MODEL---------------------------------------------

        base_wts = read_asset_weights.loc[:, 'base']
        base_assets = list(base_wts.index)
        base_weights = list(base_wts.values)
        base_returns = list(read_returns_est.loc[:, 'Annualized Returns'].values)
        base_std = list(read_returns_est.loc[:, 'Annualized Risk'].values)

        base_investment = float(read_income_inputs.loc['risky_assets', 'Base'])
        adv_fees = float(read_income_inputs.loc['advisor_fees', 'Base'])

        # -------------------required income----------------------------------
        req_annual_income = float(read_income_inputs.loc['annual_income', 'inputs'])
        income_needed = req_annual_income - social
        income_net_fia_income = max(0, income_needed - income_from_fia)
        cpn_income_base = base_investment * wtd_cpn_yield

        # ----------------------RANDOM RETURNS--------------------------
        r_cols = base_assets
        boy_value = ['bv_{}'.format(name) for name in base_assets]
        eoy_value = ['ev_{}'.format(name) for name in base_assets]

        random_returns = pd.DataFrame(index=income_df.index, columns=r_cols)

        random_returns = read_normal.copy()

        base_df = random_returns.copy()
        fia_portfolio_df = random_returns.copy()
        port_investment = float(read_income_inputs.loc['risky_assets', 'FIA'])
        cpn_income_port = port_investment * wtd_cpn_yield

        # -------------BASE PORTFOLIO----------------------------
        for name in boy_value:
            base_df.loc[:, name] = 0.0

        for counter in years:
            period_returns = list(random_returns.loc[counter, :])
            if counter == 0:

                base_df.loc[counter, boy_value] = [base_weights[c] * base_investment for c in range(len(boy_value))]

                base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
                base_df.loc[counter, 'total_net_fees'] = 0.0
                base_df.loc[counter, 'income'] = 0.0
                base_investment = base_df.loc[counter, boy_value].sum()

            elif (counter > 0) and (counter < income_starts):

                base_df.loc[counter, boy_value] = [base_weights[c] * base_investment * (1 + period_returns[c])
                                                   for c in range(len(boy_value))]
                base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
                base_df.loc[counter, 'adv_fees'] = base_df.loc[counter, 'total'] * adv_fees
                base_df.loc[counter, 'total_net_fees'] = base_df.loc[counter, 'total'] - base_df.loc[
                    counter, 'adv_fees']

                # --coupon payment is invested back into the risky portfolio until the income is withdrawn----
                base_investment = base_df.loc[counter, 'total_net_fees'] + cpn_income_base

            else:

                base_df.loc[counter, boy_value] = [base_weights[c] * base_investment * (1 + period_returns[c])
                                                   for c in range(len(boy_value))]
                base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
                base_df.loc[counter, 'adv_fees'] = base_df.loc[counter, 'total'] * adv_fees

                # ---req. income is adjusted for inflation from the second year of withdrawal. Reinvestment of coupon
                # stops from the year income starts. Req. income is reduced by the coupon payments

                if counter == income_starts:

                    income_needed = req_annual_income - social
                    base_df.loc[counter, 'income'] = income_needed - cpn_income_base
                    income_needed = req_annual_income

                else:
                    income_needed = income_needed * (1 + inflation) - social
                    base_df.loc[counter, 'income'] = income_needed - cpn_income_base
                    income_needed = income_needed + social

                base_df.loc[counter, 'total_net_fees'] = base_df.loc[counter, 'total'] - \
                                                         base_df.loc[counter, 'adv_fees'] - \
                                                         base_df.loc[counter, 'income']

                base_investment = base_df.loc[counter, 'total_net_fees']

        base_df.loc[:, 'adj_total'] = base_df.loc[:, 'total_net_fees'].apply(lambda x: x if x > 0 else 0)
        sim_base_total.loc[:, 's_{}'.format(str(runs))] = base_df.loc[:, 'total_net_fees']
        sim_base_income.loc[:, 's_{}'.format(str(runs))] = base_df.loc[:, 'income']

        # ----------------------------FIA PORTFOLIO----------------------------------------------
        for name in boy_value:
            fia_portfolio_df.loc[:, name] = 0.0

        for counter in years:
            period_returns = list(random_returns.loc[counter, :])
            if counter == 0:

                fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment
                                                            for c in range(len(boy_value))]
                fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
                fia_portfolio_df.loc[counter, 'total_net_fees'] = 0.0
                fia_portfolio_df.loc[counter, 'income'] = 0.0
                port_investment = fia_portfolio_df.loc[counter, boy_value].sum()

            elif (counter > 0) and (counter < income_starts):

                fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment * (1 + period_returns[c])
                                                            for c in range(len(boy_value))]
                fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
                fia_portfolio_df.loc[counter, 'adv_fees'] = fia_portfolio_df.loc[counter, 'total'] * adv_fees
                fia_portfolio_df.loc[counter, 'total_net_fees'] = fia_portfolio_df.loc[counter, 'total'] - \
                                                                  fia_portfolio_df.loc[counter, 'adv_fees']

                port_investment = fia_portfolio_df.loc[counter, 'total_net_fees'] + cpn_income_port

            else:
                fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment * (1 + period_returns[c])
                                                            for c in range(len(boy_value))]
                fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
                fia_portfolio_df.loc[counter, 'adv_fees'] = fia_portfolio_df.loc[counter, 'total'] * adv_fees

                # ---req. income is adjusted for inflation from the second year of withdrawal. Reinvestment of coupon
                # stops from the year income starts. Req. income is reduced by the coupon payments

                if counter == income_starts:

                    income_needed = req_annual_income - social
                    income_net_fia_income = max(0, income_needed - income_from_fia)
                    fia_portfolio_df.loc[counter, 'income'] = max(0, income_net_fia_income - cpn_income_port)
                    income_needed = req_annual_income

                else:
                    income_needed = income_needed * (1 + inflation) - social
                    income_net_fia_income = max(0, income_needed - income_from_fia)
                    fia_portfolio_df.loc[counter, 'income'] = max(0, income_net_fia_income - cpn_income_port)
                    income_needed = income_needed + social

                fia_portfolio_df.loc[counter, 'total_net_fees'] = fia_portfolio_df.loc[counter, 'total'] - \
                                                                  fia_portfolio_df.loc[counter, 'adv_fees'] - \
                                                                  fia_portfolio_df.loc[counter, 'income']

                port_investment = fia_portfolio_df.loc[counter, 'total_net_fees']

        sim_port_total.loc[:, 's_{}'.format(str(runs))] = fia_portfolio_df.loc[:, 'total_net_fees'] + \
                                                          income_df.loc[:, 'contract_value']

        sim_port_income.loc[:, 's_{}'.format(str(runs))] = fia_portfolio_df.loc[:, 'income']

        fia_portfolio_df.loc[:, 'adj_total'] = fia_portfolio_df.loc[:, 'total_net_fees'].apply(
            lambda x: x if x > 0 else 0)

        runs = runs + 1

    # ----% of trials ending value at expected life is less than 0
    inflation_factor = (1 + annual_inflation) ** (life_expectancy - clients_age - income_starts)
    required_income_horizon = income_needed * inflation_factor
    required_income_horizon_net_fia = required_income_horizon - income_from_fia

    # prob_success_base = (sim_base_total.iloc[life_expectancy - clients_age] < 0).sum() / trials
    # prob_success_fia_port = (sim_port_total.iloc[life_expectancy - clients_age] < 0).sum() / trials

    prob_failure_base = (sim_base_total.iloc[life_expectancy - clients_age] < required_income_horizon).sum() / trials
    prob_failure_fia_port = (sim_port_total.iloc[life_expectancy - clients_age] < required_income_horizon_net_fia) \
                                .sum() / trials

    # ----Calculate at total average lifetime income for base portfolio----
    base_inc = sim_base_income.copy()
    base_inc = base_inc.cumsum()
    avg_income_base = base_inc.iloc[life_expectancy - clients_age].mean()

    # ----Calculate at total average lifetime income for FIA portfolio----
    port_inc = sim_port_income.copy()
    port_inc = port_inc.cumsum()
    avg_income_port = port_inc.iloc[life_expectancy - clients_age].mean()
    avg_income_port = avg_income_port + (income_from_fia * (life_expectancy - clients_age))

    # ---------income breakdown for Base portfolio----------------------------------
    base_df.to_csv(src + 'base_port_detail.csv')
    sim_base_total.to_csv(src + 'base_ending_values.csv')
    income_breakdown_base = pd.DataFrame(sim_base_total.quantile(0.5, axis=1))
    income_breakdown_base.loc[:, 'income_from_portfolio'] = sim_base_income.quantile(0.5, axis=1)
    income_breakdown_base.loc[:, 'fia_income'] = 0.0
    income_breakdown_base.loc[:, 'social_security_income'] = social
    income_breakdown_base.loc[:, 'coupon_income'] = cpn_income_base

    income_breakdown_base.rename(columns={income_breakdown_base.columns[0]: 'portfolio_ending_value'}, inplace=True)
    income_breakdown_base.loc[:, 'income_from_portfolio'][
        income_breakdown_base.loc[:, 'portfolio_ending_value'] <= 0] = 0
    income_breakdown_base.loc[:, 'total_income'] = income_breakdown_base.loc[:, income_breakdown_base.columns[1:]].sum(
        axis=1)

    # ------------Block Ends-------------------------------------------------------------

    # ---------income breakdown for FIA portfolio----------------------------------
    fia_portfolio_df.to_csv(src + 'fia_port_detail.csv')
    sim_port_total.to_csv(src + 'fiaport_ending_values.csv')

    income_breakdown_port = pd.DataFrame(sim_port_total.quantile(0.5, axis=1))
    income_breakdown_port.loc[:, 'income_from_portfolio'] = sim_port_income.quantile(0.5, axis=1)
    income_breakdown_port.loc[:, 'fia_income'] = income_from_fia
    income_breakdown_port.loc[:, 'social_security_income'] = social
    income_breakdown_port.loc[:, 'coupon_income'] = cpn_income_port

    income_breakdown_port.rename(columns={income_breakdown_port.columns[0]: 'portfolio_ending_value'}, inplace=True)
    income_breakdown_port.loc[:, 'income_from_portfolio'][
        income_breakdown_port.loc[:, 'portfolio_ending_value'] <= 0] = 0
    income_breakdown_port.loc[:, 'total_income'] = income_breakdown_port.loc[:, income_breakdown_port.columns[1:]].sum(
        axis=1)

    # ------------Block Ends-------------------------------------------------------------
    q_cut = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    sim_base_income[sim_base_total < income_needed] = 0.0

    sim_port_income[sim_port_total < income_net_fia_income] = 0

    sim_port_income = sim_port_income + income_from_fia

    # base_quantile = sim_base_total.loc[sim_base_total.index[-1]].quantile([0.05, 0.25, 0.50, 0.75, 0.90])
    #
    # port_quantile = sim_port_total.loc[sim_port_total.index[-1]].quantile([0.05, 0.25, 0.50, 0.75, 0.90])

    base_quantile = sim_base_total.loc[sim_base_total.index[-1]].quantile(q_cut)

    port_quantile = sim_port_total.loc[sim_port_total.index[-1]].quantile(q_cut)

    # q_cut = [0.0, .05, 0.25, 0.5, 0.75, 0.95, 1.0]
    cols = ['Min', '5th', '25th', '50th', '75th', '90th', 'Max']

    # ----------------------------drop year 0--------------------------------
    sim_base_total = sim_base_total[1:]
    sim_port_total = sim_port_total[1:]

    # ---------quantile analysis for base terminal value-----------------------------
    base_qcut = pd.DataFrame(index=sim_base_total.index, columns=cols)
    for c in range(len(cols)):
        base_qcut.loc[:, cols[c]] = sim_base_total.quantile(q_cut[c], axis=1)

    base_qcut.clip(lower=0, inplace=True)

    # ---------quantile analysis for base income----------------------------
    base_income_qcut = pd.DataFrame(index=sim_base_income.index, columns=cols)
    for c in range(len(cols)):
        base_income_qcut.loc[:, cols[c]] = sim_base_income.quantile(q_cut[c], axis=1)

    # ----Remove NaN's prior to the income start years------------
    # base_income_qcut = base_income_qcut.loc[income_starts:]

    # -------------quantile analysis for portfolio terminal value ---------------

    port_qcut = pd.DataFrame(index=sim_port_total.index, columns=cols)
    for c in range(len(cols)):
        port_qcut.loc[:, cols[c]] = sim_port_total.quantile(q_cut[c], axis=1)

    port_qcut.clip(lower=0, inplace=True)

    # ---------------quantile analysis for portfolio income----------------------------
    port_income_qcut = pd.DataFrame(index=sim_port_income.index, columns=cols)
    for c in range(len(cols)):
        port_income_qcut.loc[:, cols[c]] = sim_port_income.quantile(q_cut[c], axis=1)

    # ----Remove NaN's prior to the income start years------------
    # port_income_qcut = port_income_qcut.loc[income_starts:]

    # ----------probability ending value will be less than 0 at the end of the horizon -------------------------

    base_legacy_risk = (sim_base_total.loc[sim_base_total.index[life_expectancy - clients_age]] < 0).sum() / (
            trials + 1)
    port_legacy_risk = (sim_port_total.loc[sim_port_total.index[life_expectancy - clients_age]] < 0).sum() / (
            trials + 1)

    legacy_risk = pd.DataFrame([base_legacy_risk, port_legacy_risk,
                                'Prob. of portfolio value less than 0 at the end of the expected life'],
                               index=['base', 'fia_portfolio', 'Notes'],
                               columns=['Ruin Probability'])

    # -----------Year-wise probability of ending value greater than 0 -----------------
    base_psuccess = sim_base_total.apply(lambda x: x > 0).sum(axis=1) / (trials + 1)
    port_psuccess = sim_port_total.apply(lambda x: x > 0).sum(axis=1) / (trials + 1)

    # -----------------------WRITING FILES TO EXCEL ---------------------------
    writer = pd.ExcelWriter(src + 'simulated_portfolios_summary.xlsx', engine='xlsxwriter')
    sim_base_total.to_excel(writer, sheet_name='base_ending_value')
    sim_port_total.to_excel(writer, sheet_name='fiaport_ending_value')
    base_qcut.to_excel(writer, sheet_name='base_quantile_ending')
    base_income_qcut.to_excel(writer, sheet_name='base_quantile_income')
    port_qcut.to_excel(writer, sheet_name='port_quantile_ending')
    port_income_qcut.to_excel(writer, sheet_name='port_quantile_income')

    sucess_df = pd.concat([base_psuccess, port_psuccess], axis=1)
    sucess_df.rename(columns={sucess_df.columns[0]: 'Base', sucess_df.columns[1]: 'Fia_Port'}, inplace=True)

    base_mean = sim_base_total[sim_base_total <= 0].isnull().sum().mean()
    port_mean = sim_port_total[sim_port_total <= 0].isnull().sum().mean()

    base_median = sim_base_total[sim_base_total <= 0].isnull().sum().median()
    port_median = sim_port_total[sim_port_total <= 0].isnull().sum().median()

    stats_df = pd.DataFrame([[base_mean, port_mean, 'Average years portfolio ending value > 0, out of N trials'],
                             [base_median, port_median, 'Ending Value >0, 50% of the time']],
                            index=['Mean years', 'Median years'], columns=['Base', 'fia_port', 'Comment'])

    # ---Average of terminal values at the end of horizon from N Trials
    stats_df.loc['Average Portfolio', 'Base'] = sim_base_total.iloc[-1].mean() + clients_age
    stats_df.loc['Average Portfolio', 'fia_port'] = sim_port_total.iloc[-1].mean() + clients_age
    stats_df.loc['Average Portfolio', 'Comment'] = "Average of terminal values at the end of analysis period" \
                                                   " from N Trials"

    # ----Median of terminal values at the end of horizon from N Trials
    stats_df.loc['Median Portfolio', 'Base'] = sim_base_total.iloc[-1].median() + clients_age
    stats_df.loc['Median Portfolio', 'fia_port'] = sim_port_total.iloc[-1].median() + clients_age
    stats_df.loc['Median Portfolio', 'Comment'] = "Median of terminal values at the end of analysis period " \
                                                  "from N Trials"

    # ---Average of terminal values at the end of Actuarial life from N Trials Base Portfolio
    stats_df.loc['Average Portfolio (end of expected_life)', 'Base'] = sim_base_total.iloc[
        life_expectancy - clients_age].mean()

    # ----Median of terminal values at the end of horizon from N Trials Base Portfolio
    stats_df.loc['Median Portfolio (end of expected_life)', 'Base'] = sim_base_total.iloc[
        life_expectancy - clients_age].median()

    # ---Average of terminal values at the end of Actuarial life from N Trials - FIA portfolio
    stats_df.loc['Average Portfolio (end of expected_life)', 'fia_port'] = sim_port_total.iloc[
        life_expectancy - clients_age].mean()
    stats_df.loc['Average Portfolio (end of expected_life)', 'Comment'] = "Average of terminal values at the end of " \
                                                                          "Actuarial life from N Trials"

    # ----Median of terminal values at the end of horizon from N Trials - FIA Portfolio
    stats_df.loc['Median Portfolio (end of expected_life)', 'fia_port'] = sim_port_total.iloc[
        life_expectancy - clients_age].median()
    stats_df.loc['Median Portfolio (end of expected_life)', 'Comment'] = "Median of terminal values at the end of " \
                                                                         "horizon from N Trials"

    stats_df.loc['% Success(value>0 at the end of expected_life)', 'Base'] = 1 - prob_failure_base
    stats_df.loc['% Success(value>0 at the end of expected_life)', 'fia_port'] = 1 - prob_failure_fia_port
    stats_df.loc['% Success(value>0 at the end of expected_life)', 'Comment'] = "% of N trials yearly ending value " \
                                                                                "greater than 0"

    # -----Mininum of N portfolios terminal value at the end of Actuarial Age
    stats_df.loc['Minimum Value', 'Base'] = sim_base_total.iloc[life_expectancy - clients_age].min()
    stats_df.loc['Minimum Value', 'fia_port'] = sim_port_total.iloc[life_expectancy - clients_age].min()
    stats_df.loc['Minimum Value', 'Comment'] = "Mininum of N portfolios terminal value at the end of Actuarial Age"

    # -----Maxinum of N portfolios terminal value at the end of Actuarial Age
    stats_df.loc['Maximum Value', 'Base'] = sim_base_total.iloc[life_expectancy - clients_age].max()
    stats_df.loc['Maximum Value', 'fia_port'] = sim_port_total.iloc[life_expectancy - clients_age].max()
    stats_df.loc['Maximum Value', 'Comment'] = "Maxinum of N portfolios terminal value at the end of Actuarial Age"

    # -----Average income from N portfolios at the ned of Actuarial Age
    stats_df.loc['Avg Income (lifetime)', 'Base'] = avg_income_base
    stats_df.loc['Avg Income (lifetime)', 'fia_port'] = avg_income_port
    stats_df.loc['Avg Income (lifetime)', 'Comment'] = "Average income from N portfolios at the end of Actuarial Age"

    sucess_df.to_excel(writer, sheet_name='yearly_success_rates')
    stats_df.to_excel(writer, sheet_name='Stats')

    writer.save()

    print("simulation completed.")


def income_model_asset_based_portfolio_quantile(num_of_years=30, trials=100, method='normal'):
    
    """Random assets returns are generated for N trials and Income and accumulation is simulated. The quantile analysis
    is run using the simulated N portofolios. Version 1 - Original Standard MONTE CARLO Simulation"""

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

    # read_returns_est = pd.read_csv(src + "income_assets_returns_estimates.csv", index_col='Symbol')
    read_returns_est = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_assets_returns_estimates',
                                     index_col=[0])

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

    # ---------------INCOME MODEL--------------------------------------------
    runs = 0
    returns_dict = {}
    asset_dict = {}
    fia_dict = {}
    while runs < trials:
        print(runs)

        income_df = pd.DataFrame(index=years, columns=income_cols)
        income_df.loc[:, 'year'] = years
        income_df.loc[:, 'strategy_term'] = income_df.loc[:, 'year'] % term
        income_df.loc[:, 'strategy_term'] = income_df['strategy_term'].apply(lambda x: 1 if x == 0 else 0)

        if method == 'normal':
            income_df.loc[:, 'index_returns'] = read_normal.loc[:, '{}_{}'.format('r_FIA', str(runs))]

        elif method == 'smallest':
            income_df.loc[:, 'index_returns'] = read_small.loc[:, '{}_{}'.format('r_FIA', str(runs))]

        else:
            income_df.loc[:, 'index_returns'] = read_large.loc[:, '{}_{}'.format('r_FIA', str(runs))]

        # income_df.loc[:, 'index_returns'] = np.random.normal(fia_ret, fia_risk, size=(len(years), 1))

        cumprod = (1. + income_df['index_returns']).rolling(window=term).agg(lambda x: x.prod()) - 1
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

        income_df.loc[:, 'rider_fee'] = income_df.loc[:, 'high_inc_benefit_base'] * rider_fee
        income_df.loc[:, 'eoy_income'] = np.where(income_df.loc[:, 'year'] > income_starts,
                                                  income_df.loc[:, 'high_inc_benefit_base'] * inc_payout_factor, 0)

        for counter in years:
            if counter == 0:
                income_df.loc[counter, 'contract_value'] = premium * (1 + contract_bonus)

            elif income_df.loc[counter, 'strategy_term'] == 1:
                x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee']
                x2 = (x1 * (1 + income_df.loc[counter, 'term_ret_netspr'])) - income_df.loc[counter, 'eoy_income']
                income_df.loc[counter, 'contract_value'] = x2

            else:
                x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee'] - \
                     income_df.loc[counter, 'eoy_income']

                income_df.loc[counter, 'contract_value'] = x1

        # variable stores the income number that is used in the base and fia portfolio calcs.

        income_from_fia = income_df.loc[income_df.index[-1], 'eoy_income']

        income_df.loc[:, 'contract_value'] = income_df.loc[:, 'contract_value'].apply(lambda x: 0 if x <= 0 else x)

        sim_fia_cv.loc[:, str(runs)] = income_df.loc[:, 'contract_value']

        # -------------------------------------BASE MODEL---------------------------------------------

        base_wts = read_asset_weights.loc[:, 'base']
        base_assets = list(base_wts.index)
        base_weights = list(base_wts.values)
        base_returns = list(read_returns_est.loc[:, 'Annualized Returns'].values)
        base_std = list(read_returns_est.loc[:, 'Annualized Risk'].values)

        base_investment = float(read_income_inputs.loc['risky_assets', 'Base'])
        adv_fees = float(read_income_inputs.loc['advisor_fees', 'Base'])

        # -------------------required income----------------------------------
        req_annual_income = float(read_income_inputs.loc['annual_income', 'inputs'])
        income_needed = req_annual_income - social
        income_net_fia_income = max(0, income_needed - income_from_fia)
        cpn_income_base = base_investment * wtd_cpn_yield

        # ----------------------RANDOM RETURNS--------------------------
        r_cols = ['r_{}'.format(name) for name in base_assets]
        boy_value = ['bv_{}'.format(name) for name in base_assets]
        eoy_value = ['ev_{}'.format(name) for name in base_assets]

        random_returns = pd.DataFrame(index=income_df.index, columns=r_cols)

        for c in range(len(r_cols)):
            ret = np.random.normal(base_returns[c], base_std[c], size=(len(random_returns.index), 1))

        if method == 'smallest':
            this_run_cols = ['{}_{}'.format(cname, str(runs)) for cname in r_cols]
            random_returns = read_small.loc[:, this_run_cols]

            # random_returns.loc[:, r_cols[c]] = np.sort(ret.flatten())
            # asset_dict.update({'{}_{}'.format(r_cols[c], str(runs)): np.sort(ret.flatten())})

        elif method == 'largest':
            this_run_cols = ['{}_{}'.format(cname, str(runs)) for cname in r_cols]
            random_returns = read_large.loc[:, this_run_cols]

            # random_returns.loc[:, r_cols[c]] = np.flip(np.sort(ret.flatten()))
            # asset_dict.update({'{}_{}'.format(r_cols[c], str(runs)): np.flip(np.sort(ret.flatten()))})

        else:
            this_run_cols = ['{}_{}'.format(cname, str(runs)) for cname in r_cols]
            random_returns = read_normal.loc[:, this_run_cols]

            # random_returns.loc[:, r_cols[c]] = ret.flatten()
            # asset_dict.update({'{}_{}'.format(r_cols[c], str(runs)): ret.flatten()})

        # store the simulated assets returns in one dictionary
        # returns_dict.update({str(runs): random_returns})

        # collect the asset based returns from all simulation and calculate the median returns.
        # def get_median_returns(sym):
        #     cols = [sym + '_' + str(c) for c in np.arange(trials)]
        #     asset_df = pd.DataFrame({c: asset_dict.get(c) for c in cols})
        #     return asset_df.median(axis=1)
        #
        # asset_median_returns = pd.DataFrame({symbol: get_median_returns(symbol) for symbol in r_cols})
        #
        # asset_median_returns.loc[:, 'simulated_portfolio_median_returns'] = asset_median_returns.dot(base_weights)

        base_df = random_returns.copy()
        pre_income_base_df = random_returns.copy()

        # base_investment = float(read_portfolio_inputs.loc['risky_assets', 'Base'])

        fia_portfolio_df = random_returns.copy()
        pre_income_port_df = random_returns.copy()
        port_investment = float(read_income_inputs.loc['risky_assets', 'FIA'])
        cpn_income_port = port_investment * wtd_cpn_yield

        # ---------Initial Investments for pre-income account values---------------------
        pre_income_base_inv = base_investment
        pre_income_port_inv = port_investment
        # ----------------------------------------BASE PORTFOLIO----------------------------
        for name in boy_value:
            base_df.loc[:, name] = 0.0
            pre_income_base_df.loc[:, name] = 0.0

        for counter in years:
            period_returns = list(random_returns.loc[counter, :])
            if counter == 0:
                # ---------------For year 0, the year of investment------------

                # ------------Calculate the annual portfolio returns - Gross Returns--------------------
                base_df.loc[counter, boy_value] = [base_weights[c] * base_investment for c in range(len(boy_value))]

                # -------------Record the Pre Income Base Portfolio-----------------------------

                pre_income_base_df.loc[counter, boy_value] = [base_weights[c] *
                                                              pre_income_base_inv for c in range(len(boy_value))]
                pre_income_base_df.loc[counter, 'total'] = pre_income_base_df.loc[counter, boy_value].sum()
                pre_income_base_inv = pre_income_base_df.loc[counter, boy_value].sum()

                # ------------------Pre Income Block Ends------------------------

                base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()

                base_df.loc[counter, 'total_net_fees'] = 0.0
                base_df.loc[counter, 'income'] = 0.0
                base_investment = base_df.loc[counter, boy_value].sum()
                base_df.loc[counter, 'total_pre_income'] = base_investment

            elif (counter > 0) and (counter < income_starts):

                # ----For years between the start of the investment and start if the income---------------
                base_df.loc[counter, boy_value] = [base_weights[c] * base_investment * (1 + period_returns[c])
                                                   for c in range(len(boy_value))]

                # -------------Record the Pre Income Base Portfolio-----------------------------
                pre_income_base_df.loc[counter, boy_value] = [
                    base_weights[c] * pre_income_base_inv * (1 + period_returns[c])
                    for c in range(len(boy_value))]

                pre_income_base_df.loc[counter, 'total'] = pre_income_base_df.loc[counter, boy_value].sum()
                pre_income_base_df.loc[counter, 'adv_fees'] = pre_income_base_df.loc[counter, 'total'] * adv_fees
                pre_income_base_df.loc[counter, 'total_net_fees'] = pre_income_base_df.loc[counter, 'total'] - \
                                                                    pre_income_base_df.loc[counter, 'adv_fees']
                pre_income_base_inv = pre_income_base_df.loc[counter, 'total_net_fees'] + cpn_income_base

                # ------------------Pre Income Block Ends------------------------

                base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
                base_df.loc[counter, 'adv_fees'] = base_df.loc[counter, 'total'] * adv_fees
                base_df.loc[counter, 'total_net_fees'] = base_df.loc[counter, 'total'] - base_df.loc[
                    counter, 'adv_fees']

                # --coupon payment is invested back into the risky portfolio until the income is withdrawn----
                base_investment = base_df.loc[counter, 'total_net_fees'] + cpn_income_base
                base_df.loc[counter, 'total_pre_income'] = base_df.loc[counter, 'total_net_fees']

            else:

                # -------------For Years after the income started----------------------
                base_df.loc[counter, boy_value] = [base_weights[c] * base_investment * (1 + period_returns[c])
                                                   for c in range(len(boy_value))]

                # -------------Record the Pre Income Base Portfolio-----------------------------
                pre_income_base_df.loc[counter, boy_value] = [
                    base_weights[c] * pre_income_base_inv * (1 + period_returns[c])
                    for c in range(len(boy_value))]

                pre_income_base_df.loc[counter, 'total'] = pre_income_base_df.loc[counter, boy_value].sum()
                pre_income_base_df.loc[counter, 'adv_fees'] = pre_income_base_df.loc[counter, 'total'] * adv_fees
                pre_income_base_df.loc[counter, 'total_net_fees'] = pre_income_base_df.loc[counter, 'total'] - \
                                                                    pre_income_base_df.loc[counter, 'adv_fees']
                pre_income_base_inv = pre_income_base_df.loc[counter, 'total_net_fees'] + cpn_income_base

                # ------------------Pre Income Block Ends------------------------
                base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
                base_df.loc[counter, 'adv_fees'] = base_df.loc[counter, 'total'] * adv_fees

                # ---req. income is adjusted for inflation from the second year of withdrawal. Reinvestment of coupon
                # stops from the year income starts. Req. income is reduced by the coupon payments

                if counter == income_starts:

                    income_needed = req_annual_income - social
                    base_df.loc[counter, 'income'] = income_needed - cpn_income_base
                    income_needed = req_annual_income

                else:
                    income_needed = income_needed * (1 + inflation) - social
                    base_df.loc[counter, 'income'] = income_needed - cpn_income_base
                    income_needed = income_needed + social

                base_df.loc[counter, 'total_net_fees'] = base_df.loc[counter, 'total'] - \
                                                         base_df.loc[counter, 'adv_fees'] - \
                                                         base_df.loc[counter, 'income']

                base_df.loc[counter, 'total_pre_income'] = base_df.loc[counter, 'total'] - \
                                                           base_df.loc[counter, 'adv_fees']

                base_investment = base_df.loc[counter, 'total_net_fees']

        # ------------------------Portfolio with PreIncome Values----------------------------
        sim_base_total_preincome.loc[:, 's_{}'.format(str(runs))] = pre_income_base_df.loc[:, 'total_net_fees']
        sim_base_total_preincome.fillna(float(read_income_inputs.loc['risky_assets', 'Base']), inplace=True)
        # --------------------------------PreIncome Block Ends----------------------------

        base_df.loc[:, 'adj_total'] = base_df.loc[:, 'total_net_fees'].apply(lambda x: x if x > 0 else 0)
        sim_base_total.loc[:, 's_{}'.format(str(runs))] = base_df.loc[:, 'total_net_fees']
        sim_base_income.loc[:, 's_{}'.format(str(runs))] = base_df.loc[:, 'income']
        sim_base_total_pre_income.loc[:, 's_{}'.format(str(runs))] = base_df.loc[:, 'total_pre_income']

        # ----------------------------FIA PORTFOLIO----------------------------------------------
        for name in boy_value:
            fia_portfolio_df.loc[:, name] = 0.0
            pre_income_port_df.loc[:, name] = 0.0

        for counter in years:
            period_returns = list(random_returns.loc[counter, :])
            if counter == 0:

                fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment
                                                            for c in range(len(boy_value))]

                # -------------Record the Pre Income Base Portfolio-----------------------------

                pre_income_port_df.loc[counter, boy_value] = [base_weights[c] *
                                                              pre_income_port_inv for c in range(len(boy_value))]
                pre_income_port_df.loc[counter, 'total'] = pre_income_port_df.loc[counter, boy_value].sum()
                pre_income_port_inv = pre_income_port_df.loc[counter, boy_value].sum()

                # ------------------Pre Income Block Ends------------------------

                fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
                fia_portfolio_df.loc[counter, 'total_net_fees'] = 0.0
                fia_portfolio_df.loc[counter, 'income'] = 0.0
                port_investment = fia_portfolio_df.loc[counter, boy_value].sum()
                fia_portfolio_df.loc[counter, 'total_pre_income'] = port_investment

            elif (counter > 0) and (counter < income_starts):

                fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment * (1 + period_returns[c])
                                                            for c in range(len(boy_value))]

                # ------------------Record the Pre Income Base Portfolio-----------------------------
                pre_income_port_df.loc[counter, boy_value] = [
                    base_weights[c] * pre_income_port_inv * (1 + period_returns[c])
                    for c in range(len(boy_value))]

                pre_income_port_df.loc[counter, 'total'] = pre_income_port_df.loc[counter, boy_value].sum()
                pre_income_port_df.loc[counter, 'adv_fees'] = pre_income_port_df.loc[counter, 'total'] * adv_fees
                pre_income_port_df.loc[counter, 'total_net_fees'] = pre_income_port_df.loc[counter, 'total'] - \
                                                                    pre_income_port_df.loc[counter, 'adv_fees']
                pre_income_port_inv = pre_income_port_df.loc[counter, 'total_net_fees'] + cpn_income_base

                # ------------------Pre Income Block Ends------------------------

                fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
                fia_portfolio_df.loc[counter, 'adv_fees'] = fia_portfolio_df.loc[counter, 'total'] * adv_fees
                fia_portfolio_df.loc[counter, 'total_net_fees'] = fia_portfolio_df.loc[counter, 'total'] - \
                                                                  fia_portfolio_df.loc[counter, 'adv_fees']
                fia_portfolio_df.loc[counter, 'total_pre_income'] = fia_portfolio_df.loc[counter, 'total_net_fees']

                port_investment = fia_portfolio_df.loc[counter, 'total_net_fees'] + cpn_income_port

            else:
                fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment * (1 + period_returns[c])
                                                            for c in range(len(boy_value))]

                # -------------Record the Pre Income Base Portfolio-----------------------------
                pre_income_port_df.loc[counter, boy_value] = [
                    base_weights[c] * pre_income_port_inv * (1 + period_returns[c])
                    for c in range(len(boy_value))]

                pre_income_port_df.loc[counter, 'total'] = pre_income_port_df.loc[counter, boy_value].sum()
                pre_income_port_df.loc[counter, 'adv_fees'] = pre_income_port_df.loc[counter, 'total'] * adv_fees
                pre_income_port_df.loc[counter, 'total_net_fees'] = pre_income_port_df.loc[counter, 'total'] - \
                                                                    pre_income_port_df.loc[counter, 'adv_fees']
                pre_income_port_inv = pre_income_port_df.loc[counter, 'total_net_fees'] + cpn_income_base

                # ------------------Pre Income Block Ends------------------------

                fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
                fia_portfolio_df.loc[counter, 'adv_fees'] = fia_portfolio_df.loc[counter, 'total'] * adv_fees

                # ---req. income is adjusted for inflation from the second year of withdrawal. Reinvestment of coupon
                # stops from the year income starts. Req. income is reduced by the coupon payments

                if counter == income_starts:

                    income_needed = req_annual_income - social
                    income_net_fia_income = max(0, income_needed - income_from_fia)
                    fia_portfolio_df.loc[counter, 'income'] = max(0, income_net_fia_income - cpn_income_port)
                    income_needed = req_annual_income

                else:
                    income_needed = income_needed * (1 + inflation) - social
                    income_net_fia_income = max(0, income_needed - income_from_fia)
                    fia_portfolio_df.loc[counter, 'income'] = max(0, income_net_fia_income - cpn_income_port)
                    income_needed = income_needed + social

                fia_portfolio_df.loc[counter, 'total_net_fees'] = fia_portfolio_df.loc[counter, 'total'] - \
                                                                  fia_portfolio_df.loc[counter, 'adv_fees'] - \
                                                                  fia_portfolio_df.loc[counter, 'income']

                fia_portfolio_df.loc[counter, 'total_pre_income'] = fia_portfolio_df.loc[counter, 'total'] - \
                                                                    fia_portfolio_df.loc[counter, 'adv_fees']

                port_investment = fia_portfolio_df.loc[counter, 'total_net_fees']

        sim_port_total.loc[:, 's_{}'.format(str(runs))] = fia_portfolio_df.loc[:, 'total_net_fees'] + \
                                                          income_df.loc[:, 'contract_value']

        sim_port_income.loc[:, 's_{}'.format(str(runs))] = fia_portfolio_df.loc[:, 'income']

        fia_portfolio_df.loc[:, 'adj_total'] = fia_portfolio_df.loc[:, 'total_net_fees'].apply(
            lambda x: x if x > 0 else 0)

        sim_port_total_pre_income.loc[:, 's_{}'.format(str(runs))] = fia_portfolio_df.loc[:, 'total_pre_income']

        # -------------------Portfolio with PreIncome Values----------------------------
        sim_port_total_preincome.loc[:, 's_{}'.format(str(runs))] = pre_income_port_df.loc[:, 'total_net_fees'] + \
                                                                    income_df.loc[:, 'contract_value']

        sim_port_total_preincome.fillna(float(read_income_inputs.loc['risky_assets', 'FIA']), inplace=True)
        # --------------------------------PreIncome Block Ends----------------------------

        runs += 1

    # ------------------Calculate % of portfolios ending value greater than required LIFETIME cumm. income---------
    total_income_by_age = sim_base_income.loc[:, sim_base_income.columns[0]].cumsum()
    total_income_by_acturial_age = total_income_by_age.loc[life_expectancy - clients_age]
    total_income_by_age.fillna(0, inplace=True)
    income_dataframe = pd.DataFrame(total_income_by_age)
    income_dataframe.loc[:, 'remaining_income_by_acturial_age'] = total_income_by_age.apply(
        lambda x: total_income_by_acturial_age - x)

    s = income_dataframe.loc[:, 'remaining_income_by_acturial_age']
    base_prob_of_success = sim_base_total.gt(s, axis=0).sum(axis=1)
    port_prob_of_success = sim_port_total.gt(s, axis=0).sum(axis=1)

    # ----------------------------Portfolio sufficient for NEXT YEARS income needs-------------------
    next_year_income = sim_base_income.loc[:, sim_base_income.columns[0]].shift(-1).fillna(0)  # Yearly Income Reqd.
    base_success_next_year = sim_base_total.gt(next_year_income, axis=0).sum(axis=1)

    base_for_next_year_need = sim_base_total[sim_base_total.gt(next_year_income, axis=0)]

    port_success_next_year = sim_port_total.gt(next_year_income, axis=0).sum(axis=1)

    port_for_next_year_need = sim_port_total[sim_port_total.gt(next_year_income, axis=0)]

    # ---------------Portfolio for 45 years of simulation---------------------------------------
    base_success_portfolio = sim_base_total[sim_base_total.gt(next_year_income, axis=0)]
    port_success_portfolio = sim_port_total[sim_port_total.gt(next_year_income, axis=0)]

    # ----------------Portfolio Simulation until the acturial age------------------------------
    acturial_years = life_expectancy - clients_age
    base_success_portfolio_act_age = base_success_portfolio.loc[acturial_years, :]
    port_success_portfolio_act_age = port_success_portfolio.loc[acturial_years, :]

    # -------------------------Base Portfolio TS with max Terminal Value ----------------------------
    if base_success_portfolio_act_age.isnull().sum() == trials:
        base_max_portfolio = 0.0
    else:
        base_max_portfolio = base_success_portfolio.loc[:, base_success_portfolio_act_age.idxmax()]

    # -------------------------FIA Portfolio TS with max Terminal Value ----------------------------
    if port_success_portfolio_act_age.isnull().sum() == trials:
        port_max_portfolio = 0.0
    else:
        port_max_portfolio = port_success_portfolio.loc[:, port_success_portfolio_act_age.idxmax()]

    # ------------------------------Average age with full income------------------------------
    base_mean_age = ((life_expectancy - clients_age) - base_success_portfolio.loc[1:life_expectancy - clients_age, :]
                     .isnull().sum()).mean()

    port_mean_age = ((life_expectancy - clients_age) - port_success_portfolio.loc[1:life_expectancy - clients_age, :]
                     .isnull().sum()).mean()

    # ----------------------------Median Age with full Income------------------------------------------
    base_median_age = ((life_expectancy - clients_age) - base_success_portfolio.loc[1:life_expectancy - clients_age, :]
                       .isnull().sum()).median()

    port_median_age = ((life_expectancy - clients_age) - port_success_portfolio.loc[1:life_expectancy - clients_age, :]
                       .isnull().sum()).median()

    # --------------Mean Value for all the portfolios at end of the acturial age--------------------
    base_act_avg_porfolio = base_success_portfolio.loc[life_expectancy - clients_age, :].fillna(0).mean()
    port_act_avg_porfolio = port_success_portfolio.loc[life_expectancy - clients_age, :].fillna(0).mean()

    # ---------------Median Value for all the portfolios at end of the acturial age--------------------
    base_act_median_porfolio = base_success_portfolio.loc[life_expectancy - clients_age, :].fillna(0).median()
    port_act_median_porfolio = port_success_portfolio.loc[life_expectancy - clients_age, :].fillna(0).median()

    # # --------------Mean Value for all the portfolios in the simulation--------------------
    # base_sim_mean = base_success_portfolio.loc[1:life_expectancy - clients_age, :].mean().mean()
    # port_sim_mean = port_success_portfolio.loc[1:life_expectancy - clients_age, :].mean().mean()
    #
    # # --------------Median Value for all the portfolios in the simulation--------------------
    # base_sim_mean = base_success_portfolio.loc[1:life_expectancy - clients_age, :].mean().median()
    # port_sim_mean = port_success_portfolio.loc[1:life_expectancy - clients_age, :].mean().median()

    # -------Max Portfolio value at the end of acturial age----------------------------------------
    base_act_max = base_success_portfolio.loc[life_expectancy - clients_age, :].max()
    port_act_max = port_success_portfolio.loc[life_expectancy - clients_age, :].max()

    # ----------------------Min Portfolio value at the end of acturial age----------------------------------------
    base_act_min = base_success_portfolio.loc[life_expectancy - clients_age, :].min()
    port_act_min = port_success_portfolio.loc[life_expectancy - clients_age, :].min()

    # -----------------------------------------Lifetime Average Income-----------------------------------------
    base_total_income = sim_base_income.cumsum().loc[acturial_years, :].mean()
    port_total_income = income_from_fia + sim_port_income
    port_total_income = port_total_income.cumsum().loc[acturial_years, :].mean()

    simulation_stats = pd.DataFrame(index=['Average Years', 'Median Years', 'Average Age', 'Median Age',
                                           'Average Portfolio (act.age)', 'Median Portfolio (act.age)',
                                           'Max Portfolio Value', 'Min Portfolio Value',
                                           'Average Lifetime Income'], columns=['Base', 'FIA'])

    simulation_stats.loc['Average Years', :] = [base_mean_age, base_mean_age]
    simulation_stats.loc['Median Years', :] = [base_median_age, base_median_age]
    simulation_stats.loc['Average Age', :] = [base_mean_age + clients_age, base_mean_age + clients_age]
    simulation_stats.loc['Median Age', :] = [base_median_age + clients_age, base_median_age + clients_age]
    simulation_stats.loc['Average Portfolio (act.age)', :] = [base_act_avg_porfolio, port_act_avg_porfolio]
    simulation_stats.loc['Median Portfolio (act.age)', :] = [base_act_median_porfolio, port_act_median_porfolio]
    simulation_stats.loc['Max Portfolio Value', :] = [base_act_max, port_act_max]
    simulation_stats.loc['Min Portfolio Value', :] = [base_act_min, port_act_min]
    simulation_stats.loc['Average Lifetime Income', :] = [base_total_income, port_total_income]
    comments = ['Average years of portfolios that meet the next years income needs for the lifetime',
                'Median years of portfolios that meet the next years income needs for the lifetime',
                'Average Clients Age',
                'Median Clients Age',
                'Average of terminal values for the portfolios at the end of the acturial life',
                'Median of terminal values for the portfolios at the end of the acturial life',
                'Maximum of terminal values for the portfolios at the end of the acturial life',
                'Minimum of terminal values for the portfolios at the end of the acturial life',
                'Average of total income generated by all portfolios at the end of the acturial life']

    simulation_stats.loc[:, 'Notes'] = comments

    # --------------------------------------------------------------------------------

    # # -----------------------------------income breakdown for Base portfolio----------------------------------
    # base_df.to_csv(src + 'base_port_detail.csv')
    # sim_base_total.to_csv(src + 'base_ending_values.csv')
    # income_breakdown_base = pd.DataFrame(sim_base_total.quantile(0.5, axis=1))
    # income_breakdown_base.loc[:, 'income_from_portfolio'] = sim_base_income.quantile(0.5, axis=1)
    # income_breakdown_base.loc[:, 'fia_income'] = 0.0
    # income_breakdown_base.loc[:, 'social_security_income'] = social
    # income_breakdown_base.loc[:, 'coupon_income'] = cpn_income_base
    #
    # income_breakdown_base.rename(columns={income_breakdown_base.columns[0]: 'portfolio_ending_value'}, inplace=True)
    # income_breakdown_base.loc[:, 'income_from_portfolio'][
    #     income_breakdown_base.loc[:, 'portfolio_ending_value'] <= 0] = 0
    # income_breakdown_base.loc[:, 'total_income'] = income_breakdown_base.loc[:, income_breakdown_base.columns[1:]].sum(
    #     axis=1)
    #
    # # --------------------------------------Block Ends-----------------------------------------------------------
    #
    # # ---------------------------------------income breakdown for FIA portfolio----------------------------------
    # fia_portfolio_df.to_csv(src + 'fia_port_detail.csv')
    # sim_port_total.to_csv(src + 'fiaport_ending_values.csv')
    #
    # income_breakdown_port = pd.DataFrame(sim_port_total.quantile(0.5, axis=1))
    # income_breakdown_port.loc[:, 'income_from_portfolio'] = sim_port_income.quantile(0.5, axis=1)
    # income_breakdown_port.loc[:, 'fia_income'] = income_from_fia
    # income_breakdown_port.loc[:, 'social_security_income'] = social
    # income_breakdown_port.loc[:, 'coupon_income'] = cpn_income_port
    #
    # income_breakdown_port.rename(columns={income_breakdown_port.columns[0]: 'portfolio_ending_value'}, inplace=True)
    # income_breakdown_port.loc[:, 'income_from_portfolio'][
    #     income_breakdown_port.loc[:, 'portfolio_ending_value'] <= 0] = 0
    # income_breakdown_port.loc[:, 'total_income'] = income_breakdown_port.loc[:, income_breakdown_port.columns[1:]].sum(
    #     axis=1)
    #
    # # ----------------------------------Block Ends-------------------------------------------------------------
    q_cut = [0.0, 0.1, 0.25, 0.5, 0.75, 0.95, 1.0]
    sim_base_income[sim_base_total < income_needed] = 0.0

    sim_port_income[sim_port_total < income_net_fia_income] = 0

    sim_port_income = sim_port_income + income_from_fia

    # base_quantile = sim_base_total.loc[sim_base_total.index[-1]].quantile([0.05, 0.25, 0.50, 0.75, 0.90])
    #
    # port_quantile = sim_port_total.loc[sim_port_total.index[-1]].quantile([0.05, 0.25, 0.50, 0.75, 0.90])

    base_quantile = sim_base_total.loc[sim_base_total.index[-1]].quantile(q_cut)

    port_quantile = sim_port_total.loc[sim_port_total.index[-1]].quantile(q_cut)

    # q_cut = [0.0, .05, 0.25, 0.5, 0.75, 0.95, 1.0]
    cols = ['Min', '10th', '25th', '50th', '75th', '90th', 'Max']

    # ------------------------------------------drop year 0-----------------------------------------
    sim_base_total = sim_base_total[1:]
    sim_port_total = sim_port_total[1:]

    # ---------------------------------plot for histogram for porfolios--------------------------------------
    # base_term_value = sim_base_total.loc[sim_base_total.index[:life_expectancy - clients_age], :]
    # fact = 1 / len(base_term_value)
    # base_ann_ret = (base_term_value.iloc[-1] / base_term_value.iloc[0]) ** fact - 1
    # counts, bins, bars = plt.hist(base_ann_ret)

    # ------------------------quantile analysis for base terminal value-----------------------------
    base_qcut = pd.DataFrame(index=sim_base_total.index, columns=cols)
    for c in range(len(cols)):
        base_qcut.loc[:, cols[c]] = sim_base_total.quantile(q_cut[c], axis=1)

    base_qcut.clip(lower=0, inplace=True)

    sim_base_total.clip(lower=0, inplace=True)

    # -------------------------------------quantile analysis for base income----------------------------
    base_income_qcut = pd.DataFrame(index=sim_base_income.index, columns=cols)
    for c in range(len(cols)):
        base_income_qcut.loc[:, cols[c]] = sim_base_income.quantile(q_cut[c], axis=1)

    # ----Remove NaN's prior to the income start years------------
    # base_income_qcut = base_income_qcut.loc[income_starts:]

    # ---------------------------------quantile analysis for portfolio terminal value ---------------

    port_qcut = pd.DataFrame(index=sim_port_total.index, columns=cols)
    for c in range(len(cols)):
        port_qcut.loc[:, cols[c]] = sim_port_total.quantile(q_cut[c], axis=1)

    port_qcut.clip(lower=0, inplace=True)

    # ----------------------------------quantile analysis for portfolio income----------------------------
    port_income_qcut = pd.DataFrame(index=sim_port_income.index, columns=cols)
    for c in range(len(cols)):
        port_income_qcut.loc[:, cols[c]] = sim_port_income.quantile(q_cut[c], axis=1)

    # ----Remove NaN's prior to the income start years------------
    # port_income_qcut = port_income_qcut.loc[income_starts:]

    # ----------probability ending value will be less than 0 at the end of the horizon -----------------------
    # base_legacy_risk = (sim_base_total.loc[sim_base_total.index[-1]] < 0).sum() / (trials)

    base_legacy_risk = (sim_base_total.loc[sim_base_total.index[life_expectancy - clients_age]] < 0).sum() / trials
    port_legacy_risk = (sim_port_total.loc[sim_port_total.index[life_expectancy - clients_age]] < 0).sum() / trials

    # port_legacy_risk = (sim_port_total.loc[sim_port_total.index[-1]] <= 0).sum() / (trials)

    legacy_risk = pd.DataFrame([base_legacy_risk, port_legacy_risk,
                                'Prob. of portfolio value less than 0 at the end of the expected life'],
                               index=['base', 'fia_portfolio', 'Notes'],
                               columns=['Ruin Probability'])

    # -----------Year-wise probability of ending value greater than 0 -----------------
    base_psuccess = sim_base_total.apply(lambda x: x > 0).sum(axis=1) / trials
    port_psuccess = sim_port_total.apply(lambda x: x > 0).sum(axis=1) / trials

    # ------------------------------------WRITING FILES TO EXCEL ---------------------------

    writer = pd.ExcelWriter(dest_simulation + method + '_montecarlo_income_summary.xlsx', engine='xlsxwriter')
    read_income_inputs.to_excel(writer, sheet_name='inputs_for_income')

    read_returns_est.to_excel(writer, sheet_name='asset_returns_estimates')
    # read_portfolio_inputs.to_excel(writer, sheet_name='portfolio_inputs')

    age_index = list(range(clients_age + 1, clients_age + len(base_qcut) + 1))
    # base_qcut.loc[:, 'clients_age'] = age_index
    # base_qcut.loc[:, 'comment'] = ''
    # base_qcut.loc[:, 'comment'] = np.where(base_qcut.clients_age == life_expectancy, 'expected_life', "")
    base_inv = float(read_income_inputs.loc['risky_assets', 'Base'])
    base_qcut.loc[:, 'age'] = age_index
    base_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'

    # -----------------------To start with year 0---------------------------------
    insert_col = [base_inv, base_inv, base_inv, base_inv, base_inv, base_inv,
                  base_inv, clients_age, np.nan]
    base_qcut.loc[len(base_qcut) + 1, :] = 0.0
    base_qcut = base_qcut.shift(1)
    base_qcut.iloc[0] = insert_col
    base_qcut.reset_index(drop=True, inplace=True)
    base_qcut.to_excel(writer, sheet_name='base_ending_value_quantiles')
    # base_qcut.loc[income_starts:, :].to_excel(writer, sheet_name='base_ending_value_quantiles')

    # base_income_qcut = base_income_qcut[1:] base_income_qcut.loc[:, 'clients_age'] = age_index
    # base_income_qcut.loc[:, 'comment'] = '' base_income_qcut.loc[:, 'comment'] = np.where(
    # base_income_qcut.clients_age == life_expectancy, 'expected_life', "")

    base_income_qcut = base_income_qcut.loc[1:, :]
    base_income_qcut.loc[:, 'age'] = age_index
    base_income_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    base_income_qcut.loc[income_starts:, :].to_excel(writer, sheet_name='base_income_quantiles')

    # age_index = list(range(clients_age+1, clients_age + len(port_qcut)+1))
    # port_qcut.loc[:, 'clients_age'] = age_index
    # port_qcut.loc[:, 'comment'] = ''
    # port_qcut.loc[:, 'comment'] = np.where(port_qcut.clients_age == life_expectancy, 'expected_life', "")

    port_qcut.loc[:, 'age'] = age_index
    port_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    port_qcut.loc[len(port_qcut) + 1, :] = 0.0
    port_qcut = port_qcut.shift(1)
    port_qcut.iloc[0] = insert_col
    port_qcut.reset_index(drop=True, inplace=True)
    port_qcut.to_excel(writer, sheet_name='fia_port_ending_value_quantiles')
    # port_qcut.loc[income_starts:, :].to_excel(writer, sheet_name='fia_port_ending_value_quantiles')

    # port_income_qcut = port_income_qcut[1:] port_income_qcut.loc[:, 'clients_age'] = age_index
    # port_income_qcut.loc[:, 'comment'] = '' port_income_qcut.loc[:, 'comment'] = np.where(
    # port_income_qcut.clients_age == life_expectancy, 'expected_life', "")

    port_income_qcut = port_income_qcut.loc[1:, :]
    port_income_qcut.loc[:, 'age'] = age_index
    port_income_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    port_income_qcut.loc[income_starts:, :].to_excel(writer, sheet_name='fia_port_income_quantiles')

    prob_success_df = pd.concat([base_psuccess, port_psuccess], axis=1)
    prob_success_df.rename(columns={prob_success_df.columns[0]: 'prob(ending_value>0)_base',
                                    prob_success_df.columns[1]: 'prob(ending_value>0)_port'}, inplace=True)

    # prob_success_df.loc[:, 'clients_age'] = age_index
    # prob_success_df.loc[:, 'comment'] = ''
    # prob_success_df.loc[:, 'comment'] = np.where(prob_success_df.clients_age == life_expectancy, 'expected_life', "")

    prob_success_df.loc[:, 'age'] = age_index
    prob_success_df.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    prob_success_df.loc[:, 'prob(ending_value>lifetime_req income)_base'] = base_prob_of_success / trials
    prob_success_df.loc[:, 'prob(ending_value>lifetime_req income)_port'] = port_prob_of_success / trials
    prob_success_df.loc[:, 'prob(ending_value>next_year_req_income)_base'] = base_success_next_year / trials
    prob_success_df.loc[:, 'prob(ending_value>next_year_req_income)_port'] = port_success_next_year / trials
    prob_success_df.loc[:, 'base_max_portfolio_at_acturial_age'] = base_max_portfolio
    prob_success_df.loc[:, 'port_max_portfolio_at_acturial_age'] = port_max_portfolio

    # --------------------Percentile Portfolio's based on Acturial Life------------------------
    base_success = prob_success_df.loc[life_expectancy - clients_age, 'prob(ending_value>next_year_req_income)_base']
    port_success = prob_success_df.loc[life_expectancy - clients_age, 'prob(ending_value>next_year_req_income)_port']

    # acturial_age_base_tv = sim_base_total.loc[:life_expectancy - clients_age, ]
    # percentile_base_tv = sim_base_total.apply(lambda x: np.nanpercentile(x, base_success), axis=1)

    # ----------------Year wise percentile portfolio to meet next year income. Based on the success at acturial age.
    # Yearly portfolio values that can provide the next year income below the success rate at end of life (Percentile)-

    # acturial_age_base = base_for_next_year_need.loc[:life_expectancy - clients_age, ]
    # acturial_age_base = base_for_next_year_need.copy().fillna(0)
    percentile_base = base_for_next_year_need.apply(lambda x: np.nanpercentile(x, base_success), axis=1)

    # ----Pre Income Portfolio based on the Probab. of Success to meet next year's income at the end on the Act. Age
    base_pre_income_success = sim_base_total_preincome.apply(lambda x: np.nanpercentile(x, base_success), axis=1)
    base_ann_ret_pre_income = base_pre_income_success.pct_change().fillna(0)

    # acturial_age_port_tv = sim_port_total.loc[:life_expectancy - clients_age, ]
    # percentile_port_tv = sim_port_total.apply(lambda x: np.nanpercentile(x, port_success), axis=1)

    # ----------------Year wise percentile portfolio to meet next year income. Based on the success at acturial age.
    # Yearly portfolio values that can provide the next year income below the success rate at end of life (Percentile)-

    # acturial_age_port = port_for_next_year_need.loc[:life_expectancy - clients_age, ]
    # acturial_age_base = port_for_next_year_need.copy().fillna(0)
    percentile_port = port_for_next_year_need.apply(lambda x: np.nanpercentile(x, port_success), axis=1)

    # ----Pre Income Portfolio based on the Probab. of Success to meet next year's income at the end on the Act. Age
    port_pre_income_success = sim_port_total_preincome.apply(lambda x: np.nanpercentile(x, port_success), axis=1)
    port_ann_ret_pre_income = port_pre_income_success.pct_change().fillna(0)

    prob_success_df.loc[:, 'acturial_success_percentile_base_portfolio'] = percentile_base
    prob_success_df.loc[:, 'acturial_success_percentile_port_portfolio'] = percentile_port

    prob_success_df.loc[:, 'base_pre_income_ann_ret'] = base_ann_ret_pre_income
    prob_success_df.loc[:, 'port_pre_income_ann_ret'] = port_ann_ret_pre_income

    # prob_success_df.loc[:, 'terminalVal_success_percentile_base_portfolio'] = percentile_base_tv
    # prob_success_df.loc[:, 'terminalVal_success_percentile_port_portfolio'] = percentile_port_tv

    sim_base_total_preincome.to_excel(writer, sheet_name='base_preincome_portfolios')
    # -------------Adding Premium to calculate the total initial investment--------------
    sim_port_total_preincome.iloc[0] = sim_port_total_preincome.iloc[0] + premium
    sim_port_total_preincome.to_excel(writer, sheet_name='port_preincome_portfolios')

    # -------------For Simulation slide - BASE Portfolio - Can Delete --------------------
    # base_qcut_preinc = pd.DataFrame(index=sim_base_total_preincome.index, columns=cols)
    # for c in range(len(cols)):
    #     base_qcut_preinc.loc[:, cols[c]] = sim_base_total_preincome.quantile(q_cut[c], axis=1)
    #
    # # -------------For Simulation slide - Proposed Portfolio --------------------
    # port_qcut_preinc = pd.DataFrame(index=sim_port_total_preincome.index, columns=cols)
    # for c in range(len(cols)):
    #     port_qcut_preinc.loc[:, cols[c]] = sim_port_total_preincome.quantile(q_cut[c], axis=1)
    #
    # base_qcut_preinc.to_excel(writer, sheet_name='base_preincome_quantiles')
    # port_qcut_preinc.to_excel(writer, sheet_name='port_preincome_quantiles')

    prob_success_df.to_excel(writer, sheet_name='success_probability')

    # --------------BASE - Accumulation and Income Breakdown based on the success percentile portfolio---------------
    base_df.to_csv(src + 'base_port_detail.csv')
    sim_base_total.to_csv(src + 'base_ending_values.csv')
    income_breakdown_base = pd.DataFrame(sim_base_total.quantile(base_success, axis=1))
    income_breakdown_base.loc[:, 'income_from_risky_assets'] = sim_base_income.quantile(base_success, axis=1) \
                                                               - social - cpn_income_port
    income_breakdown_base.loc[:, 'guaranteed_income'] = 0.0
    income_breakdown_base.loc[:, 'social_security_income'] = social
    income_breakdown_base.loc[:, 'coupon_income'] = cpn_income_base

    income_breakdown_base.rename(columns={income_breakdown_base.columns[0]: 'portfolio_ending_value'}, inplace=True)
    income_breakdown_base.loc[:, 'income_from_risky_assets'][
        income_breakdown_base.loc[:, 'portfolio_ending_value'] <= 0] = 0
    income_breakdown_base.loc[:, 'total_income'] = income_breakdown_base.loc[:, income_breakdown_base.columns[1:]].sum(
        axis=1)

    # ----------FIA PORTFOLIO - Accumulation and Income Breakdown based on the success percentile portfolio-----------
    fia_portfolio_df.to_csv(src + 'fia_port_detail.csv')
    sim_port_total.to_csv(src + 'fiaport_ending_values.csv')

    income_breakdown_port = pd.DataFrame(sim_port_total.quantile(port_success, axis=1))
    income_breakdown_port.loc[:, 'income_from_risky_assets'] = sim_port_income.quantile(port_success, axis=1) \
                                                               - income_from_fia - social - cpn_income_port
    income_breakdown_port.loc[:, 'guaranteed_income'] = income_from_fia
    income_breakdown_port.loc[:, 'social_security_income'] = social
    income_breakdown_port.loc[:, 'coupon_income'] = cpn_income_port

    income_breakdown_port.rename(columns={income_breakdown_port.columns[0]: 'portfolio_ending_value'}, inplace=True)
    income_breakdown_port.loc[:, 'income_from_risky_assets'][
        income_breakdown_port.loc[:, 'portfolio_ending_value'] <= 0] = 0
    income_breakdown_port.loc[:, 'total_income'] = income_breakdown_port.loc[:, income_breakdown_port.columns[1:]].sum(
        axis=1)

    # -------------------Write simulation Statistics-------------------------------------
    simulation_stats.to_excel(writer, sheet_name='simulation_statistics')

    # port_psuccess.to_excel(writer, sheet_name='fia_port_success_probability')

    income_breakdown_base = income_breakdown_base.loc[1:, :]
    income_breakdown_base.loc[:, 'age'] = age_index
    income_breakdown_base.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    income_breakdown_base.loc[income_starts:, :].to_excel(writer, sheet_name='base_income_breakdown_median')

    income_breakdown_port = income_breakdown_port.loc[1:, :]
    income_breakdown_port.loc[:, 'age'] = age_index
    income_breakdown_port.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    income_breakdown_port.loc[income_starts:, :].to_excel(writer, sheet_name='fia_income_breakdown_median')

    legacy_risk.to_excel(writer, sheet_name='ruin_probability')

    if method == 'normal':
        median_returns_normal.loc[:, 'fia_median_returns'] = median_normal_fia
        median_returns_normal.to_excel(writer, sheet_name='gr_port_median_normal')

    elif method == 'smallest':
        median_returns_smallest.loc[:, 'fia_median_returns'] = median_smallest_fia
        median_returns_smallest.to_excel(writer, sheet_name='gr_port_median_asc')

    else:
        median_returns_largest.loc[:, 'fia_median_returns'] = median_largest_fia
        median_returns_largest.to_excel(writer, sheet_name='gr_port_median_desc')

    # ---------------------Histogram for S&P Forecast---------------------------------------
    sp_returns = read_returns_est.loc['SPXT Index', 'Annualized Returns']
    sp_risk = read_returns_est.loc['SPXT Index', 'Annualized Risk']
    sp_random_ret = np.random.normal(loc=sp_returns, scale=sp_risk, size=10000)
    bins, data = np.histogram(sp_random_ret, bins=20)
    df_ret = pd.DataFrame(data, columns=['Return_range'])
    df_bins = pd.DataFrame(bins, columns=['Count'])
    df_hist = df_ret.join(df_bins)
    df_hist.to_excel(writer, sheet_name='sp500_histogram')

    # ---------------------Histogram for FIA Portfolios TV>0 at the acturial age---------------------------------------
    tval_at_horizon = sim_port_total.loc[acturial_years, :]
    fact = 1 / acturial_years
    arr_returns = np.array((tval_at_horizon / 1000000) ** fact - 1)
    clean_ann_ret = arr_returns[~np.isnan(arr_returns)]
    p_bins, p_data = np.histogram(clean_ann_ret, bins=20)
    df_ret = pd.DataFrame(p_data, columns=['Return_range'])
    df_bins = pd.DataFrame(p_bins, columns=['Count'])
    df_hist = df_ret.join(df_bins)
    df_hist.to_excel(writer, sheet_name='fia_portfolio_histogram')

    tval_df = pd.DataFrame(sim_port_total.loc[acturial_years, :])
    tval_df.rename(columns={tval_df.columns[0]:'Terminal Values'}, inplace=True)
    tval_df.to_excel(writer, sheet_name='fia_ending_values_hist')
    writer.save()

    # -----------------Plotting charts--------------------------------------------
    base_qcut.loc[income_starts:].plot(grid=True, title='Quantile Terminal Value - Base Portfolio')
    plt.savefig(src + "quantile_terminal_base.png")
    plt.close('all')

    base_income_qcut.plot(grid=True, title='Quantile Income - Base Portfolio')
    plt.savefig(src + "quantile_income_base.png")
    plt.close('all')

    base_psuccess.plot(grid=True, title='Probability of Success (Portfolio Ending Value > 0) - Base Portfolio')
    plt.savefig(src + "success_probabilty_base.png")
    plt.close('all')

    (1 - base_psuccess).plot(grid=True, title='Probability of Ruin (Portfolio Ending Value < 0) - Base Portfolio')
    plt.savefig(src + "ruin_probability_base.png")
    plt.close('all')

    port_qcut.loc[income_starts:].plot(grid=True, title='Quantile Terminal Value - FIA Portfolio')
    plt.savefig(src + "quantile_terminal_fia.png")
    plt.close('all')

    port_income_qcut.plot(grid=True, title='Quantile Income - FIA Portfolio')
    plt.savefig(src + "quantile_income_fia.png")
    plt.close('all')

    port_psuccess.plot(grid=True, title='Probability of Success (Portfolio Ending Value > 0) - FIA Portfolio')
    plt.savefig(src + "success_probabilty_fia.png")
    plt.close('all')

    (1 - port_psuccess).plot(grid=True, title='Probability of Ruin (Portfolio Ending Value < 0) - FIA Portfolio')
    plt.savefig(src + "ruin_probability_fia.png")
    plt.close('all')

    print("simulation completed....")
    

def optimized_income_model(ann_income, num_of_years=30, trials=100):
    sim_fia_cv = pd.DataFrame(index=range(num_of_years + 1))

    sim_base_total = pd.DataFrame(index=range(num_of_years + 1))
    sim_base_income = pd.DataFrame(index=range(num_of_years + 1))

    sim_port_total = pd.DataFrame(index=range(num_of_years + 1))
    sim_port_income = pd.DataFrame(index=range(num_of_years + 1))

    read_income_inputs = pd.read_csv(src + "income_model_inputs.csv", index_col='Items')
    read_returns_est = pd.read_csv(src + "income_assets_returns_estimates.csv", index_col='Symbol')
    # read_returns_est.drop(['BM', read_returns_est.index[-1]], axis=0, inplace=True)
    # read_portfolio_inputs = pd.read_csv(src + "income_portfolio_inputs.csv", index_col='Items')
    read_asset_weights = pd.read_csv(src + "asset_weights.csv", index_col='Asset')
    read_asset_weights.drop(read_asset_weights.index[-1], axis=0, inplace=True)
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

    # ---------------INCOME MODEL--------------------------------------------
    runs = 0

    while runs <= trials:
        # print(runs)

        income_df = pd.DataFrame(index=years, columns=income_cols)
        income_df.loc[:, 'year'] = years
        income_df.loc[:, 'strategy_term'] = income_df.loc[:, 'year'] % term
        income_df.loc[:, 'strategy_term'] = income_df['strategy_term'].apply(lambda x: 1 if x == 0 else 0)
        income_df.loc[:, 'index_returns'] = np.random.normal(fia_ret, fia_risk, size=(len(years), 1))
        cumprod = (1. + income_df['index_returns']).rolling(window=term).agg(lambda x: x.prod()) - 1
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

        income_df.loc[:, 'rider_fee'] = income_df.loc[:, 'high_inc_benefit_base'] * rider_fee
        income_df.loc[:, 'eoy_income'] = np.where(income_df.loc[:, 'year'] > income_starts,
                                                  income_df.loc[:, 'high_inc_benefit_base'] * inc_payout_factor, 0)

        for counter in years:
            if counter == 0:
                income_df.loc[counter, 'contract_value'] = premium * (1 + contract_bonus)

            elif income_df.loc[counter, 'strategy_term'] == 1:
                x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee']
                x2 = (x1 * (1 + income_df.loc[counter, 'term_ret_netspr'])) - income_df.loc[counter, 'eoy_income']
                income_df.loc[counter, 'contract_value'] = x2

            else:
                x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee'] - \
                     income_df.loc[counter, 'eoy_income']

                income_df.loc[counter, 'contract_value'] = x1

        # variable stores the income number that is used in the base and fia portfolio calcs.

        income_from_fia = income_df.loc[income_df.index[-1], 'eoy_income']

        income_df.loc[:, 'contract_value'] = income_df.loc[:, 'contract_value'].apply(lambda x: 0 if x <= 0 else x)

        sim_fia_cv.loc[:, str(runs)] = income_df.loc[:, 'contract_value']

        # --------------------BASE MODEL---------------------------------------------

        base_wts = read_asset_weights.loc[:, 'base']
        base_assets = list(base_wts.index)
        base_weights = list(base_wts.values)
        base_returns = list(read_returns_est.loc[:, 'Annualized Returns'].values)
        base_std = list(read_returns_est.loc[:, 'Annualized Risk'].values)

        base_investment = float(read_income_inputs.loc['risky_assets', 'Base'])
        adv_fees = float(read_income_inputs.loc['advisor_fees', 'Base'])

        # -------------------required income----------------------------------
        req_annual_income = ann_income
        income_needed = req_annual_income - social
        income_net_fia_income = max(0, income_needed - income_from_fia)
        cpn_income_base = base_investment * wtd_cpn_yield

        # ----------------------RANDOM RETURNS--------------------------
        r_cols = ['r_{}'.format(name) for name in base_assets]
        boy_value = ['bv_{}'.format(name) for name in base_assets]
        eoy_value = ['ev_{}'.format(name) for name in base_assets]

        random_returns = pd.DataFrame(index=income_df.index, columns=r_cols)

        for c in range(len(r_cols)):
            random_returns.loc[:, r_cols[c]] = np.random.normal(base_returns[c], base_std[c],
                                                                size=(len(random_returns.index), 1))

        base_df = random_returns.copy()

        # base_investment = float(read_portfolio_inputs.loc['risky_assets', 'Base'])

        fia_portfolio_df = random_returns.copy()
        port_investment = float(read_income_inputs.loc['risky_assets', 'FIA'])
        cpn_income_port = port_investment * wtd_cpn_yield

        # -------------BASE PORTFOLIO----------------------------
        for name in boy_value:
            base_df.loc[:, name] = 0.0

        for counter in years:
            period_returns = list(random_returns.loc[counter, :])
            if counter == 0:

                base_df.loc[counter, boy_value] = [base_weights[c] * base_investment for c in range(len(boy_value))]

                base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
                base_df.loc[counter, 'total_net_fees'] = 0.0
                base_df.loc[counter, 'income'] = 0.0
                base_investment = base_df.loc[counter, boy_value].sum()

            elif (counter > 0) and (counter < income_starts):

                base_df.loc[counter, boy_value] = [base_weights[c] * base_investment * (1 + period_returns[c])
                                                   for c in range(len(boy_value))]
                base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
                base_df.loc[counter, 'adv_fees'] = base_df.loc[counter, 'total'] * adv_fees
                base_df.loc[counter, 'total_net_fees'] = base_df.loc[counter, 'total'] - base_df.loc[
                    counter, 'adv_fees']

                # --coupon payment is invested back into the risky portfolio until the income is withdrawn----
                base_investment = base_df.loc[counter, 'total_net_fees'] + cpn_income_base

            else:

                base_df.loc[counter, boy_value] = [base_weights[c] * base_investment * (1 + period_returns[c])
                                                   for c in range(len(boy_value))]
                base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
                base_df.loc[counter, 'adv_fees'] = base_df.loc[counter, 'total'] * adv_fees

                # ---req. income is adjusted for inflation from the second year of withdrawal. Reinvestment of coupon
                # stops from the year income starts. Req. income is reduced by the coupon payments

                if counter == income_starts:

                    income_needed = req_annual_income - social
                    base_df.loc[counter, 'income'] = income_needed - cpn_income_base
                    income_needed = req_annual_income

                else:
                    income_needed = income_needed * (1 + inflation) - social
                    base_df.loc[counter, 'income'] = income_needed - cpn_income_base
                    income_needed = income_needed + social

                base_df.loc[counter, 'total_net_fees'] = base_df.loc[counter, 'total'] - \
                                                         base_df.loc[counter, 'adv_fees'] - \
                                                         base_df.loc[counter, 'income']

                base_investment = base_df.loc[counter, 'total_net_fees']

        base_df.loc[:, 'adj_total'] = base_df.loc[:, 'total_net_fees'].apply(lambda x: x if x > 0 else 0)
        sim_base_total.loc[:, 's_{}'.format(str(runs))] = base_df.loc[:, 'total_net_fees']
        sim_base_income.loc[:, 's_{}'.format(str(runs))] = base_df.loc[:, 'income']

        # ----------------------------FIA PORTFOLIO----------------------------------------------
        for name in boy_value:
            fia_portfolio_df.loc[:, name] = 0.0

        for counter in years:
            period_returns = list(random_returns.loc[counter, :])
            if counter == 0:

                fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment
                                                            for c in range(len(boy_value))]
                fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
                fia_portfolio_df.loc[counter, 'total_net_fees'] = 0.0
                fia_portfolio_df.loc[counter, 'income'] = 0.0
                port_investment = fia_portfolio_df.loc[counter, boy_value].sum()

            elif (counter > 0) and (counter < income_starts):

                fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment * (1 + period_returns[c])
                                                            for c in range(len(boy_value))]
                fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
                fia_portfolio_df.loc[counter, 'adv_fees'] = fia_portfolio_df.loc[counter, 'total'] * adv_fees
                fia_portfolio_df.loc[counter, 'total_net_fees'] = fia_portfolio_df.loc[counter, 'total'] - \
                                                                  fia_portfolio_df.loc[counter, 'adv_fees']

                port_investment = fia_portfolio_df.loc[counter, 'total_net_fees'] + cpn_income_port

            else:
                fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment * (1 + period_returns[c])
                                                            for c in range(len(boy_value))]
                fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
                fia_portfolio_df.loc[counter, 'adv_fees'] = fia_portfolio_df.loc[counter, 'total'] * adv_fees

                # ---req. income is adjusted for inflation from the second year of withdrawal. Reinvestment of coupon
                # stops from the year income starts. Req. income is reduced by the coupon payments

                if counter == income_starts:

                    income_needed = req_annual_income - social
                    income_net_fia_income = max(0, income_needed - income_from_fia)
                    fia_portfolio_df.loc[counter, 'income'] = max(0, income_net_fia_income - cpn_income_port)
                    income_needed = req_annual_income

                else:
                    income_needed = income_needed * (1 + inflation) - social
                    income_net_fia_income = max(0, income_needed - income_from_fia)
                    fia_portfolio_df.loc[counter, 'income'] = max(0, income_net_fia_income - cpn_income_port)
                    income_needed = income_needed + social

                fia_portfolio_df.loc[counter, 'total_net_fees'] = fia_portfolio_df.loc[counter, 'total'] - \
                                                                  fia_portfolio_df.loc[counter, 'adv_fees'] - \
                                                                  fia_portfolio_df.loc[counter, 'income']

                port_investment = fia_portfolio_df.loc[counter, 'total_net_fees']

        sim_port_total.loc[:, 's_{}'.format(str(runs))] = fia_portfolio_df.loc[:, 'total_net_fees'] + \
                                                          income_df.loc[:, 'contract_value']

        sim_port_income.loc[:, 's_{}'.format(str(runs))] = fia_portfolio_df.loc[:, 'income']

        fia_portfolio_df.loc[:, 'adj_total'] = fia_portfolio_df.loc[:, 'total_net_fees'].apply(
            lambda x: x if x > 0 else 0)

        runs += 1

    # ---------income breakdown for Base portfolio----------------------------------
    income_breakdown_base = pd.DataFrame(sim_base_total.quantile(0.5, axis=1))
    income_breakdown_base.loc[:, 'income_from_portfolio'] = sim_base_income.quantile(0.5, axis=1)
    income_breakdown_base.loc[:, 'fia_income'] = 0.0
    income_breakdown_base.loc[:, 'social_security_income'] = social
    income_breakdown_base.loc[:, 'coupon_income'] = cpn_income_base

    income_breakdown_base.rename(columns={income_breakdown_base.columns[0]: 'portfolio_ending_value'}, inplace=True)
    income_breakdown_base.loc[:, 'income_from_portfolio'][
        income_breakdown_base.loc[:, 'portfolio_ending_value'] <= 0] = 0
    income_breakdown_base.loc[:, 'total_income'] = income_breakdown_base.loc[:, income_breakdown_base.columns[1:]].sum(
        axis=1)

    # ------------Block Ends-------------------------------------------------------------

    # ---------income breakdown for FIA portfolio----------------------------------
    income_breakdown_port = pd.DataFrame(sim_port_total.quantile(0.5, axis=1))
    income_breakdown_port.loc[:, 'income_from_portfolio'] = sim_port_income.quantile(0.5, axis=1)
    income_breakdown_port.loc[:, 'fia_income'] = income_from_fia
    income_breakdown_port.loc[:, 'social_security_income'] = social
    income_breakdown_port.loc[:, 'coupon_income'] = cpn_income_port

    income_breakdown_port.rename(columns={income_breakdown_port.columns[0]: 'portfolio_ending_value'}, inplace=True)
    income_breakdown_port.loc[:, 'income_from_portfolio'][
        income_breakdown_port.loc[:, 'portfolio_ending_value'] <= 0] = 0
    income_breakdown_port.loc[:, 'total_income'] = income_breakdown_port.loc[:, income_breakdown_port.columns[1:]].sum(
        axis=1)

    # ------------Block Ends-------------------------------------------------------------
    q_cut = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    sim_base_income[sim_base_total < income_needed] = 0.0

    sim_port_income[sim_port_total < income_net_fia_income] = 0

    sim_port_income = sim_port_income + income_from_fia

    # base_quantile = sim_base_total.loc[sim_base_total.index[-1]].quantile([0.05, 0.25, 0.50, 0.75, 0.90])
    #
    # port_quantile = sim_port_total.loc[sim_port_total.index[-1]].quantile([0.05, 0.25, 0.50, 0.75, 0.90])

    base_quantile = sim_base_total.loc[sim_base_total.index[-1]].quantile(q_cut)

    port_quantile = sim_port_total.loc[sim_port_total.index[-1]].quantile(q_cut)

    # q_cut = [0.0, .05, 0.25, 0.5, 0.75, 0.95, 1.0]
    cols = ['Min', '5th', '25th', '50th', '75th', '90th', 'Max']

    # ----drop year 0--------
    sim_base_total = sim_base_total[1:]
    sim_port_total = sim_port_total[1:]

    # ---------quantile analysis for base terminal value-----------------------------
    base_qcut = pd.DataFrame(index=sim_base_total.index, columns=cols)
    for c in range(len(cols)):
        base_qcut.loc[:, cols[c]] = sim_base_total.quantile(q_cut[c], axis=1)

    base_qcut.clip(lower=0, inplace=True)

    # ---------quantile analysis for base income----------------------------
    base_income_qcut = pd.DataFrame(index=sim_base_income.index, columns=cols)
    for c in range(len(cols)):
        base_income_qcut.loc[:, cols[c]] = sim_base_income.quantile(q_cut[c], axis=1)

    # ----Remove NaN's prior to the income start years------------
    # base_income_qcut = base_income_qcut.loc[income_starts:]

    # -------------quantile analysis for portfolio terminal value ---------------

    port_qcut = pd.DataFrame(index=sim_port_total.index, columns=cols)
    for c in range(len(cols)):
        port_qcut.loc[:, cols[c]] = sim_port_total.quantile(q_cut[c], axis=1)

    port_qcut.clip(lower=0, inplace=True)

    # ---------------quantile analysis for portfolio income----------------------------
    port_income_qcut = pd.DataFrame(index=sim_port_income.index, columns=cols)
    for c in range(len(cols)):
        port_income_qcut.loc[:, cols[c]] = sim_port_income.quantile(q_cut[c], axis=1)

    # ----Remove NaN's prior to the income start years------------
    # port_income_qcut = port_income_qcut.loc[income_starts:]

    # ----------probability ending value will be less than 0 at the end of the horizon -------------------------
    # base_legacy_risk = (sim_base_total.loc[sim_base_total.index[-1]] < 0).sum() / (trials + 1)

    base_legacy_risk = (sim_base_total.loc[sim_base_total.index[life_expectancy - clients_age]] < 0).sum() / (
            trials + 1)
    port_legacy_risk = (sim_port_total.loc[sim_port_total.index[life_expectancy - clients_age]] < 0).sum() / (
            trials + 1)

    # port_legacy_risk = (sim_port_total.loc[sim_port_total.index[-1]] <= 0).sum() / (trials + 1)

    legacy_risk = pd.DataFrame([base_legacy_risk, port_legacy_risk,
                                'Prob. of portfolio value less than 0 at the end of the expected life'],
                               index=['base', 'fia_portfolio', 'Notes'],
                               columns=['Ruin Probability'])

    # -----------Year-wise probability of ending value greater than 0 -----------------
    base_psuccess = sim_base_total.apply(lambda x: x > 0).sum(axis=1) / (trials + 1)
    port_psuccess = sim_port_total.apply(lambda x: x > 0).sum(axis=1) / (trials + 1)

    # -----------------------WRITING FILES TO EXCEL ---------------------------

    writer = pd.ExcelWriter(src + 'optimized_income_summary.xlsx', engine='xlsxwriter')
    # read_income_inputs.to_excel(writer, sheet_name='inputs_for_income')
    #
    # read_returns_est.to_excel(writer, sheet_name='asset_returns_estimates')
    # read_portfolio_inputs.to_excel(writer, sheet_name='portfolio_inputs')

    age_index = list(range(clients_age + 1, clients_age + len(base_qcut) + 1))

    base_qcut.loc[:, 'age'] = age_index
    base_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    base_qcut.loc[income_starts:, :].to_excel(writer, sheet_name='base_ending_value_quantiles')

    base_income_qcut = base_income_qcut.loc[1:, :]
    base_income_qcut.loc[:, 'age'] = age_index
    base_income_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    base_income_qcut.loc[income_starts:, :].to_excel(writer, sheet_name='base_income_quantiles')

    port_qcut.loc[:, 'age'] = age_index
    port_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    port_qcut.loc[income_starts:, :].to_excel(writer, sheet_name='fia_port_ending_value_quantiles')

    port_income_qcut = port_income_qcut.loc[1:, :]
    port_income_qcut.loc[:, 'age'] = age_index
    port_income_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    port_income_qcut.loc[income_starts:, :].to_excel(writer, sheet_name='fia_port_income_quantiles')

    prob_success_df = pd.concat([base_psuccess, port_psuccess], axis=1)
    prob_success_df.rename(columns={prob_success_df.columns[0]: 'prob(ending_value>0)_base',
                                    prob_success_df.columns[1]: 'prob(ending_value>0)_port'}, inplace=True)

    prob_success_df.loc[:, 'age'] = age_index
    prob_success_df.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    prob_success_df.to_excel(writer, sheet_name='success_probability')

    income_breakdown_base = income_breakdown_base.loc[1:, :]
    income_breakdown_base.loc[:, 'age'] = age_index
    income_breakdown_base.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    income_breakdown_base.loc[income_starts:, :].to_excel(writer, sheet_name='base_income_breakdown_median')

    income_breakdown_port = income_breakdown_port.loc[1:, :]
    income_breakdown_port.loc[:, 'age'] = age_index
    income_breakdown_port.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    income_breakdown_port.loc[income_starts:, :].to_excel(writer, sheet_name='fia_income_breakdown_median')

    legacy_risk.to_excel(writer, sheet_name='ruin_probability')
    writer.save()

    # -----------------Plotting charts--------------------------------------------
    # base_qcut.loc[income_starts:].plot(grid=True, title='Quantile Terminal Value - Base Portfolio')
    # plt.savefig(src + "quantile_terminal_base.png")
    # plt.close('all')
    #
    # base_income_qcut.plot(grid=True, title='Quantile Income - Base Portfolio')
    # plt.savefig(src + "quantile_income_base.png")
    # plt.close('all')
    #
    # base_psuccess.plot(grid=True, title='Probability of Success (Portfolio Ending Value > 0) - Base Portfolio')
    # plt.savefig(src + "success_probabilty_base.png")
    # plt.close('all')
    #
    # (1 - base_psuccess).plot(grid=True, title='Probability of Ruin (Portfolio Ending Value < 0) - Base Portfolio')
    # plt.savefig(src + "ruin_probability_base.png")
    # plt.close('all')
    #
    # port_qcut.loc[income_starts:].plot(grid=True, title='Quantile Terminal Value - FIA Portfolio')
    # plt.savefig(src + "quantile_terminal_fia.png")
    # plt.close('all')
    #
    # port_income_qcut.plot(grid=True, title='Quantile Income - FIA Portfolio')
    # plt.savefig(src + "quantile_income_fia.png")
    # plt.close('all')
    #
    # port_psuccess.plot(grid=True, title='Probability of Success (Portfolio Ending Value > 0) - FIA Portfolio')
    # plt.savefig(src + "success_probabilty_fia.png")
    # plt.close('all')
    #
    # (1 - port_psuccess).plot(grid=True, title='Probability of Ruin (Portfolio Ending Value < 0) - FIA Portfolio')
    # plt.savefig(src + "ruin_probability_fia.png")
    # plt.close('all')

    print(income_breakdown_base.loc[:, ['total_income', 'comment']])
    print('*' * 100)
    print(income_breakdown_port.loc[:, ['total_income', 'comment']])

    x1 = income_breakdown_base.loc[income_breakdown_base.age == life_expectancy, 'total_income'].values[0]
    x2 = income_breakdown_base.loc[income_breakdown_base.age == life_expectancy + 1, 'total_income'].values[0]
    cond = x2 - x1
    return cond


def income_model_portfolio_based(num_of_years=30, iterations=100):
    # read_input_file = pd.read_csv(src + "income_model_inputs.csv", index_col='Items')
    read_input_file = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_model_inputs', index_col=[0])
    ann_income = float(read_input_file.loc['annual_income', 'inputs'])
    ann_inflation = float(read_input_file.loc['inflation', 'inputs'])

    # read the below inputs from the source file
    starting_investment = 1000000
    advisor_fees = .0075
    expected_std_dev = 0.10

    expected_portfolio_returns = [.02, .03, .04, .05, .06, .07, .08, .09, 0.1]
    expected_yearly_withdrawal = [round(ann_income * (1 + ann_inflation) ** c, 2) for c in
                                  np.arange(1, num_of_years + 1)]

    sim_port = {}
    average_port = {}
    for returns in expected_portfolio_returns:
        port_value = [starting_investment]
        beg_value = starting_investment
        exp_ret = np.random.normal(returns, expected_std_dev, size=(num_of_years, 1))
        print(exp_ret.mean(), exp_ret.std())
        print('#' * 100)
        for trials in np.arange(iterations):
            port_value = [starting_investment]
            beg_value = starting_investment
            exp_ret = np.random.normal(returns, expected_std_dev, size=(num_of_years, 1))

            for years in np.arange(num_of_years):
                ann_growth = exp_ret[years][0]
                portfolio_growth = beg_value * (1 + ann_growth)
                adv_fee = portfolio_growth * advisor_fees
                port_net_fees = portfolio_growth - adv_fee
                port_net_fee_and_wd = port_net_fees - expected_yearly_withdrawal[years]
                beg_value = port_net_fee_and_wd
                port_value.append(port_net_fee_and_wd)
            # print(port_value)
            sim_port.update({str(trials): port_value})

        combined = pd.DataFrame(sim_port)
        average_port.update({str(returns): combined.mean(axis=1)})
    df = pd.DataFrame(average_port)
    df.clip(lower=0, inplace=True)
    df.plot(grid=True)
    plt.show()
    df.to_csv(src + 'required_return_simulation.csv')
    print("test")


def simulation_using_historical_returns(num_of_years=30, trials=100, method='normal'):
    """Simulation run based annual return of the base portfolio from Accumulation analysis.
     Monthly portfolio returns are converted into annual returns and sorted in ascending and descending orders. The
     corresponding returns are used to generate the FIA income and income distribution model """

    print("Running method simulation_using_historical_returns()")

    sim_fia_cv = pd.DataFrame(index=range(num_of_years + 1))

    sim_base_total = pd.DataFrame(index=range(num_of_years + 1))
    sim_base_income = pd.DataFrame(index=range(num_of_years + 1))

    sim_port_total = pd.DataFrame(index=range(num_of_years + 1))
    sim_port_income = pd.DataFrame(index=range(num_of_years + 1))

    read_income_inputs = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_model_inputs',
                                       index_col=[0])

    read_returns_est = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_assets_returns_estimates',
                                     index_col=[0])

    read_asset_weights = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='asset_weights',
                                       index_col=[0])

    # read the base portfolio from accumulation model
    fia_name = read_income_inputs.loc['fia_name', 'inputs']
    path = src + fia_name + '/'
    file_name = fia_name + '_portfolio.csv'

    # base_port = pd.read_csv(path + 'base_portfolio.csv', index_col=[0], parse_dates=True)

    base_port = pd.read_csv(path + file_name, index_col=[0], parse_dates=True)

    ls1 = list(read_asset_weights.index[:-2])
    ls2 = ['Cash', fia_name, 'Total']
    combined_ls = ls1 + ls2
    base_port = base_port.loc[:, combined_ls]
    base_port.loc[:, 'port_return'] = base_port['Total'].pct_change().fillna(0)
    base_port.drop('Total', axis=1, inplace=True)
    yearly_base_port = base_port.groupby(by=base_port.index.year).apply(sum)

    # --Simulate only for first 20 years of returns--------
    first_n_years = int(read_income_inputs.loc['simulate_for_first_n_years', 'inputs'])
    yearly_base_port = yearly_base_port.loc[yearly_base_port.index[-(first_n_years + 1):], :]

    # --------Unsorted returns ---------------------
    read_normal = yearly_base_port.copy()
    read_normal.iloc[0] = 0.0
    read_normal.to_csv(src + 'base_port_ann_ret_unsorted.csv')
    read_normal.reset_index(drop=True, inplace=True)

    # ---make a copy to add to the consolidated summary file----
    save_normal = read_normal.copy()
    read_normal.drop('port_return', axis=1, inplace=True)
    read_normal.rename(columns={fia_name: 'FIA'}, inplace=True)

    # - Shift returns by one year and assign 0 returns to all assets in teh year 0, the year of initial investment.
    # read_normal = read_normal.shift().fillna(0)
    read_normal.to_csv(src + 'historical_port_return_unsorted.csv')

    # ----Best to worst returns----
    read_large = yearly_base_port.copy()

    # --Logic to sort the last 20 years of returns and assign 0 to first year, the year of investment
    read_large.iloc[0] = 1000.0
    read_large = read_large.sort_values(by=['port_return'], ascending=False)
    # --assign 0 to first year, the year of investment
    read_large.iloc[0] = 0.0
    read_large.to_csv(src + 'base_port_ann_ret_desc.csv')
    read_large.reset_index(drop=True, inplace=True)
    # read_large = read_large.shift().fillna(0)

    # ---make a copy to add to the consolidated summary file---
    save_desc = read_large.copy()
    read_large.drop('port_return', axis=1, inplace=True)
    read_large.rename(columns={fia_name: 'FIA'}, inplace=True)
    read_large.to_csv(src + 'historical_port_return_desc.csv')

    # ------Worst to Best Returns------------
    read_small = yearly_base_port.copy()
    read_small.iloc[0] = -1000.0
    read_small = read_small.sort_values(by=['port_return'], ascending=True)
    read_small.iloc[0] = 0.0
    read_small.to_csv(src + 'base_port_ann_ret_asc.csv')
    read_small.reset_index(drop=True, inplace=True)
    # read_small = read_small.shift().fillna(0)

    # ---make a copy to add to the consolidated summary file---
    save_asc = read_small.copy()
    read_small.drop('port_return', axis=1, inplace=True)
    read_small.rename(columns={fia_name: 'FIA'}, inplace=True)
    read_small.to_csv(src + 'historical_port_return_asc.csv')

    # read_asset_weights.drop(read_asset_weights.index[-1], axis=0, inplace=True)

    # read random returns for simulation
    # read_normal = pd.read_csv(src + 'median_returns_unsorted.csv', index_col=[0], parse_dates=True)
    # cols = [read_normal.columns[c].split('_')[1] for c in np.arange(len(read_normal.columns))]
    # read_normal.rename(columns=dict(zip(list(read_normal.columns), cols)), inplace=True)
    #
    # read_small = pd.read_csv(src + 'median_returns_smallest.csv', index_col=[0], parse_dates=True)
    # read_small.rename(columns=dict(zip(list(read_small.columns), cols)), inplace=True)
    #
    # read_large = pd.read_csv(src + 'median_returns_largest.csv', index_col=[0], parse_dates=True)
    # read_large.rename(columns=dict(zip(list(read_large.columns), cols)), inplace=True)

    assets_col_names = list(read_normal.columns)

    tickers = list(read_asset_weights.index)
    wts = np.array(read_asset_weights.loc[:, 'base'])

    def asset_median_returns(data, ticker):
        return data.filter(regex=ticker).median(axis=1)

    # dataframe for unsorted returns (normal)
    median_returns_normal = read_normal.copy()
    median_returns_normal.loc[:, 'portfolio_return'] = median_returns_normal.dot(wts)
    # median_returns_normal.rename(columns={fia_name: 'FIA'}, inplace=True)
    # median_normal_fia = pd.DataFrame({'FIA': asset_median_returns(read_normal, 'FIA')})

    # dataframe for smallest to largest returns
    median_returns_smallest = read_small.copy()
    median_returns_smallest.loc[:, 'portfolio_return'] = median_returns_smallest.dot(wts)
    # median_returns_smallest.rename(columns={fia_name: 'FIA'}, inplace=True)
    # median_smallest_fia = pd.DataFrame({'FIA': asset_median_returns(read_small, 'FIA')})

    # dataframe for largest to  smallest returns
    median_returns_largest = read_large.copy()
    median_returns_largest.loc[:, 'portfolio_return'] = median_returns_largest.dot(wts)
    # median_returns_largest.rename(columns={fia_name: 'FIA'}, inplace=True)
    # median_largest_fia = pd.DataFrame({'FIA': asset_median_returns(read_large, 'FIA')})

    # years = list(range(0, num_of_years + 1))
    years = np.arange(len(read_normal))
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
    # --Income start right away, not deferred ----
    income_starts = 0
    income_growth = float(read_income_inputs.loc['income_growth', 'inputs'])
    rider_fee = float(read_income_inputs.loc['rider_fee', 'inputs'])
    inc_payout_factor = float(read_income_inputs.loc['income_payout_factor', 'inputs'])
    contract_bonus = float(read_income_inputs.loc['contract_bonus', 'inputs'])
    social = float(read_income_inputs.loc['social', 'inputs'])
    inflation = float(read_income_inputs.loc['inflation', 'inputs'])
    wtd_cpn_yield = float(read_income_inputs.loc['wtd_coupon_yld', 'inputs'])
    life_expectancy = int(read_income_inputs.loc['life_expectancy_age', 'inputs'])
    clients_age = int(read_income_inputs.loc['clients_age', 'inputs'])

    # --------------------------------------INCOME MODEL--------------------------------------------
    runs = 0
    returns_dict = {}
    asset_dict = {}
    fia_dict = {}

    income_df = pd.DataFrame(index=years, columns=income_cols)
    income_df.loc[:, 'year'] = years
    income_df.loc[:, 'strategy_term'] = income_df.loc[:, 'year'] % term
    income_df.loc[:, 'strategy_term'] = income_df['strategy_term'].apply(lambda x: 1 if x == 0 else 0)

    if method == 'normal':
        income_df.loc[:, 'index_returns'] = read_normal.loc[:, 'FIA']

    elif method == 'smallest':
        income_df.loc[:, 'index_returns'] = read_small.loc[:, 'FIA']

    else:
        income_df.loc[:, 'index_returns'] = read_large.loc[:, 'FIA']

    # income_df.loc[:, 'index_returns'] = np.random.normal(fia_ret, fia_risk, size=(len(years), 1))

    cumprod = (1. + income_df['index_returns']).rolling(window=term).agg(lambda x: x.prod()) - 1
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

    income_df.loc[:, 'rider_fee'] = income_df.loc[:, 'high_inc_benefit_base'] * rider_fee
    income_df.loc[:, 'eoy_income'] = np.where(income_df.loc[:, 'year'] > income_starts,
                                              income_df.loc[:, 'high_inc_benefit_base'] * inc_payout_factor, 0)

    for counter in years:
        if counter == 0:
            income_df.loc[counter, 'contract_value'] = premium * (1 + contract_bonus)

        elif income_df.loc[counter, 'strategy_term'] == 1:
            x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee']
            x2 = (x1 * (1 + income_df.loc[counter, 'term_ret_netspr'])) - income_df.loc[counter, 'eoy_income']
            income_df.loc[counter, 'contract_value'] = x2

        else:
            x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee'] - \
                 income_df.loc[counter, 'eoy_income']

            income_df.loc[counter, 'contract_value'] = x1

    # variable stores the income number that is used in the base and fia portfolio calcs.

    income_from_fia = income_df.loc[income_df.index[-1], 'eoy_income']

    income_df.loc[:, 'contract_value'] = income_df.loc[:, 'contract_value'].apply(lambda x: 0 if x <= 0 else x)

    sim_fia_cv.loc[:, str(runs)] = income_df.loc[:, 'contract_value']

    # ---------------------------------BASE MODEL---------------------------------------------

    base_wts = read_asset_weights.loc[:, 'base']
    base_assets = list(base_wts.index)
    base_weights = list(base_wts.values)
    base_returns = list(read_returns_est.loc[:, 'Annualized Returns'].values)
    base_std = list(read_returns_est.loc[:, 'Annualized Risk'].values)

    base_investment = float(read_income_inputs.loc['risky_assets', 'Base'])
    adv_fees = float(read_income_inputs.loc['advisor_fees', 'Base'])

    # -------------------------------------required income--------------------------------
    req_annual_income = float(read_income_inputs.loc['annual_income', 'inputs'])
    income_needed = req_annual_income - social
    income_net_fia_income = max(0, income_needed - income_from_fia)
    cpn_income_base = base_investment * wtd_cpn_yield

    # --------------------------------------------RANDOM RETURNS-----------------------------------
    r_cols = base_assets
    boy_value = ['bv_{}'.format(name) for name in base_assets]
    eoy_value = ['ev_{}'.format(name) for name in base_assets]

    random_returns = pd.DataFrame(index=income_df.index, columns=r_cols)

    for c in range(len(r_cols)):
        ret = np.random.normal(base_returns[c], base_std[c], size=(len(random_returns.index), 1))

    if method == 'smallest':
        random_returns = read_small.copy()

    elif method == 'largest':
        random_returns = read_large.copy()

    else:
        random_returns = read_normal.copy()

    base_df = random_returns.copy()
    fia_portfolio_df = random_returns.copy()
    port_investment = float(read_income_inputs.loc['risky_assets', 'FIA'])
    cpn_income_port = port_investment * wtd_cpn_yield

    # -------------------------------------BASE PORTFOLIO----------------------------
    for name in boy_value:
        base_df.loc[:, name] = 0.0

    for counter in years:
        period_returns = list(random_returns.loc[counter, :])
        if counter == 0:

            base_df.loc[counter, boy_value] = [base_weights[c] * base_investment for c in range(len(boy_value))]
            base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
            base_df.loc[counter, 'total_net_fees'] = 0.0
            base_df.loc[counter, 'income'] = 0.0
            base_investment = base_df.loc[counter, boy_value].sum()

        elif (counter > 0) and (counter < income_starts):

            base_df.loc[counter, boy_value] = [base_weights[c] * base_investment * (1 + period_returns[c])
                                               for c in range(len(boy_value))]
            base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
            base_df.loc[counter, 'adv_fees'] = base_df.loc[counter, 'total'] * adv_fees
            base_df.loc[counter, 'total_net_fees'] = base_df.loc[counter, 'total'] - base_df.loc[
                counter, 'adv_fees']

            # --coupon payment is invested back into the risky portfolio until the income is withdrawn----
            base_investment = base_df.loc[counter, 'total_net_fees'] + cpn_income_base

        else:

            base_df.loc[counter, boy_value] = [base_weights[c] * base_investment * (1 + period_returns[c])
                                               for c in range(len(boy_value))]
            base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
            base_df.loc[counter, 'adv_fees'] = base_df.loc[counter, 'total'] * adv_fees

            # ---req. income is adjusted for inflation from the second year of withdrawal. Reinvestment of coupon
            # stops from the year income starts. Req. income is reduced by the coupon payments

            if counter == income_starts:

                income_needed = req_annual_income - social
                base_df.loc[counter, 'income'] = income_needed - cpn_income_base
                income_needed = req_annual_income

            else:
                income_needed = income_needed * (1 + inflation) - social
                base_df.loc[counter, 'income'] = income_needed - cpn_income_base
                income_needed = income_needed + social

            base_df.loc[counter, 'total_net_fees'] = base_df.loc[counter, 'total'] - \
                                                     base_df.loc[counter, 'adv_fees'] - \
                                                     base_df.loc[counter, 'income']

            base_investment = base_df.loc[counter, 'total_net_fees']

    base_df.loc[:, 'adj_total'] = base_df.loc[:, 'total_net_fees'].apply(lambda x: x if x > 0 else 0)
    sim_base_total.loc[:, 's_{}'.format(str(runs))] = base_df.loc[:, 'total_net_fees']
    sim_base_income.loc[:, 's_{}'.format(str(runs))] = base_df.loc[:, 'income']

    # -------------------required income----------------------------------
    req_annual_income = float(read_income_inputs.loc['annual_income', 'inputs'])
    income_needed = req_annual_income - social
    # income_net_fia_income = max(0, income_needed - income_from_fia)
    cpn_income_base = base_investment * wtd_cpn_yield

    # ----------------------------FIA PORTFOLIO----------------------------------------------
    for name in boy_value:
        fia_portfolio_df.loc[:, name] = 0.0

    for counter in years:
        period_returns = list(random_returns.loc[counter, :])
        if counter == 0:

            fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment
                                                        for c in range(len(boy_value))]
            fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
            fia_portfolio_df.loc[counter, 'total_net_fees'] = 0.0
            fia_portfolio_df.loc[counter, 'income'] = 0.0
            port_investment = fia_portfolio_df.loc[counter, boy_value].sum()

        elif (counter > 0) and (counter < income_starts):

            fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment * (1 + period_returns[c])
                                                        for c in range(len(boy_value))]
            fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
            fia_portfolio_df.loc[counter, 'adv_fees'] = fia_portfolio_df.loc[counter, 'total'] * adv_fees
            fia_portfolio_df.loc[counter, 'total_net_fees'] = fia_portfolio_df.loc[counter, 'total'] - \
                                                              fia_portfolio_df.loc[counter, 'adv_fees']

            port_investment = fia_portfolio_df.loc[counter, 'total_net_fees'] + cpn_income_port

        else:
            fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment * (1 + period_returns[c])
                                                        for c in range(len(boy_value))]
            fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
            fia_portfolio_df.loc[counter, 'adv_fees'] = fia_portfolio_df.loc[counter, 'total'] * adv_fees

            # ---req. income is adjusted for inflation from the second year of withdrawal. Reinvestment of coupon
            # stops from the year income starts. Req. income is reduced by the coupon payments

            if counter == income_starts:

                income_needed = req_annual_income - social
                income_net_fia_income = max(0, income_needed - income_from_fia)
                fia_portfolio_df.loc[counter, 'income'] = max(0, income_net_fia_income - cpn_income_port)
                income_needed = req_annual_income

            else:
                income_needed = income_needed * (1 + inflation) - social
                income_net_fia_income = max(0, income_needed - income_from_fia)
                fia_portfolio_df.loc[counter, 'income'] = max(0, income_net_fia_income - cpn_income_port)
                income_needed = income_needed + social

            fia_portfolio_df.loc[counter, 'total_net_fees'] = fia_portfolio_df.loc[counter, 'total'] - \
                                                              fia_portfolio_df.loc[counter, 'adv_fees'] - \
                                                              fia_portfolio_df.loc[counter, 'income']

            port_investment = fia_portfolio_df.loc[counter, 'total_net_fees']

    sim_port_total.loc[:, 's_{}'.format(str(runs))] = fia_portfolio_df.loc[:, 'total_net_fees'] + \
                                                      income_df.loc[:, 'contract_value']

    sim_port_income.loc[:, 's_{}'.format(str(runs))] = fia_portfolio_df.loc[:, 'income']

    fia_portfolio_df.loc[:, 'adj_total'] = fia_portfolio_df.loc[:, 'total_net_fees'].apply(
        lambda x: x if x > 0 else 0)

    # -------------------------------------income breakdown for Base portfolio----------------------------------
    base_df.to_csv(src + method + '_base_port_detail.csv')
    sim_base_total.to_csv(src + 'base_ending_values.csv')
    income_breakdown_base = pd.DataFrame(sim_base_total.quantile(0.5, axis=1))
    income_breakdown_base.loc[:, 'income_from_portfolio'] = sim_base_income.quantile(0.5, axis=1)
    income_breakdown_base.loc[:, 'fia_income'] = 0.0
    income_breakdown_base.loc[:, 'social_security_income'] = social
    income_breakdown_base.loc[:, 'coupon_income'] = cpn_income_base

    income_breakdown_base.rename(columns={income_breakdown_base.columns[0]: 'portfolio_ending_value'}, inplace=True)
    income_breakdown_base.loc[:, 'income_from_portfolio'][
        income_breakdown_base.loc[:, 'portfolio_ending_value'] <= 0] = 0
    income_breakdown_base.loc[:, 'total_income'] = income_breakdown_base.loc[:, income_breakdown_base.columns[1:]].sum(
        axis=1)

    # ------------------------------------Block Ends-------------------------------------------------------------

    # ------------------------------income breakdown for FIA portfolio----------------------------------
    fia_portfolio_df.loc[:, 'contract_value'] = income_df.loc[:, 'contract_value']
    fia_portfolio_df.to_csv(src + method + '_fia_port_detail.csv')
    sim_port_total.to_csv(src + 'fiaport_ending_values.csv')

    income_breakdown_port = pd.DataFrame(sim_port_total.quantile(0.5, axis=1))
    income_breakdown_port.loc[:, 'income_from_portfolio'] = sim_port_income.quantile(0.5, axis=1)
    income_breakdown_port.loc[:, 'fia_income'] = income_from_fia
    income_breakdown_port.loc[:, 'social_security_income'] = social
    income_breakdown_port.loc[:, 'coupon_income'] = cpn_income_port

    income_breakdown_port.rename(columns={income_breakdown_port.columns[0]: 'portfolio_ending_value'}, inplace=True)
    income_breakdown_port.loc[:, 'income_from_portfolio'][
        income_breakdown_port.loc[:, 'portfolio_ending_value'] <= 0] = 0
    income_breakdown_port.loc[:, 'total_income'] = income_breakdown_port.loc[:, income_breakdown_port.columns[1:]].sum(
        axis=1)

    # ------------------------------------Block Ends-------------------------------------------------------------
    q_cut = [0.0, .05, 0.25, 0.5, 0.75, 0.95, 1.0]
    sim_base_income[sim_base_total < income_needed] = 0.0

    sim_port_income[sim_port_total < income_net_fia_income] = 0

    sim_port_income = sim_port_income + income_from_fia

    base_quantile = sim_base_total.loc[sim_base_total.index[-1]].quantile(q_cut)

    port_quantile = sim_port_total.loc[sim_port_total.index[-1]].quantile(q_cut)

    cols = ['Min', '5th', '25th', '50th', '75th', '90th', 'Max']

    # ----------------------------------drop year 0---------------------------------
    sim_base_total = sim_base_total[1:]
    sim_port_total = sim_port_total[1:]

    # --------------------------------quantile analysis for base terminal value-----------------------------
    base_qcut = pd.DataFrame(index=sim_base_total.index, columns=cols)
    for c in range(len(cols)):
        base_qcut.loc[:, cols[c]] = sim_base_total.quantile(q_cut[c], axis=1)

    base_qcut.clip(lower=0, inplace=True)

    # ----------------------------------quantile analysis for base income----------------------------
    base_income_qcut = pd.DataFrame(index=sim_base_income.index, columns=cols)
    for c in range(len(cols)):
        base_income_qcut.loc[:, cols[c]] = sim_base_income.quantile(q_cut[c], axis=1)

    # ----Remove NaN's prior to the income start years------------
    # base_income_qcut = base_income_qcut.loc[income_starts:]

    # -------------quantile analysis for portfolio terminal value ---------------

    port_qcut = pd.DataFrame(index=sim_port_total.index, columns=cols)
    for c in range(len(cols)):
        port_qcut.loc[:, cols[c]] = sim_port_total.quantile(q_cut[c], axis=1)

    port_qcut.clip(lower=0, inplace=True)

    # ---------------quantile analysis for portfolio income----------------------------
    port_income_qcut = pd.DataFrame(index=sim_port_income.index, columns=cols)
    for c in range(len(cols)):
        port_income_qcut.loc[:, cols[c]] = sim_port_income.quantile(q_cut[c], axis=1)

    # ----Remove NaN's prior to the income start years------------
    # port_income_qcut = port_income_qcut.loc[income_starts:]

    # ----------probability ending value will be less than 0 at the end of the horizon -------------------------

    base_legacy_risk = (sim_base_total.loc[sim_base_total.index[life_expectancy - clients_age]] < 0).sum() / (
            trials + 1)
    port_legacy_risk = (sim_port_total.loc[sim_port_total.index[life_expectancy - clients_age]] < 0).sum() / (
            trials + 1)

    legacy_risk = pd.DataFrame([base_legacy_risk, port_legacy_risk,
                                'Prob. of portfolio value less than 0 at the end of the expected life'],
                               index=['base', 'fia_portfolio', 'Notes'],
                               columns=['Ruin Probability'])

    # -----------Year-wise probability of ending value greater than 0 -----------------
    base_psuccess = sim_base_total.apply(lambda x: x > 0).sum(axis=1) / (trials + 1)
    port_psuccess = sim_port_total.apply(lambda x: x > 0).sum(axis=1) / (trials + 1)

    # -----------------------WRITING FILES TO EXCEL ---------------------------
    col_names = ['50th', 'age', 'comment']
    writer = pd.ExcelWriter(src + method + '_historical_simulated_income_summary.xlsx', engine='xlsxwriter')
    read_income_inputs.to_excel(writer, sheet_name='inputs_for_income')

    read_returns_est.to_excel(writer, sheet_name='asset_returns_estimates')

    age_index = list(range(clients_age + 1, clients_age + len(base_qcut) + 1))
    base_qcut.loc[:, 'age'] = age_index
    base_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    base_qcut.loc[income_starts:, col_names].to_excel(writer, sheet_name='base_ending_value_quantiles')

    base_income_qcut = base_income_qcut.loc[1:, :]
    base_income_qcut.loc[:, 'age'] = age_index
    base_income_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    base_income_qcut.loc[income_starts:, col_names].to_excel(writer, sheet_name='base_income_quantiles')

    port_qcut.loc[:, 'age'] = age_index
    port_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    port_qcut.loc[income_starts:, col_names].to_excel(writer, sheet_name='fia_port_ending_value_quantiles')

    port_income_qcut = port_income_qcut.loc[1:, :]
    port_income_qcut.loc[:, 'age'] = age_index
    port_income_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    port_income_qcut.loc[income_starts:, col_names].to_excel(writer, sheet_name='fia_port_income_quantiles')

    # prob_success_df = pd.concat([base_psuccess, port_psuccess], axis=1)
    # prob_success_df.rename(columns={prob_success_df.columns[0]: 'prob(ending_value>0)_base',
    #                                 prob_success_df.columns[1]: 'prob(ending_value>0)_port'}, inplace=True)

    # prob_success_df.loc[:, 'age'] = age_index
    # prob_success_df.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    # prob_success_df.to_excel(writer, sheet_name='success_probability')

    income_breakdown_base = income_breakdown_base.loc[1:, :]
    income_breakdown_base.loc[:, 'age'] = age_index
    income_breakdown_base.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    income_breakdown_base.loc[income_starts:, :].to_excel(writer, sheet_name='base_income_breakdown_median')

    income_breakdown_port = income_breakdown_port.loc[1:, :]
    income_breakdown_port.loc[:, 'age'] = age_index
    income_breakdown_port.loc[:, 'ending contract value'] = income_df.loc[:, 'contract_value'].fillna(0)
    income_breakdown_port.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'

    income_breakdown_port.loc[income_starts:, :].to_excel(writer, sheet_name='fia_income_breakdown_median')

    if method == 'normal':
        # median_returns_normal.loc[:, 'fia_median_returns'] = median_normal_fia
        # median_returns_normal.to_excel(writer, sheet_name='unsorted_port_return')
        save_normal.to_excel(writer, sheet_name='port_return_unsorted')

    elif method == 'smallest':
        # median_returns_smallest.loc[:, 'fia_median_returns'] = median_smallest_fia
        # median_returns_smallest.to_excel(writer, sheet_name='port_return_asc')
        save_asc.to_excel(writer, sheet_name='port_return_asc')

    else:
        # median_returns_largest.loc[:, 'fia_median_returns'] = median_largest_fia
        # median_returns_largest.to_excel(writer, sheet_name='port_return_desc')
        save_desc.to_excel(writer, sheet_name='port_return_desc')

    # terminal_val = pd.read_csv(src + 'terminal_values.csv', index_col=[0])
    # ending_val = pd.read_csv(src + 'ending_values.csv', index_col=[0])
    # ending_val_ror = pd.read_csv(src + 'ending_values_ror.csv', index_col=[0])

    # terminal_val.to_excel(writer, sheet_name='terminal_values')
    # ending_val.to_excel(writer, sheet_name='port_ending_values')
    # ending_val_ror.to_excel(writer, sheet_name='port_annual_growth')

    writer.save()

    # -----------------Plotting charts--------------------------------------------
    # base_qcut.loc[income_starts:].plot(grid=True, title='Quantile Terminal Value - Base Portfolio')
    # plt.savefig(src + "quantile_terminal_base.png")
    # plt.close('all')
    #
    # base_income_qcut.plot(grid=True, title='Quantile Income - Base Portfolio')
    # plt.savefig(src + "quantile_income_base.png")
    # plt.close('all')
    #
    # base_psuccess.plot(grid=True, title='Probability of Success (Portfolio Ending Value > 0) - Base Portfolio')
    # plt.savefig(src + "success_probabilty_base.png")
    # plt.close('all')
    #
    # (1 - base_psuccess).plot(grid=True, title='Probability of Ruin (Portfolio Ending Value < 0) - Base Portfolio')
    # plt.savefig(src + "ruin_probability_base.png")
    # plt.close('all')
    #
    # port_qcut.loc[income_starts:].plot(grid=True, title='Quantile Terminal Value - FIA Portfolio')
    # plt.savefig(src + "quantile_terminal_fia.png")
    # plt.close('all')
    #
    # port_income_qcut.plot(grid=True, title='Quantile Income - FIA Portfolio')
    # plt.savefig(src + "quantile_income_fia.png")
    # plt.close('all')
    #
    # port_psuccess.plot(grid=True, title='Probability of Success (Portfolio Ending Value > 0) - FIA Portfolio')
    # plt.savefig(src + "success_probabilty_fia.png")
    # plt.close('all')
    #
    # (1 - port_psuccess).plot(grid=True, title='Probability of Ruin (Portfolio Ending Value < 0) - FIA Portfolio')
    # plt.savefig(src + "ruin_probability_fia.png")
    # plt.close('all')

    print("simulation completed for {}".format(method))


def income_model_user_defined_returns(num_of_years=30, trials=100, method='normal'):
    """Simulation based on the expected annual return provided by the user for S&P and FIA and portfolio assets returns
    are calculated using the regression beta and alpha to S&P 500. The calculated growth rate is assumed constant for
    the full analysis years. Perpetual Returns"""

    print("Running method income_model_user_defined_returns()")

    sim_fia_cv = pd.DataFrame(index=range(num_of_years + 1))

    sim_base_total = pd.DataFrame(index=range(num_of_years + 1))
    sim_base_income = pd.DataFrame(index=range(num_of_years + 1))

    sim_port_total = pd.DataFrame(index=range(num_of_years + 1))
    sim_port_income = pd.DataFrame(index=range(num_of_years + 1))

    read_income_inputs = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_model_inputs',
                                       index_col=[0])

    read_returns_est = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_assets_returns_estimates',
                                     index_col=[0])
    clean_names = list(read_returns_est.index)
    clean_names = [s.split(' ')[0] for s in clean_names]
    read_returns_est.loc[:, 'names'] = clean_names
    read_returns_est.set_index('names', drop=True, inplace=True)
    read_returns_est = read_returns_est[:-1]
    read_returns_est.rename(index={'SBMMTB3': 'Cash', read_returns_est.index[-1]: 'FIA'}, inplace=True)

    # ---------------Returns DataFrame based on the user input------------------------------------
    ann_ret = np.full((num_of_years + 1, len(read_returns_est)), read_returns_est.loc[:, 'Annualized Returns'])
    read_normal = pd.DataFrame(ann_ret, index=np.arange(num_of_years + 1), columns=read_returns_est.index)
    # read_normal.rename(columns={read_normal.columns[-1]: 'FIA'}, inplace=True)
    user_est_fia_return = float(read_income_inputs.loc['fia_forecast', 'inputs'])
    read_normal.loc[:, 'FIA'] = user_est_fia_return

    read_asset_weights = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='asset_weights',
                                       index_col=[0])

    # read_asset_weights.drop(read_asset_weights.index[-1], axis=0, inplace=True)

    # read random returns for simulation
    read_normal = pd.read_csv(src + 'median_returns_unsorted.csv', index_col=[0], parse_dates=True)
    cols = [read_normal.columns[c].split('_')[1] for c in np.arange(len(read_normal.columns))]
    read_normal.rename(columns=dict(zip(list(read_normal.columns), cols)), inplace=True)

    # ann_ret = np.full((num_of_years+1, len(read_returns_est)), read_returns_est.loc[:, 'Annualized Returns'])
    # read_normal = pd.DataFrame(ann_ret, index=np.arange(num_of_years+1), columns=read_returns_est.index)
    # # read_normal.rename(columns={read_normal.columns[-1]: 'FIA'}, inplace=True)
    # user_est_fia_return = float(read_income_inputs.loc['fia_forecast', 'inputs'])
    # read_normal.loc[:, 'FIA'] = user_est_fia_return

    assets_col_names = list(read_normal.columns)

    tickers = list(read_asset_weights.index)
    wts = np.array(read_asset_weights.loc[:, 'base'])

    def asset_median_returns(data, ticker):
        return data.filter(regex=ticker).median(axis=1)

    # dataframe for unsorted returns (normal)
    median_returns_normal = read_normal.copy()
    median_returns_normal.loc[:, 'portfolio_return'] = median_returns_normal.dot(wts)
    median_normal_fia = pd.DataFrame({'FIA': asset_median_returns(read_normal, 'FIA')})

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

    # ---------------INCOME MODEL--------------------------------------------
    runs = 0
    returns_dict = {}
    asset_dict = {}
    fia_dict = {}

    income_df = pd.DataFrame(index=years, columns=income_cols)
    income_df.loc[:, 'year'] = years
    income_df.loc[:, 'strategy_term'] = income_df.loc[:, 'year'] % term
    income_df.loc[:, 'strategy_term'] = income_df['strategy_term'].apply(lambda x: 1 if x == 0 else 0)
    income_df.loc[:, 'index_returns'] = read_normal.loc[:, 'FIA']

    # income_df.loc[:, 'index_returns'] = np.random.normal(fia_ret, fia_risk, size=(len(years), 1))
    cumprod = (1. + income_df['index_returns']).rolling(window=term).agg(lambda x: x.prod()) - 1
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

    income_df.loc[:, 'rider_fee'] = income_df.loc[:, 'high_inc_benefit_base'] * rider_fee
    income_df.loc[:, 'eoy_income'] = np.where(income_df.loc[:, 'year'] > income_starts,
                                              income_df.loc[:, 'high_inc_benefit_base'] * inc_payout_factor, 0)

    for counter in years:
        if counter == 0:
            income_df.loc[counter, 'contract_value'] = premium * (1 + contract_bonus)

        elif income_df.loc[counter, 'strategy_term'] == 1:
            x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee']
            x2 = (x1 * (1 + income_df.loc[counter, 'term_ret_netspr'])) - income_df.loc[counter, 'eoy_income']
            income_df.loc[counter, 'contract_value'] = x2

        else:
            x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee'] - \
                 income_df.loc[counter, 'eoy_income']

            income_df.loc[counter, 'contract_value'] = x1

    # variable stores the income number that is used in the base and fia portfolio calcs.

    income_from_fia = income_df.loc[income_df.index[-1], 'eoy_income']

    income_df.loc[:, 'contract_value'] = income_df.loc[:, 'contract_value'].apply(lambda x: 0 if x <= 0 else x)

    sim_fia_cv.loc[:, str(runs)] = income_df.loc[:, 'contract_value']

    # --------------------BASE MODEL---------------------------------------------
    base_wts = read_asset_weights.loc[:, 'base']
    base_assets = list(base_wts.index)
    base_weights = list(base_wts.values)
    base_returns = list(read_returns_est.loc[:, 'Annualized Returns'].values)
    base_std = list(read_returns_est.loc[:, 'Annualized Risk'].values)

    base_investment = float(read_income_inputs.loc['risky_assets', 'Base'])
    adv_fees = float(read_income_inputs.loc['advisor_fees', 'Base'])

    # -------------------required income----------------------------------
    req_annual_income = float(read_income_inputs.loc['annual_income', 'inputs'])
    income_needed = req_annual_income - social
    income_net_fia_income = max(0, income_needed - income_from_fia)
    cpn_income_base = base_investment * wtd_cpn_yield

    # ----------------------RANDOM RETURNS--------------------------
    r_cols = base_assets
    boy_value = ['bv_{}'.format(name) for name in base_assets]
    eoy_value = ['ev_{}'.format(name) for name in base_assets]

    random_returns = pd.DataFrame(index=income_df.index, columns=r_cols)

    for c in range(len(r_cols)):
        ret = np.random.normal(base_returns[c], base_std[c], size=(len(random_returns.index), 1))

    random_returns = read_normal.copy()

    base_df = random_returns.copy()
    fia_portfolio_df = random_returns.copy()
    port_investment = float(read_income_inputs.loc['risky_assets', 'FIA'])
    cpn_income_port = port_investment * wtd_cpn_yield

    # -------------BASE PORTFOLIO----------------------------
    for name in boy_value:
        base_df.loc[:, name] = 0.0

    for counter in years:
        period_returns = list(random_returns.loc[counter, :])
        if counter == 0:

            base_df.loc[counter, boy_value] = [base_weights[c] * base_investment for c in range(len(boy_value))]

            base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
            base_df.loc[counter, 'total_net_fees'] = 0.0
            base_df.loc[counter, 'income'] = 0.0
            base_investment = base_df.loc[counter, boy_value].sum()

        elif (counter > 0) and (counter < income_starts):

            base_df.loc[counter, boy_value] = [base_weights[c] * base_investment * (1 + period_returns[c])
                                               for c in range(len(boy_value))]
            base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
            base_df.loc[counter, 'adv_fees'] = base_df.loc[counter, 'total'] * adv_fees
            base_df.loc[counter, 'total_net_fees'] = base_df.loc[counter, 'total'] - base_df.loc[
                counter, 'adv_fees']

            # --coupon payment is invested back into the risky portfolio until the income is withdrawn----
            base_investment = base_df.loc[counter, 'total_net_fees'] + cpn_income_base

        else:

            base_df.loc[counter, boy_value] = [base_weights[c] * base_investment * (1 + period_returns[c])
                                               for c in range(len(boy_value))]
            base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
            base_df.loc[counter, 'adv_fees'] = base_df.loc[counter, 'total'] * adv_fees

            # ---req. income is adjusted for inflation from the second year of withdrawal. Reinvestment of coupon
            # stops from the year income starts. Req. income is reduced by the coupon payments

            if counter == income_starts:

                income_needed = req_annual_income - social
                base_df.loc[counter, 'income'] = income_needed - cpn_income_base
                income_needed = req_annual_income

            else:
                income_needed = income_needed * (1 + inflation) - social
                base_df.loc[counter, 'income'] = income_needed - cpn_income_base
                income_needed = income_needed + social

            base_df.loc[counter, 'total_net_fees'] = base_df.loc[counter, 'total'] - \
                                                     base_df.loc[counter, 'adv_fees'] - \
                                                     base_df.loc[counter, 'income']

            base_investment = base_df.loc[counter, 'total_net_fees']

    base_df.loc[:, 'adj_total'] = base_df.loc[:, 'total_net_fees'].apply(lambda x: x if x > 0 else 0)
    sim_base_total.loc[:, 's_{}'.format(str(runs))] = base_df.loc[:, 'total_net_fees']
    sim_base_income.loc[:, 's_{}'.format(str(runs))] = base_df.loc[:, 'income']

    # ----------------------------FIA PORTFOLIO----------------------------------------------
    for name in boy_value:
        fia_portfolio_df.loc[:, name] = 0.0

    for counter in years:
        period_returns = list(random_returns.loc[counter, :])
        if counter == 0:

            fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment
                                                        for c in range(len(boy_value))]
            fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
            fia_portfolio_df.loc[counter, 'total_net_fees'] = 0.0
            fia_portfolio_df.loc[counter, 'income'] = 0.0
            port_investment = fia_portfolio_df.loc[counter, boy_value].sum()

        elif (counter > 0) and (counter < income_starts):

            fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment * (1 + period_returns[c])
                                                        for c in range(len(boy_value))]
            fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
            fia_portfolio_df.loc[counter, 'adv_fees'] = fia_portfolio_df.loc[counter, 'total'] * adv_fees
            fia_portfolio_df.loc[counter, 'total_net_fees'] = fia_portfolio_df.loc[counter, 'total'] - \
                                                              fia_portfolio_df.loc[counter, 'adv_fees']

            port_investment = fia_portfolio_df.loc[counter, 'total_net_fees'] + cpn_income_port

        else:
            fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment * (1 + period_returns[c])
                                                        for c in range(len(boy_value))]
            fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
            fia_portfolio_df.loc[counter, 'adv_fees'] = fia_portfolio_df.loc[counter, 'total'] * adv_fees

            # ---req. income is adjusted for inflation from the second year of withdrawal. Reinvestment of coupon
            # stops from the year income starts. Req. income is reduced by the coupon payments

            if counter == income_starts:

                income_needed = req_annual_income - social
                income_net_fia_income = max(0, income_needed - income_from_fia)
                fia_portfolio_df.loc[counter, 'income'] = max(0, income_net_fia_income - cpn_income_port)
                income_needed = req_annual_income

            else:
                income_needed = income_needed * (1 + inflation) - social
                income_net_fia_income = max(0, income_needed - income_from_fia)
                fia_portfolio_df.loc[counter, 'income'] = max(0, income_net_fia_income - cpn_income_port)
                income_needed = income_needed + social

            fia_portfolio_df.loc[counter, 'total_net_fees'] = fia_portfolio_df.loc[counter, 'total'] - \
                                                              fia_portfolio_df.loc[counter, 'adv_fees'] - \
                                                              fia_portfolio_df.loc[counter, 'income']

            port_investment = fia_portfolio_df.loc[counter, 'total_net_fees']

    sim_port_total.loc[:, 's_{}'.format(str(runs))] = fia_portfolio_df.loc[:, 'total_net_fees'] + \
                                                      income_df.loc[:, 'contract_value']

    sim_port_income.loc[:, 's_{}'.format(str(runs))] = fia_portfolio_df.loc[:, 'income']

    fia_portfolio_df.loc[:, 'adj_total'] = fia_portfolio_df.loc[:, 'total_net_fees'].apply(
        lambda x: x if x > 0 else 0)

    # ---------income breakdown for Base portfolio----------------------------------
    base_df.to_csv(src + 'base_port_detail.csv')
    sim_base_total.to_csv(src + 'base_ending_values.csv')
    income_breakdown_base = pd.DataFrame(sim_base_total.quantile(0.5, axis=1))
    income_breakdown_base.loc[:, 'income_from_portfolio'] = sim_base_income.quantile(0.5, axis=1)
    income_breakdown_base.loc[:, 'fia_income'] = 0.0
    income_breakdown_base.loc[:, 'social_security_income'] = social
    income_breakdown_base.loc[:, 'coupon_income'] = cpn_income_base

    income_breakdown_base.rename(columns={income_breakdown_base.columns[0]: 'portfolio_ending_value'}, inplace=True)
    income_breakdown_base.loc[:, 'income_from_portfolio'][
        income_breakdown_base.loc[:, 'portfolio_ending_value'] <= 0] = 0
    income_breakdown_base.loc[:, 'total_income'] = income_breakdown_base.loc[:, income_breakdown_base.columns[1:]].sum(
        axis=1)

    # ------------Block Ends-------------------------------------------------------------

    # ---------income breakdown for FIA portfolio----------------------------------
    fia_portfolio_df.to_csv(src + 'fia_port_detail.csv')
    sim_port_total.to_csv(src + 'fiaport_ending_values.csv')

    income_breakdown_port = pd.DataFrame(sim_port_total.quantile(0.5, axis=1))
    income_breakdown_port.loc[:, 'income_from_portfolio'] = sim_port_income.quantile(0.5, axis=1)
    income_breakdown_port.loc[:, 'fia_income'] = income_from_fia
    income_breakdown_port.loc[:, 'social_security_income'] = social
    income_breakdown_port.loc[:, 'coupon_income'] = cpn_income_port

    income_breakdown_port.rename(columns={income_breakdown_port.columns[0]: 'portfolio_ending_value'}, inplace=True)
    income_breakdown_port.loc[:, 'income_from_portfolio'][
        income_breakdown_port.loc[:, 'portfolio_ending_value'] <= 0] = 0
    income_breakdown_port.loc[:, 'total_income'] = income_breakdown_port.loc[:, income_breakdown_port.columns[1:]].sum(
        axis=1)

    # ------------Block Ends-------------------------------------------------------------
    q_cut = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    sim_base_income[sim_base_total < income_needed] = 0.0

    sim_port_income[sim_port_total < income_net_fia_income] = 0

    sim_port_income = sim_port_income + income_from_fia

    # base_quantile = sim_base_total.loc[sim_base_total.index[-1]].quantile([0.05, 0.25, 0.50, 0.75, 0.90])

    # port_quantile = sim_port_total.loc[sim_port_total.index[-1]].quantile([0.05, 0.25, 0.50, 0.75, 0.90])

    base_quantile = sim_base_total.loc[sim_base_total.index[-1]].quantile(q_cut)

    port_quantile = sim_port_total.loc[sim_port_total.index[-1]].quantile(q_cut)

    # q_cut = [0.0, .05, 0.25, 0.5, 0.75, 0.95, 1.0]
    cols = ['Min', '5th', '25th', '50th', '75th', '90th', 'Max']

    # ----drop year 0--------
    sim_base_total = sim_base_total[1:]
    sim_port_total = sim_port_total[1:]

    # ----------------quantile analysis for base terminal value--------------------------
    base_qcut = pd.DataFrame(index=sim_base_total.index, columns=cols)
    for c in range(len(cols)):
        base_qcut.loc[:, cols[c]] = sim_base_total.quantile(q_cut[c], axis=1)

    base_qcut.clip(lower=0, inplace=True)

    # ----------------------quantile analysis for base income----------------------------
    base_income_qcut = pd.DataFrame(index=sim_base_income.index, columns=cols)
    for c in range(len(cols)):
        base_income_qcut.loc[:, cols[c]] = sim_base_income.quantile(q_cut[c], axis=1)

    # ----Remove NaN's prior to the income start years------------
    # base_income_qcut = base_income_qcut.loc[income_starts:]

    # -------------quantile analysis for portfolio terminal value ----------------
    port_qcut = pd.DataFrame(index=sim_port_total.index, columns=cols)
    for c in range(len(cols)):
        port_qcut.loc[:, cols[c]] = sim_port_total.quantile(q_cut[c], axis=1)

    port_qcut.clip(lower=0, inplace=True)

    # ---------------quantile analysis for portfolio income----------------------------
    port_income_qcut = pd.DataFrame(index=sim_port_income.index, columns=cols)
    for c in range(len(cols)):
        port_income_qcut.loc[:, cols[c]] = sim_port_income.quantile(q_cut[c], axis=1)

    # ----Remove NaN's prior to the income start years------------
    # port_income_qcut = port_income_qcut.loc[income_starts:]

    # ----------probability ending value will be less than 0 at the end of the horizon -----------------------
    base_legacy_risk = (sim_base_total.loc[sim_base_total.index[life_expectancy - clients_age]] < 0).sum() / (
            trials + 1)
    port_legacy_risk = (sim_port_total.loc[sim_port_total.index[life_expectancy - clients_age]] < 0).sum() / (
            trials + 1)

    legacy_risk = pd.DataFrame([base_legacy_risk, port_legacy_risk,
                                'Prob. of portfolio value less than 0 at the end of the expected life'],
                               index=['base', 'fia_portfolio', 'Notes'],
                               columns=['Ruin Probability'])

    # -----------Year-wise probability of ending value greater than 0 -----------------
    base_psuccess = sim_base_total.apply(lambda x: x > 0).sum(axis=1) / (trials + 1)
    port_psuccess = sim_port_total.apply(lambda x: x > 0).sum(axis=1) / (trials + 1)

    # -----------------------WRITING FILES TO EXCEL ---------------------------
    col_names = ['50th', 'age', 'comment']
    writer = pd.ExcelWriter(src + method + '_simulated_income_summary_custom.xlsx', engine='xlsxwriter')
    read_income_inputs.to_excel(writer, sheet_name='inputs_for_income')

    read_returns_est.to_excel(writer, sheet_name='asset_returns_estimates')

    age_index = list(range(clients_age + 1, clients_age + len(base_qcut) + 1))
    base_qcut.loc[:, 'age'] = age_index
    base_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    base_qcut.loc[income_starts:, col_names].to_excel(writer, sheet_name='base_ending_value_quantiles')

    base_income_qcut = base_income_qcut.loc[1:, :]
    base_income_qcut.loc[:, 'age'] = age_index
    base_income_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    base_income_qcut.loc[income_starts:, col_names].to_excel(writer, sheet_name='base_income_quantiles')

    port_qcut.loc[:, 'age'] = age_index
    port_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    port_qcut.loc[income_starts:, col_names].to_excel(writer, sheet_name='fia_port_ending_value_quantiles')

    port_income_qcut = port_income_qcut.loc[1:, :]
    port_income_qcut.loc[:, 'age'] = age_index
    port_income_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    port_income_qcut.loc[income_starts:, col_names].to_excel(writer, sheet_name='fia_port_income_quantiles')

    income_breakdown_base = income_breakdown_base.loc[1:, :]
    income_breakdown_base.loc[:, 'age'] = age_index
    income_breakdown_base.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    income_breakdown_base.loc[income_starts:, :].to_excel(writer, sheet_name='base_income_breakdown_median')

    income_breakdown_port = income_breakdown_port.loc[1:, :]
    income_breakdown_port.loc[:, 'age'] = age_index
    income_breakdown_port.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    income_breakdown_port.loc[:, 'ending_contract_value'] = income_df.loc[:, 'contract_value']
    income_breakdown_port.loc[income_starts:, :].to_excel(writer, sheet_name='fia_income_breakdown_median')

    median_returns_normal.to_excel(writer, sheet_name='gr_port_median_normal')

    terminal_val = pd.read_csv(src + 'terminal_values.csv', index_col=[0])
    ending_val = pd.read_csv(src + 'ending_values.csv', index_col=[0])
    ending_val_ror = pd.read_csv(src + 'ending_values_ror.csv', index_col=[0])

    terminal_val.to_excel(writer, sheet_name='terminal_values')
    ending_val.to_excel(writer, sheet_name='port_ending_values')
    ending_val_ror.to_excel(writer, sheet_name='port_annual_growth')

    writer.save()

    # -----------------Plotting charts--------------------------------------------
    base_qcut.loc[income_starts:].plot(grid=True, title='Quantile Terminal Value - Base Portfolio')
    plt.savefig(src + "quantile_terminal_base.png")
    plt.close('all')

    base_income_qcut.plot(grid=True, title='Quantile Income - Base Portfolio')
    plt.savefig(src + "quantile_income_base.png")
    plt.close('all')

    base_psuccess.plot(grid=True, title='Probability of Success (Portfolio Ending Value > 0) - Base Portfolio')
    plt.savefig(src + "success_probabilty_base.png")
    plt.close('all')

    (1 - base_psuccess).plot(grid=True, title='Probability of Ruin (Portfolio Ending Value < 0) - Base Portfolio')
    plt.savefig(src + "ruin_probability_base.png")
    plt.close('all')

    port_qcut.loc[income_starts:].plot(grid=True, title='Quantile Terminal Value - FIA Portfolio')
    plt.savefig(src + "quantile_terminal_fia.png")
    plt.close('all')

    port_income_qcut.plot(grid=True, title='Quantile Income - FIA Portfolio')
    plt.savefig(src + "quantile_income_fia.png")
    plt.close('all')

    port_psuccess.plot(grid=True, title='Probability of Success (Portfolio Ending Value > 0) - FIA Portfolio')
    plt.savefig(src + "success_probabilty_fia.png")
    plt.close('all')

    (1 - port_psuccess).plot(grid=True, title='Probability of Ruin (Portfolio Ending Value < 0) - FIA Portfolio')
    plt.savefig(src + "ruin_probability_fia.png")
    plt.close('all')

    print("simulation completed for {}".format(method))


def income_model_asset_based_portfolio_custom(num_of_years=30, trials=100, method='normal', income=True):
    """Simulation based on the expected annual return provided by the user for S&P and FIA and portfolio assets returns
    are calculated using the regression beta and alpha to S&P 500. The calculated growth rate is assumed constant for
    the full analysis years. Perpetual Returns"""

    sim_fia_cv = pd.DataFrame(index=range(num_of_years + 1))

    sim_base_total = pd.DataFrame(index=range(num_of_years + 1))
    sim_base_income = pd.DataFrame(index=range(num_of_years + 1))

    sim_port_total = pd.DataFrame(index=range(num_of_years + 1))
    sim_port_income = pd.DataFrame(index=range(num_of_years + 1))

    sim_base_total_pre_income = pd.DataFrame(index=range(num_of_years + 1))
    sim_port_total_pre_income = pd.DataFrame(index=range(num_of_years + 1))

    # read_income_inputs = pd.read_csv(src + "income_model_inputs.csv", index_col='Items')
    read_income_inputs = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_model_inputs',
                                       index_col=[0])

    # read_returns_est = pd.read_csv(src + "income_assets_returns_estimates.csv", index_col='Symbol')
    read_returns_est = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_assets_returns_estimates',
                                     index_col=[0])

    clean_names = list(read_returns_est.index)
    clean_names = [s.split(' ')[0] for s in clean_names]
    read_returns_est.loc[:, 'names'] = clean_names
    read_returns_est.set_index('names', drop=True, inplace=True)
    read_returns_est = read_returns_est[:-1]
    read_returns_est.rename(index={'SBMMTB3': 'Cash', read_returns_est.index[-1]: 'FIA'}, inplace=True)

    # ---------------Returns DataFrame based on the use input------------------------------------
    ann_ret = np.full((num_of_years + 1, len(read_returns_est)), read_returns_est.loc[:, 'Annualized Returns'])
    read_normal = pd.DataFrame(ann_ret, index=np.arange(num_of_years + 1), columns=read_returns_est.index)
    # read_normal.rename(columns={read_normal.columns[-1]: 'FIA'}, inplace=True)
    user_est_fia_return = float(read_income_inputs.loc['fia_forecast', 'inputs'])
    read_normal.loc[:, 'FIA'] = user_est_fia_return

    read_returns_est.loc['FIA', 'Annualized Returns'] = user_est_fia_return

    # read_returns_est.drop(['BM', read_returns_est.index[-1]], axis=0, inplace=True)
    # read_portfolio_inputs = pd.read_csv(src + "income_portfolio_inputs.csv", index_col='Items')

    # read_asset_weights = pd.read_csv(src + "asset_weights.csv", index_col='Asset')
    read_asset_weights = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='asset_weights',
                                       index_col=[0])

    read_asset_weights.drop(read_asset_weights.index[-1], axis=0, inplace=True)

    # read random returns for simulation
    # read_normal = pd.read_csv(src + 'sort_normal.csv', index_col=[0], parse_dates=True)
    # read_small = pd.read_csv(src + 'sort_small_to_large.csv', index_col=[0], parse_dates=True)
    # read_large = pd.read_csv(src + 'sort_large_to_small.csv', index_col=[0], parse_dates=True)
    assets_col_names = list(read_normal.columns)

    tickers = list(read_asset_weights.index)
    wts = np.array(read_asset_weights.loc[:, 'base'])

    def asset_median_returns(data, ticker):
        return data.filter(regex=ticker).median(axis=1)

    # dataframe for unsorted returns (normal)
    median_returns_normal = pd.DataFrame({t: asset_median_returns(read_normal, t) for t in tickers})
    median_returns_normal.loc[:, 'portfolio_return'] = median_returns_normal.dot(wts)
    median_normal_fia = pd.DataFrame({'FIA': asset_median_returns(read_normal, 'FIA')})
    #
    # # dataframe for smallest to largest returns
    # median_returns_smallest = pd.DataFrame({t: asset_median_returns(read_small, t) for t in tickers})
    # median_returns_smallest.loc[:, 'portfolio_return'] = median_returns_smallest.dot(wts)
    # median_smallest_fia = pd.DataFrame({'FIA': asset_median_returns(read_small, 'r_FIA')})
    #
    # # dataframe for unsorted returns (normal)
    # median_returns_largest = pd.DataFrame({t: asset_median_returns(read_large, t) for t in tickers})
    # median_returns_largest.loc[:, 'portfolio_return'] = median_returns_largest.dot(wts)
    # median_largest_fia = pd.DataFrame({'FIA': asset_median_returns(read_large, 'r_FIA')})

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

    # ---------------INCOME MODEL--------------------------------------------
    runs = 0
    returns_dict = {}
    asset_dict = {}
    fia_dict = {}
    while runs < trials:
        print(runs)

        income_df = pd.DataFrame(index=years, columns=income_cols)
        income_df.loc[:, 'year'] = years
        income_df.loc[:, 'strategy_term'] = income_df.loc[:, 'year'] % term
        income_df.loc[:, 'strategy_term'] = income_df['strategy_term'].apply(lambda x: 1 if x == 0 else 0)

        income_df.loc[:, 'index_returns'] = read_normal.loc[:, 'FIA']
        # income_df.loc[:, 'index_returns'] = np.random.normal(fia_ret, fia_risk, size=(len(years), 1))

        cumprod = (1. + income_df['index_returns']).rolling(window=term).agg(lambda x: x.prod()) - 1
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

        income_df.loc[:, 'rider_fee'] = income_df.loc[:, 'high_inc_benefit_base'] * rider_fee
        income_df.loc[:, 'eoy_income'] = np.where(income_df.loc[:, 'year'] > income_starts,
                                                  income_df.loc[:, 'high_inc_benefit_base'] * inc_payout_factor, 0)

        for counter in years:
            if counter == 0:
                income_df.loc[counter, 'contract_value'] = premium * (1 + contract_bonus)

            elif income_df.loc[counter, 'strategy_term'] == 1:
                x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee']
                x2 = (x1 * (1 + income_df.loc[counter, 'term_ret_netspr'])) - income_df.loc[counter, 'eoy_income']
                income_df.loc[counter, 'contract_value'] = x2

            else:
                x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee'] - \
                     income_df.loc[counter, 'eoy_income']

                income_df.loc[counter, 'contract_value'] = x1

        # variable stores the income number that is used in the base and fia portfolio calcs.

        income_from_fia = income_df.loc[income_df.index[-1], 'eoy_income']

        income_df.loc[:, 'contract_value'] = income_df.loc[:, 'contract_value'].apply(lambda x: 0 if x <= 0 else x)

        sim_fia_cv.loc[:, str(runs)] = income_df.loc[:, 'contract_value']

        # -------------------------------------BASE MODEL---------------------------------------------

        base_wts = read_asset_weights.loc[:, 'base']
        base_assets = list(base_wts.index)
        base_weights = list(base_wts.values)
        base_returns = list(read_returns_est.loc[:, 'Annualized Returns'].values)
        base_std = list(read_returns_est.loc[:, 'Annualized Risk'].values)

        base_investment = float(read_income_inputs.loc['risky_assets', 'Base'])
        adv_fees = float(read_income_inputs.loc['advisor_fees', 'Base'])

        # -------------------required income----------------------------------
        if income:
            req_annual_income = float(read_income_inputs.loc['annual_income', 'inputs'])
            income_needed = req_annual_income - social
            income_net_fia_income = max(0, income_needed - income_from_fia)
            cpn_income_base = base_investment * wtd_cpn_yield
        else:
            req_annual_income = 0.0
            income_needed = 0.0
            income_net_fia_income = 0.0
            cpn_income_base = base_investment * wtd_cpn_yield

        # ----------------------RANDOM RETURNS--------------------------
        r_cols = ['r_{}'.format(name) for name in base_assets]
        boy_value = ['bv_{}'.format(name) for name in base_assets]
        eoy_value = ['ev_{}'.format(name) for name in base_assets]

        random_returns = pd.DataFrame(index=income_df.index, columns=r_cols)

        # for c in range(len(r_cols)):
        #     ret = np.random.normal(base_returns[c], base_std[c], size=(len(random_returns.index), 1))

        # this_run_cols = ['{}_{}'.format(cname, str(runs)) for cname in r_cols]
        random_returns = read_normal.loc[:, base_assets]

        # random_returns.loc[:, r_cols[c]] = ret.flatten()
        # asset_dict.update({'{}_{}'.format(r_cols[c], str(runs)): ret.flatten()})

        # store the simulated assets returns in one dictionary
        # returns_dict.update({str(runs): random_returns})

        # collect the asset based returns from all simulation and calculate the median returns.
        # def get_median_returns(sym):
        #     cols = [sym + '_' + str(c) for c in np.arange(trials)]
        #     asset_df = pd.DataFrame({c: asset_dict.get(c) for c in cols})
        #     return asset_df.median(axis=1)
        #
        # asset_median_returns = pd.DataFrame({symbol: get_median_returns(symbol) for symbol in r_cols})
        #
        # asset_median_returns.loc[:, 'simulated_portfolio_median_returns'] = asset_median_returns.dot(base_weights)

        base_df = random_returns.copy()

        # base_investment = float(read_portfolio_inputs.loc['risky_assets', 'Base'])

        fia_portfolio_df = random_returns.copy()
        port_investment = float(read_income_inputs.loc['risky_assets', 'FIA'])
        cpn_income_port = port_investment * wtd_cpn_yield

        # ----------------------------------------BASE PORTFOLIO----------------------------
        for name in boy_value:
            base_df.loc[:, name] = 0.0

        for counter in years:
            period_returns = list(random_returns.loc[counter, :])
            if counter == 0:

                base_df.loc[counter, boy_value] = [base_weights[c] * base_investment for c in range(len(boy_value))]

                base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
                base_df.loc[counter, 'total_net_fees'] = 0.0
                base_df.loc[counter, 'income'] = 0.0
                base_investment = base_df.loc[counter, boy_value].sum()

            elif (counter > 0) and (counter < income_starts):

                base_df.loc[counter, boy_value] = [base_weights[c] * base_investment * (1 + period_returns[c])
                                                   for c in range(len(boy_value))]
                base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
                base_df.loc[counter, 'adv_fees'] = base_df.loc[counter, 'total'] * adv_fees
                base_df.loc[counter, 'total_net_fees'] = base_df.loc[counter, 'total'] - base_df.loc[
                    counter, 'adv_fees']

                # --coupon payment is invested back into the risky portfolio until the income is withdrawn----
                base_investment = base_df.loc[counter, 'total_net_fees'] + cpn_income_base

            else:

                base_df.loc[counter, boy_value] = [base_weights[c] * base_investment * (1 + period_returns[c])
                                                   for c in range(len(boy_value))]
                base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
                base_df.loc[counter, 'adv_fees'] = base_df.loc[counter, 'total'] * adv_fees

                # ---req. income is adjusted for inflation from the second year of withdrawal. Reinvestment of coupon
                # stops from the year income starts. Req. income is reduced by the coupon payments

                if counter == income_starts:

                    income_needed = req_annual_income - social
                    base_df.loc[counter, 'income'] = income_needed - cpn_income_base
                    income_needed = req_annual_income

                else:
                    income_needed = income_needed * (1 + inflation) - social
                    base_df.loc[counter, 'income'] = income_needed - cpn_income_base
                    income_needed = income_needed + social

                base_df.loc[counter, 'total_net_fees'] = base_df.loc[counter, 'total'] - \
                                                         base_df.loc[counter, 'adv_fees'] - \
                                                         base_df.loc[counter, 'income']

                base_df.loc[counter, 'total_pre_income'] = base_df.loc[counter, 'total'] - \
                                                           base_df.loc[counter, 'adv_fees']

                base_investment = base_df.loc[counter, 'total_net_fees']

        base_df.loc[:, 'adj_total'] = base_df.loc[:, 'total_net_fees'].apply(lambda x: x if x > 0 else 0)
        sim_base_total.loc[:, 's_{}'.format(str(runs))] = base_df.loc[:, 'total_net_fees']
        sim_base_income.loc[:, 's_{}'.format(str(runs))] = base_df.loc[:, 'income']
        sim_base_total_pre_income.loc[:, 's_{}'.format(str(runs))] = base_df.loc[:, 'total_pre_income']

        # ----------------------------FIA PORTFOLIO----------------------------------------------
        for name in boy_value:
            fia_portfolio_df.loc[:, name] = 0.0

        for counter in years:
            period_returns = list(random_returns.loc[counter, :])
            if counter == 0:

                fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment
                                                            for c in range(len(boy_value))]
                fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
                fia_portfolio_df.loc[counter, 'total_net_fees'] = 0.0
                fia_portfolio_df.loc[counter, 'income'] = 0.0
                port_investment = fia_portfolio_df.loc[counter, boy_value].sum()

            elif (counter > 0) and (counter < income_starts):

                fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment * (1 + period_returns[c])
                                                            for c in range(len(boy_value))]
                fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
                fia_portfolio_df.loc[counter, 'adv_fees'] = fia_portfolio_df.loc[counter, 'total'] * adv_fees
                fia_portfolio_df.loc[counter, 'total_net_fees'] = fia_portfolio_df.loc[counter, 'total'] - \
                                                                  fia_portfolio_df.loc[counter, 'adv_fees']

                port_investment = fia_portfolio_df.loc[counter, 'total_net_fees'] + cpn_income_port

            else:
                fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment * (1 + period_returns[c])
                                                            for c in range(len(boy_value))]
                fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
                fia_portfolio_df.loc[counter, 'adv_fees'] = fia_portfolio_df.loc[counter, 'total'] * adv_fees

                # ---req. income is adjusted for inflation from the second year of withdrawal. Reinvestment of coupon
                # stops from the year income starts. Req. income is reduced by the coupon payments

                if counter == income_starts:

                    income_needed = req_annual_income - social
                    income_net_fia_income = max(0, income_needed - income_from_fia)
                    fia_portfolio_df.loc[counter, 'income'] = max(0, income_net_fia_income - cpn_income_port)
                    income_needed = req_annual_income

                else:
                    income_needed = income_needed * (1 + inflation) - social
                    income_net_fia_income = max(0, income_needed - income_from_fia)
                    fia_portfolio_df.loc[counter, 'income'] = max(0, income_net_fia_income - cpn_income_port)
                    income_needed = income_needed + social

                if income:
                    fia_portfolio_df.loc[counter, 'total_net_fees'] = fia_portfolio_df.loc[counter, 'total'] - \
                                                                      fia_portfolio_df.loc[counter, 'adv_fees'] - \
                                                                      fia_portfolio_df.loc[counter, 'income']
                else:
                    fia_portfolio_df.loc[counter, 'total_net_fees'] = fia_portfolio_df.loc[counter, 'total'] - \
                                                                      fia_portfolio_df.loc[counter, 'adv_fees'] + \
                                                                      income_from_fia

                fia_portfolio_df.loc[counter, 'total_pre_income'] = fia_portfolio_df.loc[counter, 'total'] - \
                                                                    fia_portfolio_df.loc[counter, 'adv_fees']

                port_investment = fia_portfolio_df.loc[counter, 'total_net_fees']

        sim_port_total.loc[:, 's_{}'.format(str(runs))] = fia_portfolio_df.loc[:, 'total_net_fees'] + \
                                                          income_df.loc[:, 'contract_value']

        sim_port_income.loc[:, 's_{}'.format(str(runs))] = fia_portfolio_df.loc[:, 'income']

        fia_portfolio_df.loc[:, 'adj_total'] = fia_portfolio_df.loc[:, 'total_net_fees'].apply(
            lambda x: x if x > 0 else 0)

        sim_port_total_pre_income.loc[:, 's_{}'.format(str(runs))] = fia_portfolio_df.loc[:, 'total_pre_income']

        runs += 1

    # ------------------Calculate % of portfolios ending value greater than required LIFETIME cumm. income---------
    total_income_by_age = sim_base_income.loc[:, sim_base_income.columns[0]].cumsum()
    total_income_by_acturial_age = total_income_by_age.loc[life_expectancy - clients_age]
    total_income_by_age.fillna(0, inplace=True)
    income_dataframe = pd.DataFrame(total_income_by_age)
    income_dataframe.loc[:, 'remaining_income_by_acturial_age'] = total_income_by_age.apply(
        lambda x: total_income_by_acturial_age - x)

    s = income_dataframe.loc[:, 'remaining_income_by_acturial_age']
    base_prob_of_success = sim_base_total.gt(s, axis=0).sum(axis=1)
    port_prob_of_success = sim_port_total.gt(s, axis=0).sum(axis=1)

    # ----------------------------Portfolio sufficient for NEXT YEARS income needs-------------------
    next_year_income = sim_base_income.loc[:, sim_base_income.columns[0]].shift(-1).fillna(0)  # Yearly Income Reqd.
    base_success_next_year = sim_base_total.gt(next_year_income, axis=0).sum(axis=1)

    base_for_next_year_need = sim_base_total[sim_base_total.gt(next_year_income, axis=0)]

    port_success_next_year = sim_port_total.gt(next_year_income, axis=0).sum(axis=1)

    port_for_next_year_need = sim_port_total[sim_port_total.gt(next_year_income, axis=0)]

    # ---------------Portfolio for 45 years of simulation---------------------------------------
    base_success_portfolio = sim_base_total[sim_base_total.gt(next_year_income, axis=0)]
    port_success_portfolio = sim_port_total[sim_port_total.gt(next_year_income, axis=0)]

    # ----------------Portfolio Simulation until the acturial age------------------------------
    acturial_years = life_expectancy - clients_age
    base_success_portfolio_act_age = base_success_portfolio.loc[acturial_years, :]
    port_success_portfolio_act_age = port_success_portfolio.loc[acturial_years, :]

    # -------------------------Base Portfolio TS with max Terminal Value ----------------------------
    if base_success_portfolio_act_age.isnull().sum() == trials:
        base_max_portfolio = 0.0
    else:
        base_max_portfolio = base_success_portfolio.loc[:, base_success_portfolio_act_age.idxmax()]

    # -------------------------FIA Portfolio TS with max Terminal Value ----------------------------
    if port_success_portfolio_act_age.isnull().sum() == trials:
        port_max_portfolio = 0.0
    else:
        port_max_portfolio = port_success_portfolio.loc[:, port_success_portfolio_act_age.idxmax()]
    # ------------------------------Average age with full income------------------------------
    base_mean_age = ((life_expectancy - clients_age) - base_success_portfolio.loc[1:life_expectancy - clients_age, :]
                     .isnull().sum()).mean()

    port_mean_age = ((life_expectancy - clients_age) - port_success_portfolio.loc[1:life_expectancy - clients_age, :]
                     .isnull().sum()).mean()

    # ----------------------------Median Age with full Income------------------------------------------
    base_median_age = ((life_expectancy - clients_age) - base_success_portfolio.loc[1:life_expectancy - clients_age, :]
                       .isnull().sum()).median()

    port_median_age = ((life_expectancy - clients_age) - port_success_portfolio.loc[1:life_expectancy - clients_age, :]
                       .isnull().sum()).median()

    # --------------Mean Value for all the portfolios at end of the acturial age--------------------
    base_act_avg_porfolio = base_success_portfolio.loc[life_expectancy - clients_age, :].fillna(0).mean()
    port_act_avg_porfolio = port_success_portfolio.loc[life_expectancy - clients_age, :].fillna(0).mean()

    # --------------Median Value for all the portfolios at end of the acturial age--------------------
    base_act_median_porfolio = base_success_portfolio.loc[life_expectancy - clients_age, :].fillna(0).median()
    port_act_median_porfolio = port_success_portfolio.loc[life_expectancy - clients_age, :].fillna(0).median()

    # # --------------Mean Value for all the portfolios in the simulation--------------------
    # base_sim_mean = base_success_portfolio.loc[1:life_expectancy - clients_age, :].mean().mean()
    # port_sim_mean = port_success_portfolio.loc[1:life_expectancy - clients_age, :].mean().mean()
    #
    # # --------------Median Value for all the portfolios in the simulation--------------------
    # base_sim_mean = base_success_portfolio.loc[1:life_expectancy - clients_age, :].mean().median()
    # port_sim_mean = port_success_portfolio.loc[1:life_expectancy - clients_age, :].mean().median()

    # -------Max Portfolio value at the end of acturial age----------------------------------------
    base_act_max = base_success_portfolio.loc[life_expectancy - clients_age, :].max()
    port_act_max = port_success_portfolio.loc[life_expectancy - clients_age, :].max()

    # -------Min Portfolio value at the end of acturial age----------------------------------------
    base_act_min = base_success_portfolio.loc[life_expectancy - clients_age, :].min()
    port_act_min = port_success_portfolio.loc[life_expectancy - clients_age, :].min()

    # ---------------------Lifetime Average Income----------------------------------
    base_total_income = sim_base_income.cumsum().loc[acturial_years, :].mean()
    port_total_income = income_from_fia + sim_port_income
    port_total_income = port_total_income.cumsum().loc[acturial_years, :].mean()

    simulation_stats = pd.DataFrame(index=['Average Years', 'Median Years', 'Average Age', 'Median Age',
                                           'Average Portfolio (act.age)', 'Median Portfolio (act.age)',
                                           'Max Portfolio Value', 'Min Portfolio Value',
                                           'Average Lifetime Income'], columns=['Base', 'FIA'])

    simulation_stats.loc['Average Years', :] = [base_mean_age, base_mean_age]
    simulation_stats.loc['Median Years', :] = [base_median_age, base_median_age]
    simulation_stats.loc['Average Age', :] = [base_mean_age + clients_age, base_mean_age + clients_age]
    simulation_stats.loc['Median Age', :] = [base_median_age + clients_age, base_median_age + clients_age]
    simulation_stats.loc['Average Portfolio (act.age)', :] = [base_act_avg_porfolio, port_act_avg_porfolio]
    simulation_stats.loc['Median Portfolio (act.age)', :] = [base_act_median_porfolio, port_act_median_porfolio]
    simulation_stats.loc['Max Portfolio Value', :] = [base_act_max, port_act_max]
    simulation_stats.loc['Min Portfolio Value', :] = [base_act_min, port_act_min]
    simulation_stats.loc['Average Lifetime Income', :] = [base_total_income, port_total_income]
    comments = ['Average years of portfolios that meet the next years income needs for the lifetime',
                'Median years of portfolios that meet the next years income needs for the lifetime',
                'Average Clients Age',
                'Median Clients Age',
                'Average of terminal values for the portfolios at the end of the acturial life',
                'Median of terminal values for the portfolios at the end of the acturial life',
                'Maximum of terminal values for the portfolios at the end of the acturial life',
                'Minimum of terminal values for the portfolios at the end of the acturial life',
                'Average of total income generated by all portfolios at the end of the acturial life']

    simulation_stats.loc[:, 'Notes'] = comments

    # --------------------------------------------------------------------------------

    # # -----------------------------------income breakdown for Base portfolio----------------------------------
    # base_df.to_csv(src + 'base_port_detail.csv')
    # sim_base_total.to_csv(src + 'base_ending_values.csv')
    # income_breakdown_base = pd.DataFrame(sim_base_total.quantile(0.5, axis=1))
    # income_breakdown_base.loc[:, 'income_from_portfolio'] = sim_base_income.quantile(0.5, axis=1)
    # income_breakdown_base.loc[:, 'fia_income'] = 0.0
    # income_breakdown_base.loc[:, 'social_security_income'] = social
    # income_breakdown_base.loc[:, 'coupon_income'] = cpn_income_base
    #
    # income_breakdown_base.rename(columns={income_breakdown_base.columns[0]: 'portfolio_ending_value'}, inplace=True)
    # income_breakdown_base.loc[:, 'income_from_portfolio'][
    #     income_breakdown_base.loc[:, 'portfolio_ending_value'] <= 0] = 0
    # income_breakdown_base.loc[:, 'total_income'] = income_breakdown_base.loc[:, income_breakdown_base.columns[1:]].sum(
    #     axis=1)
    #
    # # --------------------------------------Block Ends-----------------------------------------------------------
    #
    # # ---------------------------------------income breakdown for FIA portfolio----------------------------------
    # fia_portfolio_df.to_csv(src + 'fia_port_detail.csv')
    # sim_port_total.to_csv(src + 'fiaport_ending_values.csv')
    #
    # income_breakdown_port = pd.DataFrame(sim_port_total.quantile(0.5, axis=1))
    # income_breakdown_port.loc[:, 'income_from_portfolio'] = sim_port_income.quantile(0.5, axis=1)
    # income_breakdown_port.loc[:, 'fia_income'] = income_from_fia
    # income_breakdown_port.loc[:, 'social_security_income'] = social
    # income_breakdown_port.loc[:, 'coupon_income'] = cpn_income_port
    #
    # income_breakdown_port.rename(columns={income_breakdown_port.columns[0]: 'portfolio_ending_value'}, inplace=True)
    # income_breakdown_port.loc[:, 'income_from_portfolio'][
    #     income_breakdown_port.loc[:, 'portfolio_ending_value'] <= 0] = 0
    # income_breakdown_port.loc[:, 'total_income'] = income_breakdown_port.loc[:, income_breakdown_port.columns[1:]].sum(
    #     axis=1)
    #
    # # ----------------------------------Block Ends-------------------------------------------------------------
    q_cut = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    sim_base_income[sim_base_total < income_needed] = 0.0

    sim_port_income[sim_port_total < income_net_fia_income] = 0

    sim_port_income = sim_port_income + income_from_fia

    # base_quantile = sim_base_total.loc[sim_base_total.index[-1]].quantile([0.05, 0.25, 0.50, 0.75, 0.90])
    #
    # port_quantile = sim_port_total.loc[sim_port_total.index[-1]].quantile([0.05, 0.25, 0.50, 0.75, 0.90])

    base_quantile = sim_base_total.loc[sim_base_total.index[-1]].quantile(q_cut)

    port_quantile = sim_port_total.loc[sim_port_total.index[-1]].quantile(q_cut)

    # q_cut = [0.0, .05, 0.25, 0.5, 0.75, 0.95, 1.0]
    cols = ['Min', '5th', '25th', '50th', '75th', '90th', 'Max']

    # ------------------------------------------drop year 0-----------------------------------------
    sim_base_total = sim_base_total[1:]
    sim_port_total = sim_port_total[1:]

    # ------------------------quantile analysis for base terminal value-----------------------------
    base_qcut = pd.DataFrame(index=sim_base_total.index, columns=cols)
    for c in range(len(cols)):
        base_qcut.loc[:, cols[c]] = sim_base_total.quantile(q_cut[c], axis=1)

    base_qcut.clip(lower=0, inplace=True)

    sim_base_total.clip(lower=0, inplace=True)

    # -------------------------------------quantile analysis for base income----------------------------
    base_income_qcut = pd.DataFrame(index=sim_base_income.index, columns=cols)
    for c in range(len(cols)):
        base_income_qcut.loc[:, cols[c]] = sim_base_income.quantile(q_cut[c], axis=1)

    # ----Remove NaN's prior to the income start years------------
    # base_income_qcut = base_income_qcut.loc[income_starts:]

    # ---------------------------------quantile analysis for portfolio terminal value ---------------

    port_qcut = pd.DataFrame(index=sim_port_total.index, columns=cols)
    for c in range(len(cols)):
        port_qcut.loc[:, cols[c]] = sim_port_total.quantile(q_cut[c], axis=1)

    port_qcut.clip(lower=0, inplace=True)

    # ----------------------------------quantile analysis for portfolio income----------------------------
    port_income_qcut = pd.DataFrame(index=sim_port_income.index, columns=cols)
    for c in range(len(cols)):
        port_income_qcut.loc[:, cols[c]] = sim_port_income.quantile(q_cut[c], axis=1)

    # ----Remove NaN's prior to the income start years------------
    # port_income_qcut = port_income_qcut.loc[income_starts:]

    # ----------probability ending value will be less than 0 at the end of the horizon -----------------------
    # base_legacy_risk = (sim_base_total.loc[sim_base_total.index[-1]] < 0).sum() / (trials)

    base_legacy_risk = (sim_base_total.loc[sim_base_total.index[life_expectancy - clients_age]] < 0).sum() / trials
    port_legacy_risk = (sim_port_total.loc[sim_port_total.index[life_expectancy - clients_age]] < 0).sum() / trials

    # port_legacy_risk = (sim_port_total.loc[sim_port_total.index[-1]] <= 0).sum() / (trials)

    legacy_risk = pd.DataFrame([base_legacy_risk, port_legacy_risk,
                                'Prob. of portfolio value less than 0 at the end of the expected life'],
                               index=['base', 'fia_portfolio', 'Notes'],
                               columns=['Ruin Probability'])

    # -----------Year-wise probability of ending value greater than 0 -----------------
    base_psuccess = sim_base_total.apply(lambda x: x > 0).sum(axis=1) / trials
    port_psuccess = sim_port_total.apply(lambda x: x > 0).sum(axis=1) / trials

    # -----------------------WRITING FILES TO EXCEL ---------------------------

    writer = pd.ExcelWriter(src + method + '_simulated_income_summary_custom.xlsx', engine='xlsxwriter')
    read_income_inputs.to_excel(writer, sheet_name='inputs_for_income')

    read_returns_est.to_excel(writer, sheet_name='asset_returns_estimates')
    # read_portfolio_inputs.to_excel(writer, sheet_name='portfolio_inputs')

    age_index = list(range(clients_age + 1, clients_age + len(base_qcut) + 1))
    # base_qcut.loc[:, 'clients_age'] = age_index
    # base_qcut.loc[:, 'comment'] = ''
    # base_qcut.loc[:, 'comment'] = np.where(base_qcut.clients_age == life_expectancy, 'expected_life', "")

    base_inv = float(read_income_inputs.loc['risky_assets', 'Base'])
    base_qcut.loc[:, 'age'] = age_index
    base_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'

    # --------To start with year 0---------------------------------
    insert_col = [base_inv, base_inv, base_inv, base_inv, base_inv, base_inv,
                  base_inv, clients_age, np.nan]
    base_qcut.loc[len(base_qcut) + 1, :] = 0.0
    base_qcut = base_qcut.shift(1)
    base_qcut.iloc[0] = insert_col
    base_qcut.reset_index(drop=True, inplace=True)
    base_qcut.loc[:, 'Annual Return'] = base_qcut.loc[:, '50th'].pct_change().fillna(0)
    base_qcut.to_excel(writer, sheet_name='base_ending_value_quantiles')
    # base_qcut.loc[income_starts:, :].to_excel(writer, sheet_name='base_ending_value_quantiles')

    # base_income_qcut = base_income_qcut[1:] base_income_qcut.loc[:, 'clients_age'] = age_index
    # base_income_qcut.loc[:, 'comment'] = '' base_income_qcut.loc[:, 'comment'] = np.where(
    # base_income_qcut.clients_age == life_expectancy, 'expected_life', "")

    base_income_qcut = base_income_qcut.loc[1:, :]
    base_income_qcut.loc[:, 'age'] = age_index
    base_income_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'

    base_income_qcut.loc[income_starts:, :].to_excel(writer, sheet_name='base_income_quantiles')

    # age_index = list(range(clients_age+1, clients_age + len(port_qcut)+1))
    # port_qcut.loc[:, 'clients_age'] = age_index
    # port_qcut.loc[:, 'comment'] = ''
    # port_qcut.loc[:, 'comment'] = np.where(port_qcut.clients_age == life_expectancy, 'expected_life', "")

    port_qcut.loc[:, 'age'] = age_index
    port_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    port_qcut.loc[len(port_qcut) + 1, :] = 0.0
    port_qcut = port_qcut.shift(1)
    port_qcut.iloc[0] = insert_col
    port_qcut.reset_index(drop=True, inplace=True)
    port_qcut.loc[:, 'Annual Return'] = port_qcut.loc[:, '50th'].pct_change().fillna(0)
    port_qcut.to_excel(writer, sheet_name='fia_port_ending_value_quantiles')
    # port_qcut.loc[income_starts:, :].to_excel(writer, sheet_name='fia_port_ending_value_quantiles')

    # port_income_qcut = port_income_qcut[1:] port_income_qcut.loc[:, 'clients_age'] = age_index
    # port_income_qcut.loc[:, 'comment'] = '' port_income_qcut.loc[:, 'comment'] = np.where(
    # port_income_qcut.clients_age == life_expectancy, 'expected_life', "")

    port_income_qcut = port_income_qcut.loc[1:, :]
    port_income_qcut.loc[:, 'age'] = age_index
    port_income_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    port_income_qcut.loc[income_starts:, :].to_excel(writer, sheet_name='fia_port_income_quantiles')

    prob_success_df = pd.concat([base_psuccess, port_psuccess], axis=1)
    prob_success_df.rename(columns={prob_success_df.columns[0]: 'prob(ending_value>0)_base',
                                    prob_success_df.columns[1]: 'prob(ending_value>0)_port'}, inplace=True)

    # prob_success_df.loc[:, 'clients_age'] = age_index
    # prob_success_df.loc[:, 'comment'] = ''
    # prob_success_df.loc[:, 'comment'] = np.where(prob_success_df.clients_age == life_expectancy, 'expected_life', "")

    prob_success_df.loc[:, 'age'] = age_index
    prob_success_df.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    prob_success_df.loc[:, 'prob(ending_value>lifetime_req income)_base'] = base_prob_of_success / trials
    prob_success_df.loc[:, 'prob(ending_value>lifetime_req income)_port'] = port_prob_of_success / trials
    prob_success_df.loc[:, 'prob(ending_value>next_year_req_income)_base'] = base_success_next_year / trials
    prob_success_df.loc[:, 'prob(ending_value>next_year_req_income)_port'] = port_success_next_year / trials
    prob_success_df.loc[:, 'base_max_portfolio_at_acturial_age'] = base_max_portfolio
    prob_success_df.loc[:, 'port_max_portfolio_at_acturial_age'] = port_max_portfolio

    # --------------------Percentile Portfolio's based on Acturial Life------------------------
    base_success = prob_success_df.loc[life_expectancy - clients_age, 'prob(ending_value>next_year_req_income)_base']
    port_success = prob_success_df.loc[life_expectancy - clients_age, 'prob(ending_value>next_year_req_income)_port']

    # acturial_age_base_tv = sim_base_total.loc[:life_expectancy - clients_age, ]
    # percentile_base_tv = sim_base_total.apply(lambda x: np.nanpercentile(x, base_success), axis=1)

    # ----------------Year wise percentile portfolio to meet next year income. Based on the success at acturial age.
    # Yearly portfolio values that can provide the next year income below the success rate at end of life (Percentile)-

    # acturial_age_base = base_for_next_year_need.loc[:life_expectancy - clients_age, ]
    # acturial_age_base = base_for_next_year_need.copy().fillna(0)
    percentile_base = base_for_next_year_need.apply(lambda x: np.nanpercentile(x, base_success), axis=1)

    # acturial_age_port_tv = sim_port_total.loc[:life_expectancy - clients_age, ]
    # percentile_port_tv = sim_port_total.apply(lambda x: np.nanpercentile(x, port_success), axis=1)

    # ----------------Year wise percentile portfolio to meet next year income. Based on the success at acturial age.
    # Yearly portfolio values that can provide the next year income below the success rate at end of life (Percentile)-

    # acturial_age_port = port_for_next_year_need.loc[:life_expectancy - clients_age, ]
    # acturial_age_base = port_for_next_year_need.copy().fillna(0)
    percentile_port = port_for_next_year_need.apply(lambda x: np.nanpercentile(x, port_success), axis=1)

    prob_success_df.loc[:, 'acturial_success_percentile_base_portfolio'] = percentile_base
    prob_success_df.loc[:, 'acturial_success_percentile_port_portfolio'] = percentile_port

    # prob_success_df.loc[:, 'terminalVal_success_percentile_base_portfolio'] = percentile_base_tv
    # prob_success_df.loc[:, 'terminalVal_success_percentile_port_portfolio'] = percentile_port_tv

    prob_success_df.to_excel(writer, sheet_name='success_probability')

    # --------------BASE - Accumulation and Income Breakdown based on the success percentile portfolio---------------
    base_df.to_csv(src + 'base_port_detail.csv')
    sim_base_total.to_csv(src + 'base_ending_values.csv')
    income_breakdown_base = pd.DataFrame(sim_base_total.quantile(base_success, axis=1))
    income_breakdown_base.loc[:, 'income_from_risky_assets'] = sim_base_income.quantile(base_success, axis=1) \
                                                               - social - cpn_income_port
    income_breakdown_base.loc[:, 'guaranteed_income'] = 0.0
    income_breakdown_base.loc[:, 'social_security_income'] = social
    income_breakdown_base.loc[:, 'coupon_income'] = cpn_income_base

    income_breakdown_base.rename(columns={income_breakdown_base.columns[0]: 'portfolio_ending_value'}, inplace=True)
    income_breakdown_base.loc[:, 'income_from_risky_assets'][
        income_breakdown_base.loc[:, 'portfolio_ending_value'] <= 0] = 0
    income_breakdown_base.loc[:, 'total_income'] = income_breakdown_base.loc[:, income_breakdown_base.columns[1:]].sum(
        axis=1)

    # ----------FIA PORTFOLIO - Accumulation and Income Breakdown based on the success percentile portfolio-----------
    fia_portfolio_df.to_csv(src + 'fia_port_detail.csv')
    sim_port_total.to_csv(src + 'fiaport_ending_values.csv')

    income_breakdown_port = pd.DataFrame(sim_port_total.quantile(port_success, axis=1))
    income_breakdown_port.loc[:, 'income_from_risky_assets'] = sim_port_income.quantile(port_success, axis=1) \
                                                               - income_from_fia - social - cpn_income_port
    income_breakdown_port.loc[:, 'guaranteed_income'] = income_from_fia
    income_breakdown_port.loc[:, 'social_security_income'] = social
    income_breakdown_port.loc[:, 'coupon_income'] = cpn_income_port

    income_breakdown_port.rename(columns={income_breakdown_port.columns[0]: 'portfolio_ending_value'}, inplace=True)
    income_breakdown_port.loc[:, 'income_from_risky_assets'][
        income_breakdown_port.loc[:, 'portfolio_ending_value'] <= 0] = 0
    income_breakdown_port.loc[:, 'total_income'] = income_breakdown_port.loc[:, income_breakdown_port.columns[1:]].sum(
        axis=1)

    # -------------------Write simulation Statistics-------------------------------------
    simulation_stats.to_excel(writer, sheet_name='simulation_statistics')

    # port_psuccess.to_excel(writer, sheet_name='fia_port_success_probability')

    income_breakdown_base = income_breakdown_base.loc[1:, :]
    income_breakdown_base.loc[:, 'age'] = age_index
    income_breakdown_base.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    income_breakdown_base.loc[income_starts:, :].to_excel(writer, sheet_name='base_income_breakdown_median')

    income_breakdown_port = income_breakdown_port.loc[1:, :]
    income_breakdown_port.loc[:, 'age'] = age_index
    income_breakdown_port.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    income_breakdown_port.loc[income_starts:, :].to_excel(writer, sheet_name='fia_income_breakdown_median')

    legacy_risk.to_excel(writer, sheet_name='ruin_probability')

    median_returns_normal.loc[:, 'fia_median_returns'] = median_normal_fia
    median_returns_normal.to_excel(writer, sheet_name='gr_port_median_normal')

    writer.save()

    # -----------------Plotting charts--------------------------------------------
    base_qcut.loc[income_starts:].plot(grid=True, title='Quantile Terminal Value - Base Portfolio')
    plt.savefig(src + "quantile_terminal_base.png")
    plt.close('all')

    base_income_qcut.plot(grid=True, title='Quantile Income - Base Portfolio')
    plt.savefig(src + "quantile_income_base.png")
    plt.close('all')

    base_psuccess.plot(grid=True, title='Probability of Success (Portfolio Ending Value > 0) - Base Portfolio')
    plt.savefig(src + "success_probabilty_base.png")
    plt.close('all')

    (1 - base_psuccess).plot(grid=True, title='Probability of Ruin (Portfolio Ending Value < 0) - Base Portfolio')
    plt.savefig(src + "ruin_probability_base.png")
    plt.close('all')

    port_qcut.loc[income_starts:].plot(grid=True, title='Quantile Terminal Value - FIA Portfolio')
    plt.savefig(src + "quantile_terminal_fia.png")
    plt.close('all')

    port_income_qcut.plot(grid=True, title='Quantile Income - FIA Portfolio')
    plt.savefig(src + "quantile_income_fia.png")
    plt.close('all')

    port_psuccess.plot(grid=True, title='Probability of Success (Portfolio Ending Value > 0) - FIA Portfolio')
    plt.savefig(src + "success_probabilty_fia.png")
    plt.close('all')

    (1 - port_psuccess).plot(grid=True, title='Probability of Ruin (Portfolio Ending Value < 0) - FIA Portfolio')
    plt.savefig(src + "ruin_probability_fia.png")
    plt.close('all')

    print("simulation completed....")


def income_model_constant_portfolio_return(num_of_years=30, trials=100, method='normal'):
    """Simulation based on the CONSTANT (Leveled) growth rate provided by the users for the risky portfolio
    and the FIA"""

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

    # read_returns_est = pd.read_csv(src + "income_assets_returns_estimates.csv", index_col='Symbol')
    read_returns_est = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_assets_returns_estimates',
                                     index_col=[0])

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

    # # dataframe for unsorted returns (normal)
    # median_returns_normal = pd.DataFrame({t: asset_median_returns(read_normal, t) for t in tickers})
    # median_returns_normal.loc[:, 'portfolio_return'] = median_returns_normal.dot(wts)
    # median_normal_fia = pd.DataFrame({'FIA': asset_median_returns(read_normal, 'r_FIA')})
    #
    # # dataframe for smallest to largest returns
    # median_returns_smallest = pd.DataFrame({t: asset_median_returns(read_small, t) for t in tickers})
    # median_returns_smallest.loc[:, 'portfolio_return'] = median_returns_smallest.dot(wts)
    # median_smallest_fia = pd.DataFrame({'FIA': asset_median_returns(read_small, 'r_FIA')})
    #
    # # dataframe for unsorted returns (normal)
    # median_returns_largest = pd.DataFrame({t: asset_median_returns(read_large, t) for t in tickers})
    # median_returns_largest.loc[:, 'portfolio_return'] = median_returns_largest.dot(wts)
    # median_largest_fia = pd.DataFrame({'FIA': asset_median_returns(read_large, 'r_FIA')})

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

    if method == 'normal':
        # income_df.loc[:, 'index_returns'] = read_normal.loc[:, '{}_{}'.format('r_FIA', str(runs))]
        # ----------CONSTANT FIA INDEX GROWTH RATE-------------------
        income_df.loc[:, 'index_returns'] = const_fia_index_ret

    elif method == 'smallest':
        income_df.loc[:, 'index_returns'] = read_small.loc[:, '{}_{}'.format('r_FIA', str(runs))]

    else:
        income_df.loc[:, 'index_returns'] = read_large.loc[:, '{}_{}'.format('r_FIA', str(runs))]

    # income_df.loc[:, 'index_returns'] = np.random.normal(fia_ret, fia_risk, size=(len(years), 1))

    cumprod = (1. + income_df['index_returns']).rolling(window=term).agg(lambda x: x.prod()) - 1
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

    income_df.loc[:, 'rider_fee'] = income_df.loc[:, 'high_inc_benefit_base'] * rider_fee
    income_df.loc[:, 'eoy_income'] = np.where(income_df.loc[:, 'year'] > income_starts,
                                              income_df.loc[:, 'high_inc_benefit_base'] * inc_payout_factor, 0)

    for counter in years:
        if counter == 0:
            income_df.loc[counter, 'contract_value'] = premium * (1 + contract_bonus)

        elif income_df.loc[counter, 'strategy_term'] == 1:
            x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee']
            x2 = (x1 * (1 + income_df.loc[counter, 'term_ret_netspr'])) - income_df.loc[counter, 'eoy_income']
            income_df.loc[counter, 'contract_value'] = x2

        else:
            x1 = income_df.loc[counter - 1, 'contract_value'] - income_df.loc[counter, 'rider_fee'] - \
                 income_df.loc[counter, 'eoy_income']

            income_df.loc[counter, 'contract_value'] = x1

    # variable stores the income number that is used in the base and fia portfolio calcs.

    income_from_fia = income_df.loc[income_df.index[-1], 'eoy_income']

    income_df.loc[:, 'contract_value'] = income_df.loc[:, 'contract_value'].apply(lambda x: 0 if x <= 0 else x)

    sim_fia_cv.loc[:, str(runs)] = income_df.loc[:, 'contract_value']

    # -------------------------------------BASE MODEL---------------------------------------------

    base_wts = read_asset_weights.loc[:, 'base']
    base_assets = list(base_wts.index)
    base_weights = list(base_wts.values)
    base_returns = list(read_returns_est.loc[:, 'Annualized Returns'].values)
    base_std = list(read_returns_est.loc[:, 'Annualized Risk'].values)

    base_investment = float(read_income_inputs.loc['risky_assets', 'Base'])
    adv_fees = float(read_income_inputs.loc['advisor_fees', 'Base'])

    # -------------------required income----------------------------------
    req_annual_income = float(read_income_inputs.loc['annual_income', 'inputs'])
    income_needed = req_annual_income - social
    income_net_fia_income = max(0, income_needed - income_from_fia)
    cpn_income_base = base_investment * wtd_cpn_yield

    # ----------------------RANDOM RETURNS--------------------------
    r_cols = ['r_{}'.format(name) for name in base_assets]
    boy_value = ['bv_{}'.format(name) for name in base_assets]
    eoy_value = ['ev_{}'.format(name) for name in base_assets]

    random_returns = pd.DataFrame(index=income_df.index, columns=r_cols)

    # for c in range(len(r_cols)):
    #     ret = np.random.normal(base_returns[c], base_std[c], size=(len(random_returns.index), 1))

    if method == 'smallest':
        this_run_cols = ['{}_{}'.format(cname, str(runs)) for cname in r_cols]
        random_returns = read_small.loc[:, this_run_cols]

        # random_returns.loc[:, r_cols[c]] = np.sort(ret.flatten())
        # asset_dict.update({'{}_{}'.format(r_cols[c], str(runs)): np.sort(ret.flatten())})

    elif method == 'largest':
        this_run_cols = ['{}_{}'.format(cname, str(runs)) for cname in r_cols]
        random_returns = read_large.loc[:, this_run_cols]

        # random_returns.loc[:, r_cols[c]] = np.flip(np.sort(ret.flatten()))
        # asset_dict.update({'{}_{}'.format(r_cols[c], str(runs)): np.flip(np.sort(ret.flatten()))})

    else:
        this_run_cols = ['{}_{}'.format(cname, str(runs)) for cname in r_cols]
        random_returns = read_normal.loc[:, this_run_cols]

        # random_returns.loc[:, r_cols[c]] = ret.flatten()
        # asset_dict.update({'{}_{}'.format(r_cols[c], str(runs)): ret.flatten()})

    # store the simulated assets returns in one dictionary
    # returns_dict.update({str(runs): random_returns})

    # collect the asset based returns from all simulation and calculate the median returns.
    # def get_median_returns(sym):
    #     cols = [sym + '_' + str(c) for c in np.arange(trials)]
    #     asset_df = pd.DataFrame({c: asset_dict.get(c) for c in cols})
    #     return asset_df.median(axis=1)
    #
    # asset_median_returns = pd.DataFrame({symbol: get_median_returns(symbol) for symbol in r_cols})
    #
    # asset_median_returns.loc[:, 'simulated_portfolio_median_returns'] = asset_median_returns.dot(base_weights)

    base_df = random_returns.copy()
    pre_income_base_df = random_returns.copy()

    # base_investment = float(read_portfolio_inputs.loc['risky_assets', 'Base'])

    fia_portfolio_df = random_returns.copy()
    pre_income_port_df = random_returns.copy()
    port_investment = float(read_income_inputs.loc['risky_assets', 'FIA'])
    cpn_income_port = port_investment * wtd_cpn_yield

    # ---------Initial Investments for pre-income account values---------------------
    pre_income_base_inv = base_investment
    pre_income_port_inv = port_investment
    # ----------------------------------------BASE PORTFOLIO----------------------------
    for name in boy_value:
        base_df.loc[:, name] = 0.0
        pre_income_base_df.loc[:, name] = 0.0

    for counter in years:
        period_returns = list(random_returns.loc[counter, :])
        if counter == 0:
            # ---------------For year 0, the year of investment------------

            # ------------Calculate the annual portfolio returns - Gross Returns--------------------
            base_df.loc[counter, boy_value] = [base_weights[c] * base_investment for c in range(len(boy_value))]

            # -------------Record the Pre Income Base Portfolio-----------------------------

            pre_income_base_df.loc[counter, boy_value] = [base_weights[c] *
                                                          pre_income_base_inv for c in range(len(boy_value))]
            pre_income_base_df.loc[counter, 'total'] = pre_income_base_df.loc[counter, boy_value].sum()
            pre_income_base_inv = pre_income_base_df.loc[counter, boy_value].sum()

            # ------------------Pre Income Block Ends------------------------

            base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()

            base_df.loc[counter, 'total_net_fees'] = 0.0
            base_df.loc[counter, 'income'] = 0.0
            # base_investment = base_df.loc[counter, boy_value].sum()
            base_df.loc[counter, 'total_pre_income'] = base_investment

        elif (counter > 0) and (counter < income_starts):

            # ----For years between the start of the investment and start if the income---------------
            base_df.loc[counter, boy_value] = [base_weights[c] * base_investment * (1 + period_returns[c])
                                               for c in range(len(boy_value))]

            # -------------Record the Pre Income Base Portfolio-----------------------------
            pre_income_base_df.loc[counter, boy_value] = [
                base_weights[c] * pre_income_base_inv * (1 + period_returns[c])
                for c in range(len(boy_value))]

            # pre_income_base_df.loc[counter, 'total'] = pre_income_base_df.loc[counter, boy_value].sum()
            pre_income_base_df.loc[counter, 'total'] = base_investment * (1 + const_risky_port_ret)
            pre_income_base_df.loc[counter, 'adv_fees'] = pre_income_base_df.loc[counter, 'total'] * adv_fees
            pre_income_base_df.loc[counter, 'total_net_fees'] = pre_income_base_df.loc[counter, 'total'] - \
                                                                pre_income_base_df.loc[counter, 'adv_fees']
            pre_income_base_inv = pre_income_base_df.loc[counter, 'total_net_fees'] + cpn_income_base

            # ------------------Pre Income Block Ends------------------------

            # base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
            base_df.loc[counter, 'total'] = base_investment * (1 + 0.06)
            base_df.loc[counter, 'adv_fees'] = base_df.loc[counter, 'total'] * adv_fees
            base_df.loc[counter, 'total_net_fees'] = base_df.loc[counter, 'total'] - base_df.loc[
                counter, 'adv_fees']

            # --coupon payment is invested back into the risky portfolio until the income is withdrawn----
            base_investment = base_df.loc[counter, 'total_net_fees'] + cpn_income_base
            base_df.loc[counter, 'total_pre_income'] = base_df.loc[counter, 'total_net_fees']

        else:

            # -------------For Years after the income started----------------------
            base_df.loc[counter, boy_value] = [base_weights[c] * base_investment * (1 + period_returns[c])
                                               for c in range(len(boy_value))]

            # -------------Record the Pre Income Base Portfolio-----------------------------
            pre_income_base_df.loc[counter, boy_value] = [
                base_weights[c] * pre_income_base_inv * (1 + period_returns[c])
                for c in range(len(boy_value))]

            # pre_income_base_df.loc[counter, 'total'] = pre_income_base_df.loc[counter, boy_value].sum()
            pre_income_base_df.loc[counter, 'total'] = base_investment * (1 + const_risky_port_ret)
            pre_income_base_df.loc[counter, 'adv_fees'] = pre_income_base_df.loc[counter, 'total'] * adv_fees
            pre_income_base_df.loc[counter, 'total_net_fees'] = pre_income_base_df.loc[counter, 'total'] - \
                                                                pre_income_base_df.loc[counter, 'adv_fees']
            pre_income_base_inv = pre_income_base_df.loc[counter, 'total_net_fees'] + cpn_income_base

            # ------------------Pre Income Block Ends------------------------

            # base_df.loc[counter, 'total'] = base_df.loc[counter, boy_value].sum()
            base_df.loc[counter, 'total'] = base_investment * (1 + const_risky_port_ret)
            base_df.loc[counter, 'adv_fees'] = base_df.loc[counter, 'total'] * adv_fees

            # ---req. income is adjusted for inflation from the second year of withdrawal. Reinvestment of coupon
            # stops from the year income starts. Req. income is reduced by the coupon payments

            if counter == income_starts:

                income_needed = req_annual_income - social
                base_df.loc[counter, 'income'] = income_needed - cpn_income_base
                income_needed = req_annual_income

            else:
                income_needed = income_needed * (1 + inflation) - social
                base_df.loc[counter, 'income'] = income_needed - cpn_income_base
                income_needed = income_needed + social

            base_df.loc[counter, 'total_net_fees'] = base_df.loc[counter, 'total'] - \
                                                     base_df.loc[counter, 'adv_fees'] - \
                                                     base_df.loc[counter, 'income']

            base_df.loc[counter, 'total_pre_income'] = base_df.loc[counter, 'total'] - \
                                                       base_df.loc[counter, 'adv_fees']

            base_investment = base_df.loc[counter, 'total_net_fees']

    # -------------------Portfolio with PreIncome Values----------------------------
    sim_base_total_preincome.loc[:, 's_{}'.format(str(runs))] = pre_income_base_df.loc[:, 'total_net_fees']
    sim_base_total_preincome.fillna(float(read_income_inputs.loc['risky_assets', 'Base']), inplace=True)
    # --------------------------------PreIncome Block Ends----------------------------

    base_df.loc[:, 'adj_total'] = base_df.loc[:, 'total_net_fees'].apply(lambda x: x if x > 0 else 0)
    sim_base_total.loc[:, 's_{}'.format(str(runs))] = base_df.loc[:, 'total_net_fees']
    sim_base_income.loc[:, 's_{}'.format(str(runs))] = base_df.loc[:, 'income']
    sim_base_total_pre_income.loc[:, 's_{}'.format(str(runs))] = base_df.loc[:, 'total_pre_income']

    # ----------------------------FIA PORTFOLIO----------------------------------------------
    for name in boy_value:
        fia_portfolio_df.loc[:, name] = 0.0
        pre_income_port_df.loc[:, name] = 0.0

    for counter in years:
        period_returns = list(random_returns.loc[counter, :])
        if counter == 0:

            fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment
                                                        for c in range(len(boy_value))]

            # -------------Record the Pre Income Base Portfolio-----------------------------

            pre_income_port_df.loc[counter, boy_value] = [base_weights[c] *
                                                          pre_income_port_inv for c in range(len(boy_value))]
            pre_income_port_df.loc[counter, 'total'] = pre_income_port_df.loc[counter, boy_value].sum()
            pre_income_port_inv = pre_income_port_df.loc[counter, boy_value].sum()

            # ------------------Pre Income Block Ends------------------------

            fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
            fia_portfolio_df.loc[counter, 'total_net_fees'] = 0.0
            fia_portfolio_df.loc[counter, 'income'] = 0.0
            # port_investment = fia_portfolio_df.loc[counter, boy_value].sum()
            fia_portfolio_df.loc[counter, 'total_pre_income'] = port_investment

        elif (counter > 0) and (counter < income_starts):

            fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment * (1 + period_returns[c])
                                                        for c in range(len(boy_value))]

            # ------------------Record the Pre Income Base Portfolio-----------------------------
            pre_income_port_df.loc[counter, boy_value] = [
                base_weights[c] * pre_income_port_inv * (1 + period_returns[c])
                for c in range(len(boy_value))]

            # pre_income_port_df.loc[counter, 'total'] = pre_income_port_df.loc[counter, boy_value].sum()
            
            # -----------------------CONSTANT GROWTH RATE-----------------
            pre_income_port_df.loc[counter, 'total'] = port_investment * (1 + const_risky_port_ret)

            pre_income_port_df.loc[counter, 'adv_fees'] = pre_income_port_df.loc[counter, 'total'] * adv_fees
            pre_income_port_df.loc[counter, 'total_net_fees'] = pre_income_port_df.loc[counter, 'total'] - \
                                                                pre_income_port_df.loc[counter, 'adv_fees']
            pre_income_port_inv = pre_income_port_df.loc[counter, 'total_net_fees'] + cpn_income_base

            # ------------------Pre Income Block Ends------------------------

            # fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
            
            # -------CONSTANT GROWTH RATE-----------------
            fia_portfolio_df.loc[counter, 'total'] = port_investment * (1 + const_risky_port_ret)

            fia_portfolio_df.loc[counter, 'adv_fees'] = fia_portfolio_df.loc[counter, 'total'] * adv_fees
            fia_portfolio_df.loc[counter, 'total_net_fees'] = fia_portfolio_df.loc[counter, 'total'] - \
                                                              fia_portfolio_df.loc[counter, 'adv_fees']
            fia_portfolio_df.loc[counter, 'total_pre_income'] = fia_portfolio_df.loc[counter, 'total_net_fees']

            port_investment = fia_portfolio_df.loc[counter, 'total_net_fees'] + cpn_income_port

        else:
            fia_portfolio_df.loc[counter, boy_value] = [base_weights[c] * port_investment * (1 + period_returns[c])
                                                        for c in range(len(boy_value))]

            # -------------Record the Pre Income Base Portfolio-----------------------------
            pre_income_port_df.loc[counter, boy_value] = [
                base_weights[c] * pre_income_port_inv * (1 + period_returns[c])
                for c in range(len(boy_value))]

            # pre_income_port_df.loc[counter, 'total'] = pre_income_port_df.loc[counter, boy_value].sum()
            pre_income_port_df.loc[counter, 'total'] = port_investment * (1 + const_risky_port_ret)
            pre_income_port_df.loc[counter, 'adv_fees'] = pre_income_port_df.loc[counter, 'total'] * adv_fees
            pre_income_port_df.loc[counter, 'total_net_fees'] = pre_income_port_df.loc[counter, 'total'] - \
                                                                pre_income_port_df.loc[counter, 'adv_fees']
            pre_income_port_inv = pre_income_port_df.loc[counter, 'total_net_fees'] + cpn_income_base

            # ------------------Pre Income Block Ends------------------------

            # fia_portfolio_df.loc[counter, 'total'] = fia_portfolio_df.loc[counter, boy_value].sum()
            fia_portfolio_df.loc[counter, 'total'] = port_investment * (1 + const_risky_port_ret)
            fia_portfolio_df.loc[counter, 'adv_fees'] = fia_portfolio_df.loc[counter, 'total'] * adv_fees

            # ---req. income is adjusted for inflation from the second year of withdrawal. Reinvestment of coupon
            # stops from the year income starts. Req. income is reduced by the coupon payments

            if counter == income_starts:

                income_needed = req_annual_income - social
                income_net_fia_income = max(0, income_needed - income_from_fia)
                fia_portfolio_df.loc[counter, 'income'] = max(0, income_net_fia_income - cpn_income_port)
                income_needed = req_annual_income

            else:
                income_needed = income_needed * (1 + inflation) - social
                income_net_fia_income = max(0, income_needed - income_from_fia)
                fia_portfolio_df.loc[counter, 'income'] = max(0, income_net_fia_income - cpn_income_port)
                income_needed = income_needed + social

            fia_portfolio_df.loc[counter, 'total_net_fees'] = fia_portfolio_df.loc[counter, 'total'] - \
                                                              fia_portfolio_df.loc[counter, 'adv_fees'] - \
                                                              fia_portfolio_df.loc[counter, 'income']

            fia_portfolio_df.loc[counter, 'total_pre_income'] = fia_portfolio_df.loc[counter, 'total'] - \
                                                                fia_portfolio_df.loc[counter, 'adv_fees']

            port_investment = fia_portfolio_df.loc[counter, 'total_net_fees']

    sim_port_total.loc[:, 's_{}'.format(str(runs))] = fia_portfolio_df.loc[:, 'total_net_fees'] + \
                                                      income_df.loc[:, 'contract_value']

    sim_port_income.loc[:, 's_{}'.format(str(runs))] = fia_portfolio_df.loc[:, 'income']

    fia_portfolio_df.loc[:, 'adj_total'] = fia_portfolio_df.loc[:, 'total_net_fees'].apply(
        lambda x: x if x > 0 else 0)

    sim_port_total_pre_income.loc[:, 's_{}'.format(str(runs))] = fia_portfolio_df.loc[:, 'total_pre_income']

    # -------------------Portfolio with PreIncome Values----------------------------
    sim_port_total_preincome.loc[:, 's_{}'.format(str(runs))] = pre_income_port_df.loc[:, 'total_net_fees'] + \
                                                                income_df.loc[:, 'contract_value']

    sim_port_total_preincome.fillna(float(read_income_inputs.loc['risky_assets', 'FIA']), inplace=True)
    
    # --------------------------------PreIncome Block Ends----------------------------

    # ------------------Calculate % of portfolios ending value greater than required LIFETIME cumm. income---------
    total_income_by_age = sim_base_income.loc[:, sim_base_income.columns[0]].cumsum()
    total_income_by_acturial_age = total_income_by_age.loc[life_expectancy - clients_age]
    total_income_by_age.fillna(0, inplace=True)
    income_dataframe = pd.DataFrame(total_income_by_age)
    income_dataframe.loc[:, 'remaining_income_by_acturial_age'] = total_income_by_age.apply(
        lambda x: total_income_by_acturial_age - x)

    s = income_dataframe.loc[:, 'remaining_income_by_acturial_age']
    base_prob_of_success = sim_base_total.gt(s, axis=0).sum(axis=1)
    port_prob_of_success = sim_port_total.gt(s, axis=0).sum(axis=1)

    # ----------------------------Portfolio sufficient for NEXT YEARS income needs-------------------
    next_year_income = sim_base_income.loc[:, sim_base_income.columns[0]].shift(-1).fillna(0)  # Yearly Income Reqd.
    base_success_next_year = sim_base_total.gt(next_year_income, axis=0).sum(axis=1)

    base_for_next_year_need = sim_base_total[sim_base_total.gt(next_year_income, axis=0)]

    port_success_next_year = sim_port_total.gt(next_year_income, axis=0).sum(axis=1)

    port_for_next_year_need = sim_port_total[sim_port_total.gt(next_year_income, axis=0)]

    # ---------------Portfolio for 45 years of simulation---------------------------------------
    base_success_portfolio = sim_base_total[sim_base_total.gt(next_year_income, axis=0)]
    port_success_portfolio = sim_port_total[sim_port_total.gt(next_year_income, axis=0)]

    # ----------------Portfolio Simulation until the acturial age------------------------------
    acturial_years = life_expectancy - clients_age
    base_success_portfolio_act_age = base_success_portfolio.loc[acturial_years, :]
    port_success_portfolio_act_age = port_success_portfolio.loc[acturial_years, :]

    # -------------------------Base Portfolio TS with max Terminal Value ----------------------------
    if base_success_portfolio_act_age.isnull().sum() == trials:
        base_max_portfolio = 0.0
    else:
        base_max_portfolio = base_success_portfolio.loc[:, base_success_portfolio_act_age.idxmax()]

    # -------------------------FIA Portfolio TS with max Terminal Value ----------------------------
    if port_success_portfolio_act_age.isnull().sum() == trials:
        port_max_portfolio = 0.0
    else:
        port_max_portfolio = port_success_portfolio.loc[:, port_success_portfolio_act_age.idxmax()]

    # ------------------------------Average age with full income------------------------------
    base_mean_age = ((life_expectancy - clients_age) - base_success_portfolio.loc[1:life_expectancy - clients_age, :]
                     .isnull().sum()).mean()

    port_mean_age = ((life_expectancy - clients_age) - port_success_portfolio.loc[1:life_expectancy - clients_age, :]
                     .isnull().sum()).mean()

    # ----------------------------Median Age with full Income------------------------------------------
    base_median_age = ((life_expectancy - clients_age) - base_success_portfolio.loc[1:life_expectancy - clients_age, :]
                       .isnull().sum()).median()

    port_median_age = ((life_expectancy - clients_age) - port_success_portfolio.loc[1:life_expectancy - clients_age, :]
                       .isnull().sum()).median()

    # --------------Mean Value for all the portfolios at end of the acturial age--------------------
    base_act_avg_porfolio = base_success_portfolio.loc[life_expectancy - clients_age, :].fillna(0).mean()
    port_act_avg_porfolio = port_success_portfolio.loc[life_expectancy - clients_age, :].fillna(0).mean()

    # --------------Median Value for all the portfolios at end of the acturial age--------------------
    base_act_median_porfolio = base_success_portfolio.loc[life_expectancy - clients_age, :].fillna(0).median()
    port_act_median_porfolio = port_success_portfolio.loc[life_expectancy - clients_age, :].fillna(0).median()

    # # --------------Mean Value for all the portfolios in the simulation--------------------
    # base_sim_mean = base_success_portfolio.loc[1:life_expectancy - clients_age, :].mean().mean()
    # port_sim_mean = port_success_portfolio.loc[1:life_expectancy - clients_age, :].mean().mean()
    #
    # # --------------Median Value for all the portfolios in the simulation--------------------
    # base_sim_mean = base_success_portfolio.loc[1:life_expectancy - clients_age, :].mean().median()
    # port_sim_mean = port_success_portfolio.loc[1:life_expectancy - clients_age, :].mean().median()

    # -------Max Portfolio value at the end of acturial age----------------------------------------
    base_act_max = base_success_portfolio.loc[life_expectancy - clients_age, :].max()
    port_act_max = port_success_portfolio.loc[life_expectancy - clients_age, :].max()

    # -------Min Portfolio value at the end of acturial age----------------------------------------
    base_act_min = base_success_portfolio.loc[life_expectancy - clients_age, :].min()
    port_act_min = port_success_portfolio.loc[life_expectancy - clients_age, :].min()

    # ---------------------Lifetime Average Income----------------------------------
    base_total_income = sim_base_income.cumsum().loc[acturial_years, :].mean()
    port_total_income = income_from_fia + sim_port_income
    port_total_income = port_total_income.cumsum().loc[acturial_years, :].mean()

    simulation_stats = pd.DataFrame(index=['Average Years', 'Median Years', 'Average Age', 'Median Age',
                                           'Average Portfolio (act.age)', 'Median Portfolio (act.age)',
                                           'Max Portfolio Value', 'Min Portfolio Value',
                                           'Average Lifetime Income'], columns=['Base', 'FIA'])

    simulation_stats.loc['Average Years', :] = [base_mean_age, base_mean_age]
    simulation_stats.loc['Median Years', :] = [base_median_age, base_median_age]
    simulation_stats.loc['Average Age', :] = [base_mean_age + clients_age, base_mean_age + clients_age]
    simulation_stats.loc['Median Age', :] = [base_median_age + clients_age, base_median_age + clients_age]
    simulation_stats.loc['Average Portfolio (act.age)', :] = [base_act_avg_porfolio, port_act_avg_porfolio]
    simulation_stats.loc['Median Portfolio (act.age)', :] = [base_act_median_porfolio, port_act_median_porfolio]
    simulation_stats.loc['Max Portfolio Value', :] = [base_act_max, port_act_max]
    simulation_stats.loc['Min Portfolio Value', :] = [base_act_min, port_act_min]
    simulation_stats.loc['Average Lifetime Income', :] = [base_total_income, port_total_income]
    comments = ['Average years of portfolios that meet the next years income needs for the lifetime',
                'Median years of portfolios that meet the next years income needs for the lifetime',
                'Average Clients Age',
                'Median Clients Age',
                'Average of terminal values for the portfolios at the end of the acturial life',
                'Median of terminal values for the portfolios at the end of the acturial life',
                'Maximum of terminal values for the portfolios at the end of the acturial life',
                'Minimum of terminal values for the portfolios at the end of the acturial life',
                'Average of total income generated by all portfolios at the end of the acturial life']

    simulation_stats.loc[:, 'Notes'] = comments

    # --------------------------------------------------------------------------------

    # # -----------------------------------income breakdown for Base portfolio----------------------------------
    # base_df.to_csv(src + 'base_port_detail.csv')
    # sim_base_total.to_csv(src + 'base_ending_values.csv')
    # income_breakdown_base = pd.DataFrame(sim_base_total.quantile(0.5, axis=1))
    # income_breakdown_base.loc[:, 'income_from_portfolio'] = sim_base_income.quantile(0.5, axis=1)
    # income_breakdown_base.loc[:, 'fia_income'] = 0.0
    # income_breakdown_base.loc[:, 'social_security_income'] = social
    # income_breakdown_base.loc[:, 'coupon_income'] = cpn_income_base
    #
    # income_breakdown_base.rename(columns={income_breakdown_base.columns[0]: 'portfolio_ending_value'}, inplace=True)
    # income_breakdown_base.loc[:, 'income_from_portfolio'][
    #     income_breakdown_base.loc[:, 'portfolio_ending_value'] <= 0] = 0
    # income_breakdown_base.loc[:, 'total_income'] = income_breakdown_base.loc[:, income_breakdown_base.columns[1:]].sum(
    #     axis=1)
    #
    # # --------------------------------------Block Ends-----------------------------------------------------------
    #
    # # ---------------------------------------income breakdown for FIA portfolio----------------------------------
    # fia_portfolio_df.to_csv(src + 'fia_port_detail.csv')
    # sim_port_total.to_csv(src + 'fiaport_ending_values.csv')
    #
    # income_breakdown_port = pd.DataFrame(sim_port_total.quantile(0.5, axis=1))
    # income_breakdown_port.loc[:, 'income_from_portfolio'] = sim_port_income.quantile(0.5, axis=1)
    # income_breakdown_port.loc[:, 'fia_income'] = income_from_fia
    # income_breakdown_port.loc[:, 'social_security_income'] = social
    # income_breakdown_port.loc[:, 'coupon_income'] = cpn_income_port
    #
    # income_breakdown_port.rename(columns={income_breakdown_port.columns[0]: 'portfolio_ending_value'}, inplace=True)
    # income_breakdown_port.loc[:, 'income_from_portfolio'][
    #     income_breakdown_port.loc[:, 'portfolio_ending_value'] <= 0] = 0
    # income_breakdown_port.loc[:, 'total_income'] = income_breakdown_port.loc[:, income_breakdown_port.columns[1:]].sum(
    #     axis=1)
    #
    # # ----------------------------------Block Ends-------------------------------------------------------------
    q_cut = [0.0, 0.1, 0.25, 0.5, 0.75, 0.95, 1.0]
    sim_base_income[sim_base_total < income_needed] = 0.0

    sim_port_income[sim_port_total < income_net_fia_income] = 0

    sim_port_income = sim_port_income + income_from_fia

    # base_quantile = sim_base_total.loc[sim_base_total.index[-1]].quantile([0.05, 0.25, 0.50, 0.75, 0.90])
    #
    # port_quantile = sim_port_total.loc[sim_port_total.index[-1]].quantile([0.05, 0.25, 0.50, 0.75, 0.90])

    base_quantile = sim_base_total.loc[sim_base_total.index[-1]].quantile(q_cut)

    port_quantile = sim_port_total.loc[sim_port_total.index[-1]].quantile(q_cut)

    # q_cut = [0.0, .05, 0.25, 0.5, 0.75, 0.95, 1.0]
    cols = ['Min', '10th', '25th', '50th', '75th', '90th', 'Max']

    # ------------------------------------------drop year 0-----------------------------------------
    sim_base_total = sim_base_total[1:]
    sim_port_total = sim_port_total[1:]

    # ---------------------------------plot for histogram for porfolios--------------------------------------
    # base_term_value = sim_base_total.loc[sim_base_total.index[:life_expectancy - clients_age], :]
    # fact = 1 / len(base_term_value)
    # base_ann_ret = (base_term_value.iloc[-1] / base_term_value.iloc[0]) ** fact - 1
    # counts, bins, bars = plt.hist(base_ann_ret)

    # ------------------------quantile analysis for base terminal value-----------------------------
    base_qcut = pd.DataFrame(index=sim_base_total.index, columns=cols)
    for c in range(len(cols)):
        base_qcut.loc[:, cols[c]] = sim_base_total.quantile(q_cut[c], axis=1)

    base_qcut.clip(lower=0, inplace=True)

    sim_base_total.clip(lower=0, inplace=True)

    # -------------------------------------quantile analysis for base income----------------------------
    base_income_qcut = pd.DataFrame(index=sim_base_income.index, columns=cols)
    for c in range(len(cols)):
        base_income_qcut.loc[:, cols[c]] = sim_base_income.quantile(q_cut[c], axis=1)

    # ----Remove NaN's prior to the income start years------------
    # base_income_qcut = base_income_qcut.loc[income_starts:]

    # ---------------------------------quantile analysis for portfolio terminal value ---------------

    port_qcut = pd.DataFrame(index=sim_port_total.index, columns=cols)
    for c in range(len(cols)):
        port_qcut.loc[:, cols[c]] = sim_port_total.quantile(q_cut[c], axis=1)

    port_qcut.clip(lower=0, inplace=True)

    # ----------------------------------quantile analysis for portfolio income----------------------------
    port_income_qcut = pd.DataFrame(index=sim_port_income.index, columns=cols)
    for c in range(len(cols)):
        port_income_qcut.loc[:, cols[c]] = sim_port_income.quantile(q_cut[c], axis=1)

    # ----Remove NaN's prior to the income start years------------
    # port_income_qcut = port_income_qcut.loc[income_starts:]

    # ----------probability ending value will be less than 0 at the end of the horizon -----------------------
    # base_legacy_risk = (sim_base_total.loc[sim_base_total.index[-1]] < 0).sum() / (trials)

    base_legacy_risk = (sim_base_total.loc[sim_base_total.index[life_expectancy - clients_age]] < 0).sum() / trials
    port_legacy_risk = (sim_port_total.loc[sim_port_total.index[life_expectancy - clients_age]] < 0).sum() / trials

    # port_legacy_risk = (sim_port_total.loc[sim_port_total.index[-1]] <= 0).sum() / (trials)

    legacy_risk = pd.DataFrame([base_legacy_risk, port_legacy_risk,
                                'Prob. of portfolio value less than 0 at the end of the expected life'],
                               index=['base', 'fia_portfolio', 'Notes'],
                               columns=['Ruin Probability'])

    # -----------Year-wise probability of ending value greater than 0 -----------------
    base_psuccess = sim_base_total.apply(lambda x: x > 0).sum(axis=1) / trials
    port_psuccess = sim_port_total.apply(lambda x: x > 0).sum(axis=1) / trials

    # -----------------------WRITING FILES TO EXCEL ---------------------------

    writer = pd.ExcelWriter(dest_simulation + method + '_leveled_growth_simulation.xlsx', engine='xlsxwriter')
    read_income_inputs.to_excel(writer, sheet_name='inputs_for_income')

    read_returns_est.to_excel(writer, sheet_name='asset_returns_estimates')
    # read_portfolio_inputs.to_excel(writer, sheet_name='portfolio_inputs')

    age_index = list(range(clients_age + 1, clients_age + len(base_qcut) + 1))
    # base_qcut.loc[:, 'clients_age'] = age_index
    # base_qcut.loc[:, 'comment'] = ''
    # base_qcut.loc[:, 'comment'] = np.where(base_qcut.clients_age == life_expectancy, 'expected_life', "")
    base_inv = float(read_income_inputs.loc['risky_assets', 'Base'])
    base_qcut.loc[:, 'age'] = age_index
    base_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'

    # -----------------------To start with year 0---------------------------------
    insert_col = [base_inv, base_inv, base_inv, base_inv, base_inv, base_inv,
                  base_inv, clients_age, np.nan]
    base_qcut.loc[len(base_qcut) + 1, :] = 0.0
    base_qcut = base_qcut.shift(1)
    base_qcut.iloc[0] = insert_col
    base_qcut.reset_index(drop=True, inplace=True)
    base_qcut.to_excel(writer, sheet_name='base_ending_value_quantiles')
    # base_qcut.loc[income_starts:, :].to_excel(writer, sheet_name='base_ending_value_quantiles')

    # base_income_qcut = base_income_qcut[1:] base_income_qcut.loc[:, 'clients_age'] = age_index
    # base_income_qcut.loc[:, 'comment'] = '' base_income_qcut.loc[:, 'comment'] = np.where(
    # base_income_qcut.clients_age == life_expectancy, 'expected_life', "")

    base_income_qcut = base_income_qcut.loc[1:, :]
    base_income_qcut.loc[:, 'age'] = age_index
    base_income_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    base_income_qcut.loc[income_starts:, :].to_excel(writer, sheet_name='base_income_quantiles')

    # age_index = list(range(clients_age+1, clients_age + len(port_qcut)+1))
    # port_qcut.loc[:, 'clients_age'] = age_index
    # port_qcut.loc[:, 'comment'] = ''
    # port_qcut.loc[:, 'comment'] = np.where(port_qcut.clients_age == life_expectancy, 'expected_life', "")

    port_qcut.loc[:, 'age'] = age_index
    port_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    port_qcut.loc[len(port_qcut) + 1, :] = 0.0
    port_qcut = port_qcut.shift(1)
    port_qcut.iloc[0] = insert_col
    port_qcut.reset_index(drop=True, inplace=True)
    port_qcut.to_excel(writer, sheet_name='fia_port_ending_value_quantiles')
    # port_qcut.loc[income_starts:, :].to_excel(writer, sheet_name='fia_port_ending_value_quantiles')

    # port_income_qcut = port_income_qcut[1:] port_income_qcut.loc[:, 'clients_age'] = age_index
    # port_income_qcut.loc[:, 'comment'] = '' port_income_qcut.loc[:, 'comment'] = np.where(
    # port_income_qcut.clients_age == life_expectancy, 'expected_life', "")

    port_income_qcut = port_income_qcut.loc[1:, :]
    port_income_qcut.loc[:, 'age'] = age_index
    port_income_qcut.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    port_income_qcut.loc[income_starts:, :].to_excel(writer, sheet_name='fia_port_income_quantiles')

    prob_success_df = pd.concat([base_psuccess, port_psuccess], axis=1)
    prob_success_df.rename(columns={prob_success_df.columns[0]: 'prob(ending_value>0)_base',
                                    prob_success_df.columns[1]: 'prob(ending_value>0)_port'}, inplace=True)

    # prob_success_df.loc[:, 'clients_age'] = age_index
    # prob_success_df.loc[:, 'comment'] = ''
    # prob_success_df.loc[:, 'comment'] = np.where(prob_success_df.clients_age == life_expectancy, 'expected_life', "")

    prob_success_df.loc[:, 'age'] = age_index
    prob_success_df.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    prob_success_df.loc[:, 'prob(ending_value>lifetime_req income)_base'] = base_prob_of_success / trials
    prob_success_df.loc[:, 'prob(ending_value>lifetime_req income)_port'] = port_prob_of_success / trials
    prob_success_df.loc[:, 'prob(ending_value>next_year_req_income)_base'] = base_success_next_year / trials
    prob_success_df.loc[:, 'prob(ending_value>next_year_req_income)_port'] = port_success_next_year / trials
    prob_success_df.loc[:, 'base_max_portfolio_at_acturial_age'] = base_max_portfolio
    prob_success_df.loc[:, 'port_max_portfolio_at_acturial_age'] = port_max_portfolio

    # --------------------Percentile Portfolio's based on Acturial Life------------------------
    base_success = prob_success_df.loc[life_expectancy - clients_age, 'prob(ending_value>next_year_req_income)_base']
    port_success = prob_success_df.loc[life_expectancy - clients_age, 'prob(ending_value>next_year_req_income)_port']

    # acturial_age_base_tv = sim_base_total.loc[:life_expectancy - clients_age, ]
    # percentile_base_tv = sim_base_total.apply(lambda x: np.nanpercentile(x, base_success), axis=1)

    # ----------------Year wise percentile portfolio to meet next year income. Based on the success at acturial age.
    # Yearly portfolio values that can provide the next year income below the success rate at end of life (Percentile)-

    # acturial_age_base = base_for_next_year_need.loc[:life_expectancy - clients_age, ]
    # acturial_age_base = base_for_next_year_need.copy().fillna(0)
    percentile_base = base_for_next_year_need.apply(lambda x: np.nanpercentile(x, base_success), axis=1)

    # ----Pre Income Portfolio based on the Probab. of Success to meet next year's income at the end on the Act. Age
    base_pre_income_success = sim_base_total_preincome.apply(lambda x: np.nanpercentile(x, base_success), axis=1)
    base_ann_ret_pre_income = base_pre_income_success.pct_change().fillna(0)

    # acturial_age_port_tv = sim_port_total.loc[:life_expectancy - clients_age, ]
    # percentile_port_tv = sim_port_total.apply(lambda x: np.nanpercentile(x, port_success), axis=1)

    # ----------------Year wise percentile portfolio to meet next year income. Based on the success at acturial age.
    # Yearly portfolio values that can provide the next year income below the success rate at end of life (Percentile)-

    # acturial_age_port = port_for_next_year_need.loc[:life_expectancy - clients_age, ]
    # acturial_age_base = port_for_next_year_need.copy().fillna(0)
    percentile_port = port_for_next_year_need.apply(lambda x: np.nanpercentile(x, port_success), axis=1)

    # ----Pre Income Portfolio based on the Probab. of Success to meet next year's income at the end on the Act. Age
    port_pre_income_success = sim_port_total_preincome.apply(lambda x: np.nanpercentile(x, port_success), axis=1)
    port_ann_ret_pre_income = port_pre_income_success.pct_change().fillna(0)

    prob_success_df.loc[:, 'acturial_success_percentile_base_portfolio'] = percentile_base
    prob_success_df.loc[:, 'acturial_success_percentile_port_portfolio'] = percentile_port

    prob_success_df.loc[:, 'base_pre_income_ann_ret'] = base_ann_ret_pre_income
    prob_success_df.loc[:, 'port_pre_income_ann_ret'] = port_ann_ret_pre_income

    # prob_success_df.loc[:, 'terminalVal_success_percentile_base_portfolio'] = percentile_base_tv
    # prob_success_df.loc[:, 'terminalVal_success_percentile_port_portfolio'] = percentile_port_tv

    sim_base_total_preincome.to_excel(writer, sheet_name='base_preincome_portfolios')
    # -------Add premium to year 0 value to get total portfolio value---------
    sim_port_total_preincome.iloc[0] = sim_port_total_preincome.iloc[0] + premium
    sim_port_total_preincome.to_excel(writer, sheet_name='port_preincome_portfolios')

    # -------------For Simulation slide - BASE Portfolio - Can Delete --------------------
    # base_qcut_preinc = pd.DataFrame(index=sim_base_total_preincome.index, columns=cols)
    # for c in range(len(cols)):
    #     base_qcut_preinc.loc[:, cols[c]] = sim_base_total_preincome.quantile(q_cut[c], axis=1)
    #
    # # -------------For Simulation slide - Proposed Portfolio --------------------
    # port_qcut_preinc = pd.DataFrame(index=sim_port_total_preincome.index, columns=cols)
    # for c in range(len(cols)):
    #     port_qcut_preinc.loc[:, cols[c]] = sim_port_total_preincome.quantile(q_cut[c], axis=1)
    #
    # base_qcut_preinc.to_excel(writer, sheet_name='base_preincome_quantiles')
    # port_qcut_preinc.to_excel(writer, sheet_name='port_preincome_quantiles')

    prob_success_df.to_excel(writer, sheet_name='success_probability')

    # --------------BASE - Accumulation and Income Breakdown based on the success percentile portfolio---------------
    base_df.to_csv(src + 'base_port_detail.csv')
    sim_base_total.to_csv(src + 'base_ending_values.csv')
    income_breakdown_base = pd.DataFrame(sim_base_total.quantile(base_success, axis=1))
    income_breakdown_base.loc[:, 'income_from_risky_assets'] = sim_base_income.quantile(base_success, axis=1) \
                                                               - social - cpn_income_port
    income_breakdown_base.loc[:, 'guaranteed_income'] = 0.0
    income_breakdown_base.loc[:, 'social_security_income'] = social
    income_breakdown_base.loc[:, 'coupon_income'] = cpn_income_base

    income_breakdown_base.rename(columns={income_breakdown_base.columns[0]: 'portfolio_ending_value'}, inplace=True)
    income_breakdown_base.loc[:, 'income_from_risky_assets'][
        income_breakdown_base.loc[:, 'portfolio_ending_value'] <= 0] = 0
    income_breakdown_base.loc[:, 'total_income'] = income_breakdown_base.loc[:, income_breakdown_base.columns[1:]].sum(
        axis=1)

    # ----------FIA PORTFOLIO - Accumulation and Income Breakdown based on the success percentile portfolio-----------
    fia_portfolio_df.to_csv(src + 'fia_port_detail.csv')
    sim_port_total.to_csv(src + 'fiaport_ending_values.csv')

    income_breakdown_port = pd.DataFrame(sim_port_total.quantile(port_success, axis=1))
    income_breakdown_port.loc[:, 'income_from_risky_assets'] = sim_port_income.quantile(port_success, axis=1) \
                                                               - income_from_fia - social - cpn_income_port
    income_breakdown_port.loc[:, 'guaranteed_income'] = income_from_fia
    income_breakdown_port.loc[:, 'social_security_income'] = social
    income_breakdown_port.loc[:, 'coupon_income'] = cpn_income_port

    income_breakdown_port.rename(columns={income_breakdown_port.columns[0]: 'portfolio_ending_value'}, inplace=True)
    income_breakdown_port.loc[:, 'income_from_risky_assets'][
        income_breakdown_port.loc[:, 'portfolio_ending_value'] <= 0] = 0
    income_breakdown_port.loc[:, 'total_income'] = income_breakdown_port.loc[:, income_breakdown_port.columns[1:]].sum(
        axis=1)

    # -------------------Write simulation Statistics-------------------------------------
    simulation_stats.to_excel(writer, sheet_name='simulation_statistics')

    # port_psuccess.to_excel(writer, sheet_name='fia_port_success_probability')

    income_breakdown_base = income_breakdown_base.loc[1:, :]
    income_breakdown_base.loc[:, 'age'] = age_index
    income_breakdown_base.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    income_breakdown_base.loc[income_starts:, :].to_excel(writer, sheet_name='base_income_breakdown_median')

    income_breakdown_port = income_breakdown_port.loc[1:, :]
    income_breakdown_port.loc[:, 'age'] = age_index
    income_breakdown_port.loc[life_expectancy - clients_age, 'comment'] = 'expected_life'
    income_breakdown_port.loc[income_starts:, :].to_excel(writer, sheet_name='fia_income_breakdown_median')

    legacy_risk.to_excel(writer, sheet_name='ruin_probability')

    # if method == 'normal':
    #     median_returns_normal.loc[:, 'fia_median_returns'] = median_normal_fia
    #     median_returns_normal.to_excel(writer, sheet_name='gr_port_median_normal')
    #
    # elif method == 'smallest':
    #     median_returns_smallest.loc[:, 'fia_median_returns'] = median_smallest_fia
    #     median_returns_smallest.to_excel(writer, sheet_name='gr_port_median_asc')
    #
    # else:
    #     median_returns_largest.loc[:, 'fia_median_returns'] = median_largest_fia
    #     median_returns_largest.to_excel(writer, sheet_name='gr_port_median_desc')

    # ---------------------Histogram for S&P Forecast---------------------------------------
    sp_returns = read_returns_est.loc['SPXT Index', 'Annualized Returns']
    sp_risk = read_returns_est.loc['SPXT Index', 'Annualized Risk']
    sp_random_ret = np.random.normal(loc=sp_returns, scale=sp_risk, size=10000)
    bins, data = np.histogram(sp_random_ret, bins=20)
    df_ret = pd.DataFrame(data, columns=['Return_range'])
    df_bins = pd.DataFrame(bins, columns=['Count'])
    df_hist = df_ret.join(df_bins)

    df_hist.to_excel(writer, sheet_name='sp500_histogram')
    writer.save()

    print("simulation completed....")


if __name__ == "__main__":
    run_trials = 100
    
    # income_model_user_defined_returns(45, run_trials, 'normal')
    # inputs_df = pd.read_csv(src + "income_model_inputs.csv", index_col='Items')

    inputs_df = pd.read_excel(src + "portfolio_information.xlsx", sheet_name='income_model_inputs', index_col=[0])
    annual_income = float(inputs_df.loc['annual_income', 'inputs'])
    annual_inflation = float(inputs_df.loc['inflation', 'inputs'])

    # Income model takes the number of years you need to simulate and the trials
    # Income model based on asset returns and CMA or historical estimates

    # -----Generate random returns for the assets----------
    generate_random_returns(45, run_trials, 'normal')
    #
    # # Simulation to find the median portfolio
    # generate_median_portfolio_from_simulation(45, run_trials)
    #
    # # run simulation for target portfolio - Normal
    # print("*****----Running simulation for Normal...*****")
    # target_portfolio_simulation(45, run_trials, method='normal')
    #
    # # run simulation for target portfolio - Smallest
    # print("*****----Running simulation for Ascending...*****")
    # target_portfolio_simulation(45, run_trials, method='smallest')
    #
    # # run simulation for target portfolio - Largest
    # print("*****----Running simulation for Descending...*****")
    # target_portfolio_simulation(45, run_trials, method='largest')
    #
    # # -----Generate random portfolios using median portfolio returns----------
    # portfolio_simulations_using_target_returns(45, run_trials)
    #
    # # ---Simulation using the first 20 years for portfolio returns from accumulation ----
    # simulation_using_historical_returns(num_of_years=30, trials=100, method='normal')
    # simulation_using_historical_returns(num_of_years=30, trials=100, method='largest')
    # simulation_using_historical_returns(num_of_years=30, trials=100, method='smallest')

    # Version 1 Simulation - normal returns
    print("Running simulation for Normal...")

    income_model_asset_based_portfolio_quantile(45, run_trials, 'normal')

    # Version 2: Based on the user defined S&P 500 and FIA forecasts and using the regression beta for assets. Assuming
    # constant growth for the assets.
    # income_model_asset_based_portfolio_custom(45, run_trials, 'normal', True)

    #  Model based on constant portfolio return and FIA index return
    income_model_constant_portfolio_return(45, run_trials, 'normal')

    print("Simulation Process Ends")

    # # returns sorted smallest to largest
    # print("Running simulation for Ascending...")
    # income_model_asset_based_portfolio_quantile(45, run_trials, 'smallest')
    #
    # # returns sorted largest to smallest
    # print("Running simulation for Descending...")
    # income_model_asset_based_portfolio_quantile(45, run_trials, 'largest')

    # # ------------Optimized Income--------------------------------
    # condition = 0.0
    # while condition >= 0.0:
    #     print("Simulation for annual income {}".format(annual_income))
    #     condition = optimized_income_model(annual_income, 45, 10)
    #     annual_income -= 1000
    #
    # print("Optimization Complete...")

    #  Simulation based on Portfolio Returns
    # income_model_portfolio_based(45, 100)
