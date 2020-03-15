import aqr.gaa.api as api
import aqr.core.algos as algos
import pandas as pd
import numpy as np
import scipy.optimize as optimize
import scipy as sp
from cluster.api import FunctionCall, run_tasks_blocking

DATA_FILE_PATH = 'N:\Research\Macro\Franklin\Dollar paper\Data_for_AugmentedUIP_allcountries (3).xls'

developed_cty_list = ['Australia', 'Canada', 'Denmark', 'Euro Area', 'Japan', 'New Zealand', 'Norway',
                      'Sweden', 'Switzerland', 'United Kingdom', 'United States']
euro_area_list = ['Austria', 'Belgium', 'France', 'Finland', 'Greece', 'Ireland', 'Netherlands', 'Portugal',
                  'Spain', 'Germany', 'Italy']
emerging_cty_list = ['China Hong Kong', 'Czech Republic', 'Hungary', 'India', 'Indonesia',
                     'Kuwait', 'Malaysia', 'Mexico', 'Philippines', 'Poland', 'Saudi Arabia',
                     'Singapore', 'South Africa', 'South Korea', 'Taiwan', 'Thailand',
                     'Turkey', 'United Arab Emirates']
verdelhan_euro_list = ['Germany', 'France', 'Italy']

START_DATE = '1988-11-30'

VERDELHAN_END_DATE = '2010-12-31'
EXTENDED_END_DATE = '2019-5-31'

CLUSTER_PROD = False

MY_CODE_PATH = [r'\\aqrcapital.com\shares\fs011\workspace\wangq\python',
                r'\\aqrcapital.com\shares\fs011\workspace\wangq\gaa-macro-research']

# euro weights from bank of england
EURO_WEIGHTS = pd.Series({
    'Germany': 33.1,
    'Belgium': 9.2,
    'Spain': 6.7,
    'Finland': 1.5,
    'France': 19.7,
    'Ireland': 1.1,
    'Italy': 14.8,
    'Netherlands': 8.2,
    'Austria': 4.4,
    'Portugal': 1.3,
    'Greece': 0.0
}) / 100.0

# KW, SI, UA commented out because AQR doesn't have data for those
AQR_CODE_MAP = {
    'AU': 'Australia',
    'BD': 'Euro Area',
    'CN': 'Canada',
    'CZ': 'Czech Republic',
    'DK': 'Denmark',
    'HK': 'China Hong Kong',
    'HN': 'Hungary',
    'ID': 'Indonesia',
    'IN': 'India',
    'JP': 'Japan',
    'KO': 'South Korea',
    # 'KW': 'Kuwait',
    'MX': 'Mexico',
    'MY': 'Malaysia',
    'NW': 'Norway',
    'NZ': 'New Zealand',
    'PH': 'Philippines',
    'PO': 'Poland',
    'SA': 'South Africa',
    'SD': 'Sweden',
    'SG': 'Singapore',
    # 'SI': 'Saudi Arabia',
    'SW': 'Switzerland',
    'TA': 'Taiwan',
    'TH': 'Thailand',
    'TK': 'Turkey',
    # 'UA': 'United Arab Emirates',
    'UK': 'United Kingdom',
    'US': 'United States'
}
pegged_currency_list = ['United Arab Emirates', 'Saudi Arabia', 'Denmark', 'China Hong Kong',
                        'Kuwait']

DATA_END_DATE = '2018-4-30'


def get_equal_weighted_home_portfolio_returns(excess_ret, exclude=[]):
    return excess_ret.drop(exclude, axis=1).mean(1)


def identify_pegged_currencies(spot_chg_usd, peg_list=['United States', 'Euro Area'],
                               override=True):
    # override with predefined list
    if override:
        return pegged_currency_list
    spot_chg_usd = spot_chg_usd.copy()
    spot_chg_usd['United States'] = 0

    pegged_countries = set([])
    for peg in peg_list:
        spot_chg = spot_chg_usd - spot_chg_usd[peg]
        spot_chg = spot_chg.drop(peg, axis=1)
        vols = spot_chg.std() * 12 ** 0.5
        pegged_countries = pegged_countries.union(set(vols[vols <= 0.015].index))

    return list(pegged_countries)


def get_returns(spot_chg_usd, interest_rate_ret_usd, home, drop_pegged_currencies, cross_section,
                dollar_factor_equal_weight_foreign_currencies=False):
    """
    compute spot, excess returns for single currencies, dollar basket, carry basket, global dollar basket

    Parameters
    ----------
    spot_chg_usd: DataFrame
        spot FX changes vs. USD
    interest_rate_ret_usd: DataFrame
        interest rate excess return vs. USD
    home: str
        home currency
    drop_pegged_currencies: boolean
        whether to drop pegged currency
    cross_section: list
        list of currencies to be used in calculations
    dollar_factor_equal_weight_foreign_currencies: boolean
        for dollar basket, whether currencies are equal-weighted; setting this to False
        would replicated Verdelhan (2018) behavior

    Returns
    -------
    dict

    """

    # make defensive copy
    spot_chg_usd = spot_chg_usd.copy()[cross_section]
    interest_rate_ret_usd = interest_rate_ret_usd.copy()[cross_section]

    if home == 'Euro Area':
        spot_chg_usd_euro_area = (spot_chg_usd * EURO_WEIGHTS).sum(1) / (EURO_WEIGHTS + spot_chg_usd * 0).sum(1)
        spot_chg_usd['Euro Area'] = spot_chg_usd['Euro Area'].combine_first(spot_chg_usd_euro_area)
        spot_chg_usd = spot_chg_usd.drop(euro_area_list, axis=1)

        interest_rate_ret_usd_euro_area = (interest_rate_ret_usd * EURO_WEIGHTS).sum(1) / (
                EURO_WEIGHTS + interest_rate_ret_usd * 0).sum(1)
        interest_rate_ret_usd['Euro Area'] = interest_rate_ret_usd['Euro Area'].combine_first(
            interest_rate_ret_usd_euro_area)
        interest_rate_ret_usd = interest_rate_ret_usd.drop(euro_area_list, axis=1)
    elif home in euro_area_list:
        raise KeyError('home country cannot be a legacy european country such as ' + home)

    spot_chg_usd['United States'] = 0
    interest_rate_ret_usd['United States'] = 0

    spot_chg = (spot_chg_usd - spot_chg_usd[home]).drop(home, 1)
    spot_chg = spot_chg.dropna(how='all', axis=1)

    interest_rate_ret = (interest_rate_ret_usd - interest_rate_ret_usd[home]).drop(home, 1)
    interest_rate_ret = interest_rate_ret.dropna(how='all', axis=1)

    # drop pegged currencies
    if drop_pegged_currencies:
        pegged_currencies_list = identify_pegged_currencies(spot_chg_usd)
        spot_chg = spot_chg.drop([x for x in pegged_currencies_list if x != home], axis=1)
        interest_rate_ret = interest_rate_ret.drop([x for x in pegged_currencies_list if x != home], axis=1)

    excess_ret = -spot_chg + interest_rate_ret.shift(1)

    if dollar_factor_equal_weight_foreign_currencies:
        dollar_spot_chg = pd.DataFrame(
            {
                cty: get_equal_weighted_home_portfolio_returns(spot_chg, exclude=[cty])
                for cty in [x for x in excess_ret.columns]
            })
        dollar_excess_ret = pd.DataFrame(
            {
                cty: get_equal_weighted_home_portfolio_returns(excess_ret, exclude=[cty])
                for cty in [x for x in excess_ret.columns]
            })

    # this would replicate Verdelhan 2008 behavior
    else:
        dollar_spot_chg = pd.DataFrame(
            {
                cty: get_dollar_carry_portfolio_returns(interest_rate_ret, spot_chg, home,
                                                        exclude=[cty])['dollar']
                for cty in [x for x in excess_ret.columns] + ['IncludeAllExHome', 'IncludeAll']
            })
        dollar_excess_ret = pd.DataFrame(
            {
                cty: get_dollar_carry_portfolio_returns(interest_rate_ret, excess_ret, home,
                                                        exclude=[cty])['dollar']
                for cty in [x for x in excess_ret.columns] + ['IncludeAllExHome', 'IncludeAll']
            })

    carry_spot_chg = pd.DataFrame(
        {
            cty: get_dollar_carry_portfolio_returns(interest_rate_ret, spot_chg, home,
                                                    exclude=[cty])['carry']
            for cty in [x for x in excess_ret.columns] + ['IncludeAllExHome', 'IncludeAll']
        })
    carry_excess_ret = pd.DataFrame(
        {
            cty: get_dollar_carry_portfolio_returns(interest_rate_ret, excess_ret, home,
                                                    exclude=[cty])['carry']
            for cty in [x for x in excess_ret.columns] + ['IncludeAllExHome', 'IncludeAll']
        })

    # dollar long short
    dollar_beta = get_currency_betas(spot_chg=spot_chg,
                                     dollar_spot_chg=dollar_spot_chg,
                                     carry_spot_chg=carry_spot_chg,
                                     interest_rate_ret=interest_rate_ret,
                                     rolling_window=60, min_periods=31)
    global_dollar_spot_chg = pd.DataFrame(
        {
            cty: get_hml_portfolio_returns_new(dollar_beta, spot_chg,
                                               exclude=[cty])
            for cty in [x for x in excess_ret.columns] + ['IncludeAllExHome']
        })
    global_dollar_excess_ret = pd.DataFrame(
        {
            cty: get_hml_portfolio_returns_new(dollar_beta, excess_ret,
                                               exclude=[cty])
            for cty in [x for x in excess_ret.columns] + ['IncludeAllExHome']
        })

    return {
        # single currency returns
        'spot_chg': spot_chg,
        'excess_ret': excess_ret,
        'interest_rate_ret': interest_rate_ret,

        # dollar basket return
        'dollar_spot_chg': dollar_spot_chg,
        'dollar_excess_ret': dollar_excess_ret,

        # global dollar basket return
        'global_dollar_spot_chg': global_dollar_spot_chg,
        'global_dollar_excess_ret': global_dollar_excess_ret,

        # carry return
        'carry_spot_chg': carry_spot_chg,
        'carry_excess_ret': carry_excess_ret
    }


# carry: sort into 6 portfolios based on interest rates
# dollar: average of 6 carry portfolios, per Lustig (2011)
def get_dollar_carry_portfolio_returns(interest_rate_ret_raw, excess_ret_raw, home, exclude=[], number_of_portfolios=6):
    assert not (home in interest_rate_ret_raw.columns)
    assert not (home in excess_ret_raw.columns)

    interest_rate_ret = interest_rate_ret_raw.copy()
    excess_ret = excess_ret_raw.copy()

    if exclude == ['IncludeAllExHome']:
        # equivalent to passing exclude = [], which includes all countries (except home)
        # hackish setting for iterating through countries
        exclude = []
    elif exclude == ['IncludeAll']:  # explicitly include home country in dollar and carry portfolio returns
        exclude = []
        interest_rate_ret[home] = 0
        excess_ret[home] = 0

    sorted_ports = get_sorted_portfolio_returns(interest_rate_ret, excess_ret, exclude, number_of_portfolios,
                                                lag=1)
    port_ret = sorted_ports['returns']
    return {
        'carry': port_ret[number_of_portfolios - 1] - port_ret[0],
        'dollar': port_ret.mean(1)
    }


def get_hml_portfolio_returns_new(signal, excess_ret, exclude=[], number_of_portfolios=6):
    if exclude == ['IncludeAllExHome']:
        # equivalent to passing exclude = [], or not excluding any countries but not include home country
        exclude = []
    port_ret = get_sorted_portfolio_returns(signal, excess_ret, exclude, number_of_portfolios)['returns']
    return port_ret[number_of_portfolios - 1] - port_ret[0]


def get_sorted_portfolio_returns(signal, excess_ret, exclude=[], number_of_portfolios=6, small_before_big=True,
                                 lag=1):
    ranks = (signal + excess_ret.shift(-lag) * 0).drop(exclude, axis=1).rank(1)
    sorted_portfolios = {i: pd.DataFrame(np.nan, index=ranks.index, columns=ranks.columns)
                         for i in range(number_of_portfolios)}
    for date in ranks.index:
        ranks_one_date = ranks.ix[date].dropna()
        if ranks_one_date.count() < number_of_portfolios:
            continue
        portfolios_left_to_form = number_of_portfolios
        port_size_dict = break_down_portfolio_size(number_of_currencies=len(ranks_one_date),
                                                   number_of_portfolios=number_of_portfolios,
                                                   small_before_big=small_before_big)
        for i in range(number_of_portfolios):
            port_size = port_size_dict[i]
            port = ranks_one_date.nsmallest(port_size) * 0 + 1.0 / port_size
            sorted_portfolios[i].loc[date, :] = port
            ranks_one_date = ranks_one_date.drop(ranks_one_date.nsmallest(port_size).index)
            portfolios_left_to_form -= 1
        assert portfolios_left_to_form == 0
        assert len(ranks_one_date) == 0

    # make sure dates are continuous
    sorted_portfolios = {i: port.asfreq('BM') for i, port in sorted_portfolios.iteritems()}

    # check that all views add up to 1 and have corresponding returns
    for i in range(number_of_portfolios):
        sum_of_views = (sorted_portfolios[i].shift(lag) + excess_ret * 0).dropna(how='all', axis=0).sum(1)
        np.testing.assert_array_almost_equal(sum_of_views.max(), 1.00)
        np.testing.assert_array_almost_equal(sum_of_views.min(), 1.00)

    sorted_portfolio_returns = pd.DataFrame(
        {
            i: (sorted_portfolios[i].shift(lag) * (excess_ret)).sum(1)
            for i in sorted_portfolios.keys()
        }).dropna(how='all', axis=0)
    _check_for_missing_values_dates(sorted_portfolio_returns)

    return {
        'returns': sorted_portfolio_returns,
        'views': sorted_portfolios}


def break_down_portfolio_size(number_of_currencies, number_of_portfolios, small_before_big):
    port_size = {}
    smaller_size = int(number_of_currencies / number_of_portfolios)
    bigger_size = number_of_currencies - int(number_of_currencies / number_of_portfolios) * (number_of_portfolios - 1)
    for i in range(number_of_portfolios):
        if small_before_big:
            port_size[i] = smaller_size if i < number_of_portfolios - 1 else bigger_size
        else:
            port_size[i] = bigger_size if i == 0 else smaller_size
    return port_size


def get_currency_betas(spot_chg, dollar_spot_chg, carry_spot_chg, interest_rate_ret, rolling_window=60, min_periods=31):
    beta = {}
    cty_list = set(spot_chg.columns).intersection(set(dollar_spot_chg.columns))
    for cty in cty_list:
        try:
            model = pd.ols(y=spot_chg[cty], x={
                'dollar': dollar_spot_chg[cty],
                'carry': carry_spot_chg[cty],
                'interest_rate_ret': interest_rate_ret[cty].shift(1),
                'interest_rate_ret*carry': interest_rate_ret[cty].shift(1) * carry_spot_chg[cty]},
                           window_type='rolling', window=rolling_window, min_periods=min_periods)
            beta[cty] = model.beta['dollar']
        except:
            print 'Not enough data to compute beta for', cty
    return pd.DataFrame(beta)


def get_portfolio_betas(port_ret, dollar_spot_chg):
    beta = {}
    for cty in port_ret.columns:
        model = pd.ols(y=port_ret[cty], x={'dollar': dollar_spot_chg})
        beta[cty] = model.beta['dollar']
    return pd.Series(beta)


def get_all_returns(spot_chg_usd, interest_rate_ret_usd):
    # all currencies
    task_data = {}
    for home in developed_cty_list + euro_area_list + emerging_cty_list:
        task_data[home] = FunctionCall('gaa_macro_research.papers.dollar_paper.helper.get_returns',
                                       spot_chg_usd.ix[: EXTENDED_END_DATE],
                                       interest_rate_ret_usd.ix[: EXTENDED_END_DATE],
                                       home, drop_pegged_currencies=False,
                                       cross_section=developed_cty_list + euro_area_list + emerging_cty_list)
    runner = run_tasks_blocking(task_data, name='return_dict_dict',
                                concurrency_limit=150,
                                code_path=MY_CODE_PATH)
    return_dict_dict = runner.get_all_results()

    # drop pegged
    task_data = {}
    for home in developed_cty_list + euro_area_list + emerging_cty_list:
        task_data[home] = FunctionCall('gaa_macro_research.papers.dollar_paper.helper.get_returns',
                                       spot_chg_usd.ix[: EXTENDED_END_DATE],
                                       interest_rate_ret_usd.ix[: EXTENDED_END_DATE],
                                       home, drop_pegged_currencies=True,
                                       cross_section=developed_cty_list + euro_area_list + emerging_cty_list)
    runner = run_tasks_blocking(task_data, name='return_dict_dict',
                                concurrency_limit=150,
                                code_path=MY_CODE_PATH)
    return_dict_dict_droppegged = runner.get_all_results()

    return return_dict_dict, return_dict_dict_droppegged


def _check_for_missing_values_dates(sorted_portfolio_returns):
    # check for missing values, break if violated
    assert len(sorted_portfolio_returns.dropna().index) == len(sorted_portfolio_returns.index)

    # check for missing dates, warning if violated
    dates = sorted_portfolio_returns.index
    date_list = pd.date_range(start=dates[0], end=dates[-1], freq='BM')
    if len(dates) != len(date_list):
        print 'dataframe missing dates:', set(date_list) - set(dates)


def get_raw_data():
    # change in spot exchange rate from USD perspective
    spot_chg_usd = pd.read_excel(DATA_FILE_PATH, sheetname='Changes in Exchange Rates')

    # extend it with AQR data
    fx = api.Table('FX')
    fx.load()
    fx = pd.DataFrame(fx).truncate(after=DATA_END_DATE).resample('BM', how='last')
    spot_chg_usd_aqr = fx.pct_change()[AQR_CODE_MAP.keys()].rename(columns=AQR_CODE_MAP)

    spot_chg_usd = spot_chg_usd.combine_first(spot_chg_usd_aqr.truncate(before='2011-1-1'))

    # 1-month risk free return
    interest_rate_ret_usd = pd.read_excel(DATA_FILE_PATH, sheetname='Interest Rate Differences')

    sr_implied = api.Table('sr_implied')
    sr_implied.load()
    sr_implied = pd.DataFrame(sr_implied).truncate(after=DATA_END_DATE).resample('BM', how='last')
    sr_implied = sr_implied[AQR_CODE_MAP.keys()].rename(columns=AQR_CODE_MAP) / 100.0 / 12.0
    interest_rate_ret_usd_aqr = sr_implied - sr_implied['United States']

    interest_rate_ret_usd = interest_rate_ret_usd.combine_first(
        interest_rate_ret_usd_aqr.truncate(before='2011-1-1'))

    # remove spot_chg values where interest_rate_ret data is not available
    spot_chg_usd = spot_chg_usd + interest_rate_ret_usd.shift(1) * 0
    return spot_chg_usd, interest_rate_ret_usd
