from helper import *
from users.wangq.tools import *

RETURN_DICT_CACHE = 'N:\Research\Macro\Franklin\Dollar paper\Cache'


def get_returns_dict(rerun=False):
    if rerun:
        spot_chg_usd, interest_rate_ret_usd = get_raw_data()
        return_dict_dict, return_dict_dict_droppegged = get_all_returns(spot_chg_usd, interest_rate_ret_usd)
        cache_data(RETURN_DICT_CACHE + '\\return_dict_dict', return_dict_dict)
        cache_data(RETURN_DICT_CACHE + '\\return_dict_dict_droppegged', return_dict_dict_droppegged)
    else:
        return_dict_dict = read_data(RETURN_DICT_CACHE + '\\return_dict_dict')
        return_dict_dict_droppegged = read_data(RETURN_DICT_CACHE + '\\return_dict_dict_droppegged')
    return return_dict_dict, return_dict_dict_droppegged


def rank_test(return_dict_dict, return_dict_dict_droppegged):
    cross_section = developed_cty_list + emerging_cty_list

    # $hml rank using dollar pairs
    hml_betas = pd.Series({cty: pd.ols(
        y=return_dict_dict['United States']['spot_chg'][cty],
        x={'dollar': return_dict_dict['United States']['global_dollar_spot_chg'][cty]}).beta['dollar']
                           for cty in cross_section if cty != 'United States'})

    betas = {}
    for home in cross_section:
        print home
        s = pd.Series()
        for cty in cross_section:
            if cty in return_dict_dict[home]['spot_chg'].columns:
                s[cty] = pd.ols(
                    y=return_dict_dict[home]['spot_chg'][cty],
                    x={'dollar': return_dict_dict[home]['dollar_spot_chg'][cty]}).beta['dollar']
        betas[home] = s
    betas = pd.DataFrame(betas)

    beta_basket = pd.Series({
        cty: pd.ols(y=return_dict_dict[cty]['dollar_spot_chg']['IncludeAllExHome'],
                    x={'dollar': return_dict_dict['United States']['global_dollar_spot_chg'][cty]}).beta['dollar']
        for cty in developed_cty_list + emerging_cty_list if cty != 'United States'
    })
    beta_basket['United States'] = pd.ols(
        y=return_dict_dict['United States']['dollar_spot_chg']['IncludeAllExHome'],
        x={'dollar': return_dict_dict['United States']['global_dollar_spot_chg']['IncludeAllExHome']}).beta['dollar']


def r2_magnitude_test(return_dict_dict, return_dict_dict_droppegged, drop_peg=False):
    bilateral_rate_cross_section = developed_cty_list + emerging_cty_list + verdelhan_euro_list

    if drop_peg:
        bilateral_rate_cross_section = list(set(bilateral_rate_cross_section) - set(pegged_currency_list))
        return_dict_dict_copy = return_dict_dict_droppegged
    else:
        return_dict_dict_copy = return_dict_dict

    home = 'United States'

    # $hml rank using dollar pairs
    f = return_dict_dict_copy[home]['global_dollar_spot_chg']['IncludeAllExHome']
    # f = return_dict_dict_copy[home]['carry_spot_chg']['IncludeAll']
    betas = pd.Series({cty: pd.ols(
        y=return_dict_dict_copy[home]['spot_chg'][cty],
        x={'dollar': f}).beta['dollar']
                       for cty in bilateral_rate_cross_section if cty != home})

    dollar_basket = return_dict_dict_copy[home]['dollar_spot_chg']['IncludeAllExHome']
    avg_beta = pd.ols(y=dollar_basket, x={'dollar': f}).beta['dollar']
    eps_dollar = pd.ols(y=dollar_basket,
                        x=f).resid

    df = pd.DataFrame()
    for cty in bilateral_rate_cross_section:
        if cty != home:
            model = pd.ols(
                y=return_dict_dict_copy['United States']['spot_chg'][cty],
                x={'dollar': dollar_basket})
            denom = model.x['dollar'].var()
            ind = model.x['dollar'].index
            df.loc[cty, 'dollar_eps'] = eps_dollar[ind].var() / denom
            df.loc[cty, 'betaf'] = betas[cty] * avg_beta * f[ind].var() / denom
            df.loc[cty, 'cov'] = model.beta['dollar'] - df.loc[cty, 'dollar_eps'] - df.loc[cty, 'betaf']
    return df


# PRICING ANALYSIS
# Table 2+3+5+6
def pricing_tables():
    spot_chg_usd, interest_rate_ret_usd = get_raw_data()
    task_data = {}
    for home in developed_cty_list + emerging_cty_list:
        task_data[home] = FunctionCall('gaa_macro_research.papers.dollar_paper.paper_tables.table_2_3_wrapper',
                                       spot_chg_usd=spot_chg_usd,
                                       interest_rate_ret_usd=interest_rate_ret_usd,
                                       home=home,
                                       start_date=START_DATE,
                                       end_date=VERDELHAN_END_DATE,
                                       drop_pegged_currencies=False,
                                       small_before_big_for_dollar=True
                                       )
    task_data['United States#no peg'] = FunctionCall(
        'gaa_macro_research.papers.dollar_paper.paper_tables.table_2_3_wrapper',
        spot_chg_usd=spot_chg_usd,
        interest_rate_ret_usd=interest_rate_ret_usd,
        home='United States',
        start_date=START_DATE,
        end_date=VERDELHAN_END_DATE,
        drop_pegged_currencies=True,
        small_before_big_for_dollar=True
        )
    runner = run_tasks_blocking(task_data, name='pricing analysis in dollar paper',
                                production=CLUSTER_PROD,
                                concurrency_limit=150, code_path=MY_CODE_PATH)
    results = runner.get_all_results()

    def _override_no_peg_label(key):
        if '#' in key:
            return 'United States'
        else:
            return key
    # Table 2+3
    dollar_sorted_port_output = pd.DataFrame({
        key: data['dollar_sorted_port_df'][_override_no_peg_label(key)] for key, data in
        results.iteritems()
    })
    dollar_sorted_port_output.T.to_clipboard()
    mono_test_table_2 = _run_mono_test_on_averages(results, 'dollar_sorted_port_ret')
    mono_test_table_2.T.to_clipboard()

    # Table 5
    correl = pd.DataFrame({
        key: data['correl'].ix[key] for key, data in results.iteritems()
    })
    correl.T.to_clipboard()

    # Table 6
    dollar_sorted_timed_port_output = pd.DataFrame({
        key: data['dollar_sorted_timed_port_df'][_override_no_peg_label(key)] for key, data in
        results.iteritems()
    })
    dollar_sorted_timed_port_output.T.to_clipboard()

    mono_test_table_6 = _run_mono_test_on_averages(results, 'dollar_sorted_timed_port_ret')
    mono_test_table_6.T.to_clipboard()

    # Version that drops pegs
    task_data = {}
    task_data['Verhelhan 1988-2010'] = FunctionCall(
        'gaa_macro_research.papers.dollar_paper.paper_tables.table_2_3_wrapper',
        spot_chg_usd=spot_chg_usd,
        interest_rate_ret_usd=interest_rate_ret_usd,
        home='United States',
        start_date='1988-11-30',
        end_date='2010-12-31'
    )
    task_data['Exclude pegged currencies 1988-2018'] = FunctionCall(
        'gaa_macro_research.papers.dollar_paper.paper_tables.table_2_3_wrapper',
        spot_chg_usd=spot_chg_usd,
        interest_rate_ret_usd=interest_rate_ret_usd,
        home='United States',
        drop_pegged_currencies=True,
        start_date='1988-11-30',
        end_date='2018-4-30'
    )
    task_data['Exclude pegged currencies 1988-2010'] = FunctionCall(
        'gaa_macro_research.papers.dollar_paper.paper_tables.table_2_3_wrapper',
        spot_chg_usd=spot_chg_usd,
        interest_rate_ret_usd=interest_rate_ret_usd,
        home='United States',
        drop_pegged_currencies=True,
        start_date='1988-11-30',
        end_date='2010-12-31'
    )
    task_data['Verdelhan 1988-2018'] = FunctionCall(
        'gaa_macro_research.papers.dollar_paper.paper_tables.table_2_3_wrapper',
        spot_chg_usd=spot_chg_usd,
        interest_rate_ret_usd=interest_rate_ret_usd,
        home='United States',
        start_date='1988-11-30',
        end_date='2018-4-30'
    )

    runner = run_tasks_blocking(task_data, name='Table 6 in dollar paper',
                                production=CLUSTER_PROD,
                                concurrency_limit=150, code_path=MY_CODE_PATH)
    results_table6 = runner.get_all_results()
    dollar_sorted_timed_port_output_table6 = pd.DataFrame({
        key: data['dollar_sorted_timed_port_df']['United States']
        for
        key, data in
        results_table6.iteritems()
    })
    dollar_sorted_port_output_table6 = pd.DataFrame({
        key: data['dollar_sorted_port_df']['United States'] for
        key, data
        in
        results_table6.iteritems()
    })


def table_4():
    return_dict_dict, return_dict_dict_droppegged = get_returns_dict(rerun=False)

    # Table 4
    home = 'United States'
    return_dict = return_dict_dict[home]

    beta = get_currency_betas(spot_chg=return_dict['spot_chg'],
                              dollar_spot_chg=return_dict['dollar_spot_chg'],
                              carry_spot_chg=return_dict['carry_spot_chg'],
                              interest_rate_ret=return_dict['interest_rate_ret'],
                              rolling_window=60, min_periods=31)
    sorted_portfolios = get_sorted_portfolio_returns(beta, return_dict['excess_ret'])['views']
    rank_df = pd.DataFrame({
        'TS mean': beta.rank(1).mean(),
        'TS Stdev': beta.rank(1).std(),
        '% in Portfolio 1': sorted_portfolios[0].count(),
        '% in Portfolio 2': sorted_portfolios[1].count(),
        '% in Portfolio 3': sorted_portfolios[2].count(),
        '% in Portfolio 4': sorted_portfolios[3].count(),
        '% in Portfolio 5': sorted_portfolios[4].count(),
        '% in Portfolio 6': sorted_portfolios[5].count()
    })
    rank_df.to_clipboard()

    # Figure 1
    beta_ranks = beta.rank(1)
    carry_ranks = return_dict['interest_rate_ret'].rank(1)
    correl_dollar_carry = beta_ranks.corrwith(carry_ranks, 1)
    interest_rate_differential = (return_dict['interest_rate_ret'][
                                      [x for x in developed_cty_list + euro_area_list if
                                       x in return_dict['interest_rate_ret'].columns]
                                  ].mean(1))
    correl_df = pd.DataFrame({
        'interest rate differential': interest_rate_differential,
        'dollar carry correl': correl_dollar_carry
    }).dropna()
    correl_df.to_clipboard()


# Table 7
# alpha of carry-timed dollar HML to carry
def table_7():
    spot_chg_usd, interest_rate_ret_usd = get_raw_data()
    task_data = {}

    # no pegs (panel A1)
    for home in developed_cty_list + emerging_cty_list:
        task_data[home] = FunctionCall(
            'gaa_macro_research.papers.dollar_paper.paper_tables.table_8_wrapper',
            spot_chg_usd=spot_chg_usd,
            interest_rate_ret_usd=interest_rate_ret_usd,
            home=home,
            drop_pegged_currencies=True
        )

    runner = run_tasks_blocking(task_data, name='Table 8 in dollar paper',
                                concurrency_limit=150, code_path=MY_CODE_PATH)
    results = runner.get_all_results()

    results['United States']['carry'].to_clipboard()

    # all currencies (panels A and B)

    for home in developed_cty_list + emerging_cty_list:
        task_data[home] = FunctionCall(
            'gaa_macro_research.papers.dollar_paper.paper_tables.table_8_wrapper',
            spot_chg_usd=spot_chg_usd,
            interest_rate_ret_usd=interest_rate_ret_usd,
            home=home
        )

    runner = run_tasks_blocking(task_data, name='Table 8 in dollar paper',
                                concurrency_limit=150, code_path=MY_CODE_PATH)
    results = runner.get_all_results()

    results['United States']['carry'].to_clipboard()
    results['United States']['carry + dynamic carry'].to_clipboard()

    # beta

    df_carry_beta = pd.DataFrame({home: results[home]['carry'].ix['beta#carry']
                                  for home in results.keys()})
    df_carry_beta.T.to_clipboard()

    df_carry_beta_t = pd.DataFrame({home: results[home]['carry'].ix['t#carry']
                                    for home in results.keys()})
    df_carry_beta_t.T.to_clipboard()

    # intercept / alpha

    df_intercept = pd.DataFrame({home: results[home]['carry'].ix['beta#intercept']
                                 for home in results.keys()})
    df_intercept.T.to_clipboard()

    df_intercept_t = pd.DataFrame({home: results[home]['carry'].ix['t#intercept']
                                   for home in results.keys()})
    df_intercept_t.T.to_clipboard()

    # alpha of dollar carry over carry (w/ home)
    df = pd.DataFrame({cty: results[cty]['dollar carry on carry'][0] for cty in results.keys()})


def residuals_correlation_test(return_dict_dict, return_dict_dict_droppegged, drop_peg=False):
    if drop_peg:
        return_dict_dict_copy = return_dict_dict_droppegged
        cross_section = list(set(developed_cty_list + emerging_cty_list) - set(pegged_currency_list))
    else:
        return_dict_dict_copy = return_dict_dict
        cross_section = developed_cty_list + emerging_cty_list

    global_dollar_basket_home = 'United States'
    resid1 = {}
    resid2 = {}
    resid3 = {}
    for home in cross_section:
        print home
        basket_ret = return_dict_dict_copy[home]['dollar_spot_chg']['IncludeAllExHome']
        carry_ret = return_dict_dict_copy[home]['carry_spot_chg']['IncludeAllExHome']

        if home == global_dollar_basket_home:
            dollarhml_ret = return_dict_dict_copy[global_dollar_basket_home]['global_dollar_spot_chg'][
                'IncludeAllExHome']
        else:
            dollarhml_ret = return_dict_dict_copy[global_dollar_basket_home]['global_dollar_spot_chg'][home]
        resid1[home] = pd.ols(y=basket_ret,
                              x={'carry': carry_ret}).resid
        resid2[home] = pd.ols(y=basket_ret,
                              x={'dollarHML': dollarhml_ret}).resid
        resid3[home] = pd.ols(y=basket_ret,
                              x={'carry': carry_ret,
                                 'dollarHML': dollarhml_ret}).resid
    resid1 = pd.DataFrame(resid1)
    resid2 = pd.DataFrame(resid2)
    resid3 = pd.DataFrame(resid3)
    cols = [
        # pegs
        'China Hong Kong', 'Kuwait', 'Saudi Arabia', 'United Arab Emirates', 'United States',
        # americas
        'Canada', 'Mexico',
        # europe
        'Euro Area', 'Czech Republic', 'Denmark', 'Hungary', 'Norway', 'Poland', 'Sweden', 'Switzerland',
        'United Kingdom',
        # asia
        'India', 'Indonesia', 'Japan', 'Malaysia', 'Philippines', 'Singapore', 'Taiwan', 'Thailand', 'South Korea',
        # AU/NZ
        'Australia', 'New Zealand',
        # others
        'South Africa', 'Turkey'
    ]
    resid1.corr().loc[cols, cols].to_clipboard()


def table_1(return_dict_dict, return_dict_dict_droppegged):
    cross_section = developed_cty_list + emerging_cty_list

    bilateral_rate_cross_section = developed_cty_list + emerging_cty_list + verdelhan_euro_list

    task_data = {}
    for home in cross_section:
        print home
        task_data[home] = FunctionCall(
            'gaa_macro_research.papers.dollar_paper.paper_tables.table_1a_wrapper',
            home=home, return_dict_dict=return_dict_dict,
            return_dict_dict_droppegged=return_dict_dict_droppegged,
            cross_section=cross_section,
            bilateral_rate_cross_section=bilateral_rate_cross_section,
            start_date=START_DATE,
            end_date=VERDELHAN_END_DATE
        )
    runner = run_tasks_blocking(task_data, name='Table 1 crosses in dollar paper',
                                production=CLUSTER_PROD,
                                concurrency_limit=150, code_path=MY_CODE_PATH)
    results = runner.get_all_results()
    r2 = pd.Panel(results)

    # Table 1a
    r2_usd = r2['United States'].T.copy()
    r2_usd['Pegs'] = r2_usd[[x for x in r2_usd.columns if x in pegged_currency_list]].mean(1)
    r2_usd['AllexUS,JP,UK,pegs'] = r2_usd[[x for x in r2_usd.columns if x not in
                                           ['United States', 'Japan',
                                            'United Kingdom'] + pegged_currency_list]].mean(1)
    r2_usd.T.to_clipboard()

    # Table 1b
    output = pd.DataFrame()
    for regression_label in ['R2_carry', 'R2_dollar', 'R2_globaldollar', 'R2_carryglobaldollar',
                             'R2_carryglobaldollar_wbase']:
        for data_type in ['mean', 'median', '90th', '10th']:
            df = r2.loc[:, :, regression_label + '#' + data_type]
            for home in df.columns:
                output.loc[home, '#'.join(['home', regression_label, data_type])] = df.loc[home, home]
                output.loc[home, '#'.join(['nonhome', regression_label, data_type])] = \
                    df[home].ix[df.index != home].mean()
    output.loc['Developed average', :] = output.loc[developed_cty_list + verdelhan_euro_list, :].mean(0)
    output.loc['Emerging average', :] = output.loc[emerging_cty_list, :].mean(0)
    output.to_clipboard()
    return r2, r2_usd.T, output


# monotonicity test on averages
def _run_mono_test_on_averages(results, ret_label):
    dollar_sorted_port_ret = pd.Panel({
        key: data[ret_label] for key, data in
        results.iteritems()
    })
    mono_test = pd.DataFrame({
        'Developed Average': pd.Series(monotonicity_test(
            dollar_sorted_port_ret.loc[developed_cty_list, :, :].mean(0))).combine_first(
            dollar_sorted_port_ret.loc[developed_cty_list, :, :].mean(0).mean().add_prefix('mean#') * 12
        ),
        'Emerging Average': pd.Series(monotonicity_test(
            dollar_sorted_port_ret.loc[emerging_cty_list, :, :].mean(0))).combine_first(
            dollar_sorted_port_ret.loc[emerging_cty_list, :, :].mean(0).mean().add_prefix('mean#') * 12
        ),
        'Grand Average': pd.Series(monotonicity_test(
            dollar_sorted_port_ret.loc[developed_cty_list + emerging_cty_list, :, :].mean(0))).combine_first(
            dollar_sorted_port_ret.loc[developed_cty_list + emerging_cty_list, :, :].mean(0).mean().add_prefix(
                'mean#') * 12
        )
    })
    return mono_test


def table_8_wrapper(spot_chg_usd, interest_rate_ret_usd, home, drop_pegged_currencies=False):
    return_dict = get_returns(spot_chg_usd.ix[: VERDELHAN_END_DATE], interest_rate_ret_usd.ix[: VERDELHAN_END_DATE],
                              home, drop_pegged_currencies=drop_pegged_currencies,
                              cross_section=developed_cty_list + euro_area_list + emerging_cty_list)

    beta = get_currency_betas(spot_chg=return_dict['spot_chg'],
                              dollar_spot_chg=return_dict['dollar_spot_chg'],
                              carry_spot_chg=return_dict['carry_spot_chg'],
                              interest_rate_ret=return_dict['interest_rate_ret'],
                              rolling_window=60, min_periods=31)

    sorted_portfolios_output = get_sorted_portfolio_returns(beta, return_dict['excess_ret'])
    sorted_portfolios = sorted_portfolios_output['views']
    sorted_portfolios_returns = sorted_portfolios_output['returns']

    cty_list = (developed_cty_list + euro_area_list) if home in (
            developed_cty_list + euro_area_list) else emerging_cty_list
    carry_timer = ((return_dict['interest_rate_ret'][
                        [x for x in cty_list if x in return_dict['interest_rate_ret'].columns]
                    ].mean(1) > 0) * 2 - 1).shift(1)

    interest_rate_diff = return_dict['interest_rate_ret'][
        [x for x in cty_list if x in return_dict['interest_rate_ret'].columns]
    ].mean(1)

    carry_ret_hml = return_dict['carry_excess_ret']['IncludeAllExHome']

    dollar_ret = sorted_portfolios_returns.mean(1)
    timed_dollar_ret = dollar_ret * carry_timer

    timed_dollar_hml_ret = (sorted_portfolios_returns[5] - sorted_portfolios_returns[0]) * carry_timer

    df = pd.DataFrame({
        '1. Dollar': return_dict['dollar_excess_ret']['IncludeAllExHome'],
        '2. Dollar HML': sorted_portfolios_returns[5] - sorted_portfolios_returns[0],
        '3. Dollar Carry': return_dict['dollar_excess_ret']['IncludeAllExHome'] * carry_timer,
        '4. Dollar Carry HML': timed_dollar_hml_ret,
        '5. Pure Carry HML': return_dict['carry_excess_ret']['IncludeAllExHome'],
        '6. Carry HML': return_dict['carry_excess_ret']['IncludeAll']
    })
    df.ix[:'2010'].corr().to_clipboard()

    # 1-6
    df1 = pd.DataFrame()
    for ind in range(6):
        model = pd.ols(y=sorted_portfolios_returns[ind] * carry_timer, x={'carry': carry_ret_hml})
        df1[ind] = model.beta.add_prefix('beta#').combine_first(model.t_stat.add_prefix('t#'))
    model = pd.ols(y=timed_dollar_hml_ret, x={'carry': carry_ret_hml})
    df1['HML'] = model.beta.add_prefix('beta#').combine_first(model.t_stat.add_prefix('t#'))

    df2 = pd.DataFrame()
    for ind in range(6):
        model = pd.ols(y=sorted_portfolios_returns[ind] * carry_timer,
                       x={'carry': carry_ret_hml,
                          'interest rate * carry': carry_ret_hml * interest_rate_diff.abs()})
        df2[ind] = model.beta.add_prefix('beta#').combine_first(model.t_stat.add_prefix('t#'))
    model = pd.ols(y=timed_dollar_hml_ret,
                   x={'carry': carry_ret_hml,
                      'interest rate * carry': carry_ret_hml * interest_rate_diff.abs()})
    df2['HML'] = model.beta.add_prefix('beta#').combine_first(model.t_stat.add_prefix('t#'))

    df3 = pd.DataFrame()
    for ind in range(6):
        model = pd.ols(y=sorted_portfolios_returns[ind] * carry_timer,
                       x={'carry': carry_ret_hml,
                          'interest rate * carry': carry_ret_hml * interest_rate_diff.abs(),
                          'dollar carry': timed_dollar_ret
                          }
                       )
        df3[ind] = model.beta.add_prefix('beta#').combine_first(model.t_stat.add_prefix('t#'))
    model = pd.ols(y=timed_dollar_hml_ret,
                   x={'carry': carry_ret_hml,
                      'interest rate * carry': carry_ret_hml * interest_rate_diff.abs(),
                      'dollar carry': timed_dollar_ret
                      })
    df3['HML'] = model.beta.add_prefix('beta#').combine_first(model.t_stat.add_prefix('t#'))

    model = pd.ols(y=timed_dollar_ret,
                   x={'carry': return_dict['carry_excess_ret']['IncludeAll']
                      }
                   )
    dollar_carry_on_carry = model.beta.add_prefix('beta#').combine_first(model.t_stat.add_prefix('t#'))

    return ({
        'carry': df1,
        'carry + dynamic carry': df2,
        'carry + dynamic carry + dollar carry': df3,
        'dollar carry on carry': dollar_carry_on_carry
    })


def table_9_ashwin(spot_chg_usd, interest_rate_ret_usd):
    home = 'United States'
    return_dict = get_returns(spot_chg_usd.ix[: VERDELHAN_END_DATE], interest_rate_ret_usd.ix[: VERDELHAN_END_DATE],
                              home, drop_pegged_currencies=False,
                              cross_section=developed_cty_list + euro_area_list + emerging_cty_list)

    cty_list = (developed_cty_list + euro_area_list) if home in (
            developed_cty_list + euro_area_list) else emerging_cty_list
    carry_timer = ((return_dict['interest_rate_ret'][
                        [x for x in cty_list if x in return_dict['interest_rate_ret'].columns]
                    ].mean(1) > 0) * 2 - 1).shift(1)

    currency_excess_ret_mean = return_dict['excess_ret'].mean() * 12
    interest_rate_ret_mean = return_dict['interest_rate_ret'].mean() * 12
    df = pd.DataFrame({
        'Excess return vs. USD': currency_excess_ret_mean,
        'Carry-timed Excess return vs. USD': (return_dict['excess_ret'] * carry_timer).mean() * 12,
        'Interest rate differential vs. USD': interest_rate_ret_mean
    })


# Computes R2 of exchange rate pairs regressed on carry and dollar and report average R2
def table_1_wrapper(home, return_dict_dict,
                    cross_section,
                    start_date,
                    end_date):
    print home
    return_dict = return_dict_dict[home]
    filter = [x for x in cross_section if (x != home) and (x in return_dict['spot_chg'].columns)]
    df = table_1_regression(return_dict['spot_chg'][filter].ix[start_date: end_date],
                            return_dict['interest_rate_ret'][filter].ix[start_date: end_date],
                            return_dict['carry_spot_chg'].ix[start_date: end_date],
                            return_dict['dollar_spot_chg'].ix[start_date: end_date],
                            return_dict['global_dollar_spot_chg'].ix[start_date: end_date],
                            home)
    r2 = df.T.mean(0)
    return r2


# similar to table_1_wrapper but allows different base currency on LHS and RHS factor returns
# similar to Table 3 in Aloosh Bekaert
def table_1a_wrapper(home,
                     return_dict_dict,
                     return_dict_dict_droppegged,
                     cross_section,
                     bilateral_rate_cross_section,
                     start_date,
                     end_date):
    r2_summary = generate_r2_summary_table(home, return_dict_dict, cross_section, bilateral_rate_cross_section,
                                           start_date, end_date)
    r2_summary_droppegged = generate_r2_summary_table(
        home, return_dict_dict_droppegged, cross_section, bilateral_rate_cross_section, start_date, end_date)
    return pd.concat([r2_summary.T, r2_summary_droppegged.T.add_prefix('HMLdroppeg_')], axis=1)


# nagel letter: JPY/GBP-denominated exchange rate R2
def nagel_letter_table(return_dict_dict):
    home = 'United States'
    base = 'United Kingdom'
    start_date = START_DATE
    end_date = VERDELHAN_END_DATE

    return_dict_home = return_dict_dict[home]
    return_dict_base = return_dict_dict[base]
    bilateral_rate_cross_section = developed_cty_list + emerging_cty_list + euro_area_list
    filter = [x for x in bilateral_rate_cross_section if (x != base) and
              (x in return_dict_base['spot_chg'].columns)]

    dollar_spot_chg = return_dict_home['dollar_spot_chg'].copy()
    dollar_spot_chg[home] = dollar_spot_chg['IncludeAllExHome']

    global_dollar_spot_chg = return_dict_home['global_dollar_spot_chg'].copy()
    global_dollar_spot_chg[home] = global_dollar_spot_chg['IncludeAllExHome']

    df = table_1_regression(return_dict_base['spot_chg'][filter].ix[start_date: end_date],
                            return_dict_base['interest_rate_ret'][filter].ix[start_date: end_date],
                            return_dict_base['carry_spot_chg'].ix[start_date: end_date],
                            # factor returns constructed for home (not base!)
                            dollar_spot_chg.ix[start_date: end_date],
                            global_dollar_spot_chg.ix[start_date: end_date],
                            base, report_tstats_for_globaldollar=True)
    df.T.to_clipboard()


def generate_r2_summary_table(home,
                              return_dict_dict,
                              cross_section,
                              bilateral_rate_cross_section,
                              start_date,
                              end_date):
    print home
    return_dict_home = return_dict_dict[home]
    r2_summary = {}
    for base in cross_section:
        print '===', base, '==='
        return_dict_base = return_dict_dict[base]

        # all pairs with base
        filter = [x for x in bilateral_rate_cross_section if (x != base) and (x != home) and
                  (x in return_dict_base['spot_chg'].columns)]
        df = table_1_regression(return_dict_base['spot_chg'][filter].ix[start_date: end_date],
                                return_dict_base['interest_rate_ret'][filter].ix[start_date: end_date],
                                return_dict_base['carry_spot_chg'].ix[start_date: end_date],
                                # factor returns constructed for home (not base!)
                                return_dict_home['dollar_spot_chg'].ix[start_date: end_date],
                                return_dict_home['global_dollar_spot_chg'].ix[start_date: end_date],
                                home)

        # ex pegs
        df_pegs = pd.DataFrame()
        # if base is pegged, no need to proceed further
        if base not in pegged_currency_list:
            filter = [x for x in bilateral_rate_cross_section if (x != base) and (x != home) and
                      (x in return_dict_base['spot_chg'].columns)
                      and (x not in pegged_currency_list)]  # remove pegged currencies
            df_pegs = table_1_regression(return_dict_base['spot_chg'][filter].ix[start_date: end_date],
                                         return_dict_base['interest_rate_ret'][filter].ix[start_date: end_date],
                                         return_dict_base['carry_spot_chg'].ix[start_date: end_date],
                                         # factor returns constructed for home (not base!)
                                         return_dict_home['dollar_spot_chg'].ix[start_date: end_date],
                                         return_dict_home['global_dollar_spot_chg'].ix[start_date: end_date],
                                         home)

        r2_summary[base] = _get_distributional_stats_in_series(df).combine_first(
            _get_distributional_stats_in_series(df_pegs).add_prefix('LHSdroppeg_'))
    return pd.DataFrame(r2_summary)


# helper function to summarize df
def _get_distributional_stats_in_series(df):
    if df.shape == (0, 0):
        return pd.Series()
    return df.T.mean(0).add_suffix('#mean').combine_first(
        df.T.median(0).add_suffix('#median')).combine_first(
        df.T.quantile(q=0.9).add_suffix('#90th')).combine_first(
        df.T.quantile(q=0.1).add_suffix('#10th')).combine_first(
        df.T.quantile(q=0.75).add_suffix('#75th')).combine_first(
        df.T.quantile(q=0.25).add_suffix('#25th'))


# main wrapper for pricing analysis
def table_2_3_wrapper(spot_chg_usd, interest_rate_ret_usd, home,
                      start_date,
                      end_date,
                      drop_pegged_currencies=False,
                      small_before_big_for_dollar=True,
                      cross_section=developed_cty_list + euro_area_list + emerging_cty_list,
                      min_lookback=31,
                      carry_timer_lag=1,
                      dollar_factor_equal_weight_foreign_currencies=False
                      ):
    print home

    dollar_sorted_port_df = pd.DataFrame()
    dollar_sorted_timed_port_df = pd.DataFrame()
    correl = pd.DataFrame()

    return_dict = get_returns(
        spot_chg_usd, interest_rate_ret_usd, home, drop_pegged_currencies, cross_section,
        dollar_factor_equal_weight_foreign_currencies=dollar_factor_equal_weight_foreign_currencies)
    beta = get_currency_betas(spot_chg=return_dict['spot_chg'],
                              dollar_spot_chg=return_dict['dollar_spot_chg'],
                              carry_spot_chg=return_dict['carry_spot_chg'],
                              interest_rate_ret=return_dict['interest_rate_ret'],
                              rolling_window=60, min_periods=min_lookback)
    dollar_sorted_port_ret = get_sorted_portfolio_returns(beta, return_dict['excess_ret'],
                                                          small_before_big=small_before_big_for_dollar
                                                          )['returns'].ix[
                             start_date: end_date]
    carry_sorted_port_ret = get_sorted_portfolio_returns(return_dict['interest_rate_ret'],
                                                         return_dict['excess_ret'])['returns'].ix[
                            start_date: end_date]
    cty_list = (developed_cty_list + euro_area_list) if home in (
            developed_cty_list + euro_area_list) else emerging_cty_list
    carry_timer = ((return_dict['interest_rate_ret'][
                        [x for x in cty_list if x in return_dict['interest_rate_ret'].columns]
                    ].mean(1) > 0) * 2 - 1).shift(carry_timer_lag)

    # dollar sorted portfolio returns
    dollar_port_return_no_exclusion = return_dict['dollar_excess_ret']['IncludeAllExHome']
    portfolio_betas = get_portfolio_betas(dollar_sorted_port_ret, dollar_port_return_no_exclusion)
    mono_test = monotonicity_test(dollar_sorted_port_ret)
    dollar_sorted_port_df[home] = (dollar_sorted_port_ret.add_prefix('mean#').mean() * 12).combine_first(
        pd.Series({'nobs': dollar_sorted_port_ret.dropna().shape[0]})).combine_first(
        dollar_sorted_port_ret.add_prefix('std#').std() * 12 ** 0.5).combine_first(
        portfolio_betas.add_prefix('beta#')
    ).combine_first(pd.Series(mono_test))

    # dollar sorted portfolio, timed by carry; Verdelhan's starts in in 1988-12-30 for some reason
    dollar_sorted_timed_port_ret = (dollar_sorted_port_ret * carry_timer).ix['1988-12-30': end_date]
    dollar_timed_port_return_no_exclusion = dollar_port_return_no_exclusion * carry_timer

    portfolio_betas = get_portfolio_betas(dollar_sorted_timed_port_ret, dollar_timed_port_return_no_exclusion)
    mono_test = monotonicity_test(dollar_sorted_timed_port_ret)
    dollar_sorted_timed_port_df[home] = (
            dollar_sorted_timed_port_ret.add_prefix('mean#').mean() * 12).combine_first(
        pd.Series({'nobs': dollar_sorted_timed_port_ret.dropna().shape[0]})).combine_first(
        dollar_sorted_timed_port_ret.add_prefix('std#').std() * 12 ** 0.5).combine_first(
        portfolio_betas.add_prefix('beta#')
    ).combine_first(pd.Series(mono_test))

    carry_hml = carry_sorted_port_ret[5] - carry_sorted_port_ret[0]
    dollar_hml = dollar_sorted_port_ret[5] - dollar_sorted_port_ret[0]

    for use_dollar_hml in [True, False]:
        dollar_port_for_correl = dollar_hml if use_dollar_hml else dollar_port_return_no_exclusion
        dollar_port_for_correl = dollar_port_for_correl.reindex(carry_timer.index)
        dollar_port_label = '#hml' if use_dollar_hml else '#avg'

        correl.loc[home, 'high' + dollar_port_label] = carry_hml.corr(dollar_port_for_correl[carry_timer == 1])
        correl.loc[home, 'high#nobs' + dollar_port_label] = (
                carry_hml + dollar_port_for_correl[carry_timer == 1]).count()
        correl.loc[home, 'low' + dollar_port_label] = carry_hml.corr(dollar_port_for_correl[carry_timer == -1])
        correl.loc[home, 'low#nobs' + dollar_port_label] = (
                carry_hml + dollar_port_for_correl[carry_timer == -1]).count()
        correl.loc[home, 'Unconditional' + dollar_port_label] = carry_hml.corr(dollar_port_for_correl)
        correl.loc[home, 'Unconditional#nobs' + dollar_port_label] = (carry_hml + dollar_port_for_correl).count()

    return {
        'dollar_sorted_port_df': dollar_sorted_port_df,
        'dollar_sorted_port_ret': dollar_sorted_port_ret,
        'dollar_sorted_timed_port_df': dollar_sorted_timed_port_df,
        'dollar_sorted_timed_port_ret': dollar_sorted_timed_port_ret,
        'correl': correl
    }


def table_1_regression(excess_ret, interest_rate_ret, carry_ret, dollar_ret, global_dollar_ret, home,
                       report_tstats_for_globaldollar=False):
    df = pd.DataFrame()
    for cty in [x for x in excess_ret.columns if x != home]:
        try:
            model_carry = pd.ols(y=excess_ret[cty], x={
                'beta': interest_rate_ret[cty].shift(1),
                'gamma': interest_rate_ret[cty].shift(1) * carry_ret[cty],
                'delta': carry_ret[cty],
            }, intercept=True)
            model_dollar = pd.ols(y=excess_ret[cty], x={
                'tau': dollar_ret[cty]
            }, intercept=True)
            model_globaldollar = pd.ols(y=excess_ret[cty], x={
                'tau_global': global_dollar_ret[cty]
            }, intercept=True)
            model_carry_dollar = pd.ols(y=excess_ret[cty], x={
                'beta': interest_rate_ret[cty].shift(1),
                'gamma': interest_rate_ret[cty].shift(1) * carry_ret[cty],
                'delta': carry_ret[cty],
                'tau': dollar_ret[cty],
            }, intercept=True)
            model_carry_globaldollar = pd.ols(y=excess_ret[cty], x={
                'beta': interest_rate_ret[cty].shift(1),
                'gamma': interest_rate_ret[cty].shift(1) * carry_ret[cty],
                'delta': carry_ret[cty],
                'tau_global': global_dollar_ret[cty]
            }, intercept=True)
            model_carry_globaldollar_wbase = pd.ols(y=excess_ret[cty], x={
                'beta': interest_rate_ret[cty].shift(1),
                'gamma': interest_rate_ret[cty].shift(1) * carry_ret['IncludeAll'],
                'delta': carry_ret['IncludeAll'],
                'tau_global': global_dollar_ret['IncludeAllExHome']
            }, intercept=True)

            df_onecty = pd.Series({
                'R2_carry': model_carry.r2_adj,
                'R2_dollar': model_dollar.r2_adj,
                'R2_globaldollar': model_globaldollar.r2_adj,
                'R2_carrydollar': model_carry_dollar.r2_adj,
                'R2_carryglobaldollar': model_carry_globaldollar.r2_adj,
                'R2_carryglobaldollar_wbase': model_carry_globaldollar_wbase.r2_adj
            })

            if report_tstats_for_globaldollar:
                df_onecty['globaldollart_carryglobaldollar_wbase'] = model_carry_globaldollar_wbase.t_stat[
                    'tau_global']
                df_onecty['globaldollart_carryglobaldollar'] = model_carry_globaldollar.t_stat['tau_global']

            df[cty] = df_onecty

        except:
            print 'regression skipped for', cty, ', home=', home
    return df


# monotonicity tests

def monotonicity_test(ret):
    # ret is 6-column monthly returns
    ret_diff = pd.DataFrame({num: ret[num + 1] - ret[num] for num in range(len(ret.columns) - 1)})
    assert ret_diff.dropna().shape == ret_diff.dropna(how='all').shape
    ret_diff = ret_diff.dropna()

    T = ret_diff.shape[0]
    N = ret_diff.shape[1]

    # Boudoukh, Richardson, Smith Is the ex ante risk premium always positive?

    cov = algos.covariance(ret_diff)
    inv_cov = pd.DataFrame(np.linalg.inv(cov), index=cov.index, columns=cov.columns)

    m = ret_diff.mean(0)

    def func(m_star):
        return (m - pd.Series(m_star)).dot(inv_cov).dot(m - pd.Series(m_star))

    def _optimize_for_monotonicity(func, x0, bnds, method='L-BFGS-B'):
        res = optimize.minimize(func, x0, method=method, bounds=bnds, tol=1e-10)
        if res['success'] == False:
            res = optimize.minimize(func, x0, method=method, bounds=bnds, tol=1e-6)
        if res['success'] == False:
            res = optimize.minimize(func, x0, method=method, bounds=bnds, tol=1e-5)
        assert res['success'] == True
        return res

    # weight function for non-centered chi^2
    def weight(size=1000000):
        mean = np.array([0] * N)
        x = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
        count = (np.sign(x) + 1).sum(1) / 2
        w = pd.Series({k: (count == N - k).mean() for k in range(N + 1)})
        # weights should add up to 1
        np.testing.assert_almost_equal(w.sum(), 1)
        return w

    def cdf(c):
        p = 0
        w = weight()
        for k in range(1, N + 1):
            p += w[k] * sp.stats.chi2.cdf(c, df=k)
        assert p >= 0
        assert p <= 1
        return p

    def find_critical_value(p=0.05):
        func = lambda c: np.abs(cdf(c) - (1 - p))
        res = optimize.minimize(func, x0=10, tol=0.0001, method='Nelder-Mead')
        assert res['success'] == True
        return res['x']

    # H0: m > 0; H1: otherwise
    bnds_increase = ((0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf))
    x0 = np.array((np.sign(m) + 1) / 2 * m) + 0.00001
    assert x0.min() > 0
    res = _optimize_for_monotonicity(func, x0, bnds_increase)
    W_increase = T * func(pd.Series(res['x']))
    p_value_W_increase = 1 - cdf(W_increase)

    # H1: m < 0; H1: otherwise
    bnds_decrease = ((-np.inf, 0), (-np.inf, 0), (-np.inf, 0), (-np.inf, 0), (-np.inf, 0))
    x0 = np.array((np.sign(m) - 1).abs() / 2 * m) - 0.00001
    assert x0.max() < 0
    res = _optimize_for_monotonicity(func, x0, bnds_decrease)
    W_decrease = T * func(pd.Series(res['x']))
    p_value_W_decrease = 1 - cdf(W_decrease)

    # H0: m <= 0; H1: m > 0
    # Patton Timmerman: Monotonicity in asset returns
    delta = ret_diff.mean(0)

    J_min = delta.min()
    J_max = delta.max()

    # bootstrap for J's distribution
    def bootstrap_for_j(size=1000):
        J_b_min = []
        J_b_max = []
        for i in range(size):
            delta_bootstrapped = ret_diff.sample(T, replace=True).mean(0)
            J_b_min += [(delta_bootstrapped - delta).min()]
            J_b_max += [(delta_bootstrapped - delta).max()]
        return np.array(J_b_min), np.array(J_b_max)

    J_b_min, J_b_max = bootstrap_for_j()

    p_value_J_increase = np.mean(J_min < J_b_min)
    p_value_J_decrease = np.mean(J_max > J_b_max)

    return {
        'W increase': W_increase,
        'p value W increase': p_value_W_increase,
        'W decrease': W_decrease,
        'p value W decrease': p_value_W_decrease,
        'J increase': J_min * 12,
        'p value J increase': p_value_J_increase,
        'J decrease': J_max * 12,
        'p value J decrease': p_value_J_decrease
    }
