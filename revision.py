import pandas as pd
import os
import numpy as np
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

from scipy import optimize
import scipy as sp


RETURNS_CACHE = os.path.expanduser("~/dollar_paper_cache")
data = pd.read_pickle(RETURNS_CACHE + '/combined_df_pickle.dms')


def get_dollar_basket_excess_returns_from_pickle():
    dic = pd.read_pickle(RETURNS_CACHE + "/return_dict_dict")
    withpeg = dic['United States']['dollar_excess_ret']['IncludeAllExHome']
    dic = pd.read_pickle(RETURNS_CACHE + "/return_dict_dict_droppegged")
    nopeg = dic['United States']['dollar_excess_ret']['IncludeAllExHome']
    output = pd.DataFrame({'withpeg': withpeg, 'nopeg': nopeg})
    return output


dollar_basket_returns = get_dollar_basket_excess_returns_from_pickle()


def get_port_ret_df(prefix, suffix):
    ret = pd.DataFrame({i: data[prefix + str(i) + suffix] for i in range(6)})
    return ret


def port_ret_summary():
    output = {}
    for prefix in ['part1_dollar_port#', 'part1_carry_timed_dollar_port#']:
        for suffix in ['', ' #no peg']:
            ret = get_port_ret_df(prefix, suffix)
            ret.loc[:, 'carry'] = data['carry' + suffix]
            ret = ret.dropna()
            label = prefix + suffix
            t = ret.mean() / ret.std() * len(ret) ** 0.5  # t on mean return
            # regress on carry
            x = sm.add_constant(ret['carry'])
            ols_series = pd.Series()
            for i in range(7):
                if i == 6:
                    olslabel = 'HML'
                    y = ret[5] - ret[0]
                else:
                    olslabel = str(i)
                    y = ret[i]
                model = OLS(y, x)
                results = model.fit()
                ols_series = ols_series.combine_first(
                    pd.Series({'alpha to carry': results.params['const'] * 12, 'beta to carry': results.params['carry'],
                               't(alpha to carry)': results.tvalues['const'], 't(beta to carry)': results.tvalues['carry']}
                              ).add_suffix('#' + olslabel))
            output[label] = (ret.mean().multiply(12)).add_prefix('mean return#').combine_first(
                t.add_prefix('t(mean return)#')).combine_first(
                pd.Series({'nobs': len(ret)})).combine_first(ols_series)
    output = pd.DataFrame(output)


def table_2_t_test():
    output = {}
    # t-test on return diffs for dollar ports, carry-timed dollar ports, and its alpha to carry
    for prefix in ['part1_dollar_port#', 'part1_carry_timed_dollar_port#']:
        for suffix in ['', ' #no peg']:
            ret = get_port_ret_df(prefix, suffix)
            ret.loc[:,'carry'] = data['carry' + suffix]
            ret = ret.dropna()
            for i in range(5):
                y = ret[i + 1] - ret[i]
                if prefix == 'part1_carry_timed_dollar_port#':
                    # t test on alpha diff
                    x = sm.add_constant(ret['carry'])
                    model = OLS(y, x)
                    results = model.fit()
                    label = 'part1_carry_timed_dollar_port_alpha_to_carry#' + suffix + str(i + 1) + '-' + str(i)
                    output[label] = pd.Series({'annualized ret diff': results.params['const'] * 12,
                                               't': results.tvalues['const'],
                                               'nobs': results.nobs})
                # t test on return diff
                x = np.ones(len(y))
                model = OLS(y, x)
                results = model.fit()
                label = prefix + suffix + str(i + 1) + '-' + str(i)
                output[label] = pd.Series({'annualized ret diff': results.params['const'] * 12,
                                           't': results.tvalues['const'],
                                           'nobs': results.nobs})

    output = pd.DataFrame(output).T
    return output


def table_2_f_test():
    output = {}
    # f-test on whether all mean returns / alphas are the same
    for prefix in ['part1_dollar_port#', 'part1_carry_timed_dollar_port#']:
        for suffix in ['', ' #no peg']:
            ret = get_port_ret_df(prefix, suffix)
            ret.loc[:,'carry'] = data['carry' + suffix]
            ret = ret.dropna()

            y = ret.drop('carry', axis=1).unstack()

            if prefix == 'part1_carry_timed_dollar_port#':
                x = {}
                for i in range(6):
                    x['meandiff#' + str(i)] = y * 0
                    x['meandiff#' + str(i)].loc[i:] = 1
                x = pd.DataFrame(x)
                # test on alpha
                for i in range(6):
                    df = pd.DataFrame(0, index=ret.index, columns=range(6))
                    df[i] = ret['carry']
                    x.loc[:, 'carry#' + str(i)] = df.unstack()

                model = OLS(y, x)
                results = model.fit(cov_type='cluster', cov_kwds={'groups': y.index.get_level_values(1)})
                rmat = np.identity(len(results.params))[1:6, :]
                f_test = results.f_test(rmat)
                label = prefix + suffix + ' f-test on alpha to carry'
                output[label] = pd.Series({'f': f_test.fvalue[0][0], 'pvalue': f_test.pvalue, 'nobs': results.nobs})

            # test on returns
            x = {}
            for i in range(6):
                x['meandiff#' + str(i)] = y * 0
                x['meandiff#' + str(i)].loc[i:] = 1
            x = sm.add_constant(pd.DataFrame(x))
            model = OLS(y, x)
            results = model.fit(cov_type='cluster', cov_kwds={'groups': y.index.get_level_values(1)})
            rmat = np.identity(len(results.params))[1:,:]
            f_test = results.f_test(rmat)
            label = prefix + suffix + ' f-test'
            output[label] = pd.Series({'f': f_test.fvalue[0][0], 'pvalue': f_test.pvalue, 'nobs': results.nobs})
    output = pd.DataFrame(output).T
    return output


# monotonicity tests
def master_func_monotonicity_test():
    output = {}
    for prefix in ['part1_dollar_port#', 'part1_carry_timed_dollar_port#']:
        for suffix in ['', ' #no peg']:
            ret = get_port_ret_df(prefix, suffix)
            ret.loc[:, 'carry'] = data['carry' + suffix]
            ret = ret.dropna()
            label = prefix + suffix
            print(label)
            if prefix == 'part1_carry_timed_dollar_port#':
                if suffix == '':
                    cov_alpha = pd.read_pickle(RETURNS_CACHE + '/cov_ctd')
                else:
                    cov_alpha = pd.read_pickle(RETURNS_CACHE + '/cov_ctd_nopeg')
                cov_alpha.columns = range(5)
                cov_alpha.index = range(5)
                cov_alpha = cov_alpha / 10 ** 8
                output[label] = monotonicity_test(ret, cov_alpha)
            else:
                output[label] = monotonicity_test(ret)
    output = pd.DataFrame(output)


def boudoukh_richardson(m, N, cov):
    inv_cov = pd.DataFrame(np.linalg.inv(cov), index=cov.index, columns=cov.columns)

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
        assert res['success']
        return res['x']


    # H0: m > 0; H1: otherwise
    bnds_increase = ((0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf))
    x0 = np.array((np.sign(m) + 1) / 2 * m) + 0.00001
    assert x0.min() > 0
    res = _optimize_for_monotonicity(func, x0, bnds_increase)
    W_increase = func(pd.Series(res['x']))
    p_value_W_increase = 1 - cdf(W_increase)

    # H1: m < 0; H1: otherwise
    bnds_decrease = ((-np.inf, 0), (-np.inf, 0), (-np.inf, 0), (-np.inf, 0), (-np.inf, 0))
    x0 = np.array((np.sign(m) - 1).abs() / 2 * m) - 0.00001
    assert x0.max() < 0
    res = _optimize_for_monotonicity(func, x0, bnds_decrease)
    W_decrease = func(pd.Series(res['x']))
    p_value_W_decrease = 1 - cdf(W_decrease)
    return W_increase, p_value_W_increase, W_decrease, p_value_W_decrease


def monotonicity_test(ret, cov_alpha=None):
    # ret is 6-column monthly returns
    ret_diff = pd.DataFrame({num: ret[num + 1] - ret[num] for num in range(5)})
    assert ret_diff.dropna().shape == ret_diff.dropna(how='all').shape
    ret_diff = ret_diff.dropna()

    T = ret_diff.shape[0]
    N = ret_diff.shape[1]

    # Boudoukh, Richardson, Smith Is the ex ante risk premium always positive?
    # test for mean diff
    cov = ret_diff.cov() / T
    m = ret_diff.mean(0)
    W_increase, p_value_W_increase, W_decrease, p_value_W_decrease = boudoukh_richardson(m, N, cov)

    # test for alpha diff
    m_alpha = pd.Series({i: OLS(ret_diff[i], sm.add_constant(ret['carry'])).fit().params['const'] for i in range(5)})
    if cov_alpha is not None:
        alpha_W_increase, p_value_alpha_W_increase, alpha_W_decrease, p_value_alpha_W_decrease = boudoukh_richardson(
            m_alpha, N, cov_alpha)
    else:
        alpha_W_increase, p_value_alpha_W_increase, alpha_W_decrease, p_value_alpha_W_decrease = (
            np.nan, np.nan, np.nan, np.nan)

    # Patton Timmerman: Monotonicity in asset returns

    # H0: m <= 0; H1: m > 0
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

    # Patton Timmerman: Monotonicity in alpha to carry

    delta_alpha = pd.Series({i: OLS(ret_diff[i], sm.add_constant(ret['carry'])).fit().params['const']
                             for i in range(5)})
    alpha_J_min = delta_alpha.min()
    alpha_J_max = delta_alpha.max()

    # bootstrap for J's distribution, alpha to carry
    def bootstrap_for_j_alpha(size=1000):
        ret_diff_null = ret_diff.subtract(delta_alpha, axis=1)
        ret_diff_null.loc[:, 'carry'] = ret['carry']
        J_b_min = []
        J_b_max = []
        for i in range(size):
            ret_diff_bs = ret_diff_null.sample(T, replace=True)
            delta_alpha_bs = pd.Series({i: OLS(ret_diff_bs[i],
                                               sm.add_constant(ret_diff_bs['carry'])).fit().params['const']
                                        for i in range(5)})
            J_b_min += [delta_alpha_bs.min()]
            J_b_max += [delta_alpha_bs.max()]
        return np.array(J_b_min), np.array(J_b_max)
    J_b_min, J_b_max = bootstrap_for_j_alpha()

    p_value_alpha_J_increase = np.mean(alpha_J_min < J_b_min)
    p_value_alpha_J_decrease = np.mean(alpha_J_max > J_b_max)

    return {
        'W increase': W_increase,
        'p value W increase': p_value_W_increase,
        'W decrease': W_decrease,
        'p value W decrease': p_value_W_decrease,
        'alpha W increase': alpha_W_increase,
        'p value alpha W increase': p_value_alpha_W_increase,
        'alpha W decrease': alpha_W_decrease,
        'p value alpha W decrease': p_value_alpha_W_decrease,
        'J increase': J_min * 12,
        'p value J increase': p_value_J_increase,
        'J decrease': J_max * 12,
        'p value J decrease': p_value_J_decrease,
        'alpha J increase': alpha_J_min * 12,
        'p value alpha J increase': p_value_alpha_J_increase,
        'alpha J decrease': alpha_J_max * 12,
        'p value alpha J decrease': p_value_alpha_J_decrease,
        'nobs': len(ret_diff)
    }


def factor_analysis():
    # load excess returns for dollar basket, dollar HML, carry-timed dollar HML, carry
    rets = {}
    rets['dollar basket'] = dollar_basket_returns['withpeg']
    ports = get_port_ret_df('part1_dollar_port#', '')
    rets['dollar HML'] = ports[5] - ports[0]
    ports = get_port_ret_df('part1_carry_timed_dollar_port#', '')
    rets['carry-timed dollar HML'] = ports[5] - ports[0]
    rets['carry'] = data['carry']
    rets = pd.DataFrame(rets).dropna()
    assert rets.first_valid_index() == pd.Timestamp('1988-12-30')
    assert rets.last_valid_index() == pd.Timestamp('2010-12-31')

    # sharpe, correl, markowitz
    sr = rets.mean() / rets.std() * 12 ** 0.5
    corr = rets.corr()
    inv_corr = np.linalg.inv(corr)
    markowitz_risk_weights = pd.Series(inv_corr.dot(sr), index=sr.index)
    markowitz_risk_weights  = markowitz_risk_weights / markowitz_risk_weights.sum()

    # univariate regressions
    reg = {}
    for y_label in rets.columns:
        for x_label in rets.columns:
            if x_label == y_label: continue
            y = rets[y_label]
            x = rets[x_label]
            results = OLS(y, sm.add_constant(x)).fit()
            reg[y_label + ' on ' + x_label] = results.params.combine_first(results.tvalues.add_prefix('t#')).combine_first(
                pd.Series({'nobs': results.nobs}))
    reg = pd.DataFrame(reg)
    reg.to_clipboard()
