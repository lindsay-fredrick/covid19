import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, FunctionTransformer
import pymc3 as pm
from pymc3.ode import DifferentialEquation
#from bs4 import BeautifulSoup
#import requests
#import re
import theano.tensor as tt
import os
import pandas as pd
from scipy.integrate import odeint
import seaborn as sns
import theano

import scipy
floatX = theano.config.floatX


import joblib

plt.style.use('seaborn-darkgrid')


def exp_model(x, y):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    with pm.Model() as model:
        # Priors for unknown model parameters
        # alpha = pm.Normal('alpha', mu=0, sigma=10)
        # beta = pm.Normal('beta', mu=0, sigma=10)
        # sigma = pm.HalfNormal('sigma', sigma=1)

        # Priors for unknown model parameters
        alpha = pm.Normal('alpha', mu=np.exp(y[0]), sigma=1)
        beta = pm.HalfNormal('beta', sigma=1)
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Expected value of outcome
        # mu = alpha*pm.math.exp(beta*x)
        mu = pm.math.log(alpha)+(beta*x)

        # Likelihood (sampling distribution) of observations
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    return model


def sig_model(x, y):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    with pm.Model() as model:
        # Priors for unknown model parameters
        # alpha = pm.Normal('alpha', mu=0, sigma=10)
        # beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
        # sigma = pm.HalfNormal('sigma', sigma=1)

        # older version
        # alpha = pm.HalfNormal('alpha', sigma=1)
        # beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
        # sigma = pm.HalfNormal('sigma', sigma=1)

        # Expected value of outcome
        # mu = alpha / (1 + pm.math.exp(-(beta[0] * x + beta[1])))

        # Priors for unknown model parameters
        alpha = pm.HalfNormal('alpha', sigma=1)
        p0 = pm.Normal('p0', mu=y[0], sigma=1)
        beta = pm.HalfNormal('beta', sigma=1)
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Expected value of outcome
        # mu = pm.math.log(alpha) - pm.math.log((1+((alpha-p0)/p0)*pm.math.exp(-(beta*x))))
        mu = alpha/(1+((alpha-p0)/p0)*pm.math.exp(-(beta*x)))
        # mu = alpha/(1+pm.math.exp(-(beta[0]*x+beta[1])))

        # Likelihood (sampling distribution) of observations
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    return model


def train_model(model, draws=5000, tune=5000, progressbar=True):
    with model:
        # Use Maximum A Posteriori (MAP) optimisation as initial value for MCMC
        # start = pm.find_MAP()

        # Use the No-U-Turn Sampler
        # step = pm.NUTS()

        trace = pm.sample(
            draws=draws,  # step=step, start=start,
            tune=tune,
            #random_seed=42,
            progressbar=progressbar  # , cores=4
        )

    return trace


def predict_model(model, trace, samples):
    with model:
        y_hat = pm.sample_posterior_predictive(trace, samples=samples)

    return y_hat['y_obs']


def predict_model_from_file(model, trace_path, samples):
    with model:
        trace = pm.load_trace(directory=trace_path)
        y_hat = pm.sample_posterior_predictive(trace[500:], samples=samples, progressbar=False)

    return y_hat['y_obs'], trace


# arbitrary country (old version, don't need anymore)
'''
def get_country(country, start_date='', end_date='', min_cases=10):
    url = 'https://www.worldometers.info/coronavirus/country/' + country.lower() + '/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    test = soup.find('script', text=re.compile('Total Coronavirus Cases'))
    save_stuff = False
    for line in test.get_text().split('\n'):
        if 'text' in line:
            if 'Linear Scale' in line or 'Total Coronavirus Cases' in line:
                save_stuff = True
            else:
                save_stuff = False
        if 'categories' in line:
            if save_stuff:
                categories = line.split(',')
                categories[0] = categories[0][25:]
                categories = categories[:-1]
                categories[-1] = categories[-1].strip('}').strip(' ').strip(']')
                categories = [x.strip('"') for x in categories]
        if 'data' in line:
            if save_stuff:
                data = line.split(',')
                data[0] = data[0][19:]
                data = data[:-1]
                data[-1] = data[-1].strip(']').strip('}').strip(' ').strip(']')
                data = [int(x) for x in data]

    start = np.where(np.array(categories) == start_date)[0]
    if len(start) == 0:
        start = 0
    else:
        start = start[0]

    end = np.where(np.array(categories) == end_date)[0]
    if len(end) == 0:
        end = len(categories) - 1
    else:
        end = end[0]

    dates = categories[start:end + 1]
    data = data[start:end + 1]

    if max(data) < min_cases:
        print('Warning, {:d} cases has not occured in this date range.')
    else:
        min_start = np.where(np.array(data) >= min_cases)[0][0]
        data = data[min_start:]
        dates = dates[min_start:]

    return dates, np.arange(1, len(data) + 1), np.array(data)
'''


# better version that can grab more info
def get_country_v2(country, start_date='', end_date='', min_cases=10):
    country = country.title().replace(' ', '_')
    file = os.path.join('csv_out', country + '.csv')
    country_df = pd.read_csv(file)

    start = country_df[country_df.Date == start_date].index
    if len(start) == 0:
        start = 0
    else:
        start = start[0]

    end = country_df[country_df.Date == end_date].index
    if len(end) == 0:
        end = country_df.index[-1]
    else:
        end = end[0]

    dates = country_df.loc[start:end + 1, 'Date'].values
    data = country_df.loc[start:end + 1, 'Confirmed'].values

    if max(data) < min_cases:
        print('Warning, {:d} cases has not occured in this date range.')
    else:
        min_start = np.where(np.array(data) >= min_cases)[0][0]
        data = data[min_start:]
        dates = dates[min_start:]

    return dates, np.arange(0, len(data)), np.array(data)


def scale_data(x, y):
    x_train = np.array(x[:-3])
    y_train = np.array(y[:-3])

    x_test = np.array(x[-3:])
    y_test = np.array(y[-3:])

    # rescale y
    # use log scaling for exponential fit and min max for sigmoid
    scale_sig = MinMaxScaler(feature_range=(0.1, 0.8))
    # scale_exp = PowerTransformer(method='box-cox', standardize=False)
    scale_exp = FunctionTransformer(np.log, np.exp, validate=True)
    y_sig = scale_sig.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_exp = scale_exp.fit_transform(y_train.reshape(-1, 1)).flatten()

    # rescale x?
    scalex = MinMaxScaler()
    x_scale = scalex.fit_transform(x_train.reshape(-1, 1)).flatten()

    return x_train, y_train, x_scale, y_sig, y_exp, x_test, y_test, scale_sig, scale_exp, scalex


def plot_country(country, num_days, ymax):
    # dates, x, y = get_country(country, min_cases=100)

    tr_path = os.path.join('traces', country.lower().replace(' ', '_'))

    dates = joblib.load(os.path.join(tr_path, 'dates.pkl'))
    x = joblib.load(os.path.join(tr_path, 'x.pkl'))
    y = joblib.load(os.path.join(tr_path, 'y.pkl'))

    scalex = joblib.load(os.path.join(tr_path, 'scalex.pkl'))
    scale_exp = joblib.load(os.path.join(tr_path, 'scale_exp.pkl'))
    scale_sig = joblib.load(os.path.join(tr_path, 'scale_sig.pkl'))

    x_scale = scalex.transform(x.reshape(-1, 1)).flatten()
    y_sig = scale_sig.transform(y.reshape(-1, 1)).flatten()
    y_exp = scale_exp.transform(y.reshape(-1, 1)).flatten()

    x_train = x[:-3]
    x_test = x[-3:]

    y_train = y[:-3]
    y_test = y[-3:]

    last = len(x)
    num_days = num_days
    extend = np.arange(last, last + num_days)
    x_updated = np.append(x, extend)
    x_updated_scaled = scalex.transform(x_updated.reshape(-1, 1)).flatten()
    y_updated = np.empty(x_updated.shape)

    # non transformed for exp
    exp_updated = exp_model(x_updated, y_updated)

    # transformed for sig
    sig_updated = sig_model(x_updated_scaled, y_updated)

    y_exp_pred, _ = predict_model_from_file(exp_updated, os.path.join(tr_path, 'exp'), 1000)
    y_sig_pred, _ = predict_model_from_file(sig_updated, os.path.join(tr_path, 'sig'), 1000)

    y_exp_avg = np.mean(y_exp_pred, axis=0).reshape(-1, 1)
    y_exp_std = 1 * np.std(y_exp_pred, axis=0).reshape(-1, 1)

    y_sig_avg = np.mean(y_sig_pred, axis=0).reshape(-1, 1)
    y_sig_std = 1 * np.std(y_sig_pred, axis=0).reshape(-1, 1)

    y_exp_high = scale_exp.inverse_transform(y_exp_avg + y_exp_std).flatten()
    y_exp_low = scale_exp.inverse_transform(y_exp_avg - y_exp_std).flatten()

    y_sig_high = scale_sig.inverse_transform(y_sig_avg + y_sig_std).flatten()
    y_sig_low = scale_sig.inverse_transform(y_sig_avg - y_sig_std).flatten()

    y_exp_avg_1 = scale_exp.inverse_transform(y_exp_avg).flatten()
    y_sig_avg_1 = scale_sig.inverse_transform(y_sig_avg).flatten()

    plt.figure(figsize=(10, 8))
    plt.plot(x_updated, y_exp_avg_1)
    plt.plot(x_updated, y_sig_avg_1)
    plt.fill_between(x_updated, y_exp_high, y_exp_low, alpha=0.5, label='Exponential')
    plt.fill_between(x_updated, y_sig_high, y_sig_low, alpha=0.5, label='Sigmoid')
    plt.scatter(x_train, y_train, label='Training Data')
    plt.scatter(x_test, y_test, label='Last 3 Days')
    plt.vlines(last + 1, -0.5, max(y) + 100 * max(y), label='Most Recent Unknown')

    if ymax == -1:
        ymax = max(y) + 3 * max(y)
    plt.ylim([-0.5, ymax])
    plt.xlim([x_updated[0], x_updated[-1]])
    plt.legend(loc='upper left')
    plt.title(country.upper() + ' -- DAY ONE = {:s}'.format(dates[0].upper()))
    plt.xlabel('Days since hitting 100 cases.')
    plt.ylabel('Total number of cases.')
    plt.show()


'''
 ---------------- FOR ODE SYSTEMS ----------------
'''


def get_country_sir(country, start_date='', end_date='', min_cases=10):
    # update for any country
    # italy
    population = 60e6

    # get population - only for canada for now
    pop_df = pd.read_csv(os.path.join('data', 'pop_canada.csv'))

    province = ''
    if country.lower() != 'canada':
        province = country[7:]
    else:
        province = 'Canada'

    if province in pop_df['Geography'].values:
        idx = pop_df[pop_df['Geography'] == province].index
        population = pop_df.iloc[idx, -1].values[0]
    else:
        population = 60e6

    if type(population) == str:
        population = population.replace(',', '')
        population = int(population)

    # print('Population of {:s}: {:d}'.format(country, population))

    country = country.title().replace(' ', '_')
    file = os.path.join('csv_out', country + '.csv')
    country_df = pd.read_csv(file)

    start = country_df[country_df.Date == start_date].index
    if len(start) == 0:
        start = 0
    else:
        start = start[0]

    end = country_df[country_df.Date == end_date].index
    if len(end) == 0:
        end = country_df.index[-1]
    else:
        end = end[0]

    dates = country_df.loc[start:end + 1, 'Date'].values
    data = country_df.loc[start:end + 1, 'Confirmed'].values
    deaths = country_df.loc[start:end + 1, 'Deaths'].values
    recovered = country_df.loc[start:end + 1, 'Recovered'].values

    if max(data) < min_cases:
        print('Warning, {:d} cases has not occurred in this date range.')
    else:
        min_start = np.where(np.array(data) >= min_cases)[0][0]
        data = data[min_start:]
        dates = dates[min_start:]
        deaths = deaths[min_start:]
        recovered = recovered[min_start:]

    # infected = total cases - deaths - recoveries
    infected = data - deaths - recovered

    # susceptible = population - infected - deaths - recovered
    susceptible = population - infected - deaths - recovered

    return dates, np.arange(0, len(data)), susceptible / population, infected / population


class DE(pm.ode.DifferentialEquation):
    def _simulate(self, y0, theta):
        # Initial condition comprised of state initial conditions and raveled sensitivity matrix
        s0 = np.concatenate([y0, self._sens_ic])

        # perform the integration
        sol = scipy.integrate.solve_ivp(
            fun=lambda t, Y: self._system(Y, t, tuple(np.concatenate([y0, theta]))),
            t_span=[self._augmented_times.min(), self._augmented_times.max()],
            y0=s0,
            method='RK23',
            t_eval=self._augmented_times[1:],
            atol=1, rtol=1,
            max_step=0.02).y.T.astype(floatX)

        # The solution
        y = sol[:, :self.n_states]

        # The sensitivities, reshaped to be a sequence of matrices
        sens = sol[0:, self.n_states:].reshape(self.n_times, self.n_states, self.n_p)

        return y, sens


def sir_delta_function(y, t, p):
    # 'constants'
    delta = p[0]  # rename to delta when testing
    lmbda = p[1]
    beta = p[2]*pm.math.exp(-t*delta)

    # y = (s, i)

    # susceptible differential
    ds = -y[0] * y[1] * beta

    # infected differential
    di = y[0] * y[1] * beta - y[1] * lmbda

    return [ds, di]


def sir_function(y, t, p):
    # 'constants'
    beta = p[0]  # rename to delta when testing
    lmbda = p[1]
    #beta = p[2]*pm.math.exp(-t*delta)

    # y = (s, i)

    # susceptible differential
    ds = -y[0] * y[1] * beta

    # infected differential
    di = y[0] * y[1] * beta - y[1] * lmbda

    return [ds, di]


def sir_delta_model(x, y, y0):

    x = np.asarray(x).flatten()
    y = np.asarray(y)

    sir_ode = DifferentialEquation(
        func=sir_delta_function,
        times=x,
        n_states=2,  # number of y (sus and inf)
        n_theta=3,  # number of parameters (delta, lambda, beta)
        t0=0
    )

    with pm.Model() as model:

        # Overall model uncertainty
        # sigma = pm.HalfNormal('sigma', 3, shape=2)
        # sigma = 0.1
        # sigma = pm.HalfCauchy('sigma', 3)

        # Note that we access the distribution for the standard
        # deviations, and do not create a new random variable.
        dim = 2
        sd_dist = pm.HalfCauchy.dist(beta=2.5)
        packed_chol = pm.LKJCholeskyCov('chol_cov', n=dim, eta=1, sd_dist=sd_dist)
        # compute the covariance matrix
        chol = pm.expand_packed_triangular(dim, packed_chol, lower=True)

        # Extract the cov matrix and standard deviations
        # cov = tt.dot(chol, chol.T)
        # sd = pm.Deterministic('sd', tt.sqrt(tt.diag(cov)))

        # R0 is bounded below by 1 because we see an epidemic has occurred
        R0 = pm.Bound(pm.Normal, lower=1)('R0', 2, 3)
        # R0 = pm.Normal('R0', 2, 2)

        # approximate lmbda as 1/9 to begin (between 1/5 and 1/13 ish)
        lmbda = pm.Normal('lambda', 1/9, 0.1)
        # lmbda = 1/10

        # allow delta to be whatever, but near 0
        delta = pm.Normal('delta', 0, 0.1)
        beta = pm.Deterministic('beta', lmbda * R0)

        # print('Setting up model')
        sir_curves = sir_ode(y0=y0, theta=[delta, lmbda, beta])  # [beta, lmbda])
        # sir_curves = sir_ode(y0=y0, theta=[beta, lmbda])

        y_obs = pm.MvNormal('y_obs', mu=sir_curves, chol=chol, observed=y)
        # y_obs = pm.Normal('y_obs', mu=sir_curves, sigma=sigma, observed=y)

    return model


def sir_model(x, y, y0):

    x = np.asarray(x).flatten()
    y = np.asarray(y)

    sir_ode = DifferentialEquation(
        func=sir_function,
        times=x,
        n_states=2,  # number of y (sus and inf)
        n_theta=2,  # number of parameters (delta, lambda, beta)
        t0=0
    )

    with pm.Model() as model:

        # Overall model uncertainty
        # sigma = pm.HalfNormal('sigma', 3, shape=2)
        # sigma = 0.1

        # Note that we access the distribution for the standard
        # deviations, and do not create a new random variable.
        dim = 2
        sd_dist = pm.HalfCauchy.dist(beta=2.5)
        packed_chol = pm.LKJCholeskyCov('chol_cov', n=dim, eta=1, sd_dist=sd_dist)
        # compute the covariance matrix
        chol = pm.expand_packed_triangular(dim, packed_chol, lower=True)

        # Extract the cov matrix and standard deviations
        cov = tt.dot(chol, chol.T)
        sd = pm.Deterministic('sd', tt.sqrt(tt.diag(cov)))

        # R0 is bounded below by 1 because we see an epidemic has occurred
        R0 = pm.Bound(pm.Normal, lower=1)('R0', 2, 3)
        # R0 = pm.Normal('R0', 2, 2)

        # approximate lmbda as 1/9 to begin (between 1/5 and 1/13 ish)
        lmbda = pm.Normal('lambda', 1/9, 0.1)
        # lmbda = 1/9

        # allow delta to be whatever, but near 0
        # delta = pm.Normal('delta', 0, 1)
        beta = pm.Deterministic('beta', lmbda * R0)

        # print('Setting up model')
        # sir_curves = sir_ode(y0=y0, theta=[delta, lmbda, beta])  # [beta, lmbda])
        sir_curves = sir_ode(y0=y0, theta=[beta, lmbda])

        y_obs = pm.MvNormal('y_obs', mu=sir_curves, chol=chol, observed=y)

    return model


def train_ode_model(model, cores=4, draws=5000, tune=5000, progressbar=True):
    with model:
        # Use Maximum A Posteriori (MAP) optimisation as initial value for MCMC
        # start = pm.find_MAP()

        # Use the No-U-Turn Sampler
        # step = pm.NUTS()

        trace = pm.sample(
            draws=draws,  # step=step, start=start,
            tune=tune,
            cores=cores,
            chains=2,
            # random_seed=42,
            progressbar=progressbar  # , cores=4
        )

    return trace


# def system of equations
def sir_function_static(y, t, p):
    # 'constants'
    delta = p[0]  # rename to delta when testing
    lmbda = p[1]
    beta = p[2] * np.exp(-t * delta)

    # y = (s, i)

    # susceptible differential
    ds = -y[0] * y[1] * beta

    # infected differential
    di = y[0] * y[1] * beta - y[1] * lmbda

    return [ds, di]


# def function that will use odeint
def sir_solution_static(t, p, y0):
    y = odeint(sir_function_static, y0, t, args=(p,))
    return y


# static plots for scipy model
def sir_plot_static(delta=0, R0=3, gamma=1 / 9):
    # R0 = 3
    # gamma = 1/9
    beta = R0*gamma
    # delta = 0
    x = np.arange(200)
    y0 = [0.99999, 0.00001]

    y = sir_solution_static(x, [delta, gamma, beta], y0)
    sus = y[:, 0]
    inf = y[:, 1]
    res = 1 - sus - inf
    total_cases = inf + res
    new_cases = np.gradient(total_cases)

    # plots : SIR together?
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.subplot(2,1,1)
    #plt.sca(ax[0, :])
    plt.plot(x, sus, color='g', label='Susceptible')
    plt.plot(x, inf, color='r', label='Infected')
    plt.plot(x, res, color='b', label='Resistant')
    plt.legend(loc='center right')
    plt.xlabel('Days')
    plt.ylim([0, 1.01])
    plt.xlim([x[0], x[-1]])
    plt.title('Infection Rates')
    #plt.show()

    #fig, ax = plt.subplots(1, 2, figsize = (16,6))
    # total cases
    plt.subplot(2,2,3)
    #plt.sca(ax[1,0])
    plt.plot(x, total_cases)
    plt.xlabel('Days')
    plt.title('Total Cases')
    plt.ylim([0, 1.01])
    plt.xlim([x[0], x[-1]])

    # new cases
    plt.subplot(2,2,4)
    #plt.sca(ax[1,1])
    subax = plt.gca()
    plt.plot(x, new_cases, label='New Cases')
    plt.xlabel('Days')
    plt.title('Number of New DAILY Cases')
    plt.ylim([0, 0.1])
    plt.xlim([x[0], x[-1]])
    plt.text(0.8, 0.8, r'$\delta = {:0.3f}$'.format(delta), transform=subax.transAxes, horizontalalignment='center',
             fontsize=14)
    # plt.legend()
    plt.show()


# plots to show bayes results
def sir_bayes_plot(country, num_days):
    # country = 'Canada British Columbia'
    # dates, x, sus, inf = get_country_sir(country, min_cases=1)

    tr_path = os.path.join('traces', country.lower().replace(' ', '_'))

    dates = joblib.load(os.path.join(tr_path, 'dates.pkl'))
    x = joblib.load(os.path.join(tr_path, 'x_sir.pkl'))
    sus = joblib.load(os.path.join(tr_path, 'sus_sir.pkl'))
    inf = joblib.load(os.path.join(tr_path, 'inf_sir.pkl'))

    # sus and inf are already normalized
    # just normalize x
    x_train = x[:-1]
    x_test = x[-1:]

    sus_train = sus[:-1]
    sus_test = sus[-1:]

    inf_train = inf[:-1]
    inf_test = inf[-1:]

    # make single array
    y_train = np.hstack((sus_train.reshape(-1, 1), inf_train.reshape(-1, 1)))
    y_test = np.hstack((sus_test.reshape(-1, 1), inf_test.reshape(-1, 1)))

    y0 = [y_train[0][0], y_train[0][1]]

    last = len(x)
    extend = np.arange(last, last + num_days)
    x_updated = np.append(x, extend)
    y_updated = np.empty((x_updated.shape[0], y_train.shape[1]))

    sir = sir_model(x_updated, y_updated, y0)
    posterior_predictive, trace = predict_model_from_file(sir, os.path.join(tr_path, 'sir'), 1000)

    all_y = posterior_predictive
    y0_array = all_y[:, :, 0]
    y1_array = all_y[:, :, 1]
    y2_array = 1 - y0_array - y1_array
    total_cases = y1_array + y2_array
    new_cases = np.gradient(total_cases, axis=1)

    y0_mean = np.mean(y0_array, axis=0)
    y0_std = 2 * np.std(y0_array, axis=0)

    y1_mean = np.mean(y1_array, axis=0)
    y1_std = 2 * np.std(y1_array, axis=0)

    y2_mean = np.mean(y2_array, axis=0)
    y2_std = 2 * np.std(y2_array, axis=0)

    total_cases_mean = np.mean(total_cases, axis=0)
    total_cases_std = 2 * np.std(total_cases, axis=0)

    new_cases_mean = np.mean(new_cases, axis=0)
    new_cases_std = 2 * np.std(new_cases, axis=0)

    # SIR Curves
    fig, ax = plt.subplots(figsize=(16, 6))
    # plt.sca(ax[0])
    plt.fill_between(x_updated, y0_mean + y0_std, y0_mean - y0_std, alpha=0.5, color='g')
    plt.plot(x_train, sus_train, c='g', label='suseptible')
    plt.scatter(x_test, sus_test, color='g')
    plt.plot(x_updated, y0_mean, '--g', alpha=0.7)

    plt.fill_between(x_updated, y1_mean + y1_std, y1_mean - y1_std, alpha=0.5, color='r')
    plt.plot(x_train, inf_train, c='r', label='infected')
    plt.scatter(x_test, inf_test, color='r')
    plt.plot(x_updated, y1_mean, '--r', alpha=0.7)

    plt.fill_between(x_updated, y2_mean + y2_std, y2_mean - y2_std, alpha=0.5, color='b')
    plt.plot(x_train, 1 - sus_train - inf_train, c='b', label='resistant')
    plt.scatter(x_test, 1 - sus_test - inf_test, color='b')
    plt.plot(x_updated, y2_mean, '--b', alpha=0.7)

    plt.xlabel('Days')
    plt.ylim([0, 1.01])
    plt.xlim([x_updated[0], x_updated[-1]])
    plt.title('Infection Rates')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    # Total Cases

    plt.sca(ax[0])
    plt.plot(x_updated, total_cases_mean, '--')
    plt.fill_between(x_updated, total_cases_mean + total_cases_std, total_cases_mean - total_cases_std)
    plt.xlabel('Days')
    plt.title('Total Cases')
    plt.ylim([0, 1.01])
    plt.xlim([x_updated[0], x_updated[-1]])

    # New Cases
    plt.sca(ax[1])
    plt.plot(x_updated, new_cases_mean, '--')
    plt.fill_between(x_updated, new_cases_mean + new_cases_std, new_cases_mean - new_cases_std)
    plt.xlabel('Days')
    plt.title('Number of New DAILY Cases')
    # plt.ylim([0, 0.1])
    plt.xlim([x_updated[0], x_updated[-1]])
    plt.show()

    # Parameters
    pm.plot_posterior(trace[:500])
    plt.show()


# plots to show bayes results v2 - doesn't use model
def sir_bayes_plot_v2(country, num_days, delta_case=False):
    # country = 'Canada British Columbia'
    # dates, x, sus, inf = get_country_sir(country, min_cases=1)

    if delta_case:
        second_dir = 'sir_delta'
    else:
        second_dir = 'sir'

    tr_path = os.path.join('traces', country.lower().replace(' ', '_'))

    dates = joblib.load(os.path.join(tr_path, 'dates_{:s}.pkl'.format(second_dir)))
    x = joblib.load(os.path.join(tr_path, 'x_{:s}.pkl'.format(second_dir)))
    sus = joblib.load(os.path.join(tr_path, 'sus_{:s}.pkl'.format(second_dir)))
    inf = joblib.load(os.path.join(tr_path, 'inf_{:s}.pkl'.format(second_dir)))

    # sus and inf are already normalized
    # just normalize x
    x_train = x[:-1]
    x_test = x[-1:]

    sus_train = sus[:-1]
    sus_test = sus[-1:]

    inf_train = inf[:-1]
    inf_test = inf[-1:]

    # make single array
    y_train = np.hstack((sus_train.reshape(-1, 1), inf_train.reshape(-1, 1)))
    y_test = np.hstack((sus_test.reshape(-1, 1), inf_test.reshape(-1, 1)))

    y0 = [y_train[0][0], y_train[0][1]]

    last = len(x)
    extend = np.arange(last, last + num_days)
    x_updated = np.append(x, extend)
    y_updated = np.empty((x_updated.shape[0], y_train.shape[1]))

    # sir = sir_model(x_updated, y_updated, y0)
    # posterior_predictive, trace = predict_model_from_file(sir, os.path.join(tr_path, 'sir'), 1000)

    posterior_predictive = joblib.load(os.path.join(tr_path, second_dir+'_y_predict.pkl'))
    all_y = posterior_predictive['y_obs'][:, :len(x_updated), :]

    y0_array = all_y[:, :, 0]
    y1_array = all_y[:, :, 1]
    y2_array = 1 - y0_array - y1_array
    total_cases = y1_array + y2_array
    new_cases = np.gradient(total_cases, axis=1)

    y0_mean = np.nanmean(y0_array, axis=0)
    y0_std = 2 * np.nanstd(y0_array, axis=0)

    y1_mean = np.nanmean(y1_array, axis=0)
    y1_std = 2 * np.nanstd(y1_array, axis=0)

    y2_mean = np.nanmean(y2_array, axis=0)
    y2_std = 2 * np.nanstd(y2_array, axis=0)

    total_cases_mean = np.nanmean(total_cases, axis=0)
    total_cases_std = 2 * np.nanstd(total_cases, axis=0)

    new_cases_mean = np.nanmean(new_cases, axis=0)
    new_cases_std = 2 * np.nanstd(new_cases, axis=0)

    # SIR Curves
    fig, ax = plt.subplots(figsize=(16, 6))
    # plt.sca(ax[0])
    plt.fill_between(x_updated, y0_mean + y0_std, y0_mean - y0_std, alpha=0.5, color='g')
    plt.plot(x_train, sus_train, c='g', label='suseptible')
    plt.scatter(x_test, sus_test, color='g')
    plt.plot(x_updated, y0_mean, '--g', alpha=0.7)

    plt.fill_between(x_updated, y1_mean + y1_std, y1_mean - y1_std, alpha=0.5, color='r')
    plt.plot(x_train, inf_train, c='r', label='infected')
    plt.scatter(x_test, inf_test, color='r')
    plt.plot(x_updated, y1_mean, '--r', alpha=0.7)

    plt.fill_between(x_updated, y2_mean + y2_std, y2_mean - y2_std, alpha=0.5, color='b')
    plt.plot(x_train, 1 - sus_train - inf_train, c='b', label='resistant')
    plt.scatter(x_test, 1 - sus_test - inf_test, color='b')
    plt.plot(x_updated, y2_mean, '--b', alpha=0.7)

    plt.xlabel('Days')
    plt.ylim([0, 1.01])
    plt.xlim([x_updated[0], x_updated[-1]])
    plt.title('Infection Rates')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    # Total Cases

    plt.sca(ax[0])
    plt.plot(x_updated, total_cases_mean, '--')
    plt.fill_between(x_updated, total_cases_mean + total_cases_std, total_cases_mean - total_cases_std, alpha=0.5)
    plt.xlabel('Days')
    plt.title('Total Cases')
    plt.ylim([0, 1.01])
    plt.xlim([x_updated[0], x_updated[-1]])

    # New Cases
    plt.sca(ax[1])
    plt.plot(x_updated, new_cases_mean, '--')
    plt.fill_between(x_updated, new_cases_mean + new_cases_std, new_cases_mean - new_cases_std, alpha=0.5)
    plt.xlabel('Days')
    plt.title('Number of New DAILY Cases')
    plt.ylim([0, 1.01*np.max(new_cases_mean+new_cases_std)])
    plt.xlim([x_updated[0], x_updated[-1]])
    plt.show()

    # Parameters
    trace = joblib.load(os.path.join(tr_path, second_dir+'_params.pkl'))
    if not delta_case:
        vars_list = ['R0', 'lambda', 'beta']
    else:
        vars_list = ['R0', 'lambda', 'beta', 'delta']
    fig, ax = plt.subplots(1, len(vars_list), figsize=(16, 6))
    for idx, var in enumerate(vars_list):
        plt.sca(ax[idx])
        sns.kdeplot(trace[var], shade=True)
        plt.title('{:s} = {:0.3f}'.format(var, np.mean(trace[var])))
    plt.show()

