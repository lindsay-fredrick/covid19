import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pymc3 as pm
from bs4 import BeautifulSoup
import requests
import re
import os
import pandas as pd

from sklearn.externals import joblib

plt.style.use('seaborn-darkgrid')


def exp_model(x, y):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    with pm.Model() as model:
        # Priors for unknown model parameters
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Expected value of outcome
        mu = alpha * pm.math.exp(beta * x)

        # Likelihood (sampling distribution) of observations
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    return model


def sig_model(x, y):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    with pm.Model() as model:
        # Priors for unknown model parameters
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Expected value of outcome
        mu = alpha / (1 + pm.math.exp(-(beta[0] * x + beta[1])))

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
            random_seed=42, progressbar=True  # , cores=4
        )

    return trace


def predict_model(model, trace, samples):
    with model:
        y_hat = pm.sample_posterior_predictive(trace, samples=samples)

    return y_hat['y_obs']


def predict_model_from_file(model, trace_path, samples):
    with model:
        trace = pm.load_trace(directory=trace_path)
        y_hat = pm.sample_posterior_predictive(trace, samples=samples, progressbar=False)

    return y_hat['y_obs']


# arbitrary country
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

    return dates, np.arange(1, len(data) + 1), np.array(data)


def scale_data(x, y):
    x_train = np.array(x[:-3])
    y_train = np.array(y[:-3])

    x_test = np.array(x[-3:])
    y_test = np.array(y[-3:])

    # rescale y
    scaley = MinMaxScaler(feature_range=(0.1, 0.8))
    y_scale = scaley.fit_transform(y_train.reshape(-1, 1)).flatten()

    # rescale x?
    scalex = MinMaxScaler()
    x_scale = scalex.fit_transform(x_train.reshape(-1, 1)).flatten()

    return x_train, y_train, x_scale, y_scale, x_test, y_test, scaley, scalex


def plot_country(country, num_days, ymax):
    # dates, x, y = get_country(country, min_cases=100)

    tr_path = os.path.join('traces', country.lower())

    dates = joblib.load(os.path.join(tr_path, 'dates.pkl'))
    x = joblib.load(os.path.join(tr_path, 'x.pkl'))
    y = joblib.load(os.path.join(tr_path, 'y.pkl'))

    scalex = joblib.load(os.path.join(tr_path, 'scalex.pkl'))
    scaley = joblib.load(os.path.join(tr_path, 'scaley.pkl'))

    x_scale = scalex.transform(x.reshape(-1, 1)).flatten()
    y_scale = scaley.transform(y.reshape(-1, 1)).flatten()

    x_train = x[:-3]
    x_test = x[-3:]

    y_train = y[:-3]
    y_test = y[-3:]

    last = len(x)
    num_days = num_days
    extend = np.arange(last + 1, last + num_days + 1)
    x_updated = np.append(x, extend)
    x_updated_scaled = scalex.transform(x_updated.reshape(-1, 1)).flatten()
    y_updated = np.empty(x_updated.shape)

    exp_updated = exp_model(x_updated_scaled, y_updated)
    sig_updated = sig_model(x_updated_scaled, y_updated)

    y_exp = predict_model_from_file(exp_updated, os.path.join(tr_path, 'exp'), 1000)
    y_sig = predict_model_from_file(sig_updated, os.path.join(tr_path, 'sig'), 1000)

    y_exp_avg = np.mean(y_exp, axis=0).reshape(-1, 1)
    y_exp_std = 2 * np.std(y_exp, axis=0).reshape(-1, 1)

    y_sig_avg = np.mean(y_sig, axis=0).reshape(-1, 1)
    y_sig_std = 2 * np.std(y_sig, axis=0).reshape(-1, 1)

    y_exp_high = scaley.inverse_transform(y_exp_avg + y_exp_std).flatten()
    y_exp_low = scaley.inverse_transform(y_exp_avg - y_exp_std).flatten()

    y_sig_high = scaley.inverse_transform(y_sig_avg + y_sig_std).flatten()
    y_sig_low = scaley.inverse_transform(y_sig_avg - y_sig_std).flatten()

    plt.figure(figsize=(10, 8))
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
