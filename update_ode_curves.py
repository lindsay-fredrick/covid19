from utils.helpers import *

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    country_list = [
        'Canada British Columbia',
        'Canada Alberta',
        'Canada Ontario',
        'Canada'
    ]
    min_cases = 1

    delta_case = True

    for country in country_list:

        print()
        print('-' * len(country))
        print(country.upper())
        print('-' * len(country))
        print()

        # get data from website
        dates, x, sus, inf = get_country_sir(country, min_cases=min_cases)  # , 'Mar 02', 'Mar 17')
        # print(y)

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

        # produce fits
        # sir is with unscaled x, unscaled y
        print('Initializing Model')
        print()
        if delta_case:
            sir = sir_delta_model(x_train, y_train, y0)
        else:
            sir = sir_model(x_train, y_train, y0)

        print('Training Model')
        print()
        sir_trace = train_ode_model(sir, draws=1000, tune=500, cores=1)

        print('Making Predictions for Future Calls')
        print()
        num_days = 500
        last = len(x)
        extend = np.arange(last, last + num_days)
        x_updated = np.append(x, extend)
        y_updated = np.empty((x_updated.shape[0], y_train.shape[1]))

        if delta_case:
            sir_pred = sir_delta_model(x_train, y_train, y0)
        else:
            sir_pred = sir_model(x_train, y_train, y0)

        with sir_pred:
            y_hat = pm.sample_posterior_predictive(sir_trace[500:], samples=1000, progressbar=True)

        print()
        print('Saving Model')
        print()
        # save to file
        tr_path = os.path.join('traces', country.lower().replace(' ', '_'))

        if delta_case:
            second_dir = 'sir_delta'
        else:
            second_dir = 'sir'

        if not os.path.isdir(tr_path):
            print('Directory for {:s} does not exist. Creating now.'.format(country.title()))
            os.mkdir(tr_path)
            os.mkdir(os.path.join(tr_path, second_dir))
        else:
            print('Directory for {:s} already exists. Will overwrite current traces.'.format(country.title()))

        if not os.path.isdir(os.path.join(tr_path, second_dir)):
            os.mkdir(os.path.join(tr_path, second_dir))

        pm.save_trace(sir_trace, directory=os.path.join(tr_path, second_dir), overwrite=True)

        # save predictions
        # from trace, save parameters of interest
        # save yhat for future models
        joblib.dump(y_hat, os.path.join(tr_path, second_dir+'_y_predict.pkl'))

        if not delta_case:
            vars_list = ['R0', 'lambda', 'beta']
        else:
            vars_list = ['R0', 'lambda', 'beta', 'delta']

        vars_dict = {}
        for idx, var in enumerate(vars_list):
            vars_dict[var] = sir_trace[var]

        joblib.dump(vars_dict, os.path.join(tr_path, second_dir+'_params.pkl'))

        # save scalers also

        # also save fitting data so comparing apples to apples
        joblib.dump(dates, os.path.join(tr_path, 'dates_{:s}.pkl'.format(second_dir)))
        joblib.dump(x, os.path.join(tr_path, 'x_{:s}.pkl'.format(second_dir)))
        joblib.dump(sus, os.path.join(tr_path, 'sus_{:s}.pkl'.format(second_dir)))
        joblib.dump(inf, os.path.join(tr_path, 'inf_{:s}.pkl'.format(second_dir)))
