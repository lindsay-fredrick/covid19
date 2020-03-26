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
        sir = sir_model(x_train, y_train, y0)

        sir_trace = train_ode_model(sir, draws=2000, tune=1000)

        # save to file
        tr_path = os.path.join('traces', country.lower().replace(' ', '_'))
        if not os.path.isdir(tr_path):
            print('Directory for {:s} does not exist. Creating now.'.format(country.title()))
            os.mkdir(tr_path)
            os.mkdir(os.path.join(tr_path, 'sir'))
        else:
            print('Directory for {:s} already exists. Will overwrite current traces.'.format(country.title()))

        if not os.path.isdir(os.path.join(tr_path, 'sir')):
            os.mkdir(os.path.join(tr_path, 'sir'))

        pm.save_trace(sir_trace, directory=os.path.join(tr_path, 'sir'), overwrite=True)

        # save scalers also

        # also save fitting data so comparing apples to apples
        joblib.dump(dates, os.path.join(tr_path, 'dates_sir.pkl'))
        joblib.dump(x, os.path.join(tr_path, 'x_sir.pkl'))
        joblib.dump(sus, os.path.join(tr_path, 'sus_sir.pkl'))
        joblib.dump(inf, os.path.join(tr_path, 'inf_sir.pkl'))
