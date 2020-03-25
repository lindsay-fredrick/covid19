from utils.helpers import *

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    country_list = [
        'Italy',
        'Germany',
        'China',
        'Spain',
        'Canada',
        'US',
        'Canada British Columbia',
        'Canada Alberta',
        'Canada Ontario',
        'China Hong Kong',
        'Korea South',
        'Singapore'
    ]
    min_cases = 100

    for country in country_list:

        print()
        print('-' * len(country))
        print(country.upper())
        print('-' * len(country))
        print()

        # get data from website
        dates, x, y = get_country_v2(country, min_cases=min_cases)  # , 'Mar 02', 'Mar 17')
        # print(y)

        # rescale
        x_train, y_train, x_scale, y_sig, y_exp, x_test, y_test, scale_sig, scale_exp, scalex = scale_data(x, y)

        # print(x_train)
        # print(y_exp)

        # produce fits
        # exp is with unscaled x, log scaled y
        exp = exp_model(x_train, y_exp)

        # sig is min max scaled for both
        sig = sig_model(x_scale, y_sig)

        exp_trace = train_model(exp, draws=10000, tune=5000)
        sig_trace = train_model(sig, draws=10000, tune=5000)

        # save to file
        tr_path = os.path.join('traces', country.lower().replace(' ', '_'))
        if not os.path.isdir(tr_path):
            print('Directory for {:s} does not exist. Creating now.'.format(country.title()))
            os.mkdir(tr_path)
            os.mkdir(os.path.join(tr_path, 'exp'))
            os.mkdir(os.path.join(tr_path, 'sig'))
        else:
            print('Directory for {:s} already exists. Will overwrite current traces.'.format(country.title()))

        pm.save_trace(exp_trace, directory=os.path.join(tr_path, 'exp'), overwrite=True)
        pm.save_trace(sig_trace, directory=os.path.join(tr_path, 'sig'), overwrite=True)

        # save scalers also
        joblib.dump(scalex, os.path.join(tr_path, 'scalex.pkl'))
        joblib.dump(scale_exp, os.path.join(tr_path, 'scale_exp.pkl'))
        joblib.dump(scale_sig, os.path.join(tr_path, 'scale_sig.pkl'))

        # also save fitting data so comparing apples to apples
        joblib.dump(dates, os.path.join(tr_path, 'dates.pkl'))
        joblib.dump(x, os.path.join(tr_path, 'x.pkl'))
        joblib.dump(y, os.path.join(tr_path, 'y.pkl'))
