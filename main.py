import pandas as pd
from datetime import datetime
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss, pacf, acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.stats.stattools import jarque_bera

def read_csv(filename: str, original_cols: list, new_cols: list = None):
    csv = pd.read_csv(
            filename,
            usecols=original_cols,
            parse_dates=[ 'Date' ],
            date_parser=lambda x: datetime.strptime('0' + x if int(str(x).split('-')[0]) < 10 else x, '%d-%b-%y'))

    if new_cols is not None:
        cols_dict = { og_col: new_col for (og_col, new_col) in zip(original_cols, new_cols) }
        csv.rename(columns=cols_dict, inplace=True)

    return csv.set_index(['date'])

def basic_statistics(data: pd.DataFrame, with_graph: bool = True):
    print(data.info())
    print(data.describe())
    print()

    if with_graph is True:
        data.plot(x='date', y='price', rot=-45)
        plt.show()
        data.plot(x='date', y='price', kind='hist', rot=-45)
        plt.show()

def descriptive_statistics(df: pd.DataFrame, cols: list):
    dataframe_adfuller_test(df, cols, with_graph=False)
    dataframe_skewness(df, cols)
    dataframe_kurtosis(df, cols)
    dataframe_jarque_bera(df, cols)

def dataframe_adfuller_test(df: pd.DataFrame, cols: list, with_graph: bool = True):
    for col in cols:
        print('\t==== ADF (with Intercept) for %s ====\n' % col)
        series_adfuller_test(df[col])
        print('\t==== ADF (with Intercept and Trend) for %s ====\n' % col)
        series_adfuller_test(df[col], 'ct')
        
        if with_graph is True:
            df.plot(x='date', y=col, kind='hist', rot=-45)
            plt.show()

def series_adfuller_test(series: pd.Series, reg: str = 'c'):
    test_result = adfuller(series, autolag='t-stat', regression=reg)

    print('ADF Statistic: \t%f' % test_result[0])
    print('p-value: \t%f' % test_result[1])
    print('\nCritical values:')
    for (key, value) in test_result[4].items():
        print('\t%s, %f' % (key, value))

    print()

def dataframe_skewness(df: pd.DataFrame, cols: list):
    print('\t==== Skewness ====\n')
    for col in cols:
        print('%s: %f' % (col, df[col].skew()))
    print()


def dataframe_kurtosis(df: pd.DataFrame, cols: list):
    print('\t==== Kurtosis ====\n')
    for col in cols:
        print('%s: %f' % (col, df[col].kurtosis()))
    print()

def dataframe_jarque_bera(df: pd.DataFrame, cols: list):
    print('\t==== Jarque Bera ====\n')
    for col in cols:
        print('%s: ' % col, jarque_bera(df[col]))
    print()

def dataframe_kpss_test(df: pd.DataFrame, cols: list):
    for col in cols:
        print('\t==== KPSS for %s ====\n' % col)
        test_result = kpss(df[col], nlags='legacy')

        print('KPSS Statistic: \t%f' % test_result[0])
        print('p-value: \t%f' % test_result[1])
        print('\nCritical values:')
        for (key, value) in test_result[3].items():
            print('\t%s, %f' % (key, value))

        print()


def plot_trends(df: pd.DataFrame, cols: list, with_graph: bool = True):
    if with_graph is True:
        nrows = int(len(cols) / 2) + 1 if len(cols) % 2 != 0 else 0
        ncols = int(len(cols) / nrows) + 1 if len(cols) % 2 != 0 else 0
        index_col = 0
        index_row = 0
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        fig.suptitle('Trend of features used in analysis')
        for col in cols:
            df.plot(ax=ax[index_row][index_col], x='date', y=col, rot=-45, subplots=True)
            ax[index_row][index_col].set_title('Trend of "%s"' % col)

            index_col += 1
            if index_col >= ncols:
                index_row += 1
                index_col = 0

        plt.show()
        print()

def correl(df: pd.DataFrame, cols: list, with_graph: bool = True):
    print('\t==== Correlogram ====\n')

    if with_graph is True:
        sns.pairplot(df)
        plt.show()
        plot_acf(df['price'])
        plot_acf(df['market_cap'])
        plot_acf(df['difficulty'])
        plt.show()

    for col in cols:
        print('%s:' % col)
        for lag in range(1, 16):
            print('%d:\t' % lag, df[col].autocorr(lag=lag))

    print()

def check_stationary_or_not(df: pd.DataFrame, cols: list, with_graph: bool = True):
    for col in cols:
        df[col].plot(title='Trend of %s' % col)
        plt.show()

def make_stationary(df: pd.DataFrame, cols: list):
    for col in cols:
        df[col] = df[col].apply(lambda x: np.log(x)).diff()

def dataframe_autocorrel_plot(df: pd.DataFrame, cols: list, with_graph: bool = True):
    for col in cols:
        print('%s' % col)

        pacf_results = pacf(df[col], nlags=15)
        for lag in range(1, 16):
            print('%d:\t' % lag, df[col].autocorr(lag=lag), pacf_results[lag])

        if with_graph is True:
            ax = pd.plotting.autocorrelation_plot(df[col])
            ax.set_title('Autocorrelation for %s' % col)
            plt.show()
        print()

def dataframe_acf_pacf(df: pd.DataFrame, cols: list):
    for col in cols:
        print('%s' % col)

        acf_results, _, q_stat = acf(df[col], nlags=15, qstat=True)
        pacf_results = pacf(df[col], nlags=15)
        for lag in range(0, 16):
            print('%d:' % (lag + 1), acf_results[lag], pacf_results[lag], '-' if lag - 1 < 0 else q_stat[lag - 1], sep='\t')
        print()

def arima(df: pd.DataFrame, cols: list, with_graph: bool = False):
    arima_model(df, cols, lag=0, order=1, moving_avg_model=0, with_graph=with_graph)

def arima_model(df: pd.DataFrame, cols: list, lag: int, order: int, moving_avg_model: int, with_graph: bool):
    for col in cols:
        model = ARIMA(df[col], order=(lag, order, moving_avg_model))
        model_fit = model.fit()

        print('\t==== Summary of ARIMA(%d, %d, %d) model for %s ====\n' % (lag, order, moving_avg_model, col))
        print(model_fit.summary())
        print()

        x_mean = df[col].mean()
        sst = df[col].apply(lambda x: (x - x_mean) ** 2).sum()
        ssr = sst - model_fit.sse
        r_squared = ssr / sst
        print('R-squared: %f\n' % r_squared)
        n = len(df[col])
        k = len(model_fit.arroots) + len(model_fit.maroots)
        print('n: %d, k: %d' % (n, k))
        adj_r_sqr = 1 - ((1 - r_squared) * (n - 1)) / (n - k - 1)
        print('Adjusted R-squared: %f' % adj_r_sqr)
        print()

        print('\t==== Correlogram of residuals ====\n')
        acf_results, _, q_stat = acf(model_fit.resid, nlags=15, qstat=True)
        pacf_results = pacf(model_fit.resid, nlags=15)
        for clag in range(0, 16):
            print('%d:' % (clag + 1), acf_results[clag], pacf_results[clag], '-' if clag - 1 < 0 else q_stat[clag - 1], sep='\t')
        print()

        if lag > 0 or moving_avg_model > 0:
            r_matrix = '(ar.L1 = 0)' if lag > 0 else ''
            if len(r_matrix) > 0 and moving_avg_model > 0:
                r_matrix = r_matrix + ','
            r_matrix = r_matrix + ('(ma.L1 = 0)' if moving_avg_model > 0 else '')
            f_test = model_fit.f_test(r_matrix)
            print('\t==== F Test ====\n', f_test.summary())
            print()

        print('\t==== Summary of residuals for %s ====\n' % col)
        residuals = pd.DataFrame(model_fit.resid)
        print(residuals.describe())
        print()

        if with_graph is True:
            plot_pacf(residuals, title='PAC plot for residuals of %s' % col)
            plt.show()

            #residuals.plot(kind='kde', title='Density of residuals %s' % col)
            #plt.show()

            ax = pd.plotting.autocorrelation_plot(residuals)
            ax.set_title('AC plot for residuals of %s' % col)
            plt.show()

def arma(df: pd.DataFrame, cols: list):
    #arma_model(df, cols, lag=0, moving_avg_model=0)
    #arma_model(df, cols, lag=0, moving_avg_model=1)
    #arma_model(df, cols, lag=1, moving_avg_model=0)
    #arma_model(df, cols, lag=1, moving_avg_model=1)

    arima_model(df, cols, lag=0, order=0, moving_avg_model=0, with_graph=False)
    arima_model(df, cols, lag=0, order=0, moving_avg_model=1, with_graph=False)
    arima_model(df, cols, lag=1, order=0, moving_avg_model=0, with_graph=False)
    arima_model(df, cols, lag=1, order=0, moving_avg_model=1, with_graph=False)

def arma_model(df: pd.DataFrame, cols: list, lag: int, moving_avg_model: int):
    for col in cols:
        model = ARMA(df[col], order=(lag, moving_avg_model))
        model_fit = model.fit()

        print('\t==== Summary of ARIMA(%d, %d) model for %s ====\n' % (lag, moving_avg_model, col))
        print(model_fit.summary())
        print()

        print('\t==== Summary of residuals for %s ====\n' % col)
        residuals = pd.DataFrame(model_fit.resid)
        print(residuals.describe())
        print()

def main():
    original_cols = [
        'Date',
        'Close',
        'MarketCap',
        'Difficulty',
    ]
    cols = [
        'date',
        'price',
        'market_cap',
        'difficulty',
    ]
    source_file = 'block_chain_ts.csv'
    btc_data = read_csv(source_file, original_cols, cols)
    stat_cols = list(filter(lambda x: x != 'date', cols))

    basic_statistics(btc_data, with_graph=False)
    descriptive_statistics(btc_data, stat_cols)
    plot_trends(btc_data, stat_cols, with_graph=False)

    correl(btc_data, stat_cols, with_graph=False)
    make_stationary(btc_data, stat_cols)
    #check_stationary_or_not(btc_data, stat_cols, with_graph=True)

    btc_data = btc_data.reset_index().replace([ -np.inf, np.inf ], np.nan).dropna()
    btc_data.set_index('date', inplace=True)

    dataframe_adfuller_test(btc_data, stat_cols, with_graph=False)
    dataframe_autocorrel_plot(btc_data, stat_cols, with_graph=False)
    dataframe_kpss_test(btc_data, stat_cols)

    dataframe_acf_pacf(btc_data, stat_cols)

    arima(btc_data, list(filter(lambda x: x != 'difficulty', stat_cols)), with_graph=False)
    arma(btc_data, list(filter(lambda x: x == 'difficulty', stat_cols)))

if __name__ == '__main__':
    main()

