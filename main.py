import pandas as pd
from datetime import datetime
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.stattools import jarque_bera

def read_csv(filename: str, original_cols: list, new_cols: list = None):
    csv = pd.read_csv(
            filename,
            usecols=original_cols,
            index_col=0,
            parse_dates=[ 'Date' ],
            date_parser=lambda x: datetime.strptime('0' + x if int(str(x).split('-')[0]) < 10 else x, '%d-%b-%y'))
    csv.index = csv.index.to_period('D')

    if new_cols is not None:
        cols_dict = { og_col: new_col for (og_col, new_col) in zip(original_cols, new_cols) }
        csv.rename(columns=cols_dict, inplace=True)

    return csv

def basic_statistics(data: pd.DataFrame, with_graph: bool = True):
    print(data.info())
    print(data.describe())
    print()

    if with_graph is True:
        data.plot(x='date', y='price', rot=-45)
        plt.show()
        data.plot(x='date', y='price', kind='hist', rot=-45)
        plt.show()

def series_adfuller_test(series: pd.Series):
    test_result = adfuller(series, autolag='AIC')

    print('ADF Statistic: \t%f' % test_result[0])
    print('p-value: \t%f' % test_result[1])
    print('\nCritical values:')
    for (key, value) in test_result[4].items():
        print('\t%s, %f' % (key, value))

    print()

def dataframe_adfuller_test(df: pd.DataFrame, cols: list, with_graph: bool = True):
    for col in cols:
        print('\t==== ADF for %s ====\n' % col)
        series_adfuller_test(df[col])
        
        if with_graph is True:
            df.plot(x='date', y=col, kind='hist', rot=-45)
            plt.show()

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

def descriptive_statistics(df: pd.DataFrame, cols: list):
    dataframe_adfuller_test(df, cols, with_graph=False)
    dataframe_skewness(df, cols)
    dataframe_kurtosis(df, cols)
    dataframe_jarque_bera(df, cols)

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
    if with_graph is True:
        for col in cols:
            df[col].diff().plot(title='Trend of %s' % col)
            plt.show()

def arima(df: pd.DataFrame, cols: list, with_graph: bool = False):
    arima_model(df, cols, lag=5, order=1, moving_avg_model=0, with_graph=with_graph)

def arima_model(df: pd.DataFrame, cols: list, lag: int, order: int, moving_avg_model: int, with_graph: bool):
    for col in cols:
        model = ARIMA(df[col], order=(lag, order, moving_avg_model))
        model_fit = model.fit()

        print('\t==== Summary of ARIMA model for %s ====\n' % col)
        print(model_fit.summary())
        print()

        print('\t==== Summary of residuals for %s ====\n' % col)
        residuals = pd.DataFrame(model_fit.resid)
        print(residuals.describe())
        print()

        if with_graph is True:
            residuals.plot(title='Residuals %s' % col)
            plt.show()

            residuals.plot(kind='kde', title='Density of residuals %s' % col)
            plt.show()

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
    check_stationary_or_not(btc_data, stat_cols, with_graph=False)

    arima(btc_data, stat_cols, with_graph=True)

if __name__ == '__main__':
    main()

