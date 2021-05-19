import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import jarque_bera

def read_csv(filename: str, original_cols: list, new_cols: list = None):
    csv = pd.read_csv(filename, usecols=original_cols)

    if new_cols is not None:
        cols_dict = { og_col: new_col for (og_col, new_col) in zip(original_cols, new_cols) }
        csv.rename(columns=cols_dict, inplace=True)

    return csv

def basic_statistics(data: pd.DataFrame, with_graph: bool = False):
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

def dataframe_adfuller_test(df: pd.DataFrame, cols: list, with_graph: bool = False):
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
    nrows = int(len(cols) / 2) + 1 if len(cols) % 2 != 0 else 0
    ncols = int(len(cols) / nrows) + 1 if len(cols) % 2 != 0 else 0
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    fig.suptitle('Trend of features used in analysis')
    index_col = 0
    index_row = 0

    if with_graph is True:
        for col in cols:
            df.plot(ax=ax[index_row][index_col], x='date', y=col, rot=-45, subplots=True)
            ax[index_row][index_col].set_title('Trend of "%s"' % col)

            index_col += 1
            if index_col >= ncols:
                index_row += 1
                index_col = 0

        plt.show()

def main():
    original_cols = [
        'Date',
        'Close',
        'MarketCap',
        'Difficulty',
        'MinersRevenue',
        'TransactionFees_BTC',
        'Total_Transactions_Per_Day',
        'NASDAQ Composite',
        'Dow Jones Industrial Avg',
        'S&P 500',
        'Google Trends Interest',
        'Breakeven Inflation Rate',
    ]
    cols = [
        'date',
        'price',
        'market_cap',
        'difficulty',
        'miners_revenue',
        'transaction_fees',
        'total_transactions_per_day',
        'nasdaq_composite',
        'dji_avg',
        'sp500',
        'google_trends',
        'inflation_rate',
    ]
    source_file = 'block_chain_ts.csv'
    btc_data = read_csv(source_file, original_cols, cols)
    stat_cols = list(filter(lambda x: x != 'date', cols))

    basic_statistics(btc_data, with_graph=False)
    descriptive_statistics(btc_data, stat_cols)
    plot_trends(btc_data, stat_cols, with_graph=True)

if __name__ == '__main__':
    main()

