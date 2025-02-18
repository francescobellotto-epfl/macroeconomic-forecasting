import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_market_cap(data, figsize=(14,6)):
    """
    Plot mean of market capitalizations and their logarithm.

    Args:
        data: pandas Series containing market capitalizations
        figsize: size of the figure
    """

    # Figure definition
    plt.figure(figsize=figsize)

    # Plot the histogram of mean market cap
    plt.subplot(1, 2, 1)
    sns.histplot(data, bins=100, color='orange', kde=True)
    plt.title('Distribution of Mean Market Capitalization', fontsize=14, fontweight='bold')
    plt.xlabel('Mean Market Capitalization', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Compute logarithm of data
    log_data = np.log(data[data > 0])

    # Plot the logarithmic histogram of mean market cap
    plt.subplot(1, 2, 2)
    sns.histplot(log_data, bins=100, color='skyblue', kde=True)
    plt.title('Log Distribution of Mean Market Capitalization', fontsize=14, fontweight='bold')
    plt.xlabel('Log Mean Market Capitalization', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout for clarity
    plt.tight_layout()
    plt.show()

def plot_mc_evolution(top20_market_cap, crsp_df, sp500_df, permno_col, comnam_col, mktcap_col, figsize=(14,6)):

    # Figure definition
    plt.figure(figsize=figsize)

    # Plots
    for permno in top20_market_cap.index:
        plt.plot(crsp_df[crsp_df[permno_col] == permno][mktcap_col], label=sp500_df[sp500_df[permno_col] == permno][comnam_col].values[0])
    plt.title('Top 20 Companies by Average Market Capitalization', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Market Capitalization', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()  

def plot_returns(Y_pred_list, Y_test_list, by, best_idx=0):
    if by=='mean':
        Y_pred = Y_pred_list[0]
        Y_test = Y_test_list[0]
        
        quarters = ['Q1-2015', 'Q2-2015', 'Q3-2015', 'Q4-2015', 'Q1-2016', 'Q2-2016', 'Q3-2016', 'Q4-2016', 
            'Q1-2017', 'Q2-2017', 'Q3-2017', 'Q4-2017', 'Q1-2018', 'Q2-2018', 'Q3-2018', 'Q4-2018', 
            'Q1-2019', 'Q2-2019', 'Q3-2019', 'Q4-2019', 'Q1-2020', 'Q2-2020', 'Q3-2020', 'Q4-2020']
        
        plt.plot(quarters, Y_pred, label='Predictions', marker='o', color='blue')
        plt.plot(quarters, Y_test, label='Actual Values', marker='o', color='red')
        plt.xlabel('Period')
        plt.ylabel('S&P500 Return')
        plt.xticks(rotation=45)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    elif by=='pca':
        Y_pred = Y_pred_list[best_idx]
        Y_test = Y_test_list[best_idx]
        
        quarters = ['Q1-2015', 'Q2-2015', 'Q3-2015', 'Q4-2015', 'Q1-2016', 'Q2-2016', 'Q3-2016', 'Q4-2016', 
            'Q1-2017', 'Q2-2017', 'Q3-2017', 'Q4-2017', 'Q1-2018', 'Q2-2018', 'Q3-2018', 'Q4-2018', 
            'Q1-2019', 'Q2-2019', 'Q3-2019', 'Q4-2019', 'Q1-2020', 'Q2-2020', 'Q3-2020', 'Q4-2020']
        
        plt.plot(quarters, Y_pred, label='Predictions', marker='o', color='blue')
        plt.plot(quarters, Y_test, label='Actual Values', marker='o', color='red')
        plt.xlabel('Period')
        plt.ylabel('S&P500 Return')
        plt.xticks(rotation=45)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()