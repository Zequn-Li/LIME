import numpy as np
import tensorflow as tf
import scipy.stats as stats
import statsmodels.api as sm

def mse(y_true, y_pred):
    """Calculate Mean Squared Error."""
    y = np.array(y_true)
    y_hat = np.array(y_pred)
    y = y.reshape(-1)
    y_hat = y_hat.reshape(-1)
    return np.mean(np.square(y-y_hat))

def r2_score(y_true, y_pred):
    """Calculate R-squared score."""
    y = np.array(y_true)
    y_hat = np.array(y_pred)
    y = y.reshape(-1)
    y_hat = y_hat.reshape(-1)
    return 1 - np.sum(np.square(y-y_hat))/np.sum(np.square(y))


def r2_metrics_tf(y_true, y_pred):
    """TensorFlow implementation of R-squared for use as a Keras metric."""
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.dtypes.cast(y_true, tf.float64)
    y_pred = tf.dtypes.cast(y_pred, tf.float64)
    return 1 - tf.reduce_sum(tf.square(y_true-y_pred))/tf.reduce_sum(tf.square(y_true))

def newey_west_t_stat(y, maxlag=None):
    """Calculate Newey-West t-statistic."""
    x = np.ones_like(y)
    nobs = len(y)
    if maxlag is None:
        maxlag = int(np.ceil(12 * np.power(nobs / 100, 1/4)))
    
    model = sm.OLS(y, x)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': maxlag, 'use_correction': True})

    # Calculate the Newey-West t-statistic for the mean of y
    t_stat = results.params[0] / results.bse[0]
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), nobs - 1))
    sharpe_ratio = np.sqrt(12) * np.mean(y) / np.std(y)

    return {
        'mean': np.mean(y),
        'std_error': results.bse[0],
        't_stat': t_stat,
        'p_value': p_value,
        'sharpe_ratio': sharpe_ratio
    }

def apply_multiple_testing_adjustment(df, method='holm', alpha=0.05):
    """
    Apply multiple testing adjustment to p-values.
    
    Parameters:
    df : DataFrame with 'p_values' column
    method : str, 'holm' or 'bhy' (Benjamini-Hochberg-Yekutieli)
    alpha : float, significance level
    """
    df = df.sort_values(by='p_values').reset_index()
    m = len(df)  # number of tests
    
    if method == 'holm':
        # Holm's adjustment
        adjusted_p_values = [min(1, (m - i) * p) for i, p in enumerate(df['p_values'])]
    else:  # BHY adjustment
        c_m = np.sum(1 / np.arange(1, m + 1))
        adjusted_p_values = []
        prev_bh_value = 0
        for i, p_value in enumerate(df['p_values']):
            bh_value = min(1, c_m * m / (m - i) * p_value)
            bh_value = max(bh_value, prev_bh_value)
            adjusted_p_values.append(bh_value)
            prev_bh_value = bh_value
    
    df['adjusted_p_values'] = adjusted_p_values
    return df[df['adjusted_p_values'] < alpha].sort_index()