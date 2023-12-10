import numpy as np
import pandas as pd

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def NMAE(pred, true):
    return np.sum(np.abs(true-pred))/np.sum(true)

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def NRMSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2))*np.sqrt(len(pred))/np.sum(true)

def AMBE(pred, true):
    return np.abs(np.mean(pred - true))

def MAPE(pred, true):
    return np.mean((np.abs((pred - true)+1) / (true+1)))

def MSPE(pred, true):
    return np.mean((np.square((pred - true) +1)/ (true+1)))

def _aggregate_fn(df):
    return pd.Series({
        'label_mean': np.mean(df['y_true']),
        'pred_mean': np.mean(df['y_pred']),
        'normalized_mae': NMAE(df['y_true'], df['y_pred']),
        'normalized_rmse': NRMSE(df['y_true'], df['y_pred']),
        'log_mae':MAE(np.log(df['y_true']+1),np.log(df['y_pred']+1)),
        'log_mae_max':np.abs(np.log(df['y_true']+1)-np.log(df['y_pred']+1)).max()
    })

def cumulative_true(y_true,y_pred):
    """Calculates cumulative sum of lifetime values over predicted rank.

    Arguments:
        y_true: true lifetime values.
        y_pred: predicted lifetime values.

    Returns:
        res: cumulative sum of lifetime values over predicted rank.
    """
    df = pd.DataFrame({
        'y_true': y_true.squeeze(),
        'y_pred': y_pred.squeeze(),
    }).sort_values(
        by='y_pred', ascending=False)

    return (df['y_true'].cumsum() / df['y_true'].sum()).values


def gini_from_gain(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates gini coefficient over gain charts.

    Arguments:
        df: Each column contains one gain chart. First column must be ground truth.

    Returns:
        gini_result: This dataframe has two columns containing raw and normalized
                    gini coefficient.
    """
    raw = df.apply(lambda x: 2 * x.sum() / df.shape[0] - 1.)
    normalized = raw / raw[0]
    return pd.DataFrame({
        'raw': raw,
        'normalized': normalized
    })[['raw', 'normalized']]



def metric(pred, true):
    pred=np.array(pred)
    true=np.array(true)
    mae = MAE(pred, true)
    nmae = NMAE(pred, true)
    rmse = RMSE(pred, true)
    nrmse = NRMSE(pred, true)
    ambe = AMBE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)


    # 分桶计算 10
    num_buckets = 10
    decile = pd.qcut(
        true.reshape(-1), q=num_buckets, labels=['%d' % i for i in range(num_buckets)])

    df = pd.DataFrame({
        'y_true': true.reshape(-1),
        'y_pred': pred.reshape(-1),
        'decile': decile,
    }).groupby('decile').apply(_aggregate_fn)

    df['decile_mape'] = np.abs(df['pred_mean'] -
                                df['label_mean']) / df['label_mean']
    print(df)

    # gain
    gain_perfect = cumulative_true(true, true)
    gain_model= cumulative_true(true, pred)
    gain = pd.DataFrame({
        'ground_truth': gain_perfect,
        'our_model': gain_model
    })
    gini = gini_from_gain(gain)
    print(gini)
    gini=gini.loc['our_model','normalized']
    return mae,nmae,rmse,nrmse,ambe,gini,mape,mspe