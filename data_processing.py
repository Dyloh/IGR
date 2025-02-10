import numpy as np
from itertools import product
from typing import Optional
from functools import partial 

def standardize(X_list: list[np.ndarray], y_list: list[np.ndarray], train_id_list: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Standardize the data by subtracting the mean and dividing by the standard deviation of training data.
    
    parameters
    ----------
    X_list: list[np.ndarray], shape=(E, n, d)
        The list of feature matrices.
    y_list: list[np.ndarray], shape=(E, n)
        The list of target variables.
    
    returns
    -------
    X_list: list[np.ndarray], shape=(E, n, d)
        The standardized feature matrices.
    y_list: list[np.ndarray], shape=(E, n)
        The standardized target variables.
    """
    assert len(X_list) == len(y_list)
    X_list = [X.copy() for X in X_list]
    y_list = [y.copy() for y in y_list]
    E = len(X_list)
    n, d = X_list[0].shape
    
    for e in range(E): # First we substract the mean for both X and Y in each environment
        X_e = X_list[e]
        y_e = y_list[e]
        X_mean = X_e.mean(axis=0, keepdims=True)
        y_mean = y_e.mean()
        X_list[e] = (X_e - X_mean)
        y_list[e] = (y_e - y_mean)
        
        assert X_mean.shape == (1, d,)
        assert np.allclose(X_list[e].mean(axis=0), 0, atol=1e-5)
        assert abs(np.mean(y_list[e])) < 1e-5
    
    train_X_std = np.vstack([X_list[e] for e in train_id_list]).std(axis=0, keepdims=True)
    train_y_std = np.hstack([y_list[e] for e in train_id_list]).std()
    assert np.vstack([X_list[e] for e in train_id_list]).shape == (sum(X_list[e].shape[0] for e in train_id_list), d)
    assert np.hstack([y_list[e] for e in train_id_list]).shape == (sum(y_list[e].shape[0] for e in train_id_list),)
    Y_INDEX = (train_X_std < 1e-8)
    assert Y_INDEX.sum() <= 1
    train_X_std[Y_INDEX] = 1.
    for e in range(E):
        X_list[e] = X_list[e] / train_X_std
        y_list[e] = y_list[e] / train_y_std

    
    assert np.allclose(np.vstack([X_list[e] for e in train_id_list]).std(axis=0) + Y_INDEX.astype(int)
                    , 1., atol=1e-5)
    assert len(np.hstack([y_list[e] for e in train_id_list]).shape) == 1
    assert abs(np.hstack([y_list[e] for e in train_id_list]).std()-1) < 1e-5
    assert np.allclose(np.vstack([X_list[e] for e in train_id_list]).mean(axis=0), 0, atol=1e-5)
    
    return X_list, y_list



def time_augment_X(X: np.ndarray, L_MIN: int, L_MAX: int, y_index: int, original_column_name: Optional[list[str]] = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Augment the feature matrix by L time steps, also extract the target variable y.
    the leftest column block is the nearest time step (t-L_MIN), the rightest column block is the (t-L_MAX) time step before the current time step
    
    Parameters:
    -----------
    X: np.ndarray, shape=(T, C)
        The design matrix
    L_MIN: int
        the nearest time step to be included, L_MIN=0 means the current time step
    L_MAX: int
        the farest time step to be included, L_MAX=1 means the previous time step
    Y_INDEX: int
        the index of the target variable
    original_column_name: Optional[list[str]]
        the name of the columns of the original feature matrix. If None, the column name will be the index of the columns
    
    Return:
    -------
    X_aug: np.ndarray, shape=(T-L_MAX, C*(L_MAX-L_MIN+1), )
        The augmented feature matrix
    y: np.ndarray, shape=(T-L_MAX,)
        The target variable
    column_name: list[str], length=C*(L_MAX-L_MIN+1)
        The name of the columns of the augmented feature matrix
    """
    n0, d0 = X.shape
    assert y_index < d0 and y_index >= 0 and L_MIN >= 0 and L_MAX >= L_MIN
    d = d0*(L_MAX-L_MIN+1)
    X_aug = np.zeros((n0-L_MAX, d), dtype=X.dtype)
    for l in range(L_MIN, L_MAX+1):
        X_aug[:, (l-L_MIN)*d0:(l-L_MIN+1)*d0] = X[L_MAX-l: n0-l, :].copy()
    
    y = X[L_MAX:, y_index]
    if original_column_name is None:
        original_column_name = [str(c) for c in range(X.shape[1])]
    assert len(original_column_name) == d0
    column_name = [str((t,original_column_name[c])) for t,c in product(range(L_MIN, L_MAX+1), range(X.shape[1]))]

    if L_MIN == 0:
        X_aug[:, y_index] = 0.
    
    return X_aug, y, column_name


def _simple_time_augment_X(X:np.ndarray, L, Y_INDEX, original_column_name=None):
    X_aug = np.zeros((X.shape[0]-L, X.shape[1]*(L+1)), dtype=X.dtype)
    for l in range(L+1):
        X_aug[:, l*X.shape[1]: (l+1)*X.shape[1]] = X[L-l:X.shape[0]-l, :].copy()
        
    y = X_aug[:, Y_INDEX].copy()
    X_aug[:, Y_INDEX] = 0.
    
    return X_aug, y, None


def loading_netsim_data(file_path : str, sim : int):
    """
    Load the netsim data from the path
    
    parameters
    ----------
    
    
    return
    ------
    X_raw: np.ndarray, shape=(T, C)
    adj_matrix: np.ndarray, shape=(C, C)
    """
    X_raw = np.load(file_path)["X"][sim,:,:]
    adj_matrix = np.load(file_path)["adj_matrix"]
    return X_raw.copy(), adj_matrix



def loading_stock_data():
    """
    Load the log-return stock data stored already.
    
    returns
    -------
    X_raw: np.ndarray, shape=(T, C)
        The log-return stock data in a daily frequency.
    adj_matrix: np.ndarray, shape=(C, C)
        The causal graph of the stock data. This is preserved to align with other datasets. It is set to be the identity matrix since the causal graph is not known.
    """
    X_raw = np.load('./data/stock_log_return.npy')
    X_raw = X_raw.T
    d = X_raw.shape[1]
    assert d==100
    adj_matrix = np.eye(d)
    return X_raw.copy(), adj_matrix



def test_time_augment():
    """
    Modified in 2025/1/16
    Suceessfully tested
    """
    X_raw, adj_matrix = loading_stock_data()
    X_aug, y, column_name = time_augment_X(X_raw, L_MIN=0, L_MAX=2, y_index=8, original_column_name=None)
    X_aug1, y1, column_name1 = _simple_time_augment_X(X_raw, L=2, Y_INDEX=8, original_column_name=None)
    assert np.allclose(X_aug, X_aug1, atol=1e-5)
    assert np.allclose(y, y1, atol=1e-5)
    assert np.allclose(X_aug[:, 100:200], X_raw[1:-1, :])
    assert np.allclose(X_aug[:, 200:300], X_raw[:-2, :])
    assert np.allclose(X_aug[:, 50:100], X_raw[2:, 50:])
    
    X_aug, y, column_name = time_augment_X(X_raw, L_MIN=1, L_MAX=3, y_index=8, original_column_name=None)
    assert np.allclose(y, y1[1:], atol=1e-5)
    assert np.allclose(X_aug[:, 100:200], X_raw[1:-2, :])
    assert np.allclose(X_aug[:, 200:300], X_raw[:-3, :])
    assert np.allclose(X_aug[:, 50:100], X_raw[2:-1, 50:])
    
    print(X_aug.shape)
    print(y.shape)
    

def angment_stock_data_and_split(y_index: int, seed: int, n_train: int, L_MAX: int = 1, L_MIN: int = 0, 
                                 train_id_list : Optional[list[int]] = None) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    r"""
    Provide split data for the time-augmented stock data. The training environments are bootstraped. This reveals the data used in experiment 1 in the paper. 
    
    The time windows are hard-coded.
    """
    assert L_MAX >= L_MIN >= 0
    from data_processing import loading_stock_data
    stock_log_return_raw, adj_matrix = loading_stock_data()
    TIME_WINDOW_DICT = {
        0: (650, 750),
        1: (750, 850),
        2: (850, 1050), # valid
        3: (1050, 1150),
        4: (1150, 1250),
        5: (1250, 1350),
        6: (1350, 1450), 
    }
    E = 7
    ###########################

    np.random.seed(seed)
    random_indeces_list = []
    for e in range(E):
        n_sample = TIME_WINDOW_DICT[e][1]-TIME_WINDOW_DICT[e][0] - L_MAX
        r_indices = np.random.choice(np.arange(n_sample), min(n_train, n_sample), replace=False)
        if n_train >= n_sample:
            r_indices = np.arange(n_sample)
        random_indeces_list.append(r_indices)


    from data_processing import time_augment_X
    time_augment_X = partial(time_augment_X, L_MAX=L_MAX, L_MIN=L_MIN, y_index=y_index, original_column_name=None)
    def prepare_data(e):
        time_start, time_end = TIME_WINDOW_DICT[e]
        return stock_log_return_raw[time_start:time_end, :].copy()
    X = stock_log_return_raw.copy()
    X_list = []
    y_list = []


    column_name = time_augment_X(prepare_data(e=0))[2]
    for e in range(E):
        X_e, y_e, _ = time_augment_X(prepare_data(e=e))
        if e in train_id_list and random_indeces_list is not None:
            X_e = X_e[random_indeces_list[e], :]
            y_e = y_e[random_indeces_list[e]]

        X_list.append(X_e)
        y_list.append(y_e)
    return X_list, y_list, column_name


if __name__ == '__main__':
    test_time_augment()


















































def debug_time_augment():
    """test completed"""
    L_MIN = 0
    L_MAX = 3
    X = np.zeros([8,3],dtype=object)
    for t in range(8):
        for c in range(3):
            X[t,c] = f'{t}-{c}'
    Y_INDEX = 2
    X_aug, y, column_name = time_augment_X(X, L_MIN, L_MAX, Y_INDEX)
    X

if __name__ == '__main__':
    pass