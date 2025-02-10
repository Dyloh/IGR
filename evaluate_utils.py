import numpy as np


def test_beta(beta, Sigma_test, u_test, y_test):
    """
    return R^2 over test set
    1 - (RSS) / (Var[y])
    """
    
    return 1 - (beta.T @ Sigma_test @ beta - 2 * u_test.T @ beta + np.mean(np.square(y_test))) / np.mean(np.square(y_test))


def test_beta_from_Xy(beta: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """
    1 - (RSS) / (Var[y])
    """
    
    n, d = X_test.shape
    assert beta.shape == (d,)
    assert y_test.shape == (n,)
    return 1 - np.mean(np.square(X_test @ beta - y_test)) / np.mean(np.square(y_test))


def _calculate_r2(beta_list: np.ndarray, Sigma_test: np.ndarray, u_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """Calculate the R2 of each beta on the test set.
    
    Input
    -----
    beta_list: np.ndarray, shape=(n_gamma, n_lambda, d)
        The estimated beta for each (gamma, lambda) pair.
    Sigma_test: np.ndarray, shape=(d, d)
        The covariance matrix of the test set.
    u_test: np.ndarray, shape=(d,)
        The X^T y of the test set.
    y_test: np.ndarray, shape=(n,)
        The target of the test set.
    
    Returns
    -------
    r2_result_list: np.ndarray, shape=(n_gamma, n_lambda)
        The R2 of each beta.
    """
    n_gamma, n_lambda, d = beta_list.shape
    assert Sigma_test.shape == (d, d) and u_test.shape == (d,) and y_test.shape[0] > 0
    r2_result_list = np.zeros([n_gamma, n_lambda], np.float64)
    
    for gamma_id in range(n_gamma):
        for lambda_id in range(n_lambda):
            _result = test_beta(beta=beta_list[gamma_id, lambda_id, :], Sigma_test=Sigma_test, u_test=u_test, y_test=y_test)
            r2_result_list[gamma_id, lambda_id] = _result
    
    return r2_result_list


def calculate_worst_r2(beta_list: np.ndarray, Sigma_list: list[np.ndarray], u_list: list[np.ndarray], y_list: list[np.ndarray], debug: bool = False) -> np.ndarray:
    """
    Return the worst R2 of each beta across all environments.
    
    Parameters
    ----------
    beta_list : np.ndarray, shape=(n_gamma, n_lambda, d)
        The estimated beta for each (gamma, lambda) pair.
    Sigma_list : List[np.ndarray], each with shape=(d, d)
        The list of covariance matrix of the test set.
    u_list : List[np.ndarray], each with shape=(d,)
        The list of (X^T y)/n of the test set.
    y_list : List[np.ndarray], each with shape=(n,)
        The list of target of the test set.
    Returns
    -------
        result_list_worst_test: np.ndarray, shape=(n_gamma, n_lambda)
            The worst R2 of each beta across all environments.
    """
    n_gamma, n_lambda, d = beta_list.shape
    E = len(Sigma_list)
    assert len(u_list) == E == len(y_list)
    result_table_test = np.zeros((E, n_gamma, n_lambda))
    
    result_list_worst_test = np.full((n_gamma, n_lambda), np.inf)
    for i_e, Sigma_e, u_e, y_e in zip(range(len(Sigma_list)), Sigma_list, u_list, y_list, strict=True):
        assert Sigma_e.shape == (d, d) and u_e.shape == (d,) and y_e.shape[0] > 0
        result_list = _calculate_r2(beta_list=beta_list, Sigma_test=Sigma_e, u_test=u_e, y_test=y_e)
        result_table_test[i_e] = result_list
        result_list_worst_test = np.minimum(result_list_worst_test, result_list)

    assert np.allclose(result_list_worst_test, np.min(result_table_test, axis=0), atol=1e-5)
    
    
    if debug:
        return result_list_worst_test, result_table_test
    else:
        return result_list_worst_test
    

def get_acc(result_valid_list: np.ndarray, result_test_list: np.ndarray) -> float:
    """
    Report the performance of a model. We first find the best hyperparameters on the validation set, and then report the performance on the test set.
    
    Input:
    ------
    result_valid_list: np.ndarray, shape=(n_gamma, n_lambda)
        The performance of each hyperparameter on the validation set.
    result_test_list: np.ndarray, shape=(n_gamma, n_lambda)
        The performance of each hyperparameter on the test set.
    
    Returns:
    --------
    acc: float
        The performance of the best hyperparameter on the test set.
    """
    assert result_valid_list.shape == result_test_list.shape
    assert len(result_valid_list.shape) == 2
    
    best_valid_index = np.unravel_index(np.argmax(result_valid_list), result_valid_list.shape)
    assert len(best_valid_index) == 2
    return result_test_list[best_valid_index]


def get_acc_per_seed(result_valid_table: np.ndarray, result_test_table: np.ndarray) -> np.ndarray:
    """
    Report the performance of a model on multiple seeds.
    
    Input:
    ------
    result_valid_table: np.ndarray, shape=(n_seed, n_gamma, n_lambda)
        The performance of each hyperparameter on the validation set.
    result_test_table: np.ndarray, shape=(n_seed, n_gamma, n_lambda)
        The performance of each hyperparameter on the test set.
    
    Returns:
    --------
    acc_list: np.ndarray, shape=(n_seed,)
        The performance of the best hyperparameter on the test set.
    """
    
    n_seed = result_valid_table.shape[0]
    return np.array([get_acc(result_valid_table[i], result_test_table[i]) for i in range(n_seed)])


