import numpy as np
from itertools import combinations, product
import math
from tqdm import tqdm
import gzip
import pickle
from typing import Optional, Union, Sequence
import cvxpy as cp
import os




def calculating_wk(k: int, Sigma_list: list[np.ndarray], u_list: list[np.ndarray], 
                   n_list: Optional[Sequence[int]] = None, silent: bool = False, path: str = None) -> tuple[np.ndarray, bool]:
    """
        Retuen w_k(j) for each j=1, 2, ..., d.
        
        Parameters
        ----------
        k: int
            The computational budget i.e. the maximal size of tuples to consider.
        Sigma_list: list[np.ndarray]
            The list of covariance matrix of training environments. They should be normalized in advance if intended.
        u_list: list[np.ndarray]
            The list of u-vectors,  i.e. X^T y. They should be normalized in advance if intended.
        n_list: Optional[Sequence[int]]
            The vector of weights of each environment. Normally it is the number of samples in each environment. If not specified, then n_e = 1.
        silent: bool
            Whether to show the progress bar.
        path: Optional[str]
            The path to load or save the result.
        
        Returns
        -------
        w_table: npndarray, shape = (d, k)
            The square rooted w_table.
        read_from_cache_flag: bool
            Whether the result is read from cache.
    """
    
    d = np.shape(Sigma_list[0])[0]
    
    assert len(Sigma_list) == len(u_list)
    n_e = len(u_list)
    
    
    Sigma_mean = np.zeros((d, d), dtype=np.float64)
    u_mean = np.zeros((d,), dtype=np.float64)
    if n_list is None:
        n_list = [1] * n_e
    for e in range(n_e):
        assert len(u_list[e].shape) == 1
        assert n_list[e] > 0
        Sigma_mean += Sigma_list[e] * n_list[e]
        u_mean += u_list[e] * n_list[e]
    Sigma_mean /= sum(n_list)
    u_mean /= sum(n_list)
    
    read_from_cache_flag = False
    if path is not None and os.path.exists(path):
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
            assert np.allclose(data["Sigma_mean"], Sigma_mean, atol=1e-6)
            assert np.allclose(data["u_mean"], u_mean, atol=1e-6)
            assert data["k"] >= k
            assert data["w_table_group"].shape[0] == d
            w_table = data['w_table_group'][:, :k]
            read_from_cache_flag = True
            return w_table, read_from_cache_flag
        
    
    
    w_table = np.full([d, k], np.inf, dtype=np.float64)
    

    for k_ in range(1, k+1): 
        if not silent:
            iterator = tqdm(combinations(range(d), k_), total=math.comb(d, k_), desc=f"searching {k_}-tuples")
        else:
            iterator = combinations(range(d), k_)
            
        for feature_indices in iterator:
            heterogeneity_of_indices = 0.
            
            feature_indices = list(feature_indices)
            
            Sigma_S = Sigma_mean[np.ix_(feature_indices, feature_indices)]
            u_S = u_mean[feature_indices]
            beta_S = np.linalg.lstsq(Sigma_S, u_S, rcond=1e-6)[0]
            
            
            for e in range(n_e):
                Sigma_e_S = Sigma_list[e][np.ix_(feature_indices, feature_indices)]
                u_e_S = u_list[e][feature_indices]
                beta_e_S = np.linalg.lstsq(Sigma_e_S, u_e_S, rcond=1e-6)[0]
                
                heterogeneity_of_indices += (beta_S - beta_e_S).T @ Sigma_e_S @ (beta_S - beta_e_S)
            heterogeneity_of_indices /= n_e
            for i_feature in feature_indices:
                w_table[i_feature, k_-1] = min(w_table[i_feature, k_-1], heterogeneity_of_indices)
                
        w_table[:, k_-1] = np.sqrt(w_table[:, k_-1])
        if k_ >= 2:
            w_table[:, k_-1] = np.minimum(w_table[:, k_-1], w_table[:, k_-2])
    
    if path is not None:
        with gzip.open(path, "wb") as f:
            pickle.dump(
            {
                "Sigma_mean": Sigma_mean,
                "u_mean": u_mean,
                "k": k,
                "w_table_group": w_table
            },
            f,
            )
    
    return w_table, read_from_cache_flag


def IGR(k : int, Sigma_list: list[np.ndarray], u_list: list[np.ndarray], n_list: Optional[Sequence[int]] = None,
        gamma_sequence: Optional[Sequence[float]] = None, lambda_sequence: Optional[Sequence[float]] = None) -> np.ndarray:
    r"""
    The IGR algorithm we propose in our paper.
    
    Minimize $$\frac{1}{2n}\|X^\top\beta-y\|_2^2 + \gamma\|weights\odot\beta\|_1 + \lambda \|\beta\|_1$$ where weights are defined by (3.7).
    
    Parameters
    ----------
    k: int
        The computational budget i.e. the maximal size of tuples to consider.
    Sigma_list: list[np.ndarray]
        The list of covariance matrix of training environments. They should be normalized in advance if intended.
    u_list: list[np.ndarray]
        The list of u-vectors,  i.e. X^T y. They should be normalized in advance if intended.
    n_list: Optional[Sequence[int]]
        The vector of weights of each environment. Normally it is the number of samples in each environment. If not specified, then n_e = 1.
    gamma_sequence: Optional[Sequence[float]]
        The sequence of gamma values of hyperparameter.
    lambda_sequence: Optional[Sequence[float]]
        The sequence of lambda values of hyperparameter.
    
    Returns
    -------
    beta_IR_k_list: np.ndarray, shape=(n_gamma, n_lambda, d,)
        The estimated beta for each lambda.
    """
    assert len(Sigma_list) == len(u_list)
    E = len(Sigma_list)
    if n_list is None:
        n_list = [1] * E
    assert len(n_list) == E
    
    d = Sigma_list[0].shape[0]
    Sigma_mean = np.zeros((d, d), dtype=np.float64)
    u_mean = np.zeros((d,), dtype=np.float64)
    
    for e in range(E):
        assert len(u_list[e].shape) == 1
        Sigma_mean += Sigma_list[e] * n_list[e]
        u_mean += u_list[e] * n_list[e]
    Sigma_mean /= sum(n_list)
    u_mean /= sum(n_list)    
    
    w_table, _ = calculating_wk(k=k, Sigma_list=Sigma_list, u_list=u_list, n_list=n_list, silent=True)
    w_array = w_table[:, k-1]
    _, beta_IR_k_list, _ = seq_solving_L1_Problem(
                            weights=w_array, gamma_sequence=gamma_sequence, lambda_sequence=lambda_sequence,
                            Sigma_train=Sigma_mean, u_train=u_mean,
                            silent=True, )
    return beta_IR_k_list


from evaluate_utils import test_beta, calculate_worst_r2



def seq_solving_L1_Problem(weights: np.ndarray, gamma_sequence: Optional[Sequence[float]] = None, lambda_sequence: Optional[Sequence[float]] = None, 
                           *,
                           Sigma_train: np.ndarray, u_train: np.ndarray, X_train: Optional[np.ndarray] = None, y_train: Optional[np.ndarray] = None,
                           Sigma_test=None, u_test=None, y_test=None, beta_truth=None, 
                           ridge_parameter : Optional[float] = None, 
                           silent: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Solving the standard L1 regularized Problem, namely,
    
    minimize $$ \frac{1}{2n}\|X^\top\beta-y\|_2^2 + \gamma\|weights\odot\beta\|_1 + \lambda \|\beta\|_1 + ridge_parameter \|beta\|_2^2$$ .
    
    To alleviate the computational burden, if X_train and y_train are not provided, we use Sigma_train and u_train to calculate the loss term.
    
    This function can be used for IGR and LASSO, and also for regression on selected covariates (e.g. PCMCI).
    
    Parameters
    ----------
    weights: np.ndarray, shape=(d,)
        the weights for each feature.
    gamma_sequence: Sequence[float], length=n_gamma
        the sequence of gamma values of hyperparameter.
    lambda_sequence: Sequence[float], length=n_lambda
        the sequence of lambda values of hyperparameter.
    Sigma_train: np.ndarray, shape=(d, d)
        the covariance matrix of the training set.
    u_train: np.ndarray, shape=(d,)
        the u vector $(X^\top y)/n$ of the training set.
    X_train: np.ndarray, shape=(n, d)
        optional. the design matrix of the training set.
    y_train: np.ndarray, shape=(n,)
        optional. the target variable of the training set.
    ridge_parameter: Optional[float]
        if provided, then add a ridge regularization term.
    silent: bool
        whether to show the progress bar.
    
    Returns
    -------
    r2_result_list: np.ndarray, shape=(n_gamma, n_lambda)
        if (Sigma_test, y_test, u_test) is provided, return the R2 of beta on the test set.
    beta_table: np.ndarray, shape=(n_gamma, n_lambda, d)
        the betas of each hyperparameter.
    est_error_list: np.ndarray, shape=(n_gamma, n_lambda)
        if beta_truth is not None, then return the estimation error.
    
    """
    assert weights.shape[0] == u_train.shape[0] == Sigma_train.shape[0]
    d = weights.shape[0]
    
    
    if gamma_sequence is None and lambda_sequence is None:
        raise ValueError("Both gamma_sequence and lambda_sequence are None.")
        
    
    beta = cp.Variable(d)
    gamma = cp.Parameter(nonneg=True)
    lambda_ = cp.Parameter(nonneg=True)
    
    l1_term = gamma * cp.norm1(cp.multiply(weights, beta)) # define L1 regularization term
    if gamma_sequence is None:
        gamma_sequence = [0.]
    
    if lambda_sequence is not None:
        l1_term += lambda_ * cp.norm1(beta)
    else:
        lambda_sequence = [0.]
        
    n_gamma, n_lambda = len(gamma_sequence), len(lambda_sequence)

    if X_train is not None and y_train is not None:
        assert X_train.shape[0] == y_train.shape[0]
        assert X_train.shape[1] == d
        n = X_train.shape[0]
        l2_term = cp.sum_squares(X_train @ beta - y_train) / (2 * n)
    else:
        l2_term = beta.T @ cp.psd_wrap(0.5 * Sigma_train) @ beta - (u_train.T) @ beta # 1e-4
    if ridge_parameter is not None:
        assert isinstance(ridge_parameter, float)
        l2_term += ridge_parameter * cp.pnorm(beta, p=2)**2
    
    loss = l2_term + l1_term

    problem = cp.Problem(cp.Minimize(loss))
    
    r2_result_list = np.zeros([n_gamma, n_lambda], np.float64)
    beta_table = np.zeros([n_gamma, n_lambda, d], np.float64)
    esti_error_list = np.zeros([n_gamma, n_lambda], np.float64)
    
    # assign hyperparameter values and solve
    iterator = product(range(n_gamma), range(n_lambda))
    if not silent:
        iterator = tqdm(iterator, total=n_gamma*n_lambda, desc="Solving L1 Problem")
    
    for gamma_id, lambda_id in iterator:
        gamma.value = gamma_sequence[gamma_id]
        lambda_.value = lambda_sequence[lambda_id]
        
        solve_result = problem.solve(solver="GUROBI")
        if problem.status != "optimal":
            raise RuntimeError(f"Optimization failed:{solve_result}, {problem.status},{gamma.value},{lambda_.value}")
        beta_table[gamma_id, lambda_id, :] = beta.value
        
        
        # evaluation stage
        if beta_truth is not None:
            esti_error_list[gamma_id, lambda_id] = np.linalg.norm(beta_truth - beta.value)
        if Sigma_test is not None and u_test is not None and y_test is not None:
            _result = test_beta(beta=beta.value, Sigma_test=Sigma_test, u_test=u_test, y_test=y_test)
        else:
            _result = None
        r2_result_list[gamma_id, lambda_id] = _result

    return r2_result_list, beta_table, esti_error_list





def train_and_evaluate_worst_R2(*, w_array: np.ndarray, gamma_sequence: Union[list[float], np.ndarray], lambda_sequence: Union[list[float], np.ndarray],
                   Sigma_list: list[np.ndarray], u_list: list[np.ndarray], y_list: list[np.ndarray], 
                   train_id_slice: slice, valid_id_slice: slice, test_id_slice: slice, 
                   ridge_parameter : Optional[float] = None, 
                   debug : bool = False):
    r"""
    For any method using seq_solve_L1_Problem, we can use this function to evaluate the worst-case R2 on validation and test environments.
    
    Parameters
    ----------
    weights: np.ndarray, shape=(d,)
        the weights for each feature.
    gamma_sequence: Sequence[float], length=n_gamma
        the sequence of gamma values of hyperparameter.
    lambda_sequence: Sequence[float], length=n_lambda
        the sequence of lambda values of hyperparameter.
    Sigma_list: List[np.ndarray]
        the list of covariance matrix of each environment.
    u_list: List[np.ndarray]
        the list of u vectors (X^\top y)/n of each environment.
    y_list: List[np.ndarray]
        the list of target variables of each environment.
    train_id_slice: slice
        the slice of training environments, e.g. slice(0, 2).
    valid_id_slice: slice
        the slice of validation environments, e.g. slice(2, 3).
    test_id_slice: slice
        the slice of test environments, e.g. slice(3, 7).
    ridge_parameter: Optional[float]
        if provided, then add a ridge regularization term.
    debug: bool
        whether to return the full list of R2 on the test set.
    """
    E = len(Sigma_list)
    assert len(u_list) == len(y_list)
    train_id_list = list(range(E))[train_id_slice]
    Sigma_train = sum([Sigma_list[i] for i in train_id_list]) / len(train_id_list)
    u_train = sum([u_list[i] for i in train_id_list]) / len(train_id_list)
    
    _, beta_IR_k_list, _ = seq_solving_L1_Problem(
                            weights=w_array, gamma_sequence=gamma_sequence, lambda_sequence=lambda_sequence,
                            Sigma_train=Sigma_train, u_train=u_train,
                            silent=True, 
                            ridge_parameter=ridge_parameter)
    result_IR_list_valid = calculate_worst_r2(beta_list=beta_IR_k_list, Sigma_list=Sigma_list[valid_id_slice], 
                                    u_list=u_list[valid_id_slice], y_list=y_list[valid_id_slice])
    if debug:
        result_IR_list_test, result_IR_fulllist_test = calculate_worst_r2(beta_list=beta_IR_k_list, Sigma_list=Sigma_list[test_id_slice], 
                                    u_list=u_list[test_id_slice], y_list=y_list[test_id_slice], debug=debug)
    else:
        result_IR_list_test = calculate_worst_r2(beta_list=beta_IR_k_list, Sigma_list=Sigma_list[test_id_slice], 
                                    u_list=u_list[test_id_slice], y_list=y_list[test_id_slice])
        
    if debug:
        return result_IR_list_valid, result_IR_list_test, beta_IR_k_list, result_IR_fulllist_test
    else:
        return result_IR_list_valid, result_IR_list_test, beta_IR_k_list




def _causal_dantzig(Sigma_list: list[np.ndarray], u_list: list[np.ndarray], lambda_list: Sequence[float]) -> np.ndarray:
    r"""
    The Causal Dantzig (Rothenh¨ausler et al., 2019).
    
    Parameters
    ----------
    Sigma_list: List[np.ndarray], each with shape=(d, d)
        The list of covariance matrix of the training set.
    u_list: List[np.ndarray], each with shape=(d,)
        The list of (X^T y)/n of the training set.
    lambda_list: Sequence[float]
        The sequence of lambda values of hyperparameter.
        
    Returns
    -------
    beta_l_CD: np.ndarray, shape=(n_lambda, d)
        The estimated beta for each lambda.
    """
    n_lambda = len(lambda_list)
    assert len(Sigma_list) == 2
    assert len(u_list) == 2
    Sigma = Sigma_list[0] - Sigma_list[1]
    u = u_list[0] - u_list[1]
    assert len(u.shape) == 1
    d = u.shape[0]
    
    beta_l_CD = np.zeros([n_lambda, d], np.float64)
    
    beta = cp.Variable(d)
    lambda_ = cp.Parameter(nonneg=True)
    
    constraints = [
               (u - Sigma @ beta) <= lambda_, 
               (u - Sigma @ beta) >= -lambda_,
               ]
    objective = cp.Minimize(cp.norm1(beta))
    problem = cp.Problem(objective, constraints)
    for lambda_id, lambda_value in enumerate(lambda_list):
        lambda_.value = lambda_value
        problem.solve(solver="GUROBI")
        beta_l_CD[lambda_id, :] = beta.value
    return beta_l_CD
    

def Causal_Dantzig_train_and_evaluate(*, Sigma_list: list[np.ndarray], u_list: list[np.ndarray], y_list: list[np.ndarray], 
                   lambda_sequence: Sequence[float],
                   train_id_slice: slice, valid_id_slice: slice, test_id_slice: slice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Apply the Causal Dantzig estimator on train environments and evaluate worst R2 on validation and test environments.
    
    Parameters
    ----------
    Sigma_list : list of np.ndarray
        List of covariance matrices for each environment.
    u_list : list of np.ndarray
        List of (X^t y)/n for each environment.
    y_list : list of np.ndarray
        List of response variables for each environment.
    lambda_sequence : Sequence[float]
        Sequence of hyperparameters to be used in the Dantzig estimator.
    train_id_slice : slice
        Slice object to index the training environments.
    valid_id_slice : slice
        Slice object to index the validation environments.
    test_id_slice : slice
        Slice object to index the test environments.
        
    Returns
    -------
    r_v_l_CD : np.ndarray
        Worst R2 on validation environments of each lambda.
    r_t_l_CD : np.ndarray
        Worst R2 on test environments of each lambda.
    beta_l_CD : np.ndarray
        Estimated coefficients of each lambda.
    """
    
    assert len(y_list) == len(Sigma_list) == len(u_list)
    beta_l_CD = _causal_dantzig(Sigma_list=Sigma_list[train_id_slice], u_list=u_list[train_id_slice], lambda_list=lambda_sequence)[None,:,:]
    r_v_l_CD = calculate_worst_r2(beta_list=beta_l_CD, Sigma_list=Sigma_list[valid_id_slice], u_list=u_list[valid_id_slice], y_list=y_list[valid_id_slice])
    r_t_l_CD = calculate_worst_r2(beta_list=beta_l_CD, Sigma_list=Sigma_list[test_id_slice], u_list=u_list[test_id_slice], y_list=y_list[test_id_slice])
    return r_v_l_CD, r_t_l_CD, beta_l_CD
    



def Anchor_Regression_train_and_evaluate(X_list: list[np.ndarray], y_list: list[np.ndarray], Sigma_list: list[np.ndarray], u_list: list[np.ndarray],
                        gamma_sequence: Sequence[float], lambda_sequence: Sequence[float],
                        train_id_slice: slice, valid_id_slice: slice, test_id_slice: slice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Perform anchor regression to estimate the regression coefficients under different regularization parameters.
    
    Parameters:
    X_list : List[np.ndarray]
        List of feature matrices for different environments.
    y_list : List[np.ndarray]
        List of response vectors for different environments.
    Sigma_list : List[np.ndarray]
        List of covariance matrices for different environments.
    u_list : List[np.ndarray]
        List of (X^\top y)/n vectors for different environments.
    gamma_sequence : Sequence[float]
        Sequence of gamma values for regularization.
    lambda_sequence : Sequence[float]
        Sequence of lambda values for regularization.
    train_id_slice : slice
        Slice object to select training environments.
    valid_id_slice : slice
        Slice object to select validation environments.
    test_id_slice : slice
        Slice object to select test environments.
    
    Returns:
    --------
    r_v_l_AR : np.ndarray
        Worst-case R^2 values on the validation set for each combination of gamma and lambda.
    r_t_l_AR : np.ndarray
        Worst-case R^2 values on the test set for each combination of gamma and lambda.
    beta_l_AR : np.ndarray
        Estimated regression coefficients for each combination of gamma and lambda.
    """
    n_gamma, n_lambda = len(gamma_sequence), len(lambda_sequence)
    beta_l_AR = np.zeros((len(gamma_sequence), len(lambda_sequence), X_list[0].shape[1]), dtype=np.float64)
    r_v_l_AR = np.zeros((len(gamma_sequence), len(lambda_sequence)), dtype=np.float64)
    r_t_l_AR = np.zeros((len(gamma_sequence), len(lambda_sequence)), dtype=np.float64)
    
    d = X_list[0].shape[1]
    n = sum(X_e.shape[0] for X_e in X_list[train_id_slice])
    
    beta = cp.Variable(d)
    gamma_rooted = cp.Parameter(nonneg=True)
    lambda_ = cp.Parameter(nonneg=True)
    
    l1_term = lambda_ * cp.norm1(beta)
    l2_term = cp.Constant(0)
    for X_e, y_e in zip(X_list[train_id_slice], y_list[train_id_slice]):
        X_e_demean = X_e - np.mean(X_e, axis=0, keepdims=True)
        y_e_demean = y_e - np.mean(y_e, keepdims=True)
        assert np.allclose(X_e-X_e_demean, 0, atol=1e-6)
        assert np.allclose(y_e-y_e_demean, 0, atol=1e-6) # in our case
        X_e_tilde = X_e_demean + gamma_rooted * (X_e-X_e_demean)
        y_e_tilde = y_e_demean + gamma_rooted * (y_e-y_e_demean)
        l2_term += cp.sum_squares(X_e_tilde @ beta - y_e_tilde) / (n * 2)
    loss = l2_term + l1_term
    problem = cp.Problem(cp.Minimize(loss))
    
    for i_gamma, i_lambda in product(range(n_gamma), range(n_lambda)):
        gamma_rooted.value = gamma_sequence[i_gamma] ** 0.5
        lambda_.value = lambda_sequence[i_lambda]
        problem = cp.Problem(cp.Minimize(loss))
        problem.solve()
        beta_l_AR[i_gamma, i_lambda, :] = beta.value
    r_v_l_AR = calculate_worst_r2(beta_list=beta_l_AR, Sigma_list=Sigma_list[valid_id_slice], u_list=u_list[valid_id_slice], y_list=y_list[valid_id_slice])
    r_t_l_AR = calculate_worst_r2(beta_list=beta_l_AR, Sigma_list=Sigma_list[test_id_slice], u_list=u_list[test_id_slice], y_list=y_list[test_id_slice])
    return r_v_l_AR, r_t_l_AR, beta_l_AR
    


def _est_drig(data, gamma, y_idx=-1, del_idx=None, unif_weight=False):
    r"""DRIG estimator from [SBT23].
    
    The method include the docstring are adopted from the original code of [SBT23]. The only difference is that we add 1e-6 * np.eye(G.shape[0]) to avoid singular matrix.

    Args:
        data (list of numpy arrays): a list of data from all environments, where the first element is the observational environment.
        gamma (float): hyperparameter in DRIG.
        y_idx (int, optional): index of the response variable. Defaults to -1.
        del_idx (int, optional): index of the variable to exclude. Defaults to None.
        unif_weight (bool, optional): whether to use uniform weights. Defaults to False.

    Returns:
        numpy array: estimated coefficients.
    """
    if del_idx is None:
        del_idx = y_idx
    ## number of environment
    m = len(data)
    if unif_weight:
        w = [1/m]*m
    else:
        w = [data[e].shape[0] for e in range(m)]
        w = [a/sum(w) for a in w]
    ## gram matrices
    gram_x = [] ## E[XX^T]
    gram_xy = [] ## E[XY]
    for e in range(m):
        data_e = data[e]
        n = data_e.shape[0]
        y = data_e[:, y_idx]
        x = np.delete(data_e, (y_idx, del_idx), 1)
        # x = data_e[:, :-1]
        # y = data_e[:, -1]
        gram_x.append(x.T.dot(x)/n)
        gram_xy.append(x.T.dot(y)/n)
    G = (1 - gamma)*gram_x[0] + gamma*sum([a*b for a,b in zip(gram_x, w)])
    Z = (1 - gamma)*gram_xy[0] + gamma*sum([a*b for a,b in zip(gram_xy, w)])
    G += 1e-6 * np.eye(G.shape[0]) # additional term to make G invertible
    return np.linalg.inv(G).dot(Z)



def DRIG_train_and_evaluate(*, X_list: list[np.ndarray], y_list: list[np.ndarray], Sigma_list: list[np.ndarray], u_list: list[np.ndarray],
                    gamma_sequence_DRIG : Sequence[float], 
                    train_id_slice: slice, valid_id_slice: slice, test_id_slice: slice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Apply the DRIG estimator on train environments and evaluate worst R2 on validation and test environments.
    
    Parameters
    ----------
    X_list : List[np.ndarray]
        List of feature matrices for different environments.
    y_list : List[np.ndarray]
        List of response vectors for different environments.
    Sigma_list : list of np.ndarray
        List of covariance matrices for each environment.
    u_list : list of np.ndarray
        List of (X^t y)/n for each environment.
    gamma_sequence_DRIG : Sequence[float]
        Sequence of hyperparameters to be used.
    train_id_slice : slice
        Slice object to index the training environments.
    valid_id_slice : slice
        Slice object to index the validation environments.
    test_id_slice : slice
        Slice object to index the test environments.
        
    Returns
    -------
    r_v_l_DRIG : np.ndarray
        Worst-case R^2 values on the validation set for each gamma.
    r_t_l_DRIG : np.ndarray
        Worst-case R^2 values on the test set for each gamma.
    beta_l_DRIG : np.ndarray
        Estimated regression coefficients for each gamma.
    """
    beta_l_DRIG = np.zeros((len(gamma_sequence_DRIG), Sigma_list[0].shape[0]), dtype=np.float64)

    for i_gamma_DRIG, gamma_DRIG in enumerate(gamma_sequence_DRIG):
        beta_temp = _est_drig([np.hstack([X_e, y_e[:,None]]) for X_e, y_e in zip(X_list[train_id_slice], y_list[train_id_slice])], 
                             gamma=gamma_DRIG, y_idx=-1, unif_weight=True)
        beta_l_DRIG[i_gamma_DRIG] = beta_temp

    beta_l_DRIG = beta_l_DRIG[:,None,:]
    r_v_l_DRIG = calculate_worst_r2(beta_list=beta_l_DRIG, Sigma_list=Sigma_list[valid_id_slice], u_list=u_list[valid_id_slice], y_list=y_list[valid_id_slice])
    r_t_l_DRIG = calculate_worst_r2(beta_list=beta_l_DRIG, Sigma_list=Sigma_list[test_id_slice], u_list=u_list[test_id_slice], y_list=y_list[test_id_slice])
    assert r_v_l_DRIG.shape == r_t_l_DRIG.shape == (len(gamma_sequence_DRIG), 1)

    return r_v_l_DRIG, r_t_l_DRIG, beta_l_DRIG



import numpy as np
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from data_processing import loading_stock_data

def pcmci_selection():
    LMIN, LMAX = 0, 1
    TARGET_INDEX_LIST = (8, 84)
    stock_log_return_raw, _ = loading_stock_data()
    X_train_for_pcmci = stock_log_return_raw[650:850, :]


    pcmci_dict = {}
    debug_data = pp.DataFrame(X_train_for_pcmci)
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=debug_data, cond_ind_test=parcorr, verbosity=1)
    results = pcmci.run_pcmci(tau_max=LMAX, tau_min=LMIN, pc_alpha=0.01)["graph"]

    for target_ind in TARGET_INDEX_LIST:
        pcmci_dict[str(target_ind)] = np.concatenate([np.where(results[:, target_ind, 0]!="")[0],np.where(results[:, target_ind, 1]!="")[0]+100])

    np.savez(f"./pre_analysis/stock_pcmci_{LMIN}{LMAX}.npz", **pcmci_dict)




if __name__ == '__main__':
    pass
    
                
                
                
                    
                    
                    
    
    