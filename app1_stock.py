import numpy as np
from functools import partial
import os
from tqdm import tqdm
from methods import Anchor_Regression_train_and_evaluate, Causal_Dantzig_train_and_evaluate, DRIG_train_and_evaluate, train_and_evaluate_worst_R2

### Configuration ###
L_MAX = 1
L_MIN = 0
K=2
n_train = 90
d = 100 * (L_MAX - L_MIN + 1)
NAME = "app1_stock"
pcmci_file_path = "./pre_analysis/stock_pcmci_01.npz"

# Split
E = 7
TRAIN_ID_SLICE = slice(0,2)
VALID_ID_SLICE = slice(2,3)
TEST_ID_SLICE = slice(3,7)
SLICE_DICT = {"train_id_slice": TRAIN_ID_SLICE, "valid_id_slice": VALID_ID_SLICE, "test_id_slice": TEST_ID_SLICE}

print(f"L_MAX: {L_MAX}, L_MIN: {L_MIN}, K: {K}, n_train: {n_train}, d: {d}, NAME: {NAME}, pcmci_file_path: {pcmci_file_path}")


### Hyperparameters ###

output_dict = {}
gamma_sequence_IGR = np.power(2.0, np.arange(-7, 6, 1, dtype=np.float64))
lambda_sequence_IGR = np.power(4.0, np.arange(-5, 3, 1, dtype=np.float64))
lambda_sequence_LASSO = np.power(2.0, np.arange(-8, 6, 1, dtype=np.float64))
lambda_sequence_AR = np.power(2.0, np.arange(-8, 6, 1, dtype=np.float64))
lambda_sequence_CD = np.power(2.0, np.arange(-6, 6, 0.5, dtype=np.float64))
gamma_sequence_DRIG = np.arange(0, 4, 0.1)

competing_method_set = ["LASSO", "AR", "CD", "DRIG", "PCMCI"]
method_set = competing_method_set + [f"IGR_{k_}" for k_ in range(1, K+1)]
gamma_sequence_dict = {"LASSO": [0.],
                        "AR": [1.5],
                        "CD": [0.],
                        "DRIG": gamma_sequence_DRIG, 
                        "PCMCI": [0.]}

lambda_sequence_dict = {"LASSO": lambda_sequence_LASSO, 
                        "AR": lambda_sequence_AR, 
                        "CD": lambda_sequence_CD, 
                        "DRIG": [0.], 
                        "PCMCI": [0.]}

for k_ in range(1, K+1):
    gamma_sequence_dict[f"IGR_{k_}"] = gamma_sequence_IGR
    lambda_sequence_dict[f"IGR_{k_}"] = lambda_sequence_IGR
for method in method_set:
    output_dict[f"r_v_t_{method}"] = np.zeros([100, len(gamma_sequence_dict[method]), len(lambda_sequence_dict[method])], dtype=np.float64)
    output_dict[f"r_t_t_{method}"] = np.zeros([100, len(gamma_sequence_dict[method]), len(lambda_sequence_dict[method])], dtype=np.float64)
    output_dict[f"beta_t_{method}"] = np.zeros([100, len(gamma_sequence_dict[method]), len(lambda_sequence_dict[method]), d], dtype=np.float64)


### Begin Experiment ###



train_id_list = list(range(E))[TRAIN_ID_SLICE]
valid_id_list = list(range(E))[VALID_ID_SLICE]
test_id_list = list(range(E))[TEST_ID_SLICE]

for Y_INDEX in [8, 84]:
    pcmci_indices = np.load(pcmci_file_path)[str(Y_INDEX)]
    pcmci_array = np.full([d,], 1e6)
    pcmci_array[pcmci_indices] = 0.
    
    for SEED in tqdm(range(100), desc=f"for {Y_INDEX}"):
        ### Data Processing ####################################
        from data_processing import angment_stock_data_and_split, standardize
        X_list, y_list, column_name = angment_stock_data_and_split(y_index=Y_INDEX, seed=SEED, n_train=n_train, L_MAX=L_MAX, L_MIN=L_MIN, train_id_list=train_id_list)
        X_list, y_list = standardize(X_list, y_list, train_id_list) # standardize
        assert X_list[0].shape[1] == d
        
        Sigma_list = []
        u_list = []
        for e in range(E):
            X_e = X_list[e]
            y_e = y_list[e]
            Sigma_e = X_e.T @ X_e / X_e.shape[0]
            u_e = X_e.T @ y_e / X_e.shape[0]
            
            Sigma_list.append(Sigma_e)
            u_list.append(u_e)
            
            assert Sigma_e.shape == (d, d)
            assert u_e.shape == (d,)

        Sigma_train = sum(Sigma for Sigma in Sigma_list[TRAIN_ID_SLICE])/len(train_id_list) # Assume that each training environment has the same weight.
        u_train = sum(u for u in u_list[TRAIN_ID_SLICE])/len(train_id_list)
        X_train = np.vstack([X_list[e] for e in train_id_list])
        y_train = np.hstack([y_list[e] for e in train_id_list])
        
        X_y_Sigma_u_dict = {"X_list": X_list, "y_list": y_list, "Sigma_list": Sigma_list, "u_list": u_list}
        Sigma_u_y_dict = {"Sigma_list": Sigma_list, "u_list": u_list, "y_list": y_list}
    
        ### Our Method ####################################
        from methods import calculating_wk, train_and_evaluate_worst_R2
        w_table, _ = calculating_wk(k=K, Sigma_list=Sigma_list[TRAIN_ID_SLICE], u_list=u_list[TRAIN_ID_SLICE], 
                                                        silent=True)
        for k_ in range(1, K+1):
            w_array = w_table[:,k_-1]
            output_dict[f"r_v_t_IGR_{k_}"][SEED], output_dict[f"r_t_t_IGR_{k_}"][SEED], output_dict[f"beta_t_IGR_{k_}"][SEED] = train_and_evaluate_worst_R2(w_array=w_array, gamma_sequence=gamma_sequence_IGR, lambda_sequence=lambda_sequence_IGR,
                                                            **Sigma_u_y_dict, 
                                                            **SLICE_DICT)
        
        ###### Competing Methods ##########################
        competing_method_function_dict = \
                        {"LASSO": partial(train_and_evaluate_worst_R2, w_array=np.zeros([d], dtype=np.float64), gamma_sequence=[0.], lambda_sequence=lambda_sequence_LASSO, **Sigma_u_y_dict, **SLICE_DICT),
                        "AR": partial(Anchor_Regression_train_and_evaluate, **X_y_Sigma_u_dict, gamma_sequence=[1.], lambda_sequence=lambda_sequence_AR, **SLICE_DICT),
                        "CD": partial(Causal_Dantzig_train_and_evaluate, **Sigma_u_y_dict, lambda_sequence=lambda_sequence_CD, **SLICE_DICT),
                        "DRIG": partial(DRIG_train_and_evaluate, **X_y_Sigma_u_dict, gamma_sequence_DRIG=gamma_sequence_DRIG, **SLICE_DICT),
                        "PCMCI": partial(train_and_evaluate_worst_R2, w_array=pcmci_array, gamma_sequence=np.array([1.]), lambda_sequence=[0.], ridge_parameter=1e-1, **Sigma_u_y_dict, **SLICE_DICT)}
        
        for method in competing_method_set:
            output_dict[f"r_v_t_{method}"][SEED], output_dict[f"r_t_t_{method}"][SEED], output_dict[f"beta_t_{method}"][SEED] = competing_method_function_dict[method]()
        
        
        

    
    result_dir = f"./saved_result/"
    os.makedirs(f"{result_dir}", exist_ok=True)
    np.savez(f"{result_dir}/{Y_INDEX}.npz", 
             **output_dict,
                )
    
    