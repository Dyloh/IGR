import netCDF4 as nc
import scipy.stats
import numpy as np
from sklearn.linear_model import Lasso,LinearRegression,Ridge
import scipy
import math
from multiprocessing import Pool
from collections import defaultdict
import cvxpy as cp
import matplotlib.pyplot as plt
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from scipy.stats import pearsonr
import csv
from tigramite.independence_tests.parcorr import ParCorr
import random
from statsmodels.tsa.stattools import grangercausalitytests
from component_analysis import pca_components_gf, orthomax
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import r2_score
from methods import Anchor_Regression_train_and_evaluate, Causal_Dantzig_train_and_evaluate, DRIG_train_and_evaluate, calculating_wk, train_and_evaluate_worst_R2

def granger(varible_id,cat_x,cat_y,x_val,y_val,x_test,y_test,tau_max,tau_min,gamma_sequence_granger,lambda_sequence_granger):
    min_granger=1e9
    for alpha in gamma_sequence_granger:
        ind=[]
        for i in range(60):
                    t=np.array([cat_x[:,varible_id],cat_x[:,i]])
                    test_result = grangercausalitytests(t.T, maxlag=tau_max+1 ,verbose=False)
                    for lag in range(tau_min, tau_max + 1):
                        p_values = test_result[lag][0]['ssr_ftest'][1] 
                        if p_values>1-alpha:
                            ind.append((lag-1)*60+i)
        if ind==[]:
            continue
        for lam in lambda_sequence_granger:
            g_estimator = Ridge(alpha=lam)
            g_estimator.fit(cat_x[:,ind],cat_y)
            val_pred=g_estimator.predict(x_val[ind].T)
            val_pred=val_pred.reshape(len(val_pred),1)
            val_loss=np.mean((val_pred-y_val.T)**2)
            min_granger=min(min_granger,val_loss)
            if min_granger==val_loss:
                test_pred=g_estimator.predict(x_test[ind].T)
                test_pred=test_pred.reshape(len(test_pred),1)
                test_loss=np.mean((test_pred-y_test.T)**2)
        return test_loss
def err(x,y,weight):
    e=0
    for i in range(x.shape[0]):
          e=e+(x[i,:].dot(weight)-y[i])**2
    return e/x.shape[0]
def l1_fair(sample1,sample2,val,test,varible_ids,tau_max,tau_min,mode='IGR'):
    igr_list=[]
    granger_list=[]
    lasso_list=[]
    pc_list=[]
    cd_list=[]
    drig_list=[]
    an_list=[]
    rand_list=[]
    select_target=[]
    sample_cat=np.concatenate((sample1,sample2),axis=1).T
    TRAIN_ID_SLICE = slice(0,2)
    VALID_ID_SLICE = slice(2,3)
    TEST_ID_SLICE = slice(3,4)
    K=2
    if 'pcmci' in mode:
        var_names=[]
        res_list=[]
        gamma_sequence_PCMCI=[0.00001,0.0001,0.001,0.01]
        lambda_sequence_PCMCI=0.1*np.power(2.0,np.arange(-4,3,1,dtype=np.float64))
        for i in range(sample_cat.shape[1]):
            var_names.append(i)
        dataframe = pp.DataFrame(sample_cat, var_names=var_names)
        parcorr = ParCorr(significance='analytic')
        for g in gamma_sequence_PCMCI:
            pcmci = PCMCI(
                dataframe=dataframe, 
                cond_ind_test=parcorr,
                verbosity=1)
            results =pcmci.run_pcmci(tau_max=tau_max,tau_min=tau_min, pc_alpha=g)['graph']
            res_list.append(results)
    for varible_id in varible_ids:
        y1=sample1[varible_id:varible_id+1,tau_max:]
        y2=sample2[varible_id:varible_id+1,tau_max:]
        y_test=test[varible_id:varible_id+1,tau_max:]
        y_val=val[varible_id:varible_id+1,tau_max:]
        for j in range(tau_min,tau_max+1):
             if j ==tau_min:
                  x1=sample1[:,tau_max-tau_min:sample1.shape[1]-tau_min]
                  x2=sample2[:,tau_max-tau_min:sample2.shape[1]-tau_min]
                  x_test=test[:,tau_max-tau_min:test.shape[1]-tau_min]
                  x_val=val[:,tau_max-tau_min:val.shape[1]-tau_min]
             else:
                  x1=np.concatenate((x1,sample1[:,tau_max-j:sample1.shape[1]-j]))
                  x2=np.concatenate((x2,sample2[:,tau_max-j:sample2.shape[1]-j]))
                  x_test=np.concatenate((x_test,test[:,tau_max-j:test.shape[1]-j]))
                  x_val=np.concatenate((x_val,val[:,tau_max-j:val.shape[1]-j]))
        sample=random.sample(range(0, 364-tau_max), 300)
        y1=y1[:,sample]- np.mean(y1[:,sample], keepdims=True)
        y2=y2[:,sample]- np.mean(y2[:,sample], keepdims=True)
        x1=x1[:,sample]- np.mean(x1[:,sample], axis=1, keepdims=True)
        x2=x2[:,sample]- np.mean(x2[:,sample], axis=1, keepdims=True)
        cat_x=np.concatenate((x1,x2),axis=1).T
        cat_y=np.concatenate((y1,y2),axis=1).T

        ind=[]
        
        model = LinearRegression()
        model.fit(cat_x, cat_y)
        y_pred = model.predict(cat_x)

        r2 = r2_score(cat_y, y_pred)
        if r2<0.75:
            continue
        else:
            select_target.append(varible_id)

        x_list=[x1.T,x2.T,x_val.T,x_test.T]
        y_list=[y1.reshape(-1),y2.reshape(-1),y_val.reshape(-1),y_test.reshape(-1)]
        sigma_list=[]
        u_list=[]
        for e in range(4):
            X_e = x_list[e]
            y_e = y_list[e]
            Sigma_e = X_e.T @ X_e / X_e.shape[0]
            u_e = X_e.T @ y_e / X_e.shape[0]
            
            sigma_list.append(Sigma_e)
            u_list.append(u_e)

        if 'IGR' in mode:
            w_table, _ = calculating_wk(k=K, Sigma_list=sigma_list[TRAIN_ID_SLICE], u_list=u_list[TRAIN_ID_SLICE], 
                                                        silent=True)
            min_igr=1e9
            gamma_sequence_IGR=[0.1,1,10,100,500,1000,3000,5000]
            lambda_sequence_IGR=[0.0001,0.001,0.01,0.1,0.5]
            for k_ in range(1, K+1):
                w_array = w_table[:,k_-1]
                val_igr,test_igr,_= train_and_evaluate_worst_R2(w_array=w_array, gamma_sequence=gamma_sequence_IGR, lambda_sequence=lambda_sequence_IGR,
                                                            y_list=y_list,Sigma_list=sigma_list,u_list=u_list,
                                                            train_id_slice=TRAIN_ID_SLICE,valid_id_slice=VALID_ID_SLICE,test_id_slice=TEST_ID_SLICE)
                min_pos = np.unravel_index(np.argmin(val_igr), val_igr.shape) 
                min_igr=min(min_igr,test_igr[min_pos])
            igr_list.append(min_igr)
        if 'anchor' in mode:
            gamma_list=[0.01,0.1]
            lam_list=[0.01,0.1,0.5,1.0]
            val_a,test_a,_=Anchor_Regression_train_and_evaluate(x_list,y_list,sigma_list,u_list,gamma_list,lam_list,TRAIN_ID_SLICE,VALID_ID_SLICE,TEST_ID_SLICE)
            min_pos = np.unravel_index(np.argmin(val_a), val_a.shape) 
            an_list.append(test_a[min_pos])
        
        if 'drig' in mode:
            gamma_list=[1e-4,0.001,0.01,0.1]
            lam_list=[0.001,0.01,0.1]
            val_d,test_d,_=DRIG_train_and_evaluate(X_list=x_list,y_list=y_list,Sigma_list=sigma_list,u_list=u_list,
                                            gamma_sequence_DRIG=gamma_list,train_id_slice=TRAIN_ID_SLICE,valid_id_slice=VALID_ID_SLICE,test_id_slice=TEST_ID_SLICE)
            min_pos = np.unravel_index(np.argmin(val_d), val_d.shape) 
            drig_list.append(test_d[min_pos])

    
        if 'dantzig' in mode:
            lam_list=[0.01,0.1,0.5,1,10,20]
            val_cd,test_cd,_=Causal_Dantzig_train_and_evaluate(y_list=y_list,Sigma_list=sigma_list,u_list=u_list,
                                            lambda_sequence=lam_list,train_id_slice=TRAIN_ID_SLICE,valid_id_slice=VALID_ID_SLICE,test_id_slice=TEST_ID_SLICE)
            min_pos = np.unravel_index(np.argmin(val_cd), val_cd.shape) 
            cd_list.append(test_cd[min_pos])

        if 'granger' in mode:
            gamma_sequence_granger=[0.001,0.01,0.1,0.2]
            lambda_sequence_granger=0.1*np.power(2.0,np.arange(-4,3,1,dtype=np.float64))
            min_granger=granger(varible_id,cat_x,cat_y,x_val,y_val,x_test,y_test,tau_max,tau_min,gamma_sequence_granger,lambda_sequence_granger)
            granger_list.append(min_granger)
          
        if 'lasso' in mode:
            a_list=[0.5,0.1,0.05,0.01,0.001]
            min_lasso=1e9
            for j in range(len(a_list)):
                alpha=a_list[j]
                lasso_model=Lasso(alpha=alpha,max_iter=20000)
                lasso_model.fit(cat_x,cat_y)

                val_pred=lasso_model.predict(x_val.T)
                val_pred=val_pred.reshape(len(val_pred),1)
                loss_val=np.mean((val_pred-y_val.T)**2)
                if min_lasso>loss_val:
                    min_lasso=loss_test
                    test_pred=lasso_model.predict(x_test.T)
                    test_pred=test_pred.reshape(len(test_pred),1)
                    loss_test=np.mean((test_pred-y_test.T)**2)
            lasso_list.append(loss_test)

        if 'rand' in mode:
            a_list=[0.5,0.1,0.05,0.01,0.001]
            min_rand=1e9
            for j in range(len(a_list)):
                weight=cp.Variable(cat_x.shape[1])
                rand_weight=np.random.uniform(0, alpha, [1,cat_x.shape[1]])
                square_loss=cp.sum_squares(cat_x@weight -cat_y[:,0])
                problem=cp.Problem(cp.Minimize(square_loss +rand_weight@cp.abs(weight)) )
                problem.solve()
                weight=weight.value
                loss=err(x_val.T,y_val.T,weight)
                min_rand=min(min_rand,loss)
                if min_rand==loss:
                    loss_test=err(x_test.T,y_test,T,weight)
            rand_list.append(loss_test)

        if 'pcmci' in mode:
            min_err=1e9
            for res in res_list:
                for j in range(tau_min,tau_max+1):
                    if j==tau_min:
                        ind=np.where(res[:,varible_id,j]!='')[0]
                    else:
                        ind=np.concatenate((ind,np.where(res[:,varible_id,j]!='')[0]))
                for a in lambda_sequence_PCMCI:
                    pc_estimator = Ridge(alpha=a)
                    pc_estimator.fit(cat_x[:,ind],cat_y)
                    val_pred=pc_estimator.predict(x_val[ind].T)
                    val_pred=val_pred.reshape(len(val_pred),1)
                    erro=np.mean((val_pred-y_val.T)**2)
                    min_err=min(min_err,erro)
                    if min_err==erro:
                        test_pred=pc_estimator.predict(x_test[ind].T)
                        test_pred=test_pred.reshape(len(test_pred),1)
                        loss_test=np.mean((test_pred-y_test.T)**2)
            pc_list.append(loss_test)

    return igr_list,granger_list,lasso_list,pc_list,drig_list,an_list,rand_list,cd_list

def tofield(d, lats, lons):
    return d.reshape([len(lats), len(lons)])
    
tau_max=1
tau_min=1
resume=False
dataset = nc.Dataset('data/air.1950.nc')
temp_data = dataset.variables['air'][:,8]
dataset1 = nc.Dataset('data/air.2000.nc') 
temp_data1 = dataset1.variables['air'][:,8]
dataset2 = nc.Dataset('data/air.2010.nc')
temp_data2 = dataset2.variables['air'][:,8]
dataset3 = nc.Dataset('data/air.2020.nc')
temp_data3 = dataset3.variables['air'][:,8]

temp_mean=np.mean(temp_data,axis=0)
lats=dataset1.variables['lat'][:]
lons=dataset1.variables['lon'][:]
for i in range(temp_mean.shape[0]):
     for j in range(temp_mean.shape[1]):
          temp_data[:,i,j]=temp_data[:,i,j]-temp_mean[i,j]
          temp_data1[:,i,j]=temp_data1[:,i,j]-temp_mean[i,j]
          temp_data2[:,i,j]=temp_data2[:,i,j]-temp_mean[i,j]
          temp_data3[:,i,j]=temp_data3[:,i,j]-temp_mean[i,j]
for i in range(len(lats)):
     cos_trans=math.cos(lats[i] * math.pi / 180) ** 0.5
     temp_data[:,i,:] *=cos_trans
     temp_data1[:,i,:] *=cos_trans
     temp_data2[:,i,:] *=cos_trans
     temp_data3[:,i,:] *=cos_trans
temp_data = temp_data.reshape(temp_data.shape[0],-1).data
temp_data1 = temp_data1.reshape(temp_data1.shape[0],-1).data
temp_data2 = temp_data2.reshape(temp_data2.shape[0],-1).data
temp_data3 = temp_data3.reshape(temp_data3.shape[0],-1).data

total_data=np.concatenate((temp_data,temp_data1),axis=0)
total_mean=np.mean(total_data,axis=0)
total_std=np.std(total_data,axis=0)

tmp=np.array(total_data)
U,_,_=pca_components_gf(tmp)
U = U[:, :60]
Ur, T, iters = orthomax(U,
                         rtol = np.finfo(np.float32).eps ** 0.5,
                         gamma =1.0,
                         maxiter = 500)

temp_data=np.dot(temp_data,Ur)
temp_data1=np.dot(temp_data1,Ur)
temp_data2=np.dot(temp_data2,Ur)
temp_data3=np.dot(temp_data3,Ur)

total_data=np.concatenate((temp_data,temp_data1),axis=0)
std=np.std(total_data,axis=0)
for i in range(60):
    temp_data[:,i]=temp_data[:,i]/std[i]
    temp_data1[:,i]=temp_data1[:,i]/std[i]
    temp_data2[:,i]=temp_data2[:,i]/std[i]
    temp_data3[:,i]=temp_data3[:,i]/std[i]

target=np.arange(1,59,1)

for i in range(1):
     tau_max=i+3
     tau_min=1
     t.append(tau_max)
     with open('air.csv','a',encoding='utf-8',newline="" ) as f:
             csv_w=csv.writer(f)
             for j in range(10):
                print('iter:',j)
                igr_list,granger_list,lasso_list,pc_list,drig_list,an_list,rand_list,cd_list=l1_fair(temp_data.T,temp_data1.T,temp_data2.T,temp_data3.T,target,tau_max,tau_min,['IGR'])
                print(np.mean(igr_list),np.mean(granger_list),np.mean(lasso_list),np.mean(pc_list),np.mean(drig_list),np.mean(an_list),np.mean(rand_list),np.mean(cd_list))
                csv_w.writerow([np.mean(igr_list),np.mean(granger_list),np.mean(lasso_list),np.mean(pc_list),np.mean(drig_list),np.mean(an_list),np.mean(rand_list),np.mean(cd_list)])
             f.close()
