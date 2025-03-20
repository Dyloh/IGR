import netCDF4 as nc
import scipy.stats
import numpy as np
from sklearn.linear_model import Lasso,LinearRegression,Ridge
import scipy
import math
from multiprocessing import Pool
from collections import defaultdict
from scipy import stats
import cvxpy as cp
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import tigramite.causal_mediation as mediation
import tigramite.toymodels.non_additive as toy_setup
from tigramite import data_processing as pp
from tigramite.causal_mediation import CausalMediation
from tigramite.pcmci import PCMCI
from scipy.stats import pearsonr
import csv
from tigramite.independence_tests.parcorr import ParCorr
import statsmodels.api as sm
import random
from estimate import drig,anchor,dantzig,granger
from component_analysis import pca_components_gf, orthomax
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import r2_score

tau_max=7
tau_min=1
dataset = nc.Dataset('data/air.1950.nc')
temp_data = dataset.variables['air'][:,8]
dataset1 = nc.Dataset('data/air.2000.nc') 
temp_data1 = dataset1.variables['air'][:,8]
dataset2 = nc.Dataset('data/air.2010.nc')
temp_data2 = dataset2.variables['air'][:,8]
dataset3 = nc.Dataset('data/air.2020.nc')
temp_data3 = dataset3.variables['air'][:,8]

def subsets(nums,l,i):
    result = [[i]]
    for num in nums:
        if num !=i:
            for element in result[:]:
                if len(element)<l:
                    x=element[:]
                    x.append(num)
                    result.append(x)        
    return result


def compute_igr(features, responses, x_val,y_val,x_test,y_test,hyper_gamma=100,subset_num=0):

        num_envs = len(features)
        dim_x = np.shape(features[0])[1]
        w=np.zeros(dim_x)
        x_cat=np.concatenate((features[0],features[1]),axis=0)
        y_cat=np.concatenate((responses[0],responses[1]),axis=0)

        sig=np.zeros((num_envs,dim_x,dim_x))
        for i in range(num_envs):
            num_feat=np.shape(features[i])[0]
            for j in range(num_feat):
                sig[i]=sig[i]+np.matmul(np.transpose(features[i][j:j+1]),features[i][j:j+1])/(num_feat)


        for i in range(dim_x):
             tgt=np.arange(dim_x)
             sub=subsets(tgt,subset_num,i)
             inf_w=1e8
             for j in sub:
                    w_now=0
                    w_cat=np.linalg.lstsq(x_cat[:,j],y_cat,rcond=1e-6)[0]
                    for k in range(num_envs):
                         y=responses[k]
                         x=features[k][:,j]
                         w_e=np.linalg.lstsq(x,y)[0]
                         sig_e=sig[k,j][:,j]
                         w_now=w_now+(w_e-w_cat).T@sig_e@(w_e-w_cat)
                    if inf_w>w_now:
                         inf_w=w_now
             w[i]=np.sqrt(inf_w/num_envs)

        h_list=[0.5,1,5,10,50,100,200,500,1000,1500,2000,2500,3000,4000,5000]
        l_list=[0.0001,0.001,0.01,0.05,0.1,0.5]
        min_loss=1e9
        min_weight=0
        min_ind=[]
        for i in range(len(h_list)):
               for lam in l_list:
                    hyper_gamma=h_list[i]
                    weight=cp.Variable(dim_x)
                    square_loss=cp.sum_squares(features[0]@weight -responses[0][:,0])+cp.sum_squares(features[1]@weight -responses[1][:,0])
                    problem=cp.Problem(cp.Minimize(square_loss/4 + hyper_gamma *(w.T @ cp.abs(weight))+lam*cp.sum(cp.abs(weight)) ))
                    problem.solve()
                    weight=weight.value
                    loss=err(x_val,y_val,weight)
                    if loss<=min_loss:
                         min_loss=loss
                         loss_test=err(x_test,y_test,weight)
                         min_weight=weight.reshape(len(weight),1)
                         min_ind=np.where(weight>1e-4)[0]
        print(x_test.shape,y_test.shape,min_ind)
        return min_weight,loss_test
def err(x,y,weight):
    e=0
    for i in range(x.shape[0]):
         # print(x[i,:].dot(weight),y[i])
          e=e+(x[i,:].dot(weight)-y[i])**2
    return e/x.shape[0]

def l1_fair(sample1,sample2,val,test,varible_ids,tau_max,tau_min,mode='IGR',resume=False):
    loss0_list=[]
    loss_list=[]
    granger_list=[]
    lasso_list=[]
    pc_list=[]
    cd_list=[]
    drig_list=[]
    an_list=[]
    rand_list=[]
    select_target=[]
    sample_cat=np.concatenate((sample1,sample2),axis=1).T
    if mode=='pcmci':
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
                  x_val=val[:,tau_max-tau_min:test.shape[1]-tau_min]
             else:
                  x1=np.concatenate((x1,sample1[:,tau_max-j:sample1.shape[1]-j]))
                  x2=np.concatenate((x2,sample2[:,tau_max-j:sample2.shape[1]-j]))
                  x_test=np.concatenate((x_test,test[:,tau_max-j:test.shape[1]-j]))
                  x_val=np.concatenate((x_val,val[:,tau_max-j:test.shape[1]-j]))
        sample=random.sample(range(0, 364-tau_max), 300)
        y1=y1[:,sample]
        y2=y2[:,sample]
        x1=x1[:,sample]
        x2=x2[:,sample]
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
            print(select_target)
        
        if mode=='anchor':
            gamma_list=[0.001,0.01,0.05,0.1,0.5]
            lam_list=[0.001,0.01,0.1]
            _,min_a=anchor([x1,x2],[y1,y2],x_val,y_val,x_test,y_test,gamma_list,lam_list)
            an_list.append(min_a)
        
        if mode=='drig':
            gamma_list=[0.001,0.01,0.05,0.1,0.5]
            lam_list=[0.001,0.01,0.1]
            _,min_d=drig([x1,x2],[y1,y2],x_val,y_val,x_test,y_test,gamma_list,lam_list)
            drig_list.append(min_d)

    
        if mode=='dantzig':
            lam_list=[0.01,0.05,0.1,0.5,1,2,4,8,16]
            min_cd=dantzig([x1,x2],[y1,y2],x_val,y_val,x_test,y_test,lam_list)
            cd_list.append(min_cd)

        if mode=='granger':
            gamma_sequence_granger=[0.001,0.01,0.1,0.2]
            lambda_sequence_granger=0.1*np.power(2.0,np.arange(-4,3,1,dtype=np.float64))
            min_granger=granger(varible_id,cat_x,cat_y,x_val,y_val,x_test,y_test,tau_max,tau_min,gamma_sequence_granger,lambda_sequence_granger)
            granger_list.append(min_granger)
          
        if mode=='lasso':
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

        if mode=='rand':
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

        if mode=='pcmci':
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


        if mode=='IGR':
            weight,min_l=compute_igr((x1.T,x2.T),(y1.T,y2.T),x_val.T,y_val.T,x_test.T,y_test.T,subset_num=3)
            loss_list.append(min_l)
            print('k=3:',min_l)
    print(np.mean(drig_list),np.mean(an_list),np.mean(loss0_list),np.mean(cd_list))
    return loss_list,granger_list,lasso_list,pc_list,drig_list,an_list,rand_list,cd_list,loss0_list

    


temp_mean=np.mean(temp_data,axis=0)
print(temp_data.shape,temp_mean.shape)
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
temp_data = temp_data.reshape(temp_data.shape[0],-1)
temp_data1 = temp_data1.reshape(temp_data1.shape[0],-1)
temp_data2 = temp_data2.reshape(temp_data2.shape[0],-1)
temp_data3 = temp_data3.reshape(temp_data2.shape[0],-1)

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
t1=[]
t2=[]
t3=[]
t4=[]
t5=[]
t6=[]
t7=[]
t=[]
for i in range(1):
     tau_max=i+3
     tau_min=1
     t.append(tau_max)
     with open('saved_results/out.csv','a',encoding='utf-8',newline="" ) as f:
             csv_w=csv.writer(f)
             for j in range(10):
                loss_list,granger_list,lasso_list,pc_list,drig_list,an_list,rand_list,cd_list,loss0_list=l1_fair(temp_data.T,temp_data1.T,temp_data2.T,temp_data3.T,target,tau_max,tau_min,'IGR')
                print(np.mean(loss_list),np.mean(granger_list),np.mean(lasso_list),np.mean(pc_list),np.mean(drig_list),np.mean(an_list),np.mean(rand_list),np.mean(cd_list),np.mean(loss0_list))
                csv_w.writerow([np.mean(loss_list),np.mean(granger_list),np.mean(lasso_list),np.mean(pc_list),np.mean(drig_list),np.mean(an_list),np.mean(rand_list),np.mean(cd_list),np.mean(loss0_list)])
             f.close()
     t1.append(np.mean(loss_list))
     t2.append(np.mean(granger_list))
     t3.append(np.mean(lasso_list))
     t4.append(np.mean(pc_list))
     t5.append(np.mean(drig_list))
     t6.append(np.mean(an_list))
     t7.append(np.mean(rand_list))
     if i%3==10:
       fig = plt.figure()
       plt.scatter(target,granger_list,label='granger causality',marker='o')
       plt.scatter(target,loss_list,label='L1 norm',marker='o')
       plt.scatter(target,lasso_list,label='lasso',marker='o')
       plt.xlabel('components')
       plt.ylabel('test loss')
       plt.legend()
       plt.savefig(f"lag_{tau_max}.png")
print(t1,t2,t3,t4,t5,t6,t7)
fig=plt.figure()
plt.plot(t,t1,label='l1 norm')
plt.plot(t,t2,label='PCMCI')
plt.plot(t,t3,label='lasso')
plt.plot(t,t5,label='DRIG')
plt.plot(t,t6,label='anchor')
plt.plot(t,t7,label='rand')
plt.xlabel('lag')
plt.ylabel('Test loss')
plt.legend()
plt.show()
