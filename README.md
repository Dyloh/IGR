# Invariance-Guided Regularization

This repository contains code to reproduce the real data application in our manuscript [Fundamental Computational Limits in Pursuing Invariant Causal Prediction and Invariance-Guided Regularization](https://arxiv.org/abs/2501.17354).

To cite this paper

```latex
@article{gu2025fundamental,
  title={Fundamental Computational Limits in Pursuing Invariant Causal Prediction and Invariance-Guided Regularization},
  author={Gu, Yihong and Fang, Cong and Xu, Yang and Guo, Zijian and Fan, Jianqing},
  journal={arXiv preprint arXiv:2501.17354},
  year={2025}
}
```

## Instructions to reproduce the results

### Real data application I: Stock Log-return Prediction
#### Reproduction
We first run the following script to provide results of our method and competing methods with 100 replicates. The results will be saved in ``saved_result/``.
```bash
python app1_stock.py
```
The results can be reported ussing ``report_app1.ipynb``.
#### Source of the data
The data can be collected through the [MCD](https://github.com/Rose-STL-Lab/MCD) repository by running ``src/utils/data_gen/generate_stock.py`` and terminate before the data is chunked.

### Real data application II: Climate Dynamic Prediction
#### Reproduction
Download NECP reanalysis data through [NECP](https://psl.noaa.gov/thredds/catalog/Datasets/ncep.reanalysis/Dailies/pressure/catalog.html). It is needed to change the directories of data in app2_climate.py. We then run the following script to provide the results of our method and competing methods with 100 replicates.
```bash
python app2_climate.py
```

## Direct usage of the IGR estimator
The IGR estimator can be directly used through calling the ``IGR`` function in ``methods.py``. The signature of the function is as follows
```python
def IGR(k : int, Sigma_list: list[np.ndarray], u_list: list[np.ndarray], n_list: Optional[Sequence[int]] = None,
        gamma_sequence: Optional[Sequence[float]] = None, lambda_sequence: Optional[Sequence[float]] = None) -> np.ndarray:
    pass
```
where
 - ``k`` is the computational budget i.e. the maximal size of tuples to consider.
 - ``Sigma_list`` and ``u_list`` are the covariance matrix $X^\top X /n$ and $X^T y/n$ of each training environment, respectively.
 - ``n_list`` is the weight of each environment. By default $n_e=1$.
 - ``gamma_sequence`` and ``lambda_sequence`` are hyperparameters.
This function returns the estimatad beta for each pair of hyperparameters.
