# SHoTClean: Bridging Soft and Hard Constraints for Multivariate Time Series Cleaning

In this paper, we propose SHoTClean series, a unified framework that bridges soft and hard constraints for effective and efficient multivariate time series cleaning.

As shown in figure below, there are four specialized algorithms for  multivariate time-series cleaning—each targeting distinct computational scenarios based on our constrained optimization formulation.

- SHoTClean-B: An offline batch algorithm employing pruned dynamic programming to achieve global optimality.  
- SHoTClean-S: Online streaming variant utilizing incremental dynamic programming to achieve local optimality.
- SHoTClean-P: Online streaming variant accelerates SHoTClean-S via CDQ divide-and-conquer and Fenwick tree to attain near-linear complexity.
- SHoTClean-C: Online streaming variant employing causal modeling, designed solely for multivariate datasets.

![Overview of SHoTClean Series](./overview.jpg)

Experiments conducted on 10 real-world datasets with 10 state-of-the-art methods demonstrate the ShoTClean's superiority. SHoTClean achieve i) 6.8%–90.0% and 7.8%–82.1% improvements  in accuracy (RMSE metric) compared to the second best methods in offline and online settings, respectively; ii) an average two-order-of-magnitude runtime speed-up on large-scale datasets; and iii)  superior robustness, with consistent high performance under extreme 80% contamination level and high-dimensional datasets.



## Environment

- Python.version = 3.9.21
- tigramite.version = 5.2.7.0
- sklearn.version = 1.6.1
- Other dependencies are listed in `requirements.txt`.



## Datasets

We evaluate SHoTClean on 10 real-world datasets, with detailed statistics summarized in the table below.

|  Dimension   |                           Datasets                           | Length  | # dim |     Source     |   Field    |
| :----------: | :----------------------------------------------------------: | :-----: | :---: | :------------: | :--------: |
|  Univariate  |    [TOTALSA](https://fred.stlouisfed.org/series/TOTALSA)     |   593   |   1   |      FRED      |   Trade    |
|  Univariate  | [STOCK](https://anonymous.4open.science/r/mtcsc-E4CC/MTCSC/data/STOCK/stock12k.data) | 12,824  |   1   |     SCREEN     |  Finance   |
|  Univariate  | [COVID-19](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series) | 14,001  |   1   |    JHU CSSE    |   Health   |
|  Univariate  | [CA](https://www.caiso.com/planning/Pages/ReliabilityRequirements/Default.aspx) | 43,824  |   1   | California ISO |   Energy   |
|  Univariate  | [ID_a40b](https://competition.aiops-challenge.com/home/competition/1484452272200032281) | 137,898 |   1   |     AIOps      |    KPI     |
| Multivariate | [Porto](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data) | 16,749  |   2   |     Kaggle     | Trajectory |
| Multivariate |      [ECG](https://physionet.org/content/nstdb/1.0.0/)       | 650,000 |   2   |      UCR       |   Health   |
| Multivariate | [Exchange](https://github.com/laiguokun/multivariate-time-series-data/tree/master/exchange_rate) |  7,588  |   8   |     LSTNet     |  Finance   |
| Multivariate | [AEP](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) | 19,735  |  21   |      UCI       |   Energy   |
| Multivariate |  [PSM](https://github.com/eBay/RANSynCoders/tree/main/data)  | 132,481 |  25   |      eBay      |   Server   |
| Multivariate | [SWaT](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#swat) | 14,996  |  26   |     iTrust     | Industrial |
| Multivariate | [WADI](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#wadi) | 784,537 |  73   |     iTrust     | Industrial |

All datasets have been processed with  `./experiments/generate_dataset.py` and the outputs are available under `./datasets`. To ensure fair comparisons and account for varying dataset sizes, we applied the following selection criteria; all others remain in their original form:

- TOTALSA: records from January 1, 1976 to May 1, 2025.
- COVID-19: records from China, France, Russia, UK, US.
- CA: all energy-consumption records from 2019 to 2023.
- KPI: only the entry with ID a40b1df87e3f1c87.
- Porto: the first ten trajectories containing more than 1,200 points.
- ECG: records from 118e06 with SNR 6dB.
- SWaT & WADI: only numeric features.

Please note that the SWaT and WADI datasets aren't included here, as their distribution rights belong to iTRUST; to gain access, please use the links provided in the table. Of course, you’re also welcome to download the original raw datasets and process them however you like.



## Baselines

To generate the ground truth for the Porto and Chengdu dataset, run the following command:

|  Dimension   |                          Algorithm                           | Scenario |          Type          |
| :----------: | :----------------------------------------------------------: | :------: | :--------------------: |
|  Univariate  |                             EWMA                             |  online  |       smoothing        |
|  Univariate  | [SCREEN](https://anonymous.4open.science/r/mtcsc-E4CC/MTCSC/src/MTCSC/SCREEN.java) |  online  |       constraint       |
|  Univariate  | [SpeedAcc](https://anonymous.4open.science/r/mtcsc-E4CC/MTCSC/src/MTCSC/SpeedAcc.java) |  online  |       constraint       |
|  Univariate  | [LsGreedy](https://anonymous.4open.science/r/mtcsc-E4CC/MTCSC/src/MTCSC/Lsgreedy.java) |  online  |      statistical       |
|  Univariate  |    [Akane](https://github.com/HatsuneHan/Akane-sigmod24/)    | offline  |      statistical       |
| Multivariate |                            ARIMA                             | offline  |       smoothing        |
| Multivariate |      [Clean4MTS](https://github.com/Aries99C/Clean4MTS)      | offline  | constraint+statistical |
| Multivariate | [MTCSC](https://anonymous.4open.science/r/mtcsc-E4CC/README.md) |  online  | constraint+statistical |
| Multivariate |      [TranAD](https://github.com/imperial-qore/TranAD/)      |  online  |   anomaly detection    |
| Multivariate |    [IMDiffusion](https://github.com/17000cyh/IMDiffusion)    |  online  |   anomaly detection    |

We’ve reimplemented SCREEN, SpeedAcc, LsGreedy, MTCSC, and other algorithms in Python and merged them into our framework (see the `./baselines` directory for details), while all remaining algorithms still run in their original environments.



## Code Structure

The repository is organized into directories and standalone scripts as follows:

- **`baselines/`**: Python implementations of some baseline algorithms
- **`datasets/`**: processed datasets ready for use
- **`experiments/`**: scripts for running each experiment (see the next section for details)
- **`results/`**:  stores the output of experiments
- **`tools/`**: utility classes and helpers
- **`Optimization_Solver.py`**: global optimization solver
- **`SHoTClean_B.py`**: multi-dimensional SHoTClean-B implementation
- **`SHoTClean_B1.py`**: single-dimensional SHoTClean-B implementation
- **`SHoTClean_C.py`**: multi-dimensional SHoTClean-C implementation
- **`SHoTClean_P1.py`**: single-dimensional SHoTClean-P implementation
- **`SHoTClean_S.py`**: multi-dimensional SHoTClean-S implementation
- **`SHoTClean_S1.py`**: single-dimensional SHoTClean-S implementation



## Experiments

We provide a set of Python scripts to reproduce all of our experiments and visualizations:

**Dataset-specific experiments**

- **`experiment_CA.py`**  Runs a suite of experiments on the CA dataset.  
- **`experiment_CA_segment.py`**  Runs a suite of experiments on the CA dataset with contiguous-segment injection strategy.
- **`experiment_COVID.py`**  Runs a suite of experiments on the COVID-19 dataset.  
- **`experiment_exchange.py`**  Runs a suite of experiments on the Exchange dataset.  
- **`experiment_FRED.py`**  Runs a suite of experiments on the TOTALSA dataset.  
- **`experiment_KPI.py`**  Runs a suite of experiments on the KPI dataset.  
- **`experiment_Porto.py`**  Runs a suite of experiments on the Porto dataset. 
- **`experiment_PSM.py`**  Runs a suite of experiments on the PSM dataset.  
- **`experiment_stock.py`**  Runs a suite of experiments on the Stock dataset.  
- **`experiment_SWaT.py`**  Runs a suite of experiments on the SWaT dataset. 
- **`experiment_SWAT_partial.py`**  Runs a suite of experiments on the SWaT dataset with partial-dimension injection strategy.
- **`experiment_UCI.py`**  Runs a suite of experiments on the AEP dataset. 
- **`experiment_UCR.py`**  Runs a suite of experiments on the ECG dataset. 
- **`experiment_WADI.py`**  Runs a suite of experiments on the WADI dataset.  

**General-purpose scripts**

- **`generate_dataset.py`**  Preprocesses all raw datasets and writes them into `./datasets/`.  
- **`example_intro.py`**  Generates the figure for the introduction.  
- **`example_B.py`** Usage example for the SHoTClean-B variant. 
- **`example_P.py`** Usage example for the SHoTClean-P variant.  
- **`experiment_window_size.py`** Benchmark across different window sizes.  
- **`experiment_tSNE.py`** Produce t-SNE visualizations of the cleaned data.  
- **`experiment_DTW.py`** Run Dynamic Time Warping (DTW) experiments. 

From the project's root directory (`./`), run an experiment with a command like:

```
python -m experiments.experiment_CA.py
```

Simply replace `experiment_CA.py` with the name of any other experiment script you wish to execute.

