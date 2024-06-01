# TimesNet-Light

### How to
- Bayesian-Optimization
  - Checkout **timesnet_bayesian_optimization.ipynb**.
  - Follow the instruction of the file
  - There's no need to change or modify.


- Detect Anomalies
  - Checkout **timesnet_tasks.py**.
  - Required command line arguments
    - **task**: Use "train" for training TimesNet before detection or use "detect" for only detection. If you want to try simulating anomaly detection with test data, use "simulate".
    - data: Data for training TimesNet or detection should be within this directory.
    - model_name: Model name for saving trained model, or detecting with corresponding model.
    ```
    python timesnet_pipeline.py --task detect --data PSM_simulation --model_name timesnet
    ```
  - The result is as follows.
    ```
    ---Start detecting anomalies---
    Anomaly occured at 2024-05-31 00:56:05
    Anomaly occured at 2024-05-31 00:56:05
    Anomaly occured at 2024-05-31 00:56:06
    Anomaly occured at 2024-05-31 00:56:06
    Anomaly occured at 2024-05-31 00:56:08
    Anomaly occured at 2024-05-31 00:56:08
    Anomaly occured at 2024-05-31 00:56:08
    Anomaly occured at 2024-05-31 00:56:09
    Anomaly occured at 2024-05-31 00:56:09
    Anomaly occured at 2024-05-31 00:56:09
    ```



### Reference
- Wu, Haixu, et al. "Timesnet: Temporal 2d-variation modeling for general time series analysis." The eleventh international conference on learning representations. 2022.
- Hongzuo Xu, Guansong Pang, Yijie Wang and Yongjun Wang, "Deep Isolation Forest for Anomaly Detection," in IEEE Transactions on Knowledge and Data Engineering, doi: 10.1109/TKDE.2023.3270293.
- https://github.com/thuml/Time-Series-Library
- https://github.com/xuhongzuo/DeepOD