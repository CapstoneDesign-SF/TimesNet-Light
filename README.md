# TimesNet-Light

### How to
- Bayesian-Optimization
  - Checkout **timesnet_bayesian_optimization.ipynb**.
  - Follow the instruction of the file
  - There's no need to change or modify.


- Detect Anomalies
  - Checkout **timesnet_pipeline.py**.
  - For real time anomaly detection, check the file and follow the instruction.
  - Modification may be required depending on whether the task to be performed is **real time detection** or **simulation**.
  - Required command line arguments
    - task: Use "train" for training TimesNet before detection or use "detect" for only detection.
    - data: Train data and test data should be within this directory.
    - model: Model name for saving trained model, or detecting with corresponding model.
    ```
    python timesnet_pipeline.py --task detect --data PSM --model timesnet
    ```
  - The result is as follows.
    ```
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