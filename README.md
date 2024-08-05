# TimesNet-Light : 경량화된 TimesNet 이상 탐지 모델
### TimesNet
TimesNet의 구조<br>
<img src="https://github.com/user-attachments/assets/59c08869-c1e9-4dab-b660-a95e635b45d0" width='500' align='center'/><br>

<br><br>

### TimesNet-Light
<img width="388" alt="image" src="https://github.com/user-attachments/assets/16d08aa2-9d09-42db-a9c4-8be163102b50"/><br>
Bayesian-Optimization을 통해 적절한 컨볼루션 수를 training 동안 최적화하여 공정 모니터링 이상 탐지에 최적화.<br>


<br><br>


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
    python timesnet_tasks.py --task detect --data PSM_simulation --model_name timesnet
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
    <table>
      <tr>
        <th>평가 지표</th>
        <th>경량화 전</th>
        <th>경량화 후</th>
      </tr>
      <tr>
        <td>모델 성능(AUC-score)</td>
        <td>0.9976</td>
        <td>0.9979</td>
      </tr>
      <tr>
        <td>모델 규모(MB)</td>
        <td>18.79</td>
        <td>0.1</td>
      </tr>
      <tr>
        <td>학습 시간(sec)</td>
        <td>418.8087</td>
        <td>15.0395</td>
      </tr>
      <tr>
        <td>추론 시간(sec)</td>
        <td>260.2487</td>
        <td>14.6195</td>
      </tr>
    </table><br>
    

<br><br>

# Citation
If you found this code helpful, please consider citing:<br>
```
Semin Kim and Soohyun Oh, Minje Park, Jiho Lee & Moongi Seock (2024). Efficient Time-Series Data Anomaly Detection
with a Lightweight TimesNet Model . 대한전자공학회 학술대회, 제주.
```

<br><br>

### Reference
- Wu, Haixu, et al. "Timesnet: Temporal 2d-variation modeling for general time series analysis." The eleventh international conference on learning representations. 2022.
- Hongzuo Xu, Guansong Pang, Yijie Wang and Yongjun Wang, "Deep Isolation Forest for Anomaly Detection," in IEEE Transactions on Knowledge and Data Engineering, doi: 10.1109/TKDE.2023.3270293.
- https://github.com/thuml/Time-Series-Library
- https://github.com/xuhongzuo/DeepOD
