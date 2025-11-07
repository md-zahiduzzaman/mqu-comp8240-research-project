## Replication of SUOD (Scalable Unsupervised Outlier Detection)

In this project, we aim to replicate the outlier detection capabilities of **SUOD**, which is a framework that performs outlier detection by forming ensembles of different existing outlier detection methods. We use the testing and evaluation methodology laid out in the research paper for SUOD

### Required Packages
SUOD is available as a Python package and is simple to setup. The following Python packages are required to run test code:

```python
pip install pyod
pip install suod
pip install scipy
pip install combo
```

### Replication 

The replication code used for testing is a modified version of the original code provided by the authors of the research paper. In order to run replication code, please use the following command:

```python
python suod-copy/examples/demo_base_v2.py
```

### Test on New Datasets
For our task, we test SUOD's performance on new unseen datasets. Running the test code has been made simple. The code for testing is the latest version with all reported datasets tested in one file. Please use the following command to run the test on new datasets:

```python
python suod_test_final.py
```
Simply clone the repository and the tests are ready to be run.

### Reference

```text
@inproceedings{zhao2021suod,
  title={SUOD: Accelerating Large-scale Unsupervised Heterogeneous Outlier Detection},
  author={Zhao, Yue and Hu, Xiyang and Cheng, Cheng and Wang, Cong and Wan, Changlin and Wang, Wen and Yang, Jianing and Bai, Haoping and Li, Zheng and Xiao, Cao and others},
  journal={Proceedings of Machine Learning and Systems},
  year={2021}
}
```
