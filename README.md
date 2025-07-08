# Python版本KF-GINS

## 简介

[KF-GINS](https://github.com/i2Nav-WHU/KF-GINS) 实现了经典的GNSS/INS松组合导航解算，其包含了相应的 [C++版本](https://github.com/i2Nav-WHU/KF-GINS) 和 [Matlab版本](https://github.com/i2Nav-WHU/KF-GINS-Matlab) 。适合初学者学习理解组合导航算法，本项目旨在补充其相应的Python版本，用以笔者自己学习理解，以及和感兴趣的其他初学者一同交流。

> **注意：该项目并非 [KFGINS](https://github.com/i2Nav-WHU/KFGINS) 官方版本。**

## 1. 使用说明

### 1.1 安装依赖
```shell
conda create --name kf-gins-env --file requirements.txt
```
或者手动安装以下内容：
```shell
# using conda
conda install matplotlib numpy pyyaml scipy
# or using pip
pip install matplotlib numpy pyyaml scipy
```

### 1.2 执行命令
```shell
python src/kf_gins.py --conf ./dataset/kf-gins.yaml
```

## 2. 致谢

- 本项目感谢武汉大学卫星导航定位技术研究中心多源智能导航实验室(i2Nav)牛小骥教授团队开源的 KF-GINS 软件平台；
- 感谢Github上的开源项目：[py_IMU](https://github.com/Dennissy23/py_IMU)。

## 3. 联系方式
如有问题、错误报告或功能请求，请在 [GitHub 仓库](https://github.com/salmoshu/KF-GINS-Py) 上提交 Issue。通用问题可联系 [winchell.hu@outlook.com](mailto:winchell.hu@outlook.com)。
