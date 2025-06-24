### 1. 安装依赖
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

### 2. 执行命令
```shell
python src/kf_gins.py --conf ./dataset/kf-gins.yaml
```

### 3. py_IMU
项目主要参考[py_IMU](https://github.com/Dennissy23/py_IMU)。
