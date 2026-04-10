# Queue-Management-System-v2

### Queue Management Sytem using Yolov9 for person detection, Uniface for face attribute analysis, and Norfair for person tracking. This system leverages Onnxruntime

Clone repo 

Download Models and move into 'models/' folder in cloned repo
https://drive.google.com/drive/folders/1gxqqcMACrjvegS0_OQypeT_lDKarco5_?usp=sharing

```
cd Queue-Management-System-v2
```

### Create Virtual environment

- create a virtual environment
```
python -m venv Qpark
```

### Activate Virtual environment
```
source Qpark/bin/activate
```

### Install dependencies
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run Standalone application
```
python queue_management_v2.py
```