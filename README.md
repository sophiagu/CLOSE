# Installation
```
conda create -n cmt python=3.8
conda activate cmt
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install -c cyclus java-jdk=8.45.14 -y

git clone https://github.com/sophiagu/CLOSE.git
cd CLOSE/l2v
pip3 install -r requirements.txt
```
