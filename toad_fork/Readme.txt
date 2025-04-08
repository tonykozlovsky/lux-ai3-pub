container:
from folder       lux-ai2
build:
./d -b
start:
./d -u
attach:
./d -a
_______________________________________________________________________________________________

To run full power training:
from folder    /lux_ai2/toad_fork

pip3 install ../Lux-Design-S3/src/ && WANDB_MODE=offline TRAINING=1 USE_TORCH_COMPILE=1 python3 ./run_monobeast.py

pip3 install ../Lux-Design-S3/src/ && WANDB_MODE=offline TRAINING=1 USE_TORCH_COMPILE=0 python3 ./run_monobeast.py
pip3 install ../Lux-Design-S3/src/ && WANDB_MODE=offline TRAINING=1 USE_TORCH_COMPILE=1 SMALL=1 python3 ./run_monobeast.py



(USE_TORCH_COMPILE=1 optional, start faster, run slower, not work in coder)

_______________________________________________________________________________________________


To run debug (training in one thread, batch size 1)
from folder    /lux_ai2/toad_fork

pip3 install ../Lux-Design-S3/src/ && WANDB_MODE=offline TRAINING=1 USE_TORCH_COMPILE=1 SMALL=1 python3 ./run_monobeast.py
pip3 install ../Lux-Design-S3/src/ && WANDB_MODE=offline TRAINING=1 USE_TORCH_COMPILE=0 SMALL=1 python3 ./run_monobeast.py


command to run on mac
python3.11 -m pip install ../Lux-Design-S3/src/ --break-system-packages  && WANDB_MODE=offline TRAINING=1 USE_TORCH_COMPILE=0 SMALL=1 MAC=1 python3.11 ./run_monobeast.py

(USE_TORCH_COMPILE=1 optional, start faster, run slower, not work in coder)

_______________________________________________________________________________________________



_______________________________________________________________________________________________


To create version
from project root

rsync -aP ./toad_fork/ ./versions/{YOUR_VERSION_NAME}/ --exclude outputs --exclude .git --delete --exclude models --exclude shared

copy model to ./versions/{YOUR_VERSION_NAME}/submission_model/
modify model file name in ./versions/{YOUR_VERSION_NAME}/agent.py


_______________________________________________________________________________________________


To run games between versions

modify version names in ./toad_fork/runner.py

pip3 install Lux-Design-S3/src/ && WANDB_MODE=offline CUDA_VISIBLE_DEVICES="0" USE_TORCH_COMPILE=0 python3 ./toad_fork/runner.py


___________________________________________________________________________________________________________

To run games between versions on mac

luxai-s3 ./versions_pushed/net_9_cont/main.py ./versions_pushed/net_9_cont/main.py


___________________________________________________________________________________________________________

file analytics.ipynb shoudn't be tracked by local git. If gitignore is not working
for this file run command from toad_fork before making any changes:

git update-index --assume-unchanged  analytics/analytics.ipynb

now you can change this file without git tracking

if you want to track it run

git update-index --no-assume-unchanged analytics/analytics.ipynb
