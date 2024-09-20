# About 

# Setup 
## General VM setup

### Download and install code
```bash
# code 
git clone https://github.com/longnguyen2706/autocode.git

cd autocode/src
python -m pip install --upgrade pip
pip install -r requirements_runpod.txt 
```

### Runpod
```bash
# runpod CLI
 wget -qO- cli.runpod.net | sudo bash

# from local
runpodctl send /home/louis/secrets/autocode-sa.json
#export GOOGLE_APPLICATION_CREDENTIALS=/home/louis/secrets/autocode-sa.json

export GOOGLE_APPLICATION_CREDENTIALS=/workspace/autocode-sa.json
mv autocode-sa.json /workspace/autocode-sa.json
cd src 
```

### Gcloud utils
```bash
# install gcloud CLI (optional)
chmod 777 script/install_gsutil.sh 
script/install_gsutil.sh
```

### VM utils
```bash 
# install vim
apt update
apt install vim

apt install screen 

# using screen for multi window SSH
screen -S autocode
```

### Copy from/ to GCS 
```bash 
cd /workspace/autocode
# copy credential to sa.json
cat > sa.json 

# activate service account
gcloud auth activate-service-account autocode-sa@auto-code-421822.iam.gserviceaccount.com --key-file=sa.json

# copy all models to GCS
gsutil -m cp -r models/* gs://auto-code-gcs/models

# copy data from GCS
gsutil -m cp -r gs://auto-code-gcs/pythoncode/* /workspace/autocode/data/pythoncode/ 
```

# Train
```bash
torchrun --standalone --nproc_per_node=4 gpt_train.py 
```

# Config
## Config google cloud
- Create auto-code project 
- Create a service account (autocode-sa)
- Export and save credentials for sa 
- Give permission to the service accounts 
  - storage.objects.create
  - storage.objects.admin

# Log 
## v1: autogressive prediction - predict next word
- v1: context len = 256/64  -> 512/32
- [y] on runpod cloud
- [x] parametrize the model
- [x] decoder only
- [x] check vocab & vocab size - vocab has chinese characters 
- [x] how about encoder
- [x] remove all the comments in the code
- [x] using longer context 

- [x] using wordpiece tokenizer
- [x] improve the model efficiency 

Big epic 
- [x] using the pretrained model and finetune




## v2: gpt2 - TODO 
- [x] Tokenizer sometimes fails for special char (need to investigate)
- [x] Process more data to reach 10B tokens 
- [x] Train the model on 10B tokens
- [x] Evaluate 4B token model
- [x] Fine-tuning model from GPT2
- [x] Create test suite with GPT



