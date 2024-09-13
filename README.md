# About 

# Setup 
```bash
huggingface-cli login





```

```bash
 wget -qO- cli.runpod.net | sudo bash
 
# runpod 
git clone https://github.com/longnguyen2706/autocode.git

# install gcloud
chmod 777 script/install_gsutil.sh 
script/install_gsutil.sh

apt update
apt install vim

cd src
python -m pip install --upgrade pip
pip install -r requirements_runpod.txt 


# from local
runpodctl send /home/louis/secrets/autocode-sa.json
#export GOOGLE_APPLICATION_CREDENTIALS=/home/louis/secrets/autocode-sa.json

export GOOGLE_APPLICATION_CREDENTIALS=/workspace/autocode-sa.json
mv autocode-sa.json /workspace/autocode-sa.json
cd src 
python decoder_v1.py

```
# Config google cloud
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

Vocab size is not fixed (depends on text, which is bad, cannot use trained model)