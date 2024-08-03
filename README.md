# About 

# Setup 
```bash
huggingface-cli login


export GOOGLE_APPLICATION_CREDENTIALS=/home/louis/secrets/autocode-sa.json

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
- [x] decoder only
- [x] check vocab & vocab size - vocab has chinese characters 
- [x] how about encoder
- [x] remove all the comments in the code
- [x] using longer context 
- [x] using wordpiece tokenizer
