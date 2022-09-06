# environment
python3.8

# store datasets
- .vector_cache  # word vector ,auto download to the .vector_cache
- .data          # if selected imdb, auto download dataset to the directory.

# Depends on
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
python3 -m spacy download en_core_web_sm
```

# Datasets download
链接：https://pan.baidu.com/s/1XCmt5xlAE0yKcYWLKpFQqQ 
提取码：GZZ0 

# example

```
# step 1
# create FasterSentiment object
# abs_dataset_path your datasets dir
sobj = FasterSent(abs_dataset_path=root_path, epochs=20)

# step 2 
# start training
sobj.start_train()

# step 3
# evaluate model
res = sobj.start_eval('tut3-model.pt')
print(res)

# step 3
# predict text
res = sobj.predict("He is a bad bad boy!")
print(res)

```