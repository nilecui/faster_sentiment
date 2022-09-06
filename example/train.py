#########################################################################
#Author: nilecui
#Date: 2022-09-06 11:20:44
#LastEditors: nilecui
#LastEditTime: 2022-09-06 14:25:40
#FilePath: /faster_sentiment/example/train.py
#Description: 
#Details do not determine success or failure!
#Copyright (c) 2022 by nilecui, All Rights Reserved. 
#########################################################################

import os
from faster_sentiment import FasterSent
root_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(os.path.join(os.getcwd(), ".", "datasets"))

print(root_path)

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
