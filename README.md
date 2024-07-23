This repository is built for fine-tuning GPT2 and making it more knowledgeable and better at answering questions. NOTE: It only gets better with more of the same format of data used for training. Using different kinds of data sort of breaks the model.

First, run **GetGPT2Checkpoint.py**. 

Running that gives a checkpoint for GPT2 with all the weights. Next, use the newly generated checkpoint's name in **TrainingMultiGPU.py** to begin training the model on more data. The path for the new dataset also has to be passed in. 
