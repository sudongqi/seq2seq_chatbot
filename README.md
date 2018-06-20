# DEPRECATED
__This project was built in 2016 (the old tensorflow era), and some design choices no longer
make sense under the new tensorflow. You are likely to find better seq2seq tutorial/implementation somewhere else.__

## seq2seq_chatbot
An implementation of Seq2seq chatbot in tensorflow.

### Features
* dynamic rnn with smart loader **(padding free)**
* beam search on prediction **(fast approximation on global optimum)**
* signal indicator for decoder **(partial control on decoder)**

A technical report: 
https://docs.google.com/gview?url=http://sudongqi.com/Documents/2016_02.pdf&embedded=true

### Python 2.7 dependency
* tensorflow 1.8
* numpy
* json

### Instruction
* run "python train.py" and wait (5 minutes on GTX 1080 Ti with cuda 9.0 and cudnn 7.0) until training is completed
* run "python test.py" to enter the interactive session with the chatbot

### Try your own data
it's possible to run it on your own data, but you need to generate at least 2 files with the same format like those in bbt_data.
* text.txt      this is the training data contatining the pair in number token format
* dict.json     this is the dicitonary to translate from number token to English word token in test time
* actors.json   (optional) this is for signal indication in test time
* summary.json  (optional) this file contain the length info for selecting the right bucket options for training

### OpenSubtitles data 
If you want to train on openSubtitles (english 2016) dataset, this project provide a data processing script (data_processing/openSub_data_generator.py) for openSubtitles.
Get OpenSubtitles data from here: http://opus.lingfil.uu.se/OpenSubtitles2016.php
