# TensorFlow_ChatBot
a tutorial implementation of Seq2seq chatbot with a techniqal report: [read](https://docs.google.com/gview?url=http://sudongqi.com/Documents/2016_02.pdf&embedded=true)

###Python dependency
* tensorflow 0.12
* numpy
* json

###Features
* dynamic rnn with smart loader
* beam search on prediction
* signal indicator for decoder (read the report for more details)

###Instruction
* run "python train.py"
* after training is completed, a folder containing the model will appear inside the data folder
* run "python test.py" to see the result

###Try your own data
it's possible to run it on your own data, but you need to generate the same format like those in bbt_data
* text.txt      this is the training data contatining the pair in number token format
* dict.json     this is the dicitonary to translate from number token to English word token in test time
* actors.json   (optional), this is for signal indication in test time
* summary.json  (optional), this file contain the length info for selecting the right bucket options for training
