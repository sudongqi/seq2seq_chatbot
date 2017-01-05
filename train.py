import tensorflow as tf
import json
import numpy as np
import os
import util.s2s_reader as s2s_reader






#-----------------------------------parameters------------------------------------------

#the file where we read the data/model from
file_name = "bbt_data"
#if set to true, then load model from file instead of start a new model
load_model = True
#if set to true, use adam optimizer instead of sgd
adam_opt = True
#batch size for training
batch_size = 128

#data params
#bucket_option = [i for i in range(1, 20+1)]
bucket_option = [5,10,15,20,25,31]
buckets = s2s_reader.create_bucket(bucket_option)

# load the data set into s2s_reader
# the total bucket numbers = bucket options number ^ 2
# if clean mode is true, the leftover data in the bucket will be used before the epoch is over
reader = s2s_reader.reader(file_name = file_name, batch_size = batch_size, buckets = buckets, bucket_option = bucket_option, clean_mode=True)
vocab_size = len(reader.dict)

# if load_model = true, then we need to define the same parameter in the saved_model inorder to load it 
hidden_size = 300
recurrent_size = 512
embedding_size = 300
num_layers = 1

#training params, truncated_norm will resample x > 2std; so when std = 0.1, the range of x is [-0.2, 0.2]
truncated_std = 0.1
keep_prob = 0.95
max_epoch = 20
norm_clip = 8

#training params for adam
adam_learning_rate = 0.001

#training params for sgd
learning_rate = 0.1
momentum = 0.9

#model name & save path
model_name = "r"+str(recurrent_size)+"_h"+str(hidden_size)+"_x"+str(num_layers)
save_path = file_name+"/"+model_name








#---------------------------------model definition------------------------------------------

tf.nn.ops.reset_default_graph()
sess = tf.InteractiveSession()

#placeholder
enc_inputs = tf.placeholder(tf.int32, shape=(None, batch_size), name="enc_inputs")
targets = tf.placeholder(tf.int32, shape=(None, batch_size), name="targets")
dec_inputs = tf.placeholder(tf.int32, shape=(None, batch_size), name="dec_inputs")

#input embedding layers
emb_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=truncated_std), name="emb_weights")
enc_inputs_emb = tf.nn.embedding_lookup(emb_weights, enc_inputs, name="enc_inputs_emb")
dec_inputs_emb = tf.nn.embedding_lookup(emb_weights, dec_inputs, name="dec_inputs_emb")

#cell definiton
cell_list=[]

for i in xrange(num_layers):

	single_cell = tf.nn.rnn_cell.LSTMCell(
		num_units=recurrent_size, 
		num_proj=hidden_size, 
		#initializer=tf.truncated_normal_initializer(stddev=truncated_std),
		state_is_tuple=True
		)
	if i < num_layers-1 or num_layers == 1:
		single_cell = tf.nn.rnn_cell.DropoutWrapper(cell=single_cell, output_keep_prob=keep_prob)
	cell_list.append(single_cell)


cell = tf.nn.rnn_cell.MultiRNNCell(cells=cell_list, state_is_tuple=True)


#encoder & decoder defintion
_, enc_states = tf.nn.dynamic_rnn(cell = cell, 
	inputs = enc_inputs_emb, 
	dtype = tf.float32, 
	time_major = True, 
	scope="encoder")

dec_outputs, dec_states = tf.nn.dynamic_rnn(cell = cell, 
	inputs = dec_inputs_emb, 
	initial_state = enc_states, 
	dtype = tf.float32, 
	time_major = True, 
	scope="decoder")


#output layers
project_w = tf.Variable(tf.truncated_normal(shape=[hidden_size, embedding_size], stddev=truncated_std), name="project_w")
project_b = tf.Variable(tf.constant(shape=[embedding_size], value = 0.1), name="project_b")
softmax_w = tf.Variable(tf.truncated_normal(shape=[embedding_size, vocab_size], stddev=truncated_std), name="softmax_w")
softmax_b = tf.Variable(tf.constant(shape=[vocab_size], value = 0.1), name="softmax_b")

dec_outputs = tf.reshape(dec_outputs, [-1, hidden_size], name="dec_ouputs")
dec_proj = tf.matmul(dec_outputs, project_w) + project_b
logits = tf.nn.log_softmax(tf.matmul(dec_proj, softmax_w) + softmax_b, name="logits")

#loss function
flat_targets = tf.reshape(targets, [-1])
total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, flat_targets)
avg_loss = tf.reduce_mean(total_loss)

#optimization
if adam_opt:
	optimizer = tf.train.AdamOptimizer(adam_learning_rate)
else:
	optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum = momentum, use_nesterov=True)


gvs = optimizer.compute_gradients(avg_loss)
capped_gvs = [(tf.clip_by_norm(grad, norm_clip), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs)



#initialization or load model
saver = tf.train.Saver()

if load_model:
	cwd = os.getcwd()
	saver.restore(sess, cwd+"/"+save_path+"/model.ckpt")
	with open(save_path+'/summary.json') as json_data:
		losses = json.load(json_data)
		reader.epoch = len(losses)+1
	print("Model restored.")

else:
	os.mkdir(save_path)
	sess.run(tf.global_variables_initializer())
	losses = []






#-----------------------------------training-------------------------------------------





def update_summary(save_path, losses):
	summary_location = save_path + "/summary.json"
	if os.path.exists(summary_location):
		os.remove(summary_location)
	with open(summary_location, 'w') as outfile:
		json.dump(losses, outfile)



#local variables
count = 0
epoch_loss = 0
epoch_count = 0

while True:

	curr_epoch = reader.epoch
	data,index = reader.next_batch()
	
	enc_inp, dec_inp, dec_tar = s2s_reader.data_processing(data, buckets[index], batch_size)

	if reader.epoch != curr_epoch:
		
		print "\n----------end of epoch:" + str(reader.epoch-1) + "----------"
		print "    avg loss: " + str(epoch_loss/epoch_count)
		print "\n"

		losses.append(epoch_loss/epoch_count)

		epoch_loss = 0
		epoch_count = 0

		update_summary(save_path, losses)
		cwd = os.getcwd()
		saver.save(sess, cwd+"/"+save_path+"/model.ckpt")
		print("Model saved")


		if reader.epoch == (max_epoch+1):
			break

	feed_dict = {enc_inputs: enc_inp, dec_inputs:dec_inp, targets: dec_tar}
	_, loss_t = sess.run([train_op, avg_loss], feed_dict)
	epoch_loss += loss_t

	count+=1
	epoch_count+=1

	if count%10 == 0:
		print str(loss_t) + " @ epoch: " + str(reader.epoch) + " count: "+ str(epoch_count * batch_size) 




sess.close()
