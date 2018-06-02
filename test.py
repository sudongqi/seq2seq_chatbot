import tensorflow as tf
import numpy as np
import util.s2s_reader as s2s_reader
import sys
import os
import re

# -----------------------------------parameters------------------------------------------


# the file where we read the data/model from
file_name = "bbt_data"
# interactive mode allow user to talk to the model directly, if set to false, it will test on the training data instead
iteracitve = True
# regular expression for parsing user input
expression = r"[0-9]+|[']*[\w]+"
# signal mode allow user to insert signal token before the decoder generate sentence
signal = False
# batch size for testing
batch_size = 1

# data params
# bucket_option = [i for i in xrange(1, 20+1)]
bucket_option = [5, 10, 15, 20, 25, 31]
buckets = s2s_reader.create_bucket(bucket_option)

reader = s2s_reader.reader(file_name=file_name, batch_size=batch_size, buckets=buckets, bucket_option=bucket_option,
                           signal=signal)
vocab_size = len(reader.dict)

# if load_model = true, then we need to define the same parameter in the saved_model inorder to load it 
hidden_size = 512
projection_size = 300
embedding_size = 300
num_layers = 1

# ouput_size for softmax layer
output_size = hidden_size
if projection_size != None:
    output_size = projection_size

# model name & save path
model_name = "p" + str(projection_size) + "_h" + str(hidden_size) + "_x" + str(num_layers)
save_path = file_name + "/" + model_name

# prediction params
beam_size = 10
top_k = 10
max_sequence_len = 20

# ---------------------------------model definition------------------------------------------

tf.reset_default_graph()
sess = tf.InteractiveSession()

# placeholder
enc_inputs = tf.placeholder(tf.int32, shape=(None, batch_size), name="enc_inputs")
targets = tf.placeholder(tf.int32, shape=(None, batch_size), name="targets")
dec_inputs = tf.placeholder(tf.int32, shape=(None, batch_size), name="dec_inputs")

# input embedding layers
emb_weights = tf.Variable(tf.random_normal([vocab_size, embedding_size]), name="emb_weights")
enc_inputs_emb = tf.nn.embedding_lookup(emb_weights, enc_inputs, name="enc_inputs_emb")
dec_inputs_emb = tf.nn.embedding_lookup(emb_weights, dec_inputs, name="dec_inputs_emb")


# cell definiton
def getStackedLSTM():
    cell_list = []
    for i in xrange(num_layers):
        single_cell = tf.contrib.rnn.LSTMCell(
            num_units=hidden_size,
            num_proj=projection_size,
            # initializer=tf.truncated_normal_initializer(stddev=truncated_std),
            state_is_tuple=True
        )
        cell_list.append(single_cell)
    return tf.contrib.rnn.MultiRNNCell(cells=cell_list, state_is_tuple=True)


# encoder & decoder defintion
_, enc_states = tf.nn.dynamic_rnn(cell=getStackedLSTM(),
                                  inputs=enc_inputs_emb,
                                  dtype=tf.float32,
                                  time_major=True,
                                  scope="encoder")

dec_outputs, dec_states = tf.nn.dynamic_rnn(cell=getStackedLSTM(),
                                            inputs=dec_inputs_emb,
                                            initial_state=enc_states,
                                            dtype=tf.float32,
                                            time_major=True,
                                            scope="decoder")

# output layers
project_w = tf.Variable(tf.truncated_normal([output_size, embedding_size], stddev=0.1), name="project_w")
project_b = tf.Variable(tf.constant(0.1, shape=[embedding_size]), name="project_b")
softmax_w = tf.Variable(tf.truncated_normal([embedding_size, vocab_size], stddev=0.1), name="softmax_w")
softmax_b = tf.Variable(tf.constant(0.1, shape=[vocab_size]), name="softmax_b")

dec_outputs = tf.reshape(dec_outputs, [-1, output_size])
dec_proj = tf.matmul(dec_outputs, project_w) + project_b
logits = tf.nn.log_softmax(tf.matmul(dec_proj, softmax_w) + softmax_b)

# prediction
logit = logits[-1]
top_values, top_indexs = tf.nn.top_k(logit, k=beam_size, sorted=True)

# load variable
saver = tf.train.Saver()
cwd = os.getcwd()
saver.restore(sess, cwd + "/" + save_path + "/model.ckpt")
print("\nModel restored.")


# ----------------------------prediciton helper function-----------------------------

def build_input(sequence):
    dec_inp = np.zeros((1, len(sequence)))
    dec_inp[0][:] = sequence
    return dec_inp.T


def print_sentence(index_list):
    for index in index_list:
        sys.stdout.write(reader.id_dict[index])
        sys.stdout.write(' ')
    sys.stdout.write('\n')


def predict(enc_inp):
    dec_inp = np.zeros((1, 1))
    dec_inp[0][0] = 2
    index_output = []
    feed_dict = {enc_inputs: enc_inp, dec_inputs: dec_inp}
    indexs, state = sess.run([top_indexs, dec_states], feed_dict)
    index_output.append(indexs[0])

    while True:
        dec_inp[0][0] = indexs[0]
        feed_dict = {enc_states: state, dec_inputs: dec_inp}
        indexs, state = sess.run([top_indexs, dec_states], feed_dict)
        if indexs[0] == 3:
            break
        index_output.append(indexs[0])

    return index_output


def beam_predict(enc_inp, signal=None):
    sequnece = [2]
    if signal != None:
        sequnece.append(signal)

    dec_inp = build_input(sequnece)

    candidates = []
    options = []

    feed_dict = {enc_inputs: enc_inp, dec_inputs: dec_inp}
    values, indexs, state = sess.run([top_values, top_indexs, dec_states], feed_dict)

    for i in xrange(len(values)):
        candidates.append([values[i], [indexs[i]]])

    best_sequence = None
    highest_score = -sys.maxint - 1

    while True:

        # print candidates
        for i in xrange(len(candidates)):

            sequence = candidates[i][1]
            score = candidates[i][0]

            # if sequence end, evaluate
            if sequence[-1] == 3 or len(sequence) >= max_sequence_len:
                if score > highest_score:
                    highest_score = score
                    best_sequence = sequence
                continue

            # if not, continue searching
            dec_inp = build_input(sequence)

            feed_dict = {enc_states: state, dec_inputs: dec_inp}
            values, indexs = sess.run([top_values, top_indexs], feed_dict)

            for j in xrange(len(values)):
                new_sequence = list(sequence)
                new_sequence.append(indexs[j])
                options.append([score + values[j], new_sequence])

        # sort all options and keep top k
        options.sort(reverse=True)
        candidates = []

        for i in xrange(min(len(options), top_k)):
            if options[i][0] > highest_score:
                candidates.append(options[i])

        options = []
        if len(candidates) == 0:
            break

    if signal:
        best_sequence = [signal] + best_sequence

    return best_sequence[:-1]


def translate(token_list):
    enc = []
    for token in token_list:
        if token in reader.dict:
            enc.append(reader.dict[token])
        else:
            enc.append(reader.dict['[unk]'])
    # dec will be append with 2 inside the model
    return enc


# ---------------------------prediction loop---------------------------

if iteracitve:

    print("\n--------------------------")
    print("--Interactive mode is on--")
    print("--------------------------\n")

    while True:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            print ("\nsession close")
            break

        token_list = re.findall(expression, line.lower())
        print("-------------")

        sequence = translate(token_list)
        enc_inp = build_input(sequence[::-1])
        response = beam_predict(enc_inp, None)
        sys.stdout.write('src: ')
        print_sentence(sequence)
        sys.stdout.write('-->: ')
        print_sentence(response)
        print(' ')

else:

    while True:
        try:
            data, index = reader.next_batch()
            enc_inp, dec_inp, dec_tar = s2s_reader.data_processing(data, buckets[index], batch_size)
            response = beam_predict(enc_inp)

            sys.stdout.write('src: ')
            print_sentence(data[0][0])
            sys.stdout.write('tar: ')
            print_sentence(data[0][1])
            sys.stdout.write('-->: ')
            print_sentence(response)
            print(' ')
        except KeyboardInterrupt:
            print ("\nsession close")
            break

sess.close()
