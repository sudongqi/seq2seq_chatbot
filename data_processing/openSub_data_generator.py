import json
import re
import os

# ---------------------------parameter & file setup---------------------------
# no punctuation for the source sentence
nopun_exp = r"[0-9]+|[']*[\w]+"
# complete expression for response sentence
expression = r"[0-9]+|[']*[\w]+|[.]+|[,?!\"()]"

max_lines = 40000000
max_length = 20
max_dict_size = 12000

file_name = "raw_data/OpenSubtitles2016.raw.en"
dict_out_name = "dict.json"
text_out_name = "text.txt"
summary_out_name = "summary.txt"

zero_token = "[non]"
unknown_token = "[unk]"
start_token = "[beg]"
end_token = "[end]"

if os.path.exists(dict_out_name):
    os.remove(dict_out_name)
if os.path.exists(text_out_name):
    os.remove(text_out_name)
if os.path.exists(summary_out_name):
    os.remove(summary_out_name)

dict_f = open(dict_out_name, "a")
text_f = open(text_out_name, "a")
summary_f = open(summary_out_name, "a")

length_info = []
for i in xrange(max_length + 2):
    length_info.append(0)

# ---------------------------build dictionary-----------------------------
# count the occurrence of all tokens
token_count_dict = {}
with open(file_name) as f:
    line_count = 0

    for line in f:
        # break

        if (line_count == (max_lines)):
            break

        token_list = re.findall(expression, line.lower())

        # add to dictionary
        for token in token_list:
            if token not in token_count_dict:
                token_count_dict[token] = 1
            else:
                token_count_dict[token] += 1

        if line_count % 1000 == 0:
            print ("building dictionary: {}".format(line_count))

        line_count += 1

# pick the most frequent tokens from all tokens
from Queue import PriorityQueue

q = PriorityQueue()
for t in token_count_dict:
    q.put([-token_count_dict[t], t])

token_dict = {}
# add special token
token_dict[zero_token] = 0
token_dict[unknown_token] = 1
token_dict[start_token] = 2
token_dict[end_token] = 3
token_index = 4

token_count_dict = {}

# priority queue
while not q.empty():
    get = q.get_nowait()
    if token_index == max_dict_size:
        break
    token_dict[get[1]] = token_index
    token_index += 1

# -------------------------build data pair------------------------------
# write to file
with open(file_name) as f:
    line_count = 0
    last_exist = False
    last_list = []
    pair_count = 0
    total_token = 0

    # one way flag
    write_file = text_f

    for line in f:

        line_count += 1

        if (line_count == max_lines):
            break

        # parse input
        token_list = re.findall(expression, line.lower())
        token_no_pun = re.findall(nopun_exp, line.lower())

        # if the sentence is too long, discard the sentence
        if len(token_list) > max_length or len(token_list) == 0:
            last_list = []
            last_exist = False
            continue

        # new list in index
        new_list = []
        # new list in index with no punctuation
        new_list_nopun = []

        # add to dictionary, if [unk] appear, then this sentence can only be source sentence
        signal = False
        for token in token_list:
            if token in token_dict:
                new_list.append(token_dict[token])
            else:
                new_list.append(1)
                signal = True

        # build no punctuation sentence for source
        for token in token_no_pun:
            if token in token_dict:
                new_list_nopun.append(token_dict[token])
            else:
                new_list_nopun.append(1)

        # if we have [unk] token, continue
        if signal or len(token_no_pun) == 0:
            last_exist = False
            continue

        # count valid pair
        if (last_exist == True):
            write_file.write(json.dumps([last_list, new_list]) + "\n")
            pair_count += 1

        # source sentence have no punctuation
        last_list = new_list_nopun
        last_exist = True

        length_info[len(new_list)] += 1
        total_token += len(new_list)

        if line_count % 1000 == 0:
            print ("preparing data: {}".format(str(line_count)))

# ---------------------------write summary----------------------------------
summary_f.write("-----summary-----" + "\n")
summary_f.write("total pair:" + str(pair_count) + "\n")
summary_f.write("total token:" + str(total_token) + "\n")
summary_f.write("unique token:" + str(len(token_dict)) + "\n")

for i in xrange(len(length_info)):
    summary_f.write(str(i) + ": " + str(length_info[i]) + "\n")

dict_f.write(json.dumps(token_dict))

# free memory
dict_f.close()
text_f.close()
summary_f.close()
