import os
import json
import numpy as np


def data_processing(data, size, batch_size):

	enc_len = size[0]
	dec_len = size[1]

	enc_inp = np.zeros((enc_len, batch_size))
	dec_inp = np.zeros((dec_len, batch_size))
	dec_tar = np.zeros((dec_len, batch_size))

	# pair[0] = encoder sequence, pair[1] = target sequence
	for i in xrange(len(data)):
		pair = data[i]
		# copy data to np-array
		enc_inp[enc_len-len(pair[0]):enc_len,i] = pair[0][::-1]
		dec_inp[1:len(pair[1])+1,i] = pair[1]
		dec_tar[0:len(pair[1]),i] = pair[1]
		# add start end token
		dec_inp[0,i]=2
		dec_tar[len(pair[1]),i]=3

	return enc_inp, dec_inp, dec_tar

def create_bucket (options):
	buckets=[]
	for i in xrange(len(options)):
		for j in xrange(len(options)):
			buckets.append((options[i],options[j]+1))
	return buckets



class reader:

	def __init__(self, file_name, batch_size, buckets, bucket_option, signal=False, clean_mode=False):

		self.epoch = 1
		self.batch_size = batch_size
		self.file_name = file_name
		self.buckets = buckets
		self.bucket_option = bucket_option
		self.output = []
		self.bucket_list = []
		self.clean_stock = False
		self.clean_mode = clean_mode
		self.bucket_dict = self.build_bucket_dict()

		for i in xrange (len(buckets)):
			self.bucket_list.append([])

		self.file = open(file_name + "/text.txt", "r")
		with open(file_name + "/dict.json", "r") as f:
			self.dict = json.load(f)

		if signal:
			with open(file_name + "/signal.json", "r") as f:
				self.signal_dict = json.load(f)


		self.id_dict = {}
		for key in self.dict:
			self.id_dict[self.dict[key]]=key



	#make batch for the model
	def next_batch(self):

		# normal mode
		if not self.clean_stock: 
			index = self.fill_bucket()
			if index >= 0: 
				output = self.bucket_list[index]
				#clean the bucket
				self.bucket_list[index] = []
				return output, index
			else:
				self.clean_stock = True
				for i in xrange(len(self.bucket_list)):
					if len(self.bucket_list[i]) > 0:
						output = self.bucket_list[i]
						self.bucket_list[i] = []
						return output, i

		# clean the stock
		elif self.clean_mode:
			for i in xrange(len(self.bucket_list)):
				if len(self.bucket_list[i]) > 0:
					output = self.bucket_list[i]
					self.bucket_list[i] = []
					return output, i
			#if all bucket are cleaned, reset the batch
			self.clean_stock = False
			self.reset()
			return self.next_batch()
		# ignore the stock
		else:
			for i in xrange(len(self.bucket_list)):
				self.bucket_list[i] = []
			self.clean_stock = False
			self.reset()
			return self.next_batch()

	#reset when epoch finished
	def reset(self):
		self.epoch += 1
		self.file.close()
		self.file = open(self.file_name + "/text.txt", "r")

	def build_bucket_dict(self):
		bucket_dict={}
		for i in xrange (1, self.bucket_option[-1]+1):
			count = len(self.bucket_option)-1
			for options in reversed(self.bucket_option):
				if options >= i:
					bucket_dict[i]=count
				count = count - 1
		return bucket_dict

	#pick bucket
	def check_bucket(self, pair):
		best_i = self.bucket_dict[len(pair[0])]
		best_j = self.bucket_dict[len(pair[1])]
		return best_i * len(self.bucket_option) + best_j


	#fill bucket
	def fill_bucket(self):
		while True:
			line = self.file.readline()
			if not line:
				break
			pair = json.loads(line)
			index = self.check_bucket(pair)
			#if size exceed all buckets, continue
			if index == -1:
				continue

			self.bucket_list[index].append(pair)
			if len(self.bucket_list[index]) == self.batch_size:
				return index

		#if hit the end of epoch
		return -1

