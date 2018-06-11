from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import system things
import tensorflow as tf
import numpy as np
import os

from ray_data_generator_for_siamese import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
from tensorflow.contrib.data import Dataset
from tqdm import tqdm
import siamese
from read_tfrecords import read_and_decode

# prepare data and tf.session
sess = tf.InteractiveSession()
batch_size = 128
# setup siamese network

siamese_model = siamese.siamese_network(batch_size);
train_step = tf.train.AdamOptimizer(0.001).minimize(siamese_model.loss)
saver = tf.train.Saver()

tfrecords_filename = 'siamese_train.tfrecords'

filename_queue = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs=10)

image, annotation, labels = read_and_decode(filename_queue)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())


sess.run(init_op)

siamese_model.load_initial_weights(sess)
num_epochs = 300
new = True

if new:
    for epoch in tqdm(range(num_epochs)):
        for step in range(781):
            batch_x1, batch_x2, batch_y = sess.run([image, annotation, labels])
            #print(batch_x1.shape)
            
            _, loss_v = sess.run([train_step, siamese_model.loss, siamese_model.op_labels], feed_dict={
                siamese_model.x1: batch_x1,
                siamese_model.x2: batch_x2,
                siamese_model.y_: batch_y})

            if np.isnan(loss_v):
                print('Model diverged with loss = NaN')
                quit()

            print('epoch %d: loss %.3f %f' % (epoch, loss_v, float(np.sum(batch_y==op))/len(loss_v)))
	if (epoch+1)%51==0:
	    saver.save(sess, 'siamese_with_more_fonts'+str(epoch)+'.ckpt')
    #     embed = siamese.o1.eval({siamese.x1: mnist.test.images})
    #     embed.tofile('embed.txt')
else:
    saver.restore(sess, 'model.ckpt')