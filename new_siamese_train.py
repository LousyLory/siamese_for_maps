from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import system things
import tensorflow as tf
import numpy as np
import os
#from util import get_test_results

from ray_data_generator_for_siamese import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
from tensorflow.contrib.data import Dataset
from tqdm import tqdm
import siamese

# prepare data and tf.session
sess = tf.InteractiveSession()
batch_size = 128
# setup siamese network

siamese_model = siamese.siamese_network(batch_size);
train_step = tf.train.AdamOptimizer(0.0001).minimize(siamese_model.loss)
saver = tf.train.Saver()

# Place data loading and preprocessing on the cpu
#with tf.device('/cpu:0'):
tr_data = ImageDataGenerator(mode='training',
                                 batch_size=batch_size,
                                 shuffle=True)

val_data = ImageDataGenerator(mode='validation',
                                 batch_size=batch_size,
                                 shuffle=True)


#train_writer = tf.summary.FileWriter('.' + '/train', sess.graph)

sess.run(tf.global_variables_initializer())
# start training

siamese_model.load_initial_weights(sess)
num_epochs = 300
new = True


if new:
    for epoch in tqdm(range(num_epochs)):
        for step in range(781):
            #batch_x1, batch_y1, batch_x2, batch_y2 = tr_data.get_run_time_batch()
            #batch_x1, batch_x2, _y = tr_data.get_run_time_batch_from_files()
            batch_x1, batch_x2, _y = tr_data.get_runtime_batch_from_RAM()
            #print(batch_x1.shape, batch_x2.shape)
            batch_y = _y.astype('float')
            #batch_y = map(float, _y)
            #print(batch_y.shape)
            #print(siamese_model.o1.eval({siamese_model.x1: batch_x1}))
            
            _, loss_v, op = sess.run([train_step, siamese_model.loss, siamese_model.op_labels], 
                feed_dict={siamese_model.x1: batch_x1, 
                siamese_model.x2: batch_x2, siamese_model.y_: batch_y})
            '''
            op = sess.run([siamese_model.op_labels], feed_dict={
                siamese_model.x1: batch_x1,
                siamese_model.x2: batch_x2,
                siamese_model.y_: batch_y})
            '''
            #tf.summary.scalar('loss', loss_v)
            #train_writer.add_summary(loss_v, step)

            if np.isnan(loss_v):
                print('Model diverged with loss = NaN')
                quit()

            #print(op.shape)
            print('epoch %d: loss %.3f' % (epoch, loss_v))#, float(np.sum(batch_y==op))/128.0))

        tr_acc = []
        val_acc = []
        for i in range(0, 128*6-1, 128):
            start = i
            batch_x1, batch_x2, batch_y = tr_data.get_all_files(start)
            val_batch_x1, val_batch_x2, val_y = val_data.get_all_files(start)

            
            op1 =  sess.run([siamese_model.op_labels], 
                feed_dict={siamese_model.x1: batch_x1, 
                siamese_model.x2: batch_x2, siamese_model.y_: batch_y}) 
            op2 = sess.run([siamese_model.op_labels], 
                feed_dict={siamese_model.x1: val_batch_x1, 
                siamese_model.x2: val_batch_x2, siamese_model.y_: val_y}) 

            tr_acc.append(np.sum(batch_y == op1) / 128.0)
            val_acc.append(np.sum(val_y == op2) / 128.0)

        tr_acc = np.sum(tr_acc)/len(tr_acc)
        val_acc = np.sum(val_acc)/len(val_acc)
        print('epoch %d: %f %f' % (epoch, tr_acc, val_acc))

    if (epoch+1)%51==0:
        saver.save(sess, 'siamese_with_more_fonts'+str(epoch)+'.ckpt')
    #     embed = siamese.o1.eval({siamese.x1: mnist.test.images})
    #     embed.tofile('embed.txt')
else:
    saver.restore(sess, 'model.ckpt')

# # visualize result
# x_test = mnist.test.images.reshape([-1, 28, 28])
# visualize.visualize(embed, x_test)
