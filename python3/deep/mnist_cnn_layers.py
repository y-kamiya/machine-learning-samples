from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def main(_):
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.int64, [None])

  input_layer = tf.reshape(x, [-1, 28, 28, 1])

  conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[5,5],
    padding="same",
    activation=tf.nn.relu,
  )

  pool1 = tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=[2,2],
    strides=2,
  )

  conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5,5],
    padding="same",
    activation=tf.nn.relu,
  )

  pool2 = tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=[2,2],
    strides=2,
  )

  pool2_flat = tf.reshape(pool2, [-1, 7*7*64])

  dense = tf.layers.dense(
    inputs=pool2_flat,
    units=256,
    activation=tf.nn.relu,
  )

  dropout = tf.layers.dropout(
    inputs=dense,
    rate=0.4,
    training=True,
  )

  logits = tf.layers.dense(
    inputs=dropout,
    units=10,
  )
  
  loss = tf.losses.sparse_softmax_cross_entropy(
    labels=y_,
    logits=logits,
  )

  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
  train_op = optimizer.minimize(
    loss=loss,
    global_step=tf.train.get_global_step(),
  )

  y = tf.nn.softmax(logits)
  correct_prediction = tf.equal(tf.argmax(y,1), y_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100) 
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
        print("step %d, training accuracy %g" % (i, train_accuracy))
      sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})


if __name__ == '__main__':
    tf.app.run()
