import numpy as np
import tensorflow as tf
from ConvPool_CNN_C import ConvPool_CNN
from tensorflow.keras import datasets
import time
import matplotlib.pyplot as plt

#the path name represetn different configs
#save the train and test error
path = 'wnmobn'
wnmobn_train_err = []
wnmobn_test_err = []


#set random seed
seed = 12345


num_epochs = 200 # toal epochs that data pass through
batch_size = 100 # mini batch size


# indicators of using different normalizations
use_WN,use_BN,use_MOBN=True,False,True
if (use_WN  and use_BN ):
    print("Cannot use weight norm and batch norm together")
    exit(0)
if (use_BN  and use_MOBN ):
    print("Cannot use mean-only batch norm and batch norm together")
    exit(0)

#print out what normalization is used
print('Use weight-normalization : ' + str(use_WN))
print('Use batch-normalization : ' + str(use_BN))
print('Use mean-only-batch-normalization: ' + str(use_MOBN))


randgen = np.random.RandomState(seed)
tf.set_random_seed(seed)

#get train, test data from cifar-10
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
#apply normaliation to images
X_train, X_test = X_train / 255.0, X_test / 255.0
#make them 1d vector
y_train = y_train.reshape(y_train.size)
y_test = y_test.reshape(y_test.size)

#different batch size for train and test
train_batch_size = int(X_train.shape[0] / batch_size)
test_batch_size = int(X_test.shape[0] / batch_size)
#the batchsize for initializatioon
init_batch_size = 500

#our model
model = tf.make_template('CONVPool_model', ConvPool_CNN)

#x is image y is label, x_init for initializtion
x = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, 3])
y = tf.placeholder(tf.int32, shape=[batch_size])
x_initialization = tf.placeholder(tf.float32, shape=[init_batch_size, 32, 32, 3])

#model initialization
initialize_model = model(x_initialization, drop_rate=0.5, is_test=False, init=True,use_WN=use_WN,use_BN=use_BN,use_MOBN=use_MOBN)
#model training
train_model = model(x, drop_rate=0.5, is_test=False, init=False,use_WN=use_WN,use_BN=use_BN,use_MOBN=use_MOBN)
#train losee
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=train_model, name='cross_entropy'))
#model testing
test_model = model(x, drop_rate=0.5, is_test=True, init=False,use_WN=use_WN,use_BN=use_BN,use_MOBN=use_MOBN)
#make predictions and calculate test error
prediction = tf.argmax(test_model, axis=1, output_type=tf.int32)
not_eq = tf.cast(tf.not_equal(prediction, y), tf.float32)
test_error = tf.reduce_mean(not_eq)

# using Adam optimizer, similar to the paper
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

# save the model
saver = tf.train.Saver()
init_global_variables = tf.global_variables_initializer()

start = time.time()
print('start time: ', time.time())
with tf.Session() as sess:
    sess.run(init_global_variables)
    for epoch in range(num_epochs):
        print('epochs: ', epoch)
        #shuffle data
        idx = randgen.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]

        #initialize the model in the first epoch
        if epoch == 0:
            sess.run(initialize_model, feed_dict={x_initialization: X_train[:init_batch_size]})
            # shuffle  data
            idx =randgen.permutation(X_train.shape[0])
            X_train = X_train[idx]
            y_train = y_train[idx]
        #calculate train error
        train_err = 0.
        for t in range(train_batch_size):
            feed_dict = {x: X_train[t * batch_size:(t + 1) * batch_size],
                         y: y_train[t * batch_size:(t + 1) * batch_size]}
            l, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
            train_err += l
        train_err /= train_batch_size
        wnmobn_train_err.append(train_err)
        print('train error: ', train_err)

        #calculate test error
        test_err = 0.
        for t in range(test_batch_size):
            feed_dict = {x: X_test[t * batch_size:(t + 1) * batch_size], y: y_test[t * batch_size:(t + 1) * batch_size]}
            test_error_out = sess.run(test_error, feed_dict=feed_dict)
            test_err += test_error_out
        test_err /= test_batch_size
        wnmobn_test_err.append(test_err)
        print('test error: ', test_err)
        #save parameters
    saved_path = saver.save(sess, './' + path + '/' + path + '_saved_variable')
#show the training time
print(time.time())
train_time = time.time() - start
print('train time: ', train_time)

#plot train and test error
plt.plot(wnmobn_test_err, label='test error')
plt.plot(wnmobn_train_err, label='train error')
plt.legend()
plt.show()
#save traintime ,train and test error in three files, they are all numpy array
np.save('./' + path + '/' + path + '_test_err.npy', wnmobn_test_err)
np.save('./' + path + '/' + path + '_train_err.npy', wnmobn_train_err)
np.save('./' + path + '/' + path + '_train_time.npy', train_time)
plt.savefig('./' + path + '/' + path + '_loss.png')