import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg

mnist = tf.keras.datasets.mnist
(x_train, _), (x_test, _) = mnist.load_data()

x_train, x_test = x_train/255.0, x_test/255.0

SIZE = 28
BATCH_SIZE = 600
LEARNING_RATE = 0.002
TOTAL_EPOCHES = 200

g = tf.Graph()
filters = [1,3,4]
sizes = [1,5,3]
strides = [1,2,2]
with g.as_default():
    with tf.device('/gpu:1'):
        X = tf.placeholder(tf.float32,[None,SIZE,SIZE,1])

        # enc_layer1 : [BATCH_SIZE, 28, 28, 1]
        #              -> [BATCH_SIZE, 12, 12, 3]
        enc_layer1 = tf.layers.conv2d(
            X,
            filters = filters[1],
            kernel_size = sizes[1],
            strides = strides[1],
            padding='VALID'
            )
        enc_layer1 = tf.nn.relu(enc_layer1)

        # enc_layer2 : [BATCH_SIZE, 12, 12, 3]
        #              -> [BATCH_SIZE, 5, 5, 4]
        enc_layer2 = tf.layers.conv2d(
            enc_layer1,
            filters = filters[2],
            kernel_size = sizes[2],
            strides = strides[2],
            padding='VALID'
            )
        enc_layer2 = tf.nn.relu(enc_layer2)

        enc_flatten = tf.layers.flatten(enc_layer2)
        enc_dense = tf.layers.dense(enc_flatten,units=50,activation=tf.nn.relu)
        mu = tf.layers.dense(enc_dense,units=120,activation=None)
        log_sigma = tf.layers.dense(enc_dense,units=120,activation=None)

        noises = tf.random.normal([tf.shape(X)[0],120])
        samples = noises*tf.exp(log_sigma/2)+mu
        dec_dense = tf.layers.dense(samples,units=50,activation=tf.nn.relu)
        dec_init = tf.layers.dense(dec_dense,units=100,activation=tf.nn.relu)
        dec_init = tf.reshape(dec_init,[-1,5,5,4])

        # dec_layer1 : [BATCH_SIZE, 5, 5, 4]
        #              -> [BATCH_SIZE, 12, 12, 3]
        dec_layer1 = tf.layers.conv2d_transpose(
            dec_init,
            filters = filters[1],
            kernel_size = sizes[2]+1,
            strides = strides[2],
            padding = 'VALID'
            )
        dec_layer1 = tf.nn.relu(dec_layer1)

        # dec_layer2 : [BATCH_SIZE, 12, 12, 3]
        #              -> [BATCH_SIZE, 28, 28, 1]
        dec_layer2 = tf.layers.conv2d_transpose(
            dec_layer1,
            filters = filters[0],
            kernel_size = sizes[1]+1,
            strides = strides[1],
            padding = 'VALID'
            )
        X_ = tf.nn.sigmoid(dec_layer2)

        cross_entropy_loss = tf.reduce_mean((-1)*(X*tf.log(X_+1e-10)+(1-X)*tf.log(1-X_+1e-10)))
        kl_divergense = 0.5 * tf.reduce_mean(tf.exp(log_sigma) + tf.square(mu)-1.-log_sigma)
        loss = cross_entropy_loss+kl_divergense
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        train = optimizer.minimize(loss)


with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(TOTAL_EPOCHES):
        for i in range(len(x_train)//BATCH_SIZE):
            batch_x = np.asarray(x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
            batch_x = batch_x.reshape((BATCH_SIZE,SIZE,SIZE,1))
            sess.run(train, feed_dict={X:batch_x})

        if epoch%5==4:
            print('-'*20)
            print('epoch: {}'.format(epoch+1))
            rand = int(np.random.random()*(len(x_train)//BATCH_SIZE))
            batch_x = np.asarray(x_train[rand*BATCH_SIZE:(rand+1)*BATCH_SIZE])
            batch_x = np.reshape(batch_x, (BATCH_SIZE, SIZE, SIZE,1))
            loss_p=sess.run(loss,feed_dict={X:batch_x})
            print('loss: {:6f}'.format(loss_p))


    rand = int(np.random.random()*len(x_test))
    batch_x = np.asarray([x_test[rand]])
    batch_x = batch_x.reshape((1,SIZE,SIZE,1))
    x_p = sess.run(X_,feed_dict={X:batch_x})
    img = x_p[0].reshape([SIZE,SIZE])
    mpimg.imsave('./result-VAE.png',img,format='png')
    img_ori = batch_x[0].reshape([SIZE,SIZE])
    mpimg.imsave('./origin-VAE.png',img_ori,format='png')
