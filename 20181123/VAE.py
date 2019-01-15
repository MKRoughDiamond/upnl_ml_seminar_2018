import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg

mnist = tf.keras.datasets.mnist
(x_train, _), (x_test, _) = mnist.load_data()

x_train, x_test = x_train/255.0, x_test/255.0

SIZE = 28
BATCH_SIZE = 256
LEARNING_RATE = 0.001
TOTAL_EPOCHES = 80

g = tf.Graph()
filters = [1,8,16]
sizes = [1,5,3]
strides = [1,2,2]
xavier_initializer = tf.contrib.layers.xavier_initializer()
with g.as_default():
    with tf.device('/gpu:1'):
        X = tf.placeholder(tf.float32,[None,SIZE,SIZE,1])

        # enc_layer1 : [BATCH_SIZE, 28, 28, 1]
        #              -> [BATCH_SIZE, 12, 12, 8]
        enc_layer1 = tf.layers.conv2d(
            X,
            filters = filters[1],
            kernel_size = sizes[1],
            strides = strides[1],
            padding='VALID',
            kernel_initializer = xavier_initializer
            )
        enc_layer1 = tf.nn.elu(enc_layer1)

        # enc_layer2 : [BATCH_SIZE, 12, 12, 8]
        #              -> [BATCH_SIZE, 5, 5, 16]
        enc_layer2 = tf.layers.conv2d(
            enc_layer1,
            filters = filters[2],
            kernel_size = sizes[2],
            strides = strides[2],
            padding='VALID',
            kernel_initializer = xavier_initializer
            )
        enc_layer2 = tf.nn.elu(enc_layer2)

        enc_flatten = tf.layers.flatten(enc_layer2)
        enc_dense = tf.layers.dense(enc_flatten,units=64,activation=tf.nn.tanh,kernel_initializer=xavier_initializer)
        mu = tf.layers.dense(enc_dense,units=8,activation=None,kernel_initializer=xavier_initializer)
        sigma = tf.layers.dense(enc_dense,units=8,activation=tf.nn.elu,kernel_initializer=xavier_initializer)

        noises = tf.random.normal([tf.shape(X)[0],8])
        samples = noises*sigma+mu
        dec_dense = tf.layers.dense(samples,units=64,activation=tf.nn.tanh,kernel_initializer=xavier_initializer)
        dec_init = tf.layers.dense(dec_dense,units=400,activation=tf.nn.elu,kernel_initializer=xavier_initializer)
        dec_init = tf.reshape(dec_init,[-1,5,5,16])

        # dec_layer1 : [BATCH_SIZE, 5, 5, 16]
        #              -> [BATCH_SIZE, 12, 12, 8]
        dec_layer1 = tf.layers.conv2d_transpose(
            dec_init,
            filters = filters[1],
            kernel_size = sizes[2]+1,
            strides = strides[2],
            padding = 'VALID',
            kernel_initializer = xavier_initializer
            )
        dec_layer1 = tf.nn.elu(dec_layer1)

        # dec_layer2 : [BATCH_SIZE, 12, 12, 8]
        #              -> [BATCH_SIZE, 28, 28, 1]
        dec_layer2 = tf.layers.conv2d_transpose(
            dec_layer1,
            filters = filters[0],
            kernel_size = sizes[1]+1,
            strides = strides[1],
            padding = 'VALID',
            kernel_initializer = xavier_initializer
            )
        X_ = tf.nn.sigmoid(dec_layer2)
        X_ = tf.clip_by_value(X_,1e-8,1-1e-8)

        cross_entropy_loss = (-1)*(X*tf.log(X_)+(1-X)*tf.log(1-X_))
        #l2_loss = tf.reduce_mean(tf.square(X-X_))
        kl_divergence = 0.5 * (tf.square(sigma) + tf.square(mu)-1.-tf.log(1e-8+tf.square(sigma)))
        loss = tf.reduce_sum(cross_entropy_loss)+tf.reduce_sum(kl_divergence)
        #loss = l2_loss + kl_divergence
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        train = optimizer.minimize(loss)


with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(TOTAL_EPOCHES):
        for i in range(len(x_train)//BATCH_SIZE):
            c = int(np.random.random()*(len(x_train)-BATCH_SIZE))
            batch_x = np.asarray(x_train[c:c+BATCH_SIZE])
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
    mpimg.imsave('./result-VAE.png',img,format='png',cmap='gray')
    img_ori = batch_x[0].reshape([SIZE,SIZE])
    mpimg.imsave('./origin-VAE.png',img_ori,format='png',cmap='gray')
