import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg

# dataset
mnist = tf.keras.datasets.mnist
(x_train, _), (x_test, _) = mnist.load_data()

# normalize
x_train, x_test = x_train/255.0, x_test/255.0

# hyperparameters
SIZE = 28
BATCH_SIZE = 300
LEARNING_RATE = 0.001
TOTAL_EPOCHES = 200

# 그래프을 설정
g = tf.Graph()
# CNN에 필요한 hyperparameter들을 리스트로 선언
# 이런식으로 모아두면 바꾸기 편하다.
filters = [1,3,4]
sizes = [1,5,3]
strides = [1,2,2]
with g.as_default():
    # tf.device : 특정 device(cpu, gpu)에게 처리할 일을 정해줄 수 있다.
    with tf.device('/gpu:0'):
        # autoencoder에 필요한 input은 원본 데이터 뿐
        X = tf.placeholder(tf.float32,[None,SIZE,SIZE,1])

        # enc_layer1 : [BATCH_SIZE, 28, 28, 1]
        #              -> [BATCH_SIZE, 12, 12, 3]
        # leaky_relu : leaky_relu는 activation function의 역할을 수행하면서 R^n 공간을 모두 mapping한다.
        enc_layer1 = tf.layers.conv2d(
            X,
            filters = filters[1],
            kernel_size = sizes[1],
            strides = strides[1],
            padding='VALID'
            )
        enc_layer1 = tf.nn.leaky_relu(enc_layer1)

        # enc_layer2 : [BATCH_SIZE, 12, 12, 3]
        #              -> [BATCH_SIZE, 5, 5, 4]
        enc_layer2 = tf.layers.conv2d(
            enc_layer1,
            filters = filters[2],
            kernel_size = sizes[2],
            strides = strides[2],
            padding='VALID'
            )
        enc_layer2 = tf.nn.leaky_relu(enc_layer2)
        
        enc_flatten = tf.layers.flatten(enc_layer2)
        # VAE는 'Z'로 표현하던 mapping을 Z의 각 차원마다의 gaussian 분포들로 표현함
        # mu와 log_sigma(코딩 편의상)는 gaussian분포를 나타냄
        ## mu : 평균
        ## log_sigma = 2*ln(sigma)
        ## -> KL divergence 계산을 편하게 하기 위함
        mu = tf.layers.dense(enc_flatten,units=100,activation=None)
        log_sigma = tf.layers.dense(enc_flatten,units=100,activation=None)

        # VAE나 나중에 배울 GAN들은 학습에 random한 noise가 필요함.
        noises = tf.random.normal([tf.shape(X)[0],100])
        # gaussian 분포를 (mu,sigma^2)로 변환
        samples = noises*tf.exp(log_sigma/2)+mu
        dec_init = tf.layers.dense(samples,units=100,activation=tf.nn.leaky_relu)
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
        dec_layer1 = tf.nn.leaky_relu(dec_layer1)

        # X_ : [BATCH_SIZE, 12, 12, 3]
        #      -> [BATCH_SIZE, 28, 28, 1]
        dec_layer2 = tf.layers.conv2d_transpose(
            dec_layer1,
            filters = filters[0],
            kernel_size = sizes[1]+1,
            strides = strides[1],
            padding = 'VALID'
            )
        X_ = tf.nn.sigmoid(dec_layer2)

        cross_entropy_loss = (-1)* tf.reduce_mean(X*tf.log(X_+1e-10)+(1-X)*tf.log(1-X_+1e-10))
        # KL divergence : Kullback–Leibler_divergence
        # KL(N(mu,sigma^2)||N(0,1)) = 1/2 * sum(sigma^2+mu^2-ln(sigma^2)-1)
        # = 1/2*sum(exp(log_sigma)+mu^2-log_sigma-1)
        # loss = (cross entropy loss) + (kl divergence)
        kl_divergense = 0.5 * tf.reduce_mean(tf.exp(log_sigma) + tf.square(mu)-log_sigma-1.)
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

        if epoch%20==19:
            print('-'*20)
            print('epoch: {}'.format(epoch+1))
            rand = int(np.random.random()*(len(x_train)//BATCH_SIZE))
            batch_x = np.asarray(x_train[rand*BATCH_SIZE:(rand+1)*BATCH_SIZE])
            batch_x = np.reshape(batch_x, (BATCH_SIZE, SIZE, SIZE,1))
            loss_p=sess.run(loss,feed_dict={X:batch_x})
            print('loss: {:6f}'.format(loss_p))

    # test 데이터의 예시를 이미지로 출력
    rand = int(np.random.random()*len(x_test))
    batch_x = np.asarray([x_train[rand]])
    batch_x = batch_x.reshape((1,SIZE,SIZE,1))
    x_p = sess.run(X_,feed_dict={X:batch_x})
    img = x_p[0].reshape([SIZE,SIZE])
    mpimg.imsave('./result-VAE.png',img,format='png')
    img_ori = batch_x[0].reshape([SIZE,SIZE])
    mpimg.imsave('./origin-VAE.png',img_ori,format='png')
