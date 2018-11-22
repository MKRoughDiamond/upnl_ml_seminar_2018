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
BATCH_SIZE = 500
LEARNING_RATE = 0.002
TOTAL_EPOCHES = 50

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
        # Z는 autoencoder가 학습한 각 data에 대한 '어떠한 정보'를 담고 있다.
        Z = tf.layers.flatten(enc_layer2)

        # tf.layers.conv2d_transpose
        ## conv2d : data에서 feature를 뽑아냄
        ## conv2d_transpose : feature를 유지하며 큰 data를 생성
        # dec_layer1 : [BATCH_SIZE, 5, 5, 4]
        #              -> [BATCH_SIZE, 12, 12, 3]
        dec_layer1 = tf.layers.conv2d_transpose(
            enc_layer2,
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

        # AE에 사용된 loss는 l2 loss
        loss = tf.reduce_mean(tf.square(X-X_))
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

    # test 데이터의 예시를 이미지로 출력
    rand = int(np.random.random()*len(x_test))
    batch_x = np.asarray([x_test[rand]])
    batch_x = batch_x.reshape((1,SIZE,SIZE,1))
    x_p = sess.run(X_,feed_dict={X:batch_x})
    img = x_p[0].reshape([SIZE,SIZE])
    mpimg.imsave('./result-AE-CNN.png',img,format='png')
    img_ori = batch_x[0].reshape([SIZE,SIZE])
    mpimg.imsave('./origin-AE-CNN.png',img_ori,format='png')
