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
LEARNING_RATE = 0.001
TOTAL_EPOCHES = 50

# 그래프을 설정
g = tf.Graph()
# DNN에 필요한 hyperparameter들을 리스트로 선언
# 이런식으로 모아두면 편하다.
layers = [SIZE*SIZE,100, 50, 30]
with g.as_default():
    with tf.device('/gpu:0'):
        X = tf.placeholder(tf.float32,[None,SIZE,SIZE])

        flat = tf.layers.flatten(X)
        enc_layer1 = tf.layers.dense(flat,units=layers[1],activation=tf.nn.relu)
        enc_layer2 = tf.layers.dense(enc_layer1,units=layers[2],activation=tf.nn.relu)
        # Z는 autoencoder가 학습한 각 data에 대한 '어떠한 정보'를 담고 있다.
        Z = tf.layers.dense(enc_layer2,units=layers[3],activation=tf.nn.relu)
        
        dec_layer1 = tf.layers.dense(Z,units=layers[2],activation=tf.nn.relu)
        dec_layer2 = tf.layers.dense(dec_layer1, units=layers[1],activation=tf.nn.relu)
        dec_layer3 = tf.layers.dense(dec_layer2, units=layers[0],activation=tf.nn.sigmoid)
        X_ = tf.reshape(dec_layer3,[-1,SIZE,SIZE])

        # AE에 사용된 loss는 l2 loss
        loss = tf.reduce_mean(tf.square(X-X_))
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        train = optimizer.minimize(loss)

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(TOTAL_EPOCHES):
        for i in range(len(x_train)//BATCH_SIZE):
            batch_x = x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            sess.run(train, feed_dict={X:batch_x})

        if epoch%5==4:
            print('-'*20)
            print('epoch: {}'.format(epoch+1))
            rand = int(np.random.random()*(len(x_train)//BATCH_SIZE))
            batch_x = x_train[rand*BATCH_SIZE:(rand+1)*BATCH_SIZE]
            loss_p=sess.run(loss,feed_dict={X:batch_x})
            print('loss: {:6f}'.format(loss_p))
    
    # test 데이터의 예시를 이미지로 출력
    rand = int(np.random.random()*(len(x_test)))
    batch_x = [x_test[rand]]
    x_p = sess.run(X_,feed_dict={X:batch_x})
    img = x_p[0].reshape([SIZE,SIZE])
    mpimg.imsave('./result-AE.png',img,format='png')
    img_ori = batch_x[0]
    mpimg.imsave('./origin-AE.png',img_ori,format='png')
