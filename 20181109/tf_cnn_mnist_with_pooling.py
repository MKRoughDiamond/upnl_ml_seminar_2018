import tensorflow as tf
import numpy as np

# MNIST classification using CNN

# MNIST DATASET을 받는다
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 이미지 정보가 그대로 들어있으므로(int), 이를 [0, 1]로 normalize
x_train, x_test = x_train/255.0, x_test/255.0


# 중요한 변수 설정
# SIZE: 이미지의 크기
# BATCH_SIZE: BATCH의 크기, 적절하게 설정함
# LEARNING_RATE
# TOTAL_EPOCHS: 총 반복 횟수
SIZE = 28
BATCH_SIZE = 256
LEARNIN_RATE = 0.003
TOTAL_EPOCHS = 100


# graph 설정. 우리가 설정한 구조가 들어감
g = tf.Graph()
with g.as_default():

    # 입력 받는 placeholder
    # X : [ BATCH_SIZE, SIZE, SIZE ], BATCH_SIZE개의 (SIZE * SIZE) 크기의 이미지, channel은 흑백이미지니깐 1
    # Y : [ BATCH_SIZE ], 각 이미지의 classification 들(0-9사이의 값)
    X = tf.placeholder(tf.float32,[None,SIZE,SIZE,1])
    Y = tf.placeholder(tf.int32,[None])

    # Y에 대한 one-hot encoding
    y = tf.one_hot(Y,10) 

    # CNN은 이미지를 그대로 입력 받으므로 입력 할 때 flatten 필요 없음


    # CNN을 설정
    # tf.layers.conv2d(<input>, <filters>, <kernel_size>, strides=<strides>, activation=<activation>)
    # <input> : layer를 통과하고자 하는 입력 tensor
    # <kernel_size> : filter의 [fh, fw] 값.
    # <strides> : stride의 [sh, sw] 값.
    # <activation> : activation function들이 들어감
    ### tf.nn.sigmoid, tf.nn.softmax, tf.nn.relu, ...

    # layer가 진행됨에 따라 이미지 크기는 가로 세로 절반씩 1/4 로 줄고 channel은 2배가 돼서 결과적으로 1/2 씩 줄어듦

    layer1 = tf.layers.conv2d(X, 32, [3, 3], activation=tf.nn.relu)     # [BATCH_SIZE, 28, 28, 1] -> [BATCH_SIZE, 28, 28, 32]
    layer1 = tf.layers.max_pooling2d(layer1, [2, 2], [2, 2])            # [BATCH_SZIE, 28, 28, 32] -> [BATCH_SIZE, 14, 14, 32]


    layer2 = tf.layers.conv2d(layer1, 64, [3, 3], activation=tf.nn.relu) # [BATCH_SIZE, 14, 14, 32] -> [BATCH_SZIE, 14, 14, 64]
    layer2 = tf.layers.max_pooling2d(layer2, [2, 2], [2, 2])             # [BATCH_SZIE, 14, 14, 64] -> [BATCH_SZIE, 7, 7, 64]

    # 마지막엔 이전 layer에서 뽑아낸 추상화된 정보들(feature라고 부름)을 이용해 fc layer 학습
    layer3 = tf.layers.flatten(layer2)                                   # [BATCH_SIZE, 7, 7, 64] -> [BATCH_SIZE, 3136]
    layer3 = tf.layers.dense(layer3, 256, activation=tf.nn.relu)         # [BATCH_SIZE, 3136] -> [BATCH_SIZE, 256]
    y_ = tf.layers.dense(layer3, 10, activation=None)                    # [BATCH_SIZE, 256] -> [BATCH_SIZE, 10]


    # loss : cross-entropy
    #        -1*(y*tf.log(y_+1e-6) + (1-y) * tf.log(1-y_+1e-6)) 이걸 써도 되고
    #        식이 기억나지 않으면 tf가 제공하는걸 씁시다.
    # optimizer : adam
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y))
    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    # tf.argmax : 최대값의 index값을 return, 즉 예측 결과
    y_pred = tf.argmax(y_,axis=-1,output_type=tf.int32)
    

# 정확도 계산(같은 것으로 예측할 확률)
def get_acc(label,prediction):
    corr = .0
    for i in range(BATCH_SIZE):
        if label[i]==prediction[i]:
            corr+=1
    return corr/BATCH_SIZE

# 학습
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    # train
    for epoch in range(TOTAL_EPOCHS):
        for i in range((len(x_train)//BATCH_SIZE)):
            batch_x = x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            batch_y = y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            # [28, 28] -> [28, 28, 1] 채널을 명시해 줘야 함
            batch_x = batch_x.reshape(-1, SIZE, SIZE, 1)

            sess.run(train,feed_dict={X:batch_x,Y:batch_y})

        if epoch%5 == 4:
            print('-'*20)
            print('epoch : {}'.format(epoch+1))
            x_t = x_train[0:BATCH_SIZE].reshape(-1, SIZE, SIZE, 1)
            loss_p = sess.run(loss,feed_dict={X:x_t,Y:y_train[0:BATCH_SIZE]})
            print('loss : {:.6f}'.format(loss_p))

    # test
    acc_batch=.0
    for i in range((len(x_test)//BATCH_SIZE)):
        batch_x = x_test[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        # [28, 28] -> [28, 28, 1] 채널을 명시해 줘야 함
        batch_x = batch_x.reshape(-1, SIZE, SIZE, 1)
        batch_y = y_test[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        predictions=sess.run(y_pred,feed_dict={X:batch_x})
        acc_batch+=get_acc(batch_y,predictions)
    acc=acc_batch/(len(x_test)//BATCH_SIZE)
    print('-'*20)
    print('acc : {:.6f}'.format(acc))
    print('='*20)
