import tensorflow as tf
import numpy as np

# MNIST CLASSIFICATION

# MNIST DATASET을 받는다.
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# 이미지 정보가 그대로 들어있으므로(int), 이를 [0,1]로 normalize
x_train, x_test = x_train/255.0, x_test/255.0


# 중요한 변수 설정
# SIZE : 이미지의 크기
# BATCH_SIZE : BATCH의 크기, 적절하게 설정함
# LEARNING_RATE
# TOTAL_EPOCHES : 총 반복 횟수
SIZE = 28
BATCH_SIZE = 500
LEARNING_RATE = 0.005
TOTAL_EPOCHES = 100

# graph 설정, 우리가 설정한 구조가 들어감
g = tf.Graph()
with g.as_default():

    # 입력 받는 placeholder
    # X : [ BATCH_SIZE, SIZE, SIZE ], BATCH_SIZE개의 (SIZE * SIZE) 크기의 이미지
    # Y : [ BATCH_SIZE ], 각 이미지의 classification 들(0-9사이의 값)
    X = tf.placeholder(tf.float32,[None,SIZE,SIZE])
    Y = tf.placeholder(tf.int32,[None])

    # Y에 대한 one-hot encoding
    y = tf.one_hot(Y,10)

    # X의 형태는 [ BATCH_SIZE, SIZE, SIZE ]이므로 DNN에 넣기 쉽게 풀어줌
    flat = tf.reshape(X,[-1,SIZE*SIZE])

    # DNN을 설정
    # tf.layers.dense(<input>, units=<units>, activation=<activation>)
    # <input> : layer를 통과하고자 하는 입력 tensor
    # <units> : layer에 있는 총 perceptron의 개수, layer를 통과한 출력 tensor
    # <activation> : activation function들이 들어감
    ### tf.nn.sigmoid, tf.nn.softmax, tf.nn.relu, ...
    
    layer1 = tf.layers.dense(flat,units=50,activation=tf.nn.relu) # [ BATCH_SIZE, 784 ] -> [ BATCH_SIZE, 50 ]
    layer2 = tf.layers.dense(layer1,units=20,activation=tf.nn.relu) # [ BATCH_SIZE, 50 ] -> [ BATCH_SIZE, 20 ]
    y_ = tf.layers.dense(layer2,units=10,activation=tf.nn.softmax) # [ BATCH_SIZE, 20 ] -> [ BATCH_SIZE, 10 ]

    # loss : cross-entropy
    # optimizer : adam
    loss = tf.reduce_mean(-1*(y*tf.log(y_+1e-6) + (1-y) * tf.log(1-y_+1e-6)))
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
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
    for epoch in range(TOTAL_EPOCHES):
        for i in range((len(x_train)//BATCH_SIZE)):
            batch_x = x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            batch_y = y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            sess.run(train,feed_dict={X:batch_x,Y:batch_y})

        if epoch%5 == 4:
            print('-'*20)
            print('epoch : {}'.format(epoch+1))
            loss_p = sess.run(loss,feed_dict={X:x_train[0:BATCH_SIZE],Y:y_train[0:BATCH_SIZE]})
            print('loss : {:.6f}'.format(loss_p))

    # test
    acc_batch=.0
    for i in range((len(x_test)//BATCH_SIZE)):
        batch_x = x_test[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        batch_y = y_test[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        predictions=sess.run(y_pred,feed_dict={X:batch_x})
        acc_batch+=get_acc(batch_y,predictions)
    acc=acc_batch/(len(x_test)//BATCH_SIZE)
    print('-'*20)
    print('acc : {:.6f}'.format(acc))
    print('='*20)
