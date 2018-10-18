import tensorflow as tf
import numpy as np
import pickle

# pickle file input
# pickle이라는 라이브러리는 데이터를 쉽게 dump/load할 수 있게 해줌
# tensorflow가 아니더라도 자주 사용하게 되는 라이브러리임

def pickle_input(filename):
    with open('./{}'.format(filename),'rb') as f:
        data = pickle.load(f)
    return data

# parameters
# NUM_DATA : 데이터의 총 개수
# LEARNING_RATE : 학습을 진행하는 정도 (다음 세미나에서 다룰 예정)
# NUM_EPOCHES : 데이터를 학습하는 횟수, 한 epoch마다 모든 데이터를 입력함

NUM_DATA = 2000
LEARNING_RATE = 0.01
NUM_EPOCHES = 10

# input
data = pickle_input('linear_regression_input')
print(data.shape)

# architecture
# y_ : 학습된 추세선의 x에 대한 y좌표 값
# x, y : 주어진 input
# W, b : y_ = W * x + b에서의 기울기와 y절편 값(학습의 대상)
# loss : 우리가 최소화 해야할 대상, 추세선의 y값과 실제 y값의 차이의 제곱 (다음 세미나에서 다룰 예정)
# optimizer : W와 b의 값을 조정해주는 최적화 알고리즘 (다음 세미나에서 다룰 예정)
# train : optimizer가 loss를 최소화 하도록 하는 tensor, sess.run()의 대상 (다음 세미나에서 다룰 예정)

W = tf.Variable(0.0)
b = tf.Variable(0.0)
x = tf.placeholder(tf.float32,[])
y = tf.placeholder(tf.float32,[])
y_ = W * x + b

loss = tf.square(y_-y)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
train = optimizer.minimize(loss)

# training

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(NUM_EPOCHES):
        for i in range(NUM_DATA):
            data_x,data_y = data[i]
            sess.run(train,feed_dict={x : data_x, y : data_y})
        p_W, p_b, p_loss = sess.run((W,b,loss),feed_dict={x:data[0][0],y:data[0][1]})

        # print
        print('-'*20)
        print('epoch : {}'.format(epoch+1))
        print('loss : {:.6f}'.format(p_loss))
        print('W : {:.6f}'.format(p_W))
        print('b : {:.6f}'.format(p_b))
    print('='*20)
