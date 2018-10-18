import tensorflow as tf
import numpy as np
import pickle

# pickle file input
# pickle이라는 라이브러리는 데이터를 쉽게 dump/load할 수 있게 해줌
# tensorflow가 아니더라도 자주 사용하게 되는 라이브러리임

def pickle_input(filename):
    with open('./{}'.format(filename),'rb') as f:
        input_data,validate = pickle.load(f)
    return input_data,validate

# parameters
# NUM_DATA : 데이터의 총 개수
# LEARNING_RATE : 학습을 진행하는 정도 (다음 세미나에서 다룰 예정)
# NUM_EPOCHES : 데이터를 학습하는 횟수, 한 epoch마다 모든 데이터를 입력함

NUM_DATA = 1000
LEARNING_RATE = 0.01
NUM_EPOCHES = 10

# input
input_data,validate = pickle_input('single_perceptron_separator_input_and')
print(input_data.shape,validate.shape)

# architecture
# x1,x2 : 주어진 좌표 값
# y : 주어진 분류 값, (1 == on, 0 == off)
# y_ : sigmoid(W1 * x1 + W2 * x2 - b)로서 계산된 분류 값
# W1, W2, b : 학습할 변수들
# r : normalize에 쓰이는 값, W1, W2, b로 이루어진 벡터의 크기
# loss : 우리가 최소화 해야할 대상, 계산된 분류 값과 주어진 분류 값의 차이의 제곱 (다음 세미나에서 다룰 예정)
# optimizer : W1, W2와 b의 값을 조정해주는 최적화 알고리즘 (다음 세미나에서 다룰 예정)
# train : optimizer가 loss를 최소화 하도록 하는 tensor, sess.run()의 대상 (다음 세미나에서 다룰 예정)

W1 = tf.Variable(0.0)
W2 = tf.Variable(0.0)
b = tf.Variable(0.0)
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
y_ = tf.nn.sigmoid(W1 * x1 +W2*x2-b)

r = tf.sqrt(W1*W1+W2*W2+b*b)
norm_W1 = W1/r
norm_W2 = W2/r
norm_b = b/r

loss = tf.square(y-y_)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
train = optimizer.minimize(loss)

# training

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(NUM_EPOCHES):
        for i in range(NUM_DATA):
            data_x1,data_x2 = input_data[i]
            val_y = validate[i]
            sess.run(train,feed_dict={x1 : data_x1, x2 : data_x2,y:val_y})
        p_W1,p_W2,p_b,p_loss = sess.run((norm_W1,norm_W2,norm_b,loss),feed_dict={x1 : input_data[0][0], x2 : input_data[0][1],y:validate[0]})

        # print
        print('-'*20)
        print('epoch : {}'.format(epoch+1))
        print('loss : {:.6f}'.format(p_loss))
        print('W1 : {:.6f}'.format(p_W1))
        print('W2 : {:.6f}'.format(p_W2))
        print('b : {:.6f}'.format(p_b))
    print('='*20)        
