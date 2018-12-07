import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

train_x = mnist.train.images
train_y = mnist.train.labels

total_epochs = 450
batch_size = 100
learning_rate = 0.0001

# Generator의 입력으로 넣어줄 z의 크기
random_size = 100
image_size = 28*28

# z를 받아서 이미지 생성
def generator(z , reuse = False ):
    l = [random_size, 128, 256, image_size]

    # 텐플은 trainable Variable에 이름을 붙일 수 있음
    # Scope의 이름을 정하면 그 scope안의 모든 Variable는 scope의 이름으로 시작하는 이름을 가짐
    # 아래의 경우 모든 변수는 'Gen/어쩌구' 형태의 이름을 가짐
    with tf.variable_scope(name_or_scope = "Gen") as scope:
        layer1 = tf.layers.dense(z, l[1], activation=tf.nn.relu)
        layer2 = tf.layers.dense(layer1, l[2], activation=tf.nn.relu)

        # 이미지의 각 픽셀은 0~1사이의 값을 가져야 하므로 sigmoid로 범위 제한 
        layer3 = tf.layers.dense(layer2, l[3], activation=tf.nn.sigmoid)

    return layer3


# 이미지를 받아서 진짜인지 가짜인지 판단
def discriminator(x , reuse = False):
    l = [image_size, 256, 128, 1]

    # Scope가 여러 번 사용되는 경우(밑에서 discriminator 함수가 2번 호출됨) reuse 여부를 명시해야 함
    with tf.variable_scope(name_or_scope="Dis", reuse=reuse) as scope:
        layer1 = tf.layers.dense(x, l[1], activation=tf.nn.relu)
        layer2 = tf.layers.dense(layer1, l[2], activation=tf.nn.relu)

        # 0인지 1인지만 판단하므로 softmax가 아니라 sigmoid 사용
        layer3 = tf.layers.dense(layer2, l[3], activation=tf.nn.sigmoid)

    return layer3


def random_noise(batch_size):
    return np.random.normal(size=[batch_size , random_size])




g = tf.Graph()

with g.as_default():
    X = tf.placeholder(tf.float32, [None, 784])
    Z = tf.placeholder(tf.float32, [None, random_size])

    fake_x = generator(Z) # G(z)
    result_of_fake = discriminator(fake_x) # D(G(z))

    # discriminator 함수를 두 번째 호출 -> reuse = True로 해줘야 변수를 새로 만들지 않음
    result_of_real = discriminator(X , True) # D(x)

    g_loss = tf.reduce_mean( tf.log(result_of_fake) ) # log D(G(z))
    d_loss = tf.reduce_mean( tf.log(result_of_real) + tf.log(1 - result_of_fake) ) # log D(x) + log(1 - D(G(z)) )

    # G와 D를 각각 학습해야 하므로 위에서 정해준 scope에 따라 변수의 이름이 달라졌다는 것을 활용
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if "Gen" in var.name]
    d_vars = [var for var in t_vars if "Dis" in var.name]
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # loss 를 최대화 하고 싶을땐 -loss를 최소화
    g_train = optimizer.minimize(-g_loss, var_list= g_vars)
    d_train = optimizer.minimize(-d_loss, var_list = d_vars)



with tf.Session(graph = g) as sess:
    sess.run(tf.global_variables_initializer())
    total_batchs = int(train_x.shape[0] / batch_size)

    for epoch in range(total_epochs):
        for batch in range(total_batchs):
            batch_x = train_x[batch * batch_size : (batch+1) * batch_size]
            batch_y = train_y[batch * batch_size : (batch+1) * batch_size]
            noise = random_noise(batch_size)

            sess.run(g_train , feed_dict = {Z : noise})
            sess.run(d_train, feed_dict = {X : batch_x , Z : noise})

            gl, dl = sess.run([g_loss, d_loss], feed_dict = {X : batch_x , Z : noise})


        if epoch % 2 == 0:
            print("======= Epoch : ", epoch , " =======")
            print("generator: " , -gl )
            print("discriminator: " , -dl )


        samples = 20
        if epoch % 2 == 0:
            sample_noise = random_noise(samples)
            gen = sess.run(fake_x , feed_dict = { Z : sample_noise})

            for i in range(samples):
                img = gen[i].reshape([28,28])
                mpimg.imsave('./epoch/epoch'+str(epoch)+'_'+str(i)+'.png', img, format='png')
