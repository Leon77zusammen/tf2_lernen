import tensorflow as tf

# # basic Bespiel 1
# x = tf.Variable(initial_value=3.)
# with tf.GradientTape() as tape:
#     y = tf.square(x)
# y_grad = tape.gradient(y, x)
# print(y, y_grad)

# # basic Bespiel 2
# X = tf.constant([[1., 2.], [3., 4.]])
# y = tf.constant([[1.], [2.]])
# w = tf.Variable(initial_value=[[1.], [2.]])
# b = tf.Variable(initial_value=1.)
# with tf.GradientTape() as tape:
#     L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
# w_grad, b_grad = tape.gradient(L, [w, b])        # 计算L(w, b)关于w, b的偏导数
# print(L, w_grad, b_grad)

# # basic Bespiel 3
# # use numpy 
# import numpy as np

# X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
# y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

# X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
# y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

# a, b = 0, 0

# num_epoch = 10000
# learning_rate = 5e-4
# for e in range(num_epoch):
#     # 手动计算损失函数关于自变量（模型参数）的梯度
#     y_pred = a * X + b
#     grad_a, grad_b = 2 * (y_pred - y).dot(X), 2 * (y_pred - y).sum()

#     # 更新参数
#     a, b = a - learning_rate * grad_a, b - learning_rate * grad_b

# print(a, b)

# use tf2
import numpy as np
X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
X = tf.constant(X)
y = tf.constant(y)

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]

num_epoch = 10000
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)
for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = tf.reduce_sum(tf.square(y_pred - y))
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)
    # TensorFlow自动根据梯度更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

print(a, b)