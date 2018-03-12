#xavier 
->
```
W1 = tf.get_variable("W1", shape=[784, 256],                  initializer=tf.contrib.layers.xavier_initializer())
```

xavier도 해주고 모델도 더 깊고 넓게 해줬음에도 Accuracy가 더 줄어들었다 -> 오버피팅 -> dropout

#dropout
```
keep_prob = tf.placeholder(tf.float32)
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
```
layer의 마지막에 dropout을 추가해준다. 통상적으로 학습을 할때는 keep_prob를 0.5~0.7로 설정해주고 testing할때는 1로해주어서 한다. 

#MNIST
softmax -> 90%
Neural Nets -> 94.5%
Xavier initialization -> 97.8%
Deep Neural Nets with Dropout -> 98% 

#Batch Normalization
Gradient vanishing -> backpropagation algorithm에서 아래 layer로 내려갈수록, 현재 parameter의 gradient를 계산했을 때 앞에서 받은 미분 값들이 곱해지면서 그 값이 거의 없어지는 (vanish하는) 현상을 의미

Gradient exploding -> learning rate가 너무 높아 diverge하는 현상을 말한다. 

Learning rate의 값이 클수록 이 두 가지 현상이 발생활 확률이 높다.

exploding이 일어나지 않을 정도의 learning rate를 설정하여 속도를 향상시키고프다.

Q. Gradient vanishing/exploding problem이 발생하지 않도록 하면서 learning rate 값을 크게 설정할 수 있는 neural network model을 design하는 법은?
[http://sanghyukchun.github.io/88/]




