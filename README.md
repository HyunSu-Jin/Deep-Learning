# Deep-Learning
- 모두를 위한 딥러닝(김성훈 교수님)

# Lab9-1
Neural Network

Neural Network라는 단어는 처음 인공지능이라는 개념이 생성되어질 때, 사람이 뇌에서 생각하는 과정인 뉴런-시냅스를 통한 화학물질 전달과정을 머신에 그대로 묘사한 것에서 명칭이 사용되었다.
즉, 뉴런에서 input 데이터를 받고, input에 대한 처리를 W(weight)곱과 b(bias) 의 합을 통해 얻어진 결과값에 activation fuction을 적용, 최종적으로 얻어진 결과값을 다음 뉴런에 전달한다.
이러한 뉴런의 메커니즘을 모방하여, 머신러닝의 일종으로 Neural Network를 사용한다.

![NN](/lab9-1/img/NN.png)

1. 기존의 머신러닝모델(단일 뉴런)의 문제점
기존의 단일 뉴런모델의 가장 큰 문제점은 XOR문제를 해결할 hyperplane을 만들 수 없다는 데 있다.

![xor](/lab9-1/img/xor.png)

위 그림에서와 같이 data들의 구분자(hyperplane)을 linear한 모델로 표현할 수 있는 data 군집이 있는 반면,linear한 모델로 표현이 불가능한 군집도 존재한다. XOR같은 모양의 데이터를 구분하는 hyperplane은 linear하게 표현될 수 없다. 따라서, 기존의 단일뉴런모델을 사용하게 된다면 아래와같은 학습결과를 얻는다.
<pre><code>
x_data = [
	[0,0],
	[0,1],
	[1,0],
	[1,1]
]
y_data = [0,1,1,0]

X = tf.placeholder(tf.float32,shape=(None,2))
Y = tf.placeholder(tf.float32,shape=(None))

W = tf.Variable(tf.random_normal([2,1]),name="weight")
b = tf.Variable(tf.random_normal([1]),name = "bias")

logits = tf.matmul(X,W) +b
hypothesis = tf.sigmoid(logits)
</code></pre>

![xor_impo](/lab9-1/result/lab9-1_result1.png)

2. 해결책
이러한 문제를 해결하기 위한 방법으로 Neural Network를 사용할 수 있다. 단일 뉴런을 사용하는 것이 아닌, 다중 뉴런을 사용하게 된다면,(hidden layer를 사용한다면) hyperplane의 non-linearity가 증가하게 되어 위 문제를 해결할 수 있다.
<pre><code>
# layer1
with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([2,2]),name="weight1")
    b1 = tf.Variable(tf.random_normal([2]),name="bias1")
    layer1 = tf.sigmoid(tf.matmul(X,W1)+b1)

    w1_hist = tf.summary.histogram("weights1", W1)
    b1_hist = tf.summary.histogram("biases1", b1)
    layer1_hist = tf.summary.histogram("layer1", layer1)
    
# layer2
with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal([2,2]),name="weight2")
    b2 = tf.Variable(tf.random_normal([2]),name="bias2")
    layer2 = tf.sigmoid(tf.matmul(layer1,W2)+ b2)
    
    w2_hist = tf.summary.histogram("weights2", W2)
    b2_hist = tf.summary.histogram("biases2", b2)
    layer2_hist = tf.summary.histogram("layer2", layer2)

# layer3
with tf.name_scope("layer3") as scope:
    W3 = tf.Variable(tf.random_normal([2,1]),name="weight3")
    b3 = tf.Variable(tf.random_normal([1]),name="bias3")
    hypothesis = tf.sigmoid(tf.matmul(layer2,W3)+ b3)
    
    w3_hist = tf.summary.histogram("weights3", W3)
    b3_hist = tf.summary.histogram("biases3", b3)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)
</code></pre>

![board1](/lab9-1/result/tensorboard_1.png)

![board2](/lab9-1/result/tensorboard_2.png)

3. 초기 Neural Network의 문제점
- Neural Network의 시초에는 위 모델을 사용 했을때, 각각의 뉴런이 가지고 있는 weight와 bias를 학습 할 수 없을 것이라는 문제가 제기되었다. 실제로, forward propagation을 통한 방법으로는 불가능 했다.
### Backpropagation
- 이러한 문제에 대한 해결책으로써, "Back propagation"이 제시되었다. back-propagation은미분의 chain rule을 적용해서, Neural Network를 통해 최종적인 output으로 부터 얻어지는 Error를 거꾸로 탐색하여 최종 결과값에 영향을 주는 각각의 인자들에 대한 미분값을 구해내는 것이다. 이 방법을 사용하게 되면, 각각의 인자에 대해 GradientDescentOptimizer과 같은 방식의 cost function을 최소화 시킬 방법이 정의되어지므로 Neural Network를 통한 학습이 가능한 것이다.
### Vanishing Gradient
- 하지만, 위 모델의 경우 Neural Network가 Deep,Wild해 질수록 학습이 진행되어 지지 않는다는 한계를 가지고 잇는데, 그 이유는 다음과 같다. activation function으로 logis=WX+b 에 대해 sigmoid를 사용하게 될 경우 output이 0-1 사잇값을 가지므로 back propagation이 진행되어질수록 gradient값이 점차 0에 수렴해진다. 따라서, Neural Network가 deep,wild 한 경우 학습이 이루어지지 않았다.

### Activation fuction
- 이를 위한 해결책으로써, activation function(activation unit)면으로 sigmoid를 사용하지않고 ReLU,leacky Relu,..etc를 사용하는 방법이 있다. 이 중에서 Relu는 f(x)=max(0,x)로 표현할 수 있어, 기존의 sigmoid와 달리 back propagation을 진행하더라도 vanishing gradient 문제를 해결할 수 있다.
### Overfitting
- Neural Network를 deep,wild 하게 디자인해서 accuacry를 측정한 경우, 기대치 이하의 성능을 보이는 경우가 있다. 이는 DNN의 overfitting성질과 관련이 있는데 overfitting은 모델이 traning data에만 너무 치중되어 학습된바람에 실제 test dataset에 대해 높은 정확성을 보여주지 못하는 경우를 의미한다.
### Dropout
- Overfitting 문제에 대한 해결책으로써 Regularization방법과 Dropout 방법이 있다. 이 중 Dropout은 DNN에서 사용하는 방법인데, 학습도중 Network에 존재하는 일부 Neuron을 비활성화하여 학습과정에 참가시키지 않는 것을 의미한다. 이렇게 하게 되면 설정한 rate에 대해 모델이 training dataset에 치중되게 학습하지 않게하는것임을 의미하므로 overfitting문제를 해결할 수 있다. 단, test dataset에 대해서는 keep_prob 속성을 1.0으로 설정하여 test과정에서는 dropout설정을 하지 않는것에 유의하여야 한다.

4. DNN
위 주의사항을 토대로, Deep Neural Network 모델을 디자인하게 되면 단일 뉴런모델보다 높은 정확도를 보여주게 되는데 그 이유는 다음과 같다. 앞서, XOR문제에 대해 단일뉴런모델은 해결할 수 없다는 것을 살펴보았다. 그 이유는 무엇이었나? 바로 hyperplane이 linear하므로 non-linear 한 분포의 dataset에 대해서 학습이 불가능하기 때문이었다. 이원리를 기반으로 DNN을 디자인 한다는 것은 모델의 hyperplane의 non-linearity를 증가 시킨다는 것을 의미하고, 이는 dataset의 랜덤한 분포에 대해서 효율적인 hyperplane을 만든다는 것을 의미하므로 단일 뉴런모델보다 높은 성능을 보이는 것이다.

# Lab10-1
NN, ReLU, Xavier, Dropout and Adam

1. Activation fuction
- maxout
- ReLU
- VLReLU
- tanh
- Sigmoid

2. Weights Initializer
- LSUV
- OrthoNorm
- OrthoNorm-MSRA scaled
- Xavier
- MSRA

3. Overfitting Solution
- Dropout
- Regularization

4. Optimizer
- sgd
- momentum
- nag
- adagrad
- adadelta
- rmsprop
- adam

5. 예시
DNN, ReLU, Xavier, Dropout, Adam을 사용한 MNIST예제
![lab10-1](/lab10-1/result/result.png)

![lab10-1-b](/lab10-1/result/accuracy.png)

![lab10-1-b2](/lab10-1/result/graph.png)
