### Deep Learning, 딥 러닝

- 인공 신경망(Artificial Neural Network)의 층을 연속적으로 깊게 쌓아올려 데이터를 학습하는 방식을 의미한다.
- 인간이 학습하고 기억하는 매커니즘을 모방한 기계학습이다.
- 인간은 학습 시, 뇌에 있는 뉴런이 자극을 받아들여서 일정 자극 이상이 되면, 화학물질을 통해 다른 뉴런과 연결되며 해당 부분이 발달한다.
- 자극이 약하거나 기준치를 넘지 못하면, 뉴런은 연결되지 않는다.
- 입력한 데이터가 활성 함수에서 임계점을 넘게 되면 출력된다.
- 초기 인공 신경망(Perceptron)에서 깊게 층을 쌓아 학습하는 딥 러닝으로 발전한다.
- 딥 러닝은 Input nodes layer, Hidden nodes layer, Output nodes layer, 이렇게 3가지 층이 존재한다.

<img src="https://github.com/SOYOUNGdev/study_deep_learning/assets/115638411/5658102f-39c5-4a1b-ae4a-99c35d64b174" style="margjin-left: 0">

### 📍<a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/CH01.-SLP-%E2%80%90-Single-Layer-Perceptron">SLP (Single Layer Perceptron), 단층 퍼셉트론, 단일 퍼셉트론 </a>

- 가장 단순한 형태의 신경망으로서, Hidden Layer가 없고 Single Layer로 구성되어 있다.
- 퍼셉트론의 구조는 입력 feature와 가중치, activation function, 출력 값으로 구성되어 있다.
- 신경 세포에서 신호를 전달하는 축삭돌기의 역할을 퍼셉트론에서는 가중치가 대신하고,  
  입력 값과 가중치 값은 모두 인공 뉴런(활성 함수)으로 도착한다.
- 가중치의 값이 클수록 해당 입력 값이 중요하다는 뜻이고, 인공 뉴런(활성 함수)에 도착한 각 입력 값과 가중치 값을 곱한 뒤 전체 합한 값을 구한다.
- 인공 뉴런(활성 함수)은 보통 시그모이드 함수와 같은 계단 함수를 사용하여,
  합한 값을 확률로 변환하고 이 때, 임계치를 기준으로 0 또는 1을 출력한다.

#### SGD (Stochastic Gradient Descent), 확률적 경사 하강법
- 전체 학습 데이터 중, 단 한 건만 임의로 선택하여 경사 하강법을 실시하는 방식을 의미한다.
- 일반적으로 사용되지 않고, SGD를 얘기할 때에는 보통 미니 배치 경사 하강법을 의미한다.

#### Mini-Batch Gradient Descent, 미니 배치 경사 하강법
- 전체 학습 데이터 중, 특정 크기(Batch 크기)만큼 임의로 선택해서 경사 하강법을 실시한다. 이 또한, 확률적 경사 하강법

### 📍<a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/CH02.-MLP-%E2%80%90-Multi-Layer-Perceptron">Multi Layer Perceptron, 다층 퍼셉트론, 다중 퍼셉트론</a>
- 보다 복잡한 문제의 해결을 위해서 입력층과 출력층 사이에 은닉층이 포함되어 있다.
- 퍼셉트론을 여러층 쌓은 인공 신경망으로서, 각 층에서는 활성함수를 통해 입력을 처리한다.
- 층이 깊어질 수록 정확한 분류가 가능해지지만, 너무 깊어지면 Overfitting이 발생한다.

#### ANN (Artificial Neural Network), 인공 신경망
- 은닉층이 1개일 경우 이를 인공 신경망이라고 한다.

#### DNN (Deep Neural Network), 심층 신경망
- 은닉층이 2개 이상일 경우 이를 심층 신경망이라고 한다.

#### Back-propagation, 역전파
- 심층 신경망에서 최종 출력(예측)을 하기 위한 식이 생기지만 식이 너무 복잡해지기 때문에 편미분을 진행하기에 한계가 있다.
- 즉, 편미분을 통해 가중치 값을 구하고, 경사 하강법을 통해 가중치 값을 업데이트 하며, 손실 함수의 최소값을 찾아야 하는데,  
  순방향으로는 복잡한 미분식을 계산할 수가 없다. 따라서 미분의 연쇄 법칙(Chain Rule)을 사용하여 역방향으로 편미분을 진행한다.

### 📍<a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/CH03.-Activation-Fuction-%E2%80%90-sigmoid,-softmax,-tanh,-relu">Activation Function, 활성화 함수</a>

- 인공 신경망에서 입력 값에 가중치를 곱한 뒤 합한 결과를 적용하는 함수이다.
---
1. 시그모이드 함수
   - 은닉층이 아닌 최종 활성화 함수 즉, 출력층에서 사용된다.
   
2. 소프트맥스 함수
    - 은닉층이 아닌 최종 활성화 함수(출력층)에서 사용된다.
    - 여러 개의 타겟 데이터를 분류하는 다중 분류의 최종 활성화 함수(출력층)로 사용된다.
  
3. 탄젠트 함수
    - 은닉층이 아닌 최종 활성화 함수(출력층)에서 사용된다.
   
4. 렐루 함수
   - 대표적인 은닉층의 활성 함수이다.

### 📍<a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/CH04.-Optimizer-%E2%80%90-Momentum,-AdaGrad,-RMSProp,-*Adam">Optimizer, 최적화</a>
- 최적의 경사 하강법을 적용하기 위해 필요하며, 최소값을 찾아가는 방법들을 의미한다.

#### Momentum
- 가중치를 계속 업데이트할 때마다 이전의 값을 일정 수준 반영시키면서 새로운 가중치로 업데이트한다.

#### AdaGrad (Adaptive Gradient)
- 가중치 별로 서로 다른 학습률을 동적으로 적용한다.
- 적게 변화된 가중치는 보다 큰 학습률을 적용하고, 맣이 변화된 가중치는 보다 작은 학습률을 적용시킨다.
- 과거의 모든 기울기를 사용하기 때문에 학습률이 급격히 감소하여, 분모가 커짐으로써 학습률이 0에 가까워지는 문제가 있다.

#### RMSProp (Root Mean Square Propagation)
- AdaGrad의 단점을 보완한 기법으로서, 학습률이 지나치게 작아지는 것을 막기 위해 지수 가중 평균법(Exponentially weighted average)을 통해 구한다.
- 이전의 기울기들을 똑같이 더해가는 것이 아니라 훨씬 이전의 기울기는 조금 반영하고 최근의 기울기를 많이 반영한다.
- feature마다 적절한 학습률을 적용하여 효율적인 학습을 진행할 수 있고, AdaGrad보다 학습을 오래 할 수 있다.

#### Adam (Adaptive Moment Estimation)
- Momentum과 RMSProp 두 가지 방식을 결합한 형태로서, 진행하던 속도에 관성을 주고, 지수 가중 평균법을 적용한 알고리즘이다.

### 📍<a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/CH05.-Tensorflow">Tensorflow, 텐서플로우</a>
- 구글이 개발한 오픈소스 소프트웨어 라이브러리이며, 머신러닝과 딥러닝을 쉽게 사용할 수 있도록 다양한 기능을 제공한다.
- 주로 이미지 인식이나 반복 신경망 구성, 기계 번역, 필기 숫자 판별 등을 위한 각종 신경망 학습에 사용된다.

### Keras, 케라스
- 일반 사용 사례에 "최적화, 간단, 일관, 단순화"된 인터페이스를 제공한다.
- 손쉽게 딥러닝 모델을 개발하고 활용할 수 있도록 직관적인 API를 제공한다.

#### Grayscale, RGB
- 흑백 이미지와 컬러 이미지는 각 2차원과 3차원으로 표현될 수 있다.
- 흑백 이미지는 0 ~ 255를 갖는 2차원 배열(높이 X 너비)이고,
  컬러 이미지는 0 ~ 255를 갖는 R, G, B 2차원 배열 3개를 갖는 3차원 배열(높이 X 너비 X 채널)이다.

### Grayscale Image Matrix
- 검은색에 가까운 색은 0에 가깝고 흰색에 가까우면 255에 가깝다.
- 모든 픽셀이 feature이다.

### 📍<a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/CH06.-API(Sequential,-Functional,-Callback)">📍Sequential API, Functional API</a>

#### Sequential API
- 간단한 모델을 구현하기에 적합하고 단순하게 층을 쌓는 방식으로 쉽고 사용하기가 간단하다.

#### Functional API
- Functional API는 Sequential API로는 구현하기 어려운 복잡한 모델들을 구현할 수 있다.

### 📍<a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/CH06.-API(Sequential,-Functional,-Callback)#callback-api">Callback API</a>
- 모델이 학습 중에 충돌이 발생하거나 네트워크가 끊기면, 모든 훈련 시간이 낭비될 수 있고,
  과적합을 방지하기 위해 훈련을 중간에 중지해야 할 수도 있다.
- 모델이 학습을 시작하면 학습이 완료될 때까지 아무런 제어를 하지 못하게 되고,
  신경망 훈련을 완료하는 데에는 몇 시간 또는 며칠이 걸릴 수 있기 때문에 모델을 모니터링하고 제어할 수 있는 기능이 필요하다.
- 훈련 시(fit()) Callback API를 등록시키면 반복 내에서 특정 이벤트 발생마다 등록된 callback이 호출되어 수행된다.

**ModelCheckPoint(filepath, monitor='val_loss', verbose=0, save_best_only=False  
save_weight_only=False, mode='auto')**
- 특정 조건에 따라서 모델 또는 가중치를 파일로 저장한다.

**ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_lr=0)**
- 특정 반복동안 성능이 개선되지 않을 때, 학습률을 동적으로 감소시킨다.

**EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')**
- 특정 반복동안 성능이 개선되지 않을 때, 학습을 조기에 중단한다.

