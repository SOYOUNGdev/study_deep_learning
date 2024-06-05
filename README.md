### Deep Learning, 딥 러닝

- 인공 신경망(Artificial Neural Network)의 층을 연속적으로 깊게 쌓아올려 데이터를 학습하는 방식을 의미한다.
- 인간이 학습하고 기억하는 매커니즘을 모방한 기계학습이다.
- 인간은 학습 시, 뇌에 있는 뉴런이 자극을 받아들여서 일정 자극 이상이 되면, 화학물질을 통해 다른 뉴런과 연결되며 해당 부분이 발달한다.
- 자극이 약하거나 기준치를 넘지 못하면, 뉴런은 연결되지 않는다.
- 입력한 데이터가 활성 함수에서 임계점을 넘게 되면 출력된다.
- 초기 인공 신경망(Perceptron)에서 깊게 층을 쌓아 학습하는 딥 러닝으로 발전한다.
- 딥 러닝은 Input nodes layer, Hidden nodes layer, Output nodes layer, 이렇게 3가지 층이 존재한다.

<img src="https://github.com/SOYOUNGdev/study_deep_learning/assets/115638411/5658102f-39c5-4a1b-ae4a-99c35d64b174" style="margjin-left: 0">

---

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

---

### 📍<a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/CH03.-Activation-Fuction-%E2%80%90-sigmoid,-softmax,-tanh,-relu">Activation Function, 활성화 함수</a>

- 인공 신경망에서 입력 값에 가중치를 곱한 뒤 합한 결과를 적용하는 함수이다.

1. 시그모이드 함수
   - 은닉층이 아닌 최종 활성화 함수 즉, 출력층에서 사용된다.
   
2. 소프트맥스 함수
    - 은닉층이 아닌 최종 활성화 함수(출력층)에서 사용된다.
    - 여러 개의 타겟 데이터를 분류하는 다중 분류의 최종 활성화 함수(출력층)로 사용된다.
  
3. 탄젠트 함수
    - 은닉층이 아닌 최종 활성화 함수(출력층)에서 사용된다.
   
4. 렐루 함수
   - 대표적인 은닉층의 활성 함수이다.

---

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

---

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

---

### 📍<a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/CH06.-API(Sequential,-Functional,-Callback)">Sequential API, Functional API</a>

#### Sequential API
- 간단한 모델을 구현하기에 적합하고 단순하게 층을 쌓는 방식으로 쉽고 사용하기가 간단하다.

#### Functional API
- Functional API는 Sequential API로는 구현하기 어려운 복잡한 모델들을 구현할 수 있다.

---

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

### <a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/CH07.-CNN,-%ED%95%A9%EC%84%B1%EA%B3%B1-%EC%8B%A0%EA%B2%BD%EB%A7%9D">📍 CNN (Convloutional Neural Network), 합성곱 신경망</a>
- 실제 이미지 데이터는 분류 대상이 이미지에서 고정된 위치에 있지 않은 경우가 대부분이다.
- 실제 이미지 데이터를 분류하기 위해서는, 이미지의 각 feature들을 그대로 학습하는 것이 아닌, CNN으로 패턴을 인식한 뒤 학습해야 한다.
- CNN은 인간의 시신경 구조를 모방한 기술로서, 이미지의 패턴을 찾을 때 사용한다.

#### Filter
- 보통 정방 행렬로 구성되어 있고, 원본 이미지에 슬라이딩 윈도우 알고리즘을 사용하여 순차적으로 새로운 픽셀값을 만들면서 적용한다.
- 사용자가 목적에 맞는 특정 필터를 만들거나 기존에 설계된 다양한 필터를 선택하여 이미지에 적용한다.
  하지만, CNN은 최적의 필터값을 학습하여 스스로 최적화한다.
- 필터 하나 당, 이미지의 채널 수 만큼 Kernel이 존재하고, 각 채널에 할당된 필터의 커널을 적용하여 출력 이미지를 생성한다.
- 출력 feature map의 개수는 필터의 개수와 동일하다.

#### Kernel
- filter 안에 1 ~ n개의 커널이 존재한다. 커널의 개수는 반드시 이미지의 채널 수와 동일해야 한다.
- Kernel Size는 가로 X 세로를 의미하며, 가로와 세로는 서로 다를 수 있지만 보통은 일치시킨다.

#### Stride
- 입력 이미지에 Convolution Filter를 적용할 때 Sliding Window가 이동하는 간격을 의미한다.

#### Padding
- Filter를 적용하여 Convolution 수행 시 출력 feature map이 입력 feature map 대비 계속해서 작아지는 것을 막기 위해 사용한다.

#### Pooling
- Convolution이 적용된 feature map의 일정 영역별로 하나의 값을 추출하여 feature map의 사이즈를 줄인다.
- Max Pooling과 Average Pooling이 있으며, Max Pooling은 중요도가 가장 높은 feature를 추출하고, Average Pooling은 전체를 버무려서 추출한다.

---

### 📍<a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/CH08.-CNN-performance-%E2%80%90-Weight-Initialization,-BN,-GAP,-Weight-Regularization">CNN Performance</a>
- CNN 모델을 제작할 때, 다양한 기법을 통해 성능 개선 및 과적합 개선이 가능하다.

#### Weight Initialization, 가중치 초기화
- 처음 가중치를 어떻게 줄 것인지를 정하는 방법이며, 처음 가중치를 어떻게 설정하느냐에 따라 모델의 성능이 크게 달라질 수 있다.
  
> 1. 사비에르 글로로트 초기화
> - 고정된 표준편차를 사용하지 않고, 이전 층의 노드 수에 맞게 현재 층의 가중치를 초기화한다.

> 2. 카이밍 히 초기화
> - 고정된 표준편차를 사용하지 않고, 이전 층의 노드 수에 맞게 현재 층의 가중치를 초기화한다.

#### Batch Normalization, 배치 정규화
- 가중치를 초기화할 때 민감도를 감소시키고, 학습 속도 증가시키며, 모델을 일반화하기 위해서 사용한다.

#### Batch Size
- batch size를 작게 하면, 적절한 noise가 생겨서 overfitting을 방지하게 되고, 모델의 성능을 향상시키는 계기가 될 수 있지만, 너무 작아서는 안된다.
- batch size를 너무 작게 하는 경우에는 batch당 sample수가 작아져서 훈련 데이터를 학습하는 데에 부족할 수 있다.

#### Global Avereage Pooling
- 이전의 Pooling들은 면적을 줄이기 위해 사용했지만, Global Average Pooling은 면적을 없애고 채널 수 만큼 값을 나오게 한다.
- Flatten 후에 Classification Dense Layer로 이어지면서 많은 파라미터들로 인한 overfitting 유발 가능성 증대 및 학습 시간 증가로 이어지기 때문에  
  맨 마지막 feature map의 채널 수가 크다면 Global Average Pooling을 적용하는 것이 더 나을 수 있다.

#### Weight Regularization (가중치 규제), Weight Decay (가중치 감소)
- 기존 가중치에 특정 연산을 수행하여 loss function의 출력 값과 더해주면 loss function의 결과를 어느정도 제어할 수 있게 된다.
- kernel_regularizer 파라미터에서 l1, l2을 선택할 수 있다.

---

### 📍<a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/CH09.-Data-Augmentation">Data Augmentation, 데이터 증강</a>
- Data Augmentation을 통해 원본 이미지에 다양한 변형을 주어서 학습 이미지 데이터를 늘리는 것과 유사한 효과를 볼 수 있다.
- 원본 학습 이미지의 개수를 늘리는 것이 아닌 매 학습 마다 개별 원본 이미지를 변형해서 학습을 수행한다.

#### 공간 레벨 변형
- 좌우 또는 상하 반전, 특정 영역만큼 확대, 축소, 회전 등으로 변형시킨다.

#### 픽셀 레벨 변형
- 밝기, 명암, 채도, 색상 등을 변형시킨다.

---

### 📍<a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/CH10.-Pretrained-Model">Pretrained Model</a>
- 대규모 데이터 세트에서 훈련되고 저장된 네트워크로서, 일반적으로 대규모 이미지 분류 작업에서 훈련된 것을 뜻한다.
- 입력 이미지는 대부분 244 * 244 크기이며, 모델 별로 차이가 있다.
- 자동차나 고양이 등을 포함한 1000개의 클래스, 총 1400만개의 이미지로 구성된 ImageNet 데이터 세트로 사전 훈련되었다.

#### ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
- 2017년까지 대회가 주최되었으며, 이후에도 좋은 모델들이 등장했고, 앞으로도 계속 등장할 것이다.

---

### 📍<a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/CH10.-Pretrained-Model#vggnet-%EC%98%A5%EC%8A%A4%ED%8F%AC%EB%93%9C-%EB%8C%80%ED%95%99%EC%9D%98-%EC%97%B0%EA%B5%AC%ED%8C%80">VGGNet (옥스포드 대학의 연구팀)</a>
- 2014년 ILSVRC에서 GoogleNet이 1위, VGG는 2위를 차지했다.
- 네트워크 깊이에 따른 모델 성능의 영향에 대한 연구에 집중하여 만들어진 네트워크이다.
- 따라서 kernel 크기를 3X3으로 단일화했으며, Padding, Strides 값을 조정하여 단순한 네트워크로 구성되었다.

---

### 📍 <a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/CH10.-Pretrained-Model#inception-network-googlenet">Inception Network (GoogleNet)</a>
- 여러 사이즈의 커널들을 한꺼번에 결합하는 방식을 사용하며, 이를 묶어서 inception module이라고 한다.
- 여러 개의 inception module을 연속적으로 이어서 구성하고 여러 사이즈의 필터들이 서로 다른 공간 기반으로 feature들을 추출한다.
- 1X1 Convolution을 적용하면 입력 데이터의 특징을 함축적으로 표현하면서 파라미터 수를 줄이는 차원 축소 역할을 수행하게 된다.

#### 1X1 Convolution
- 행과 열의 사이즈를 줄이고 싶다면, Pooling을 사용하면 되고, 채널 수만 줄이고 싶다면 1X1 Convolution을 사용하면 된다.

---

### 📍<a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/CH10.-Pretrained-Model#resnet-%EB%A7%88%EC%9D%B4%ED%81%AC%EB%A1%9C%EC%86%8C%ED%94%84%ED%8A%B8">ResNet (마이크로소프트)</a>
- VGG 이후 더 깊은 Network에 대한 연구가 증가했지만, Network 깊이가 깊어질 수록 오히려 accuracy가 떨어지는 문제가 있었다.
- 이를 해결하고자 층을 만들되, Input 데이터와 결과가 동일하게 나올 수 있도록 하는 층을 연구하기 시작했다.  

---

### 📍<a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/CH10.-Pretrained-Model#transfer-learning-%EC%A0%84%EC%9D%B4-%ED%95%99%EC%8A%B5">Transfer Learning, 전이 학습</a>
- 이미지 분류 문제를 해결하는 데에 사용했던 모델을 다른 데이터세트 혹은 다른 문제에 적용시켜 해결하는 것을 의미한다.
- Pretrained Model의 Convolutional Base 구조(Conv2D + Pooling)를 그대로 두고 분류기(FC)를 붙여서 학습시킨다.

---

### 📍<a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/CH10.-Pretrained-Model#scaling-preprocessing">Scaling Preprocessing</a>
- 0 ~ 1, -1 ~ 1, z-score 변환 중에서 한 개를 선택하여 범위를 축소하는 작업을 의미한다.
- Pretrained Model은 주로 tf와 torch 프레임워크 방식을 사용한다.

---

### 📍<a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/CH10.-Pretrained-Model#fine-tuning-%EB%AF%B8%EC%84%B8-%EC%A1%B0%EC%A0%95">Fine Tuning, 미세 조정</a>
- ImageNet으로 학습된 Pretrained Model을 다른 목적 또는 다른 용도로 활용할 때 Feature Extractor의 Weight를 제어하기 위한 기법이다.
- 특정 Layer들을 Freeze시켜서 학습에서 제외시키고 Learning Rate를 점차 감소시켜 적용한다.
- 먼저 Classification Layers에만 학습을 시킨 뒤 전체에 학습을 시키는 순서로 진행하게 되며, 이를 위해 fit()을 최소 2번 사용한다.
- 층별로 Freeze 혹은 UnFreeze 결정을 위해 미세 조정을 진행 시, 학습률이 높으면 이전 지식을 잃을 수 있기 때문에 작은 학습률을 사용한다.
