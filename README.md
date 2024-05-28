### Deep Learning, 딥 러닝

- 인공 신경망(Artificial Neural Network)의 층을 연속적으로 깊게 쌓아올려 데이터를 학습하는 방식을 의미한다.
- 인간이 학습하고 기억하는 매커니즘을 모방한 기계학습이다.
- 인간은 학습 시, 뇌에 있는 뉴런이 자극을 받아들여서 일정 자극 이상이 되면, 화학물질을 통해 다른 뉴런과 연결되며 해당 부분이 발달한다.
- 자극이 약하거나 기준치를 넘지 못하면, 뉴런은 연결되지 않는다.
- 입력한 데이터가 활성 함수에서 임계점을 넘게 되면 출력된다.
- 초기 인공 신경망(Perceptron)에서 깊게 층을 쌓아 학습하는 딥 러닝으로 발전한다.
- 딥 러닝은 Input nodes layer, Hidden nodes layer, Output nodes layer, 이렇게 3가지 층이 존재한다.

<img src="https://github.com/SOYOUNGdev/study_deep_learning/assets/115638411/5658102f-39c5-4a1b-ae4a-99c35d64b174" style="margjin-left: 0">

### <a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/SLP-%E2%80%90-Single-Layer-Perceptron">SLP (Single Layer Perceptron), 단층 퍼셉트론, 단일 퍼셉트론 </a>

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

### <a href="https://github.com/SOYOUNGdev/study_deep_learning/wiki/MLP-%E2%80%90-Multi-Layer-Perceptron">Multi Layer Perceptron, 다층 퍼셉트론, 다중 퍼셉트론</a>
- 보다 복잡한 문제의 해결을 위해서 입력층과 출력층 사이에 은닉층이 포함되어 있다.
- 퍼셉트론을 여러층 쌓은 인공 신경망으로서, 각 층에서는 활성함수를 통해 입력을 처리한다.
- 층이 깊어질 수록 정확한 분류가 가능해지지만, 너무 깊어지면 Overfitting이 발생한다.

#### ANN (Artificial Neural Network), 인공 신경망
- 은닉층이 1개일 경우 이를 인공 신경망이라고 한다.

### DNN (Deep Neural Network), 심층 신경망
- 은닉층이 2개 이상일 경우 이를 심층 신경망이라고 한다.

#### Back-propagation, 역전파
- 심층 신경망에서 최종 출력(예측)을 하기 위한 식이 생기지만 식이 너무 복잡해지기 때문에 편미분을 진행하기에 한계가 있다.
- 즉, 편미분을 통해 가중치 값을 구하고, 경사 하강법을 통해 가중치 값을 업데이트 하며, 손실 함수의 최소값을 찾아야 하는데,  
  순방향으로는 복잡한 미분식을 계산할 수가 없다. 따라서 미분의 연쇄 법칙(Chain Rule)을 사용하여 역방향으로 편미분을 진행한다.

#### 합성 함수의 미분
<img src="https://github.com/SOYOUNGdev/study_deep_learning/assets/115638411/1ffe8f54-eb63-4f93-95c7-ccdbe3db1360" width="150" style="margin-left: 0">  

---
<img src="https://github.com/SOYOUNGdev/study_deep_learning/assets/115638411/5b32989b-b818-4cc4-9288-edffad81ef60" width="550" style="margin-left: 0">  

<img src="https://github.com/SOYOUNGdev/study_deep_learning/assets/115638411/9a1ad3d4-f687-47fe-b228-d27942356f13" width="800" style="margin-left: 0">  
<img src="https://github.com/SOYOUNGdev/study_deep_learning/assets/115638411/c586deb3-d599-4097-9899-1e90e199ed46" width="800" style="margin-left: 0">  
<img src="https://github.com/SOYOUNGdev/study_deep_learning/assets/115638411/0117bd23-8427-49f9-ade0-9c5901ca34d4" width="500" style="margin-left: 0">  








