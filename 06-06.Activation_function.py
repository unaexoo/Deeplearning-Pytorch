import numpy as np 
import matplotlib.pyplot as plt

# 활성화 함수의 특징 -> 비선형 함수
# 선형 함수로는 은닉층을 여러 번 추가하더라고 1회 추가한 것과 차이를 줄 수 없음

# 1. 시그모이드와 기울기 손실
def sigmoid(x):
    return 1/(1+np.exp(-x))
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()

# 역전파 과정에서 0에 가까운 아주 작은 기울기가 곱해지게 되면, 앞단에는 기울기가 잘 전달되지 않게 됨 -> 기울기 손실 문제(Vanishing Gradient)
# 0에 가까운 기울기가 계속 곱해지면 앞단에서는 거의 기울기를 전파받을 수 없게 됨
# 다시 말해 매개변수 W가 업데이트 되지 않아 학습이 되지를 않음

# 2. 하이퍼볼릭탄젠트 함수(Hyperbolic tangent function)
# 입력값을 -1과 1 사이의 값으로 변환
x = np.arange(-5.0, 5.0, 0.1)
y = np.tanh(x)

plt.plot(x, y)
plt.plot([0,0],[1.0,-1.0], ':')
plt.axhline(y=0, color='orange', linestyle='--')
plt.title('Tanh Function')
plt.show()

# -1과 1에 가까운 출력 -> 손실 문제 발생 
# 다만 0을 중심으로 하고 있어서 반환값의 변화폭이 더 큼 -> 시그모이드 함수보다 기울기 소실 증상이 적은 편

'''
렐루 함수(ReLu)
- 인공신경망에서 가장 인기 있는 함수
- f(x) = max(0,x)
- 음식를 입력하면 0을 출력하고 양수를 입력하면 입력값이 그대로 반환
- 특정 양수값에 수렴하지 않으므로 깊은 신경망에서 잘 작동
- 시그모이드 함수와 하이퍼볼릭 탄젠트 함수와 같이 연산이 필요한 것이 아니라 단순 임계값이므로 연산 속도도 빠름
- 문제점 : 입력값이 음수면 기울도 0 -> 죽은 렐루(dying ReLu)
'''

def relu(x) :
    return np.maximum(0,x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

plt.plot(x, y)
plt.plot([0,0],[5.0,0.0], ':')
plt.title('Relu Function')
plt.show()

'''
리키 렐루(Leaky ReLu)
- 죽은 렐루를 보완하기 위해 등장
- 입력값이 임수일 경우에 0이 아닌 0.001과 같은 매우 작은 수 반환
- 수식 : f(x) = max(ax, x)
- a는 하이퍼파라미터로 Leaky(새는) 정도 결정 일반적으로 0.01의 값을 가짐
- 새는 정도 : 입력값의 음수일 때의 기울기를 비유
- 입력값이 음수라도 기울기가 0이 되지 않으면 ReLU는 죽지 않음
'''

a = 0.1
def leaky_relu(x) :
    return np.maximum(a*x,x)

x = np.arange(-5.0, 5.0, 0.1)
y = leaky_relu(x)

plt.plot(x, y)
plt.plot([0,0],[5.0,0.0], ':')
plt.title('Leaky ReLU Function')
plt.show()

'''
소프트맥스 함수(Softmax function)
- 분류 문제를 로지스틱 회귀와 소프트맥스 회귀를 출력층에 적용하여 사용
- 소프트맥스 함수는 시그모이드 함수처럼 출력층의 뉴런에서 주로 사용
- 세 가지 이상의 (상호 배타적인) 선택지 중 하나를 고르는 다중 클래스 분류(MultiClass Classification) 문제에 주로 사용
'''
x = np.arange(-5.0, 5.0, 0.1) 
y = np.exp(x) / np.sum(np.exp(x))

plt.plot(x, y)
plt.title('Softmax Function')
plt.show()