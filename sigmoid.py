import numpy as np
import matplotlib.pyplot as plt 

# 시그모이드 함수 정의
def sigmoid(x) :
    return 1/(1+np.exp(-x))

# w-1, b=0
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x,y,'g')
plt.plot([0,0],[1.0,0.0],':')
plt.title('Sigmoid Function')
plt.show()

# W 값의 변화에 따른 경사도의 변화
x = np.arange(-5.0,5.0,0.1)
y1 = sigmoid(0.5 * x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)

plt.plot(x, y1, 'r', linestyle='--') # W의 값이 0.5일때
plt.plot(x, y2, 'g') # W의 값이 1일때
plt.plot(x, y3, 'b', linestyle='--') # W의 값이 2일때
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()

# b 값의 변화에 따른 좌, 우 이동
x = np.arange(-5.0,5.0,0.1)
y1 = sigmoid(x+0.5)
y2 = sigmoid(x+1)
y3 = sigmoid(x+1.5)

plt.plot(x, y1, 'r', linestyle='--') # x + 0.5
plt.plot(x, y2, 'g') # x + 1
plt.plot(x, y3, 'b', linestyle='--') # x + 1.5
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()

# 시그모이드 함수의 출력값은 0과 1 사이의 값을 가짐 이 특성을 이용하여 분류 작업에 사용할 수 있음

