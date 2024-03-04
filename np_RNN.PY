import numpy as np

'''
NLP에서는
timestpes -> 시점의 수는 보통 문자의 길이
input_size -> 입력의 차원은 보통 단어 베거의 차원
hidden_size -> 은닉 상태의 크기. 메모리 셀의 용량
'''
timesteps = 10 # 시점의 수.
input_size = 4 # 입력의 차원
hidden_size = 8 #은닉 상태의 크기

# 입력에 해당하는 2d 텐서
inputs = np.random.random((timesteps,  input_size))

# 초기 은닉 상태는 0로 초기화
hidden_state_t = np.zeros((hidden_size,))

print(hidden_state_t)

# 입력에 대한 가중치(은닉 상태의 크기 x 입력의 차원)
Wx = np.random.random((hidden_size,input_size))

# 은닉 상태에 대한 가중치(은닉 상태의 크기 x 은닉 상태의 크기)
Wh = np.random.random((hidden_size,hidden_size))

# 편향(은닉 상태의 크기)
b = np.random.random((hidden_size,))

print(np.shape(Wx))
print(np.shape(Wh))
print(np.shape(b))

'''
RNN 층
'''
total_hidden_states = []

# 메모리 셀 동작
for input_t in inputs : 
    # Wx * Wt + Wh * Ht - 1 + b
    output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh,hidden_state_t) + b)
    
    # 각 시점의 은닉 상태의 값을 계속해서 축적
    total_hidden_states.append(list(output_t))
    
    # 각 시점 t별 메모리 셀의 크기는 (timestep, output_dim)
    print(np.shape(total_hidden_states))
    hidden_state_t = output_t
    
total_hidden_states = np.stack(total_hidden_states, axis = 0)

# (timesteps, output_dim)의 크기
print(total_hidden_states)