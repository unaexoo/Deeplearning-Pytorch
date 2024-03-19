'''
불용어(stopword) 
- 자주 등장하지만 실제 의미 분석을 하는데 기여하는 바가 없는 경우
'''

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt

# nltk에서 불용어 확인
stop_words_list = stopwords.words('english')
print('불용어 개수 :', len(stop_words_list))
print('불용어 10개 출력 :', stop_words_list[:10])

# nltk를 통한 불용어 제거
example = "Family is not an important thing. It's everything."
stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example)

result = []
for word in word_tokens :
    if word not in stop_words:
        result.append(word)
        
print('불용어 제거 전 : ', word_tokens)
print('불용어 제거 후 : ',result)

# 한국어에서 불용어 제거하기
'''
토큰화 후에 조사, 접속사 등을 제거하는 방법 
-> 제거하다 보면 조사나 접속사 ,명사, 형용사와 같은 단어들 중 제거하는 경우도 발생
임의로 불용어 선정해서 테스트
'''
okt = Okt()
example = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
stop_words = "를 아무렇게나 구 우려 고 안 돼 같은 게 구울 때 는"

stop_words = set(stop_words.split(' '))
word_tokens = okt.morphs(example)

result = [word for word in word_tokens if not word in stop_words]

print('불용어 제거 전 : ', word_tokens)
print('불용어 제거 후 : ',result)