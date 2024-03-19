from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# 영어 문장에 대해 토큰화 수행 -> 품사 태깅을 수행
'''
Penn Treebank POG Tags
PRP : 인칭 대명사
VBP : 동사
RB : 부사
VBG : 현재부사
IN : 전치사
NNP : 고유 명사
NNS : 복수형 명사
CC : 접속사
DT : 관사
'''
text = "I am actively looking for Ph.D. students. and you are a Ph.D. student. "
tokenized_sentece = word_tokenize(text)

print('단어 토큰화 : ', tokenized_sentece)
print('품사 태깅 : ',pos_tag(tokenized_sentece))