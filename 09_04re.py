import re

# . : 한 개의 임의의 문자
r = re.compile("a.c")
r.search("kkk")
r.search("abc")

# ? : ?앞의 문자가 존재할 수도 있고 존재하지 않을 수도 있는 경우
r = re.compile("ab?c")
r.search("abbc")
r.search("abc")

# * : 바로 앞의 문자가 0개 이상일 경우
r = re.compile("ab*c")
r.search("a")
r.search("ac")
r.search("abc")
r.search("abbbbc")

# + : 앞의 문자가 최소 1개 이상
r = re.compile("ab+c")
r.search("ac")
r.search("abc")
r.search("abbbbc") 

# ^ : 시작되는 문자열 지정
r = re.compile("^ab")
r.serach("bbc")
r.search("zab")
r.search("abz")

# {숫자} : 해당 문자를 숫자만큼 반복한 것
r = re.compile("ab{2}c")
r.search("abbc")

# {숫자1, 숫자2} : 해당 문자를 숫자1 이상 숫자2 이하 반복
r = re.compile("ab{2,8}c")
r.search("abbc")
r.search("abbbbbbbbc")

# {숫자, } : 해당 문자를 숫자 이상 만큼 반복
r =re.compile("a{2,}bc")
r.search('aabc')
r.search("aaaaaaaabc")

# [] : 안에 문자들을 넣으면 그 문자들 중 한 개의 문자와 매치
r = re.compile("[abc]")
r.search("a")

r = re.compile("[a-z]")
r.search("aBC")

# [^문자] : ^기호 뒤에 붙은 문자들을 제외한 모든 문자를 매치하는 역할
r = re.compile("[^abc]")
r.search("d")
r.search("1")

# re.match()와 re.search() 차이
# search() : 정규 표현식 전체에 대해서 문자열이 매치하는지
# match() : 문자열의 첫 부분부터 정규 표현식과 매치하는지 확인
#           - 문자열 중간에 찾을 패턴이 있더라도 문자열의 시작에서 패턴이 일치하지 않으면 찾지 않음

r = re.compile("ab.")
r.match("kkkabc")
r.search("kkkabc")
r.match("abckkk")

# re.split() : 정규 표현식을 기준으로 문자열들을 분리하여 리스트로 리턴
#              - 토큰화에 유용하게 쓰일 수 있음
#              - 공백을 기준으로 문자열 분리를 수행하고 결과로서 리스트를 리턴

text = "사과 딸기 수박 메론 바나나"
re.split(" ", text)

# 줄바꿈 기준 분리
text = """사과
딸기
수박
메론
바나나"""

re.split("\n", text)

# '+'를 기준으로 분리
text = "사과+딸기+수박+메론+바나나"

re.split("\+", text)

# re.findall() : 정규 표현식과 매치되는 모든 문자열들을 리스트로 리턴
#               - 매치되는 문자열이 없다면 빈 리스트를 티런

text = """이름 : 김철수
전화번호 : 010 - 1234 - 1234
나이 : 30
성별 : 남"""

re.findall("\d+", text)

# re.sub() : 정규 표현식 패턴과 일치하는 문자열을 찾아 다른 문자열로 대체
text = "Regular expression : A regular expression, regex or regexp[1] (sometimes called a rational expression)[2][3] is, in theoretical computer science and formal language theory, a sequence of characters that define a search pattern."

preprocessed_text = re.sub('[^a-zA-Z]', ' ', text)
print(preprocessed_text)
