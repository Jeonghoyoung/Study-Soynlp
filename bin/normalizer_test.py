import soynlp
from soynlp.normalizer import *

'''
대화 데이터 또는 댓글 데이터에 등장하는 반복되는 이모티콘 정리 및 한글 혹은 텍스트만 남기기 위한 함수 제공 모듈.

emoticon_normalize : 반복되는 글자들 사이에 완전한글이 있는경우 글자를 줄이고 , 자음/모음 이모티콘을 분해한다. (num_repeats : 반복되는 글자의 축약 횟수)
repeat_normalize : 반복되는 글자 축약 (n_repeats : 축약 횟수 )
only_hangle : 한글과 숫자만 남기며, 이외의 글자는 공백 처리되고 연속된 공백은 하나만 남겨진다.
only_text : 알파벳, 한글, 숫자, 문법 기호 (!?.,"') 만 남기고 나머지는 공백처리.
'''

print(emoticon_normalize('ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ쿠ㅜㅜㅜㅜㅜㅜ', num_repeats=3))

print(repeat_normalize('우오아아아아아', num_repeats=3))

print(only_hangle('가나다랃ㄷㄷㄷㄷㄷaaaaaa123'))

print(only_text('가다나!!!abs123@@'))
