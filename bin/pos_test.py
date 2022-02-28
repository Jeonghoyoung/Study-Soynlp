# 사전 기반 품사 판별기 - 품사를 알고 있는 단어 사전으로 부터 주어진 문장의 단어열을 분석하는 것.
from pprint import pprint
from soynlp.postagger import Dictionary
from soynlp.postagger import LRTemplateMatcher
from soynlp.postagger import LREvaluator
from soynlp.postagger import SimpleTagger
from soynlp.postagger import UnknowLRPostprocessor

'''
pos_dict : 사용자가 입력한 사전

- Dictionary -
get_pos : 주어진 단어 word 에 대하여 등록되어 있는 모든 품사를 list 형식으로 return
word_is_tag : 주어진 단어 word 가 품사 tag 인지 확인하는 함수
max_length : 현재 사전에 등록된 단어 중에서 가장 긴 단어의 길이를 찾는 함수
add_words : 사전에 품사가 tag 인 단어들을 추가하는 함수. words 는 하나의 str 이어도 되며, 여러 개의 단어로 이뤄진 collection of str
            force=True 로 설정하면 현재 등록되지 않은 품사일지라도 사전에 추가 가능. (default = False)
remove_words : 품사 tag 에 해당하는 words 를 사전에서 제거(word 순서에 유의해서 제거), 만약 tag만 입력한 경우 해당 품사를 모두 제거한다.

- LRTemplateMatcher - 
LRTemplateMatcher : 단어열 후보 생성
.generate : LR 이라는 namedtuple (한국어의 어절을 명사/형용사/동사/부사/감탄사 의 L parts (left-subsection) 와 조사, 동사, 형용사로 이루어져있다.)
'''
pos_dict = {
    'Adverb': {'너무', '매우'},
    'Noun': {'너무너무너무', '아이오아이', '아이', '노래', '오', '이', '고양'},
    'Josa': {'는', '의', '이다', '입니다', '이', '이는', '를', '라', '라는'},
    'Verb': {'하는', '하다', '하고'},
    'Adjective': {'예쁜', '예쁘다'},
    'Exclamation': {'우와'}
}

dictionary = Dictionary(pos_dict)
pprint(dictionary.pos_dict)

print(dictionary.get_pos('이'))
print(dictionary.word_is_tag('아이오아이', 'Noun'))
print(dictionary.max_length)

dictionary.add_words('Noun', ['워너원', '양순이'])
pprint(dictionary.pos_dict)

dictionary.remove_words('Noun', {'워너원', '앙순이'} )
pprint(dictionary.pos_dict)

# Dictionary 를 이용하여 LRTemplateMatcher 생성
sent = '너무너무너무는아이오아이의노래입니다!!'
generator = LRTemplateMatcher(dictionary)
pprint(generator.generate(sent))


evaluator = LREvaluator()
postprocessor = UnknowLRPostprocessor()

tagger = SimpleTagger(generator, evaluator, postprocessor)
t = tagger.tag(sent)
print(t)
# postprocessor 가 입력되지 않으면, 사전 매칭이 되지 않은 단어들은 출력되지 않는다.
# debug mode 로 tag() 를 실행할 경우 문장 내의 단어열 뿐 아니라 디버깅용 LR 후보 리스트들이 출력할 수 있다.
# tags, debugs = tagger.tag(sent, debug=True)

# 만약 특정 품사의 단어에 대하여 점수의 가중치를 더하고 싶은 경우 preference를 이용할 수 있다.
# dict[tag][word] = score 형식의 dict-dict 인 preference 를 Evaluator 에 넣어주면 되며, debug mode로 확인할 수 있다.
preference = {
    'Noun': {'아이오아이':10.0, '너무너무너무':5}
}

evaluator = LREvaluator(preference=preference)
tagger = SimpleTagger(generator, evaluator, postprocessor)
tags, debugs = tagger.tag(sent, debug=True)

pprint(debugs)