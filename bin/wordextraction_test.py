from soynlp.utils import DoublespaceLineCorpus
from soynlp.word import WordExtractor
import math

'''
WordExtractor : Branching Entropy(BE), Accessor Variety(AV), Cohesion Score(CS) 세가지 단어 가능 점수를 계산하는 모듈.
BE : 단어 내부에서는 불확실성과 엔트로피가 줄어들고 단어 경계에서는 증가하는 현상을 모델링 한것.
     엔트로피가 높을수록 형태소일 확률이 크다.
     BE가 높은 문자열을 단어 취급해도 크게 나쁘지 않은 결과를 낼수 있다.
     

parameters:
def __init__(max_left_length=10, 
            max_right_length=6, 
            min_frequency=5, 
            verbose_points=100000, 
            min_cohesion_forward=0.1, 
            min_cohesion_backward=0.0, 
            max_droprate_cohesion=0.95, 
            max_droprate_leftside_frequency=0.95, 
            min_left_branching_entropy=0.0,
            min_right_branching_entropy=0.0, 
            min_left_accessor_variety=0, 
            min_right_accessor_variety=0, 
            remove_subwords=True)
            
.train()은 substrings의 빈도수를 카운팅 하는 것이며, .extract()는 init에 들어가는 파라미터값을 기준으로 단어를 선택.

점수 산출 방법 : cohesion_forward * right_branching_entropy 
            (1) 주어진 글자가 유기적으로 연결되어 함께 자주 나타나고, (2) 그 단어의 우측에 다양한 조사, 어미, 혹은 다른 단어가 등장하여 단어 우측의 
            branching entropy가 높다는 의미.

'''


def word_score(score):
    return (score.cohesion_forward * math.exp(score.right_branching_entropy))


corpus_path = '../data/test/2016-10-20.txt'
sents = DoublespaceLineCorpus(corpus_path, iter_sent=True)
word_extractor = WordExtractor(min_frequency=100,
                               min_cohesion_forward=0.05,
                               min_right_branching_entropy=0.0)
word_extractor.train(sents)
words = word_extractor.extract()
print(len(words))
# words는 {word:score} 형식의 dict 이다.
print(words['아이오아이'])

print('단어   (빈도수, cohesion, branching entropy)\n')
for word, score in sorted(words.items(), key=lambda x:word_score(x[1]), reverse=True)[:30]:
    print('%s     (%d, %.3f, %.3f)' % (
            word,
            score.leftside_frequency,
            score.cohesion_forward,
            score.right_branching_entropy
            )
         )

# Cohesion score, Branching Entropy, Accessor Variety 에 대하여 각각의 점수만 이용
cohesion_scores = word_extractor.all_cohesion_scores()
print(cohesion_scores['아이오아이']) # (cohesion_forward, cohesion_backward)

branching_entropy = word_extractor.all_branching_entropy()
print(branching_entropy['아이오아이']) # (left_branching_entropy, right_branching_entropy)

accessor_variety = word_extractor.all_accessor_variety()
print(accessor_variety['아이오아이']) # (left_accessor_variety, right_accessor_variety)