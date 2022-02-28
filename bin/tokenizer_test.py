from soynlp.utils import DoublespaceLineCorpus
from soynlp.tokenizer import LTokenizer
from soynlp.word import WordExtractor
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import LTokenizer, RegexTokenizer, MaxScoreTokenizer
from gensim.models import Word2Vec

# 모든 토크나이징 모듈의 input = dict {word:score} 구조이다.
'''
L parts 에는 명사/동사/형용사/부사가 위치할 수 있다.
LTokenizer 에는 L parts 의 단어 점수를 입력한다.
데이터는 띄어쓰기가 잘 되어 있는 국문 텍스트여야 한다.

LTokenizer의 파라미터중 tolerance는 한 어절에서 subword들의 점수의 차이가 그 어절의 점수 최대값과 tolerance 이하로 난다면, 길이가 가장 긴 어절을 선택한다. 
만약 scores = {'데이':0.5, '데이터':0.5, '데이터마이닝':0.5, '공부':0.5, '공부중':0.45} 이고 sent = '데이터마이닝을 공부중이다.' 라면,
tolerance=0.0 일때 tokenize = ['데이터마이닝', '을', '공부', '중이다']
tolerance=0.1 일땐 tokenize = ['데이터마이닝', '을', '공부중', '이다']

'''

corpus_path = '../data/test/2016-10-20.txt'
corpus = DoublespaceLineCorpus(corpus_path, iter_sent=True)
word_extractor = WordExtractor()
word_extractor.train(corpus)
word_scores = word_extractor.extract()

# 데이터 토크나이징, Use Cohesion Score and LTokenizer
cohesion_scores = {word:score.cohesion_forward for word, score in word_scores.items()}
l_tokenizer = LTokenizer(scores=cohesion_scores)
l_tok_corpus = [l_tokenizer.tokenize(sent) for sent in corpus]

# 데이터 토크나이징 , Use Cohesion Score and LRNounExtractor_v2
# cs 점수 외의 BE, AV 점수를 사용해서 산출 가능하다.
noun_extractor = LRNounExtractor_v2()
nouns = noun_extractor.train_extract(corpus)

noun_scores = {noun:score.score for noun, score in nouns.items()}
combined_scores = {noun:score + cohesion_scores.get(noun,0) for noun, score in noun_scores.items()}

combined_scores.update({subword:cohesion for subword, cohesion in cohesion_scores.items() if not (subword in combined_scores)})
l_tokenizer2 = LTokenizer(scores=combined_scores)
l_tok_corpus2 = [l_tokenizer2.tokenize(sent) for sent in corpus]

# Word2vec 모델 학습
# word2vec = Word2Vec(word2vec_corpus)
print(l_tok_corpus[:10])
print(l_tok_corpus2[:10])

'''
MaxScoreTokenizer : LTokenizer 의 경우 모든 데이터의 띄어쓰기가 제대로 지켜진 데이터여야만 한다는 조건이 붙는 반면,
                    MaxScoreTokenizer의 경우 띄어쓰기가 제대로 지켜지지 않은 데이터여도 토크닝이 가능하다.
                    만약 모든 데이터의 띄어쓰기가 잘 이루어진 경우에는 사용할 필요 없다.
'''
max_tokenizer = MaxScoreTokenizer(scores=combined_scores)
max_tok_corpus = [max_tokenizer.tokenize(sent) for sent in corpus]
print(max_tok_corpus[:10])

'''
RegexTokenizer : 한글과 숫자, 영어(라틴), 기호가 바뀌는 지점에서 토크나이징
                 띄어쓰기가 제대로 되어있지 않은 경우엔 사용이 불가능하다.
'''

tokenizer = RegexTokenizer()

sents = [
    '이렇게연속된문장은잘리지않습니다만',
    '숫자123이영어abc에섞여있으면ㅋㅋ잘리겠죠',
    '띄어쓰기가 포함되어있으면 이정보는10점!꼭띄워야죠'
]

for sent in sents:
    print('   %s\n->%s\n' % (sent, tokenizer.tokenize(sent)))