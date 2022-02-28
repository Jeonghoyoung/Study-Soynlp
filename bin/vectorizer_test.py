from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from soynlp.vectorizer import BaseVectorizer

'''
Vectorizer : 토크나이저를 학습하거나, 학습된 토크나이저를 이용하여 문서를 sparse matrix(희소행렬 : 행렬의 값 대부분 0인 행렬)로 만든다.
    -> corpus를 term frequency matrix로 변환.
    -> scipy.sparse.csr.csr_matrix  사용.

BaseVectorizer parameters :
    tokenizer : tokernizer (default : lambda x:x.split() - 영어 같은 언어들이 띄어쓰기 기준으로 나눠도 가능하기 때문.)
    min_tf, max_tf : TF(Term-Frequency 단어빈도) 즉 단어 빈도수의 최소값과 최대값을 의미.
    min_df, max_df : 특정 단어를 포함한 문서의 비율 (min_df가 0.02인 경우 100개의 문서에서 1번만 등장한 단어는 제외)
                     default : min_df = 0, max_df = 1.0 으로 모든 데이터 사용.
    stopwords : 불용어 처리 
    lowercase : True인 경우 영어의 경우 모든 글자를 소문자로 변환.
    verbose : True 인 경우 현재의 vectorizing 상황을 print 한다.

    vocabulary_ :  {str:int} 형식으로 각 단어가 어떤 idx 에 해당하는지를 나타내는 dict 가 저장
    
    주로 min_tf, max_tf, min_df, max_df를 조절한다.
'''

corpus_path = '../data/test/2016-10-20.txt'
sents = DoublespaceLineCorpus(corpus_path, iter_sent=True)
word_extractor = WordExtractor()
word_extractor.train(sents)

# (leftside cohesion, rightside cohesion)
cohesion_scores = word_extractor.all_cohesion_scores()

# use only leftside cohesion
scores = {word:score[0] for word, score in cohesion_scores.items()}
tokenizer = LTokenizer(scores=scores)

# Vectorizing
vectorizer = BaseVectorizer(
    tokenizer=tokenizer,
    min_tf=0,
    max_tf=10000,
    min_df=0,
    max_df=1.0,
    stopwords=None,
    lowercase=True,
    verbose=True
)

sents.iter_sent = False
# x = vectorizer.fit_transform(sents)

# 대량의 문서에 대한 sparse matrix를 메모리에 올리지 않고 파일에 저장.
vectorizer = BaseVectorizer(min_tf=1, tokenizer=tokenizer)
sents.iter_sent = False

matrix_path = '../data/test.txt'
vectorizer.fit_to_file(sents, matrix_path)

# 하나의 문장을 sparse matrix 가 아닌 list of int 로 출력이 가능합니다. 이 때 vectorizer.vocabulary_ 에 학습되지 않은 단어는 encoding 이 되지 않습니다.
print(vectorizer.encode_a_doc_to_bow('오늘 뉴스는 이것이 전부다'))

# list of int 는 list of str 로 decoding 이 가능합니다.
print(vectorizer.decode_from_bow({3: 1, 258: 1, 428: 1, 1814: 1}))
# dict 형식의 bag of words 로도 encoding 이 가능합니다.
print(vectorizer.encode_a_doc_to_list('오늘 뉴스는 이것이 전부다'))
# dict 형식의 bag of words 는 decoding 이 가능합니다.
print(vectorizer.decode_from_list([258, 4, 428, 3, 333]))
