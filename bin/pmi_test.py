from pprint import pprint
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from soynlp.vectorizer import sent_to_word_contexts_matrix
from soynlp.word import pmi as pmi_func
from soynlp.utils import most_similar

'''
PMI : Point Mutual Information 는 (word, contexts) 이나 (input, outputs) 와의 상관성을 측정하는 방법.
      서로 상관이 없는 경우 0이며, 그 값이 클수록 positive correlated 이다. 음의 값을 갖는 경우 0으로 치환된다.
      주로 연관 분석에 사용된다.
sent_to_word_contexts_matrix : (word, context words) matrix 를 만들 수 있다.

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

# 아래의 x는 scipy.sparse.csr_matrix 이며 (n_vocabs, n_vocabs) 크기이고, idx2vocab는 x의 각 row, column 에 해당 하는 단어가 포함된 list of str 이다.
# windows 단어를 context로 인식하며, min_tf 이상의 빈도수로 등장한 단어에 대해서만 계산 한다.
# dynamic_weight : context 길이에 반비례하여 weighting 한다.
# if window = 3 , 1,2,3 칸 떨어진 단어의 co-occurrence 는 1, 2/3, 1/3으로 계산된다.
x, idx2vocab = sent_to_word_contexts_matrix(
    sents,
    windows = 3,
    min_tf = 10,
    tokenizer = tokenizer, # (default) lambda x:x.split(),
    dynamic_weight = False,
    verbose = True)

# print(x[0])
# print(idx2vocab)

# x의 (rows, columns) 에 대한 pmi 계산, row = x, column = y
pmi, px, py = pmi_func(
    x,
    min_pmi = 0,
    alpha = 0.0,
    beta = 0.75
)

vocab2idx = {vocab:idx for idx, vocab in enumerate(idx2vocab)}
query = vocab2idx['아이오아이']
# 단어 '아이오아이' 와 pmi 가 높은 ( 아이오아이와 자주 등장한) 단어 를 찾습니다.

submatrix = pmi[query,:].tocsr() # get the row of query
contexts = submatrix.nonzero()[1] # nonzero() return (rows, columns)
pmi_i = submatrix.data

most_relateds = [(idx, pmi_ij) for idx, pmi_ij in zip(contexts, pmi_i)]
most_relateds = sorted(most_relateds, key=lambda x:-x[1])[:10]
most_relateds = [(idx2vocab[idx], pmi_ij) for idx, pmi_ij in most_relateds]


pprint(most_relateds)

most_similar('아이오아이', pmi, vocab2idx, idx2vocab)