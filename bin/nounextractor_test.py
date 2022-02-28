import pandas as pd
import util.file_util as ft
from soynlp.utils import DoublespaceLineCorpus
from soynlp.noun import LRNounExtractor_v2

corpus_path = '../data/test/2016-10-20.txt'
sents = DoublespaceLineCorpus(corpus_path, iter_sent=True)
ft.write_list_file(sents, '../data/dslc.txt')
print('end')
# iter_sent = True : for loop를 돌때와 corpus의 길이를 계산할 때, 문장 단위로 계산.

# .train_extract : 학습과 명사 점수 계산을 동시에 진행함, .train() 과 .extract()로 따로 진행할수도 있다.
noun_extractor = LRNounExtractor_v2()
nouns = noun_extractor.train_extract(sents)
# nouns 는 {str: NounScore} 형식의 dict 어절에서 '뉴스'가 실제로 명사로 이용된 경우만 카운팅 됩니다. '뉴스방송'과 같은 복합명사의 빈도수는 '뉴스'에 포함되지 않음.

# compounds_components는 복합명사의 components가 저장되어 있으며, tuple로 구성되어 있다. ex: ('호매실지구', ('호매실', '지구'))
# 직전의 훈련된 데이터와 모델을 바탕으로 데이터 산출.
print(list(noun_extractor.compounds_components.items())[:10])
print(nouns['잠수함발사탄도미사일'])

# decompose_compound 는 입력된 str 이 복합 명사일 경우. 이를 단일 명사의 tuple로 분해한다.
print(noun_extractor.decompose_compound('잠수함발사탄도미사일'))

# LRNounExtractor_v2 는 soynlp.utils 의 LRGraph 를 이용한다. 데이터의 L-R 구조를 살펴볼 수 있다.
# topk = -1 로 설정하면 모든 L 또는 R set 이 출력됩니다.
print(noun_extractor.lrgraph.get_r('아이오아이'))
print(noun_extractor.lrgraph.get_l('었다고', topk=-1))