Machine Reading Comprehension with Deep Learning

Implementation of <a href='https://arxiv.org/pdf/1707.09098.pdf'>MEMEN: Multi-layer Embedding with Memory Networks for Machine Comprehension</a>, which list the 2rd place on Aug 2017 for standford machine reading comprehension competition <a href='https://rajpurkar.github.io/SQuAD-explorer/'>SQuAD</a>


MEMEN: Multi-layer Embedding with Memory Networks for Machine Comprehension 

Process: 
--------------------------------------------------------------------------------------------------------------
1.encoder with bi-LSTM to word/char/ner/pos embeddings; 

2.matching of query and context(3 different attentions/alginment matrix),concat mathings, gate, bi-LSTM; 

3.use pointer network(PNet) to get start point and end point; 

PNet initialize with query-awere representation==>

do attention for query and context==>

update query==> 

repeat attention process. 

Notice: you can pretrain word/character/NER/POS embedding, and load from outside.


Toy task: 
--------------------------------------------------------------------------------------------------------------
find three successive elements that sum up most(least) in an array

input:(paragraphs,query). for example,paragraphs=[2, 6, 9, 5, 4, 0, 8, 3, 7, 1];query=0,stand for find values sum up most;

output:(start_point,end_point). for example,start_point: 1,end_point: 3,means the following three elements sum up most in the array:[6, 9, 5]

for more detail,check train() and predict() of memen_model.py

Reference:
--------------------------------------------------------------------------------------------------------------
MEMEN: Multi-layer Embedding with Memory Networks for Machine Comprehension,
Boyuan Pan, Hao Li, Zhou Zhao, Bin Cao, Deng Cai, Xiaofei He



