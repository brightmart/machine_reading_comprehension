# machine_reading_comprehension
machine reading comprehension with deep learning

Implementation of <a href='https://arxiv.org/pdf/1707.09098.pdf'>MEMEN: Multi-layer Embedding with Memory Networks for Machine Comprehension</a>


MEMEN: Multi-layer Embedding with Memory Networks for Machine Comprehension 

process: 

1.encoder with bi-LSTM to word/char/ner/pos embeddings; 
2. matching of query and context(3 different attentions/alginment matrix),concat mathings, gate, bi-LSTM 
3. use pointer network(PNet) to get start point and end point. 

PNet initialize with query-awere representation==>

do attention for query and context==>

update query==> 

repeat attention process. 

Notice: you can pretrain word/character/NER/POS embedding, and load from outside.



