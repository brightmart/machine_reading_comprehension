# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib as tf_contrib
import random
import numpy as np
#MEMEN: Multi-layer Embedding with Memory Networks for Machine Comprehension
#process:
# 1.encoder with bi-LSTM to word/char/ner/pos embeddings;
# 2. matching of query and context(3 different attentions/alginment matrix),concat mathings, gate, bi-LSTM
# 3. use pointer network(PNet) to get start point and end point. PNet initialize with query-awere representation==>do attention for query and context==>update query==>repeat attention process.
# Notice: you can pretrain word/character/NER/POS embedding, and load from outside.
#https://arxiv.org/pdf/1707.09098.pdf

class MemenNetwork():
    def __init__(self,l2_lambda,article_len,sentence_len,vocab_sz,embed_sz,ner_vocab_sz,pos_vocab_sz,
                 lr=0.0001,matching_hop_times=3,is_training=True,output_hop_times=3,clip_gradients=5.0): #matching_hop_times=3,output_hop_times=3 TODO
        self.l2_lambda=l2_lambda
        self.article_len=article_len
        self.sentence_len=sentence_len
        self.vocab_sz=vocab_sz
        self.embed_sz=embed_sz
        self.ner_vocab_sz=ner_vocab_sz
        self.pos_vocab_sz=pos_vocab_sz
        self.matching_hop_times=matching_hop_times
        self.output_hop_times=output_hop_times
        self.is_training=is_training
        print("self.is_training:",self.is_training)

        self.hidden_size=embed_sz
        self.lr=lr
        self.clip_gradients=clip_gradients

        self.query = tf.placeholder(tf.int32, [None, self.sentence_len], name="query")
        #self.query_ner=tf.placeholder(tf.int32, [None, self.sentence_len], name="query_ner")   #NER tag:PERSON,LOCATION,ORGANIZATION TODO
        #self.query_pos = tf.placeholder(tf.int32, [None, self.sentence_len], name="query_pos") #POS tag:part of speech:noun,verb     TODO

        self.context = tf.placeholder(tf.int32, [None, self.article_len], name="context")
        #self.context_ner=tf.placeholder(tf.int32, [None, self.article_len], name="context_ner")   #NER tag:PERSON,LOCATION,ORGANIZATION TODO
        #self.context_pos=tf.placeholder(tf.int32,[None,self.self.article_len],name="context_pos") #POS tag:part of speech:noun,verb     TODO

        self.label_start=tf.placeholder(tf.int32,[None],name="start_point") #start point
        self.label_end = tf.placeholder(tf.int32, [None], name="end_point")  #end point

        self.instantiate_weights()

        self.logits_start = self.inference() #(?, article_len)
        self.loss_val = self.loss()

        self.predictions_start = tf.argmax(self.logits_start,axis=1, name="predictions_start")  # shape:(?,)
        #self.predictions_end = tf.argmax(self.logits_end, axis=1,name="predictions_end")  # shape:(?,)

        correct_prediction_start = tf.equal(tf.cast(self.predictions_start, tf.int32), self.label_start)
        self.accuracy_start = tf.reduce_mean(tf.cast(correct_prediction_start, tf.float32), name="accuracy_start")  # shape=()
        #correct_prediction_end = tf.equal(tf.cast(self.predictions_end, tf.int32), self.label_end)
        #self.accuracy_end = tf.reduce_mean(tf.cast(correct_prediction_end, tf.float32), name="accuracy_end")  # shape=()
        if self.is_training:
            self.train_op = self.train()

    def instantiate_weights(self):
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.embedding_word = tf.get_variable("embedding_word", [self.vocab_sz, self.embed_sz]) #word embedding matrix      #TODO YOU CAN PRETRAIN THIS, AND LOAD FROM OUTSIDE
        self.embedding_char = tf.get_variable("embedding_char", [self.vocab_sz, self.embed_sz]) #character embedding matrix #TODO YOU CAN PRETRAIN THIS, AND LOAD FROM OUTSIDE
        self.embedding_ner = tf.get_variable("embedding_ner", [self.ner_vocab_sz, self.embed_sz]) #embedding matrix for NER #TODO YOU CAN PRETRAIN THIS, AND LOAD FROM OUTSIDE
        self.embedding_pos = tf.get_variable("embedding_pos", [self.pos_vocab_sz, self.embed_sz]) #embedding matrix for POS #TODO YOU CAN PRETRAIN THIS, AND LOAD FROM OUTSIDE
        self.embedding_query = tf.get_variable("embedding_query", [self.vocab_sz, self.embed_sz]) #word embedding matrix      #TODO YOU CAN PRETRAIN THIS, AND LOAD FROM OUTSIDE
        self.weight1=tf.get_variable("weight1", [self.hidden_size*6,1])
        self.weight_s = tf.get_variable("weight_s", [self.hidden_size * 2, 1])

    def inference(self):
        # 1.embedding
        query_embeddings_word = tf.nn.embedding_lookup(self.embedding_query,self.query)  # [none,self.sentence_len,self.embed_sz] TODO should use self.embedding_word for real task. toy task should use embedding_query
        context_embeddings_word = tf.nn.embedding_lookup(self.embedding_word,self.context)  # [none,self.article_len,self.embed_sz]
        query = tf.reduce_sum(query_embeddings_word, axis=1)  # [none,self.embed_sz]
        query_expand=tf.expand_dims(query,axis=2) #[none,self.embed_sz,1]

        # 2.attention. result should be same as [none,self.article_len]
        c_w=tf.layers.dense(context_embeddings_word,self.hidden_size) #[none,self.article_len,self.embed_sz]

        c_w_q=tf.matmul(c_w,query_expand) #[none,self.article_len,1]
        c_w_q=tf.squeeze(c_w_q,axis=2) #[none,self.article_len]

        p=tf.nn.softmax(c_w_q,dim=1) #[none,self.article_len]
        # 3. predict
        return p

    def loss(self):
        #label_start:[none]; self.logits_start:[none, article_len]
        loss_start=tf.losses.sparse_softmax_cross_entropy(self.label_start, self.logits_start) #[none,]
        loss_start=tf.reduce_mean(loss_start) #scalar
        #loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_end, logits=self.logits_end) #[none,]
        #loss_end = tf.reduce_mean(loss_end)  # scalar
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('bias' not in v.name)]) * self.l2_lambda #scalar
        loss=loss_start+l2_losses #+loss_end
        return loss #scalar


    def train(self):
        #learning_rate = tf.train.exponential_decay(self.lr, self.global_step, self.decay_steps, self.decay_rate,staircase=True)
        train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=self.lr, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

#toy task: find three successive elements that sum up most(least) in an array
#input:(paragraphs,query). for example,paragraphs=[2, 6, 9, 5, 4, 0, 8, 3, 7, 1];query=0,stand for find values sum up most;
#output:(start_point,end_point). for example,start_point: 1,end_point: 3,means the following three elements sum up most in the array:[6, 9, 5]

#train and predict.
article_len=10
sentence_len=1

l2_lambda=0 #0.001
vocab_sz=10
embed_sz=8 #128
ner_vocab_sz=10
pos_vocab_sz=10
pos_vocab_sz=10
lr=0.001

def train():
    ckpt_dir='checkpoint/'
    #.create model
    model=MemenNetwork(l2_lambda,article_len,sentence_len,vocab_sz,embed_sz,ner_vocab_sz,pos_vocab_sz,lr=lr)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(5000):
            #generate data
            query=0#random.choice([0,1])
            paragraph, query, start_point, end_point = generate_data(n=article_len,query=query)
            #feed dict
            fetch_dict=[model.loss_val, model.accuracy_start,model.predictions_start,model.train_op]
            feed_dict={model.context:paragraph,model.query:query,model.label_start:start_point,model.label_end:end_point}
            #run with session
            loss,accuracy_start,predictions_start,_=sess.run(fetch_dict,feed_dict=feed_dict)
            #print result and status
            print(i,"paragraph:",paragraph,";query:",query,";start_point:",start_point,";end_point:",end_point)
            print(i,";loss:",loss,";accuracy_start:",accuracy_start,";predictions_start:",predictions_start)
            #save model
            if i % 300 == 0:
                save_path = ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=i)

def predict():
    #1.generate data

    #2.create model

    #3.feed data

    #4.predict use model in the checkpoint
    pass

def generate_data(n=10,query=0):#query is 0, means find max values; query is 1,means find min values.
    paragraph=None; start_point=None; end_point=None
    paragraph=[i for i in range(n)]
    random.shuffle(paragraph)
    sum=0
    sum_max=-1000
    index_max=0
    max_elements=[]
    sum_mini=10000
    index_mini=0
    mini_elements=[]
    for i,element in enumerate(paragraph):
        if i<n-2:
            if query==0:#find three successive elements that sum up most
                sum=paragraph[i] +paragraph[i+1]+paragraph[i+2]
                if sum>sum_max:
                    max_elements=[]
                    sum_max=sum
                    index_max=i
                    max_elements.append(paragraph[i]);max_elements.append(paragraph[i+1]);max_elements.append(paragraph[i+2])
            else:#query==1: find mini values
                sum=paragraph[i] +paragraph[i+1]+paragraph[i+2]
                if sum<sum_mini:
                    mini_elements=[]
                    sum_mini=sum
                    index_mini=i
                    mini_elements.append(paragraph[i]);mini_elements.append(paragraph[i+1]);mini_elements.append(paragraph[i+2])
    if query==0:
        #print("index_max:",index_max,";sum_max:",sum_max,"max_elements :",max_elements)
        start_point=index_max
    else:
        #print("index_mini:", index_mini, ";sum_mini:", sum_mini, "mini_elements :", mini_elements)
        start_point = index_mini
    end_point = start_point + 2
    #print("end.paragraph:",paragraph,";query:",query,";start_point:",start_point,";end_point:",end_point)
    paragraph=np.reshape(np.array(paragraph),[1,n])
    query=np.reshape(np.array(query),[1,1])
    start_point = np.reshape(np.array(start_point), [1])
    end_point = np.reshape(np.array(end_point), [1])
    return paragraph,query,start_point,end_point

#generate_data()
train()