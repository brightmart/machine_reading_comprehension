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
                 lr=0.0001,matching_hop_times=1,is_training=True,output_hop_times=1,clip_gradients=5.0): #matching_hop_times=3,output_hop_times=3 TODO
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

        self.logits_start, self.logits_end = self.inference() #(?, article_len)
        self.loss_val = self.loss()

        self.predictions_start = tf.argmax(self.logits_start,axis=1, name="predictions_start")  # shape:(?,)
        self.predictions_end = tf.argmax(self.logits_end, axis=1,name="predictions_end")  # shape:(?,)

        correct_prediction_start = tf.equal(tf.cast(self.predictions_start, tf.int32), self.label_start)
        self.accuracy_start = tf.reduce_mean(tf.cast(correct_prediction_start, tf.float32), name="accuracy_start")  # shape=()
        correct_prediction_end = tf.equal(tf.cast(self.predictions_end, tf.int32), self.label_end)
        self.accuracy_end = tf.reduce_mean(tf.cast(correct_prediction_end, tf.float32), name="accuracy_end")  # shape=()
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
        self.encode_layer() #1.encode
        self.matching_layer() #2.matching
        start_logits,end_logits=self.output_layer() #3.output layer. [none],[none]
        return start_logits,end_logits #[none],[none]

    def encode_layer(self):
        # 1.1 embedding of word/character/ner/pos for query
        query_embeddings_word = tf.nn.embedding_lookup(self.embedding_query, self.query)   # [none,self.sentence_len,self.embed_sz] TODO should use self.embedding_word for real task. toy task should use embedding_query
        #query_embeddings_char = tf.nn.embedding_lookup(self.embedding_char, self.query)   # [none,self.sentence_len,self.embed_sz] TODO
        #query_embeddings_ner = tf.nn.embedding_lookup(self.embedding_ner,self.query_ner)  # [none,self.sentence_len,self.embed_sz] TODO
        #query_embeddings_pos = tf.nn.embedding_lookup(self.embedding_pos,self.query_pos)  # [none,self.sentence_len,self.embed_sz] TODO

        # 1.2.concat embeddings
        # query = tf.concat([query_embeddings_word,query_embeddings_char, query_embeddings_ner, query_embeddings_pos],axis=2)  # [none,self.sentence_len,self.embed_sz*4] TODO

        # 1.3.use bi-directional rnn to encoding inputs(query)
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        bi_outputs_query, bi_state_query = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, query_embeddings_word, dtype=tf.float32,time_major=False,swap_memory=True,scope="query") #TODO
        self.r_q=tf.concat([bi_outputs_query[0],bi_outputs_query[1]],axis=-1) #[none,sentence_len,hidden_size*2] query representation. bi_outputs_query contain two elements.one for forward part, another for backward part
        self.u_q=tf.concat([bi_state_query[0][1],bi_state_query[1][1]],axis=-1) #[none,hidden_size*2].the concatenation of both directions' last hidden state.

        # 2.1 embedding of word/character/ner/pos for context
        context_embeddings_word = tf.nn.embedding_lookup(self.embedding_word, self.context)   # [none,self.article_len,self.embed_sz]
        #context_embeddings_char = tf.nn.embedding_lookup(self.embedding_char, self.context)   # [none,self.article_len,self.embed_sz] TODO
        #context_embeddings_ner = tf.nn.embedding_lookup(self.embedding_ner,self.context_ner)  # [none,self.article_len,self.embed_sz] TODO
        #context_embeddings_pos = tf.nn.embedding_lookup(self.embedding_pos,self.context_pos)  # [none,self.article_len,self.embed_sz] TODO

        # 1.2.concat embeddings
        #context = tf.concat([context_embeddings_word,context_embeddings_char, context_embeddings_ner, context_embeddings_pos],axis=2)  # TODO [none,self.article_len,self.embed_sz*4]

        # 1.3.use bi-directional rnn to encoding inputs(context)
        bi_outputs_context, bi_state_context = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, context_embeddings_word, dtype=tf.float32,time_major=False,swap_memory=True,scope="context") #TODO
        self.r_p=tf.concat([bi_outputs_context[0],bi_outputs_context[1]],axis=-1) #context representation #[none,article_len,hidden_size*2]


    def matching_layer(self):
        """
        memory network of full-orientation matching
        :return:
        """
        # 1. full-oriention matching layer
        # 1.1 integral query matching.
        with tf.variable_scope("matching_layer"):
            for i in range(self.matching_hop_times): #each loop is one hop. totally run hop_times times. by default hop_times can be set to 1 or 3.
                if i>0:
                    tf.get_variable_scope().reuse_variables()
                self.m_1=self.integral_query_matching() #[none,hidden_size*2]

                # 1.2. query-based similarity matching. input:context r_p:[none,article_len,hidden_size*2],query r_q:[none,sentence_len,hidden_size*2]
                self.M_2=self.query_based_similarity_matching() #[none,article_len,hidden_size*2]

                # 1.3. context-based similarity matching
                self.m_3=self.context_based_similarity_matching() #[none,hidden_size*2]

                # 1.4. transform three differents matching output to get integrated hierarchical matching results.
                M=self.transform_matchings_concat()

                # 2. add an additional gate: to filtrates the part of tokens that are helpful in understanding the relation between passage and query.
                M=self.add_gate(M) #[none,article_len,hidden_size*2]
                # 3. pass through a bi-directional LSTM.
                self.O=self.bi_directional_LSTM(M,"matching")    #[none,article_len,hidden_size*4]
                # 4.update original passage(context) representation r_p for the use of next layer.
                with tf.variable_scope("dimension_reduction"):
                    self.r_p=tf.layers.dense(self.O,self.hidden_size*2)   #[none,article_len,hidden_size*2] #IMPORT in multiple layers, the output of directional LSTM O can be regarded as (refined) context r_p


    def integral_query_matching(self):
        """
        integral query matching. input:u_q,r_p. obtain the importance of each word in passage according to the integral query
        #  by means of computing the match between u_q and each rprsnt r_p by taking inner product.
        :return:  m_1:[none,hidden_size*2]
        """
        u_q_expand = tf.expand_dims(self.u_q, axis=1)  # [none,1,hidden_size*2]
        score = tf.multiply(u_q_expand, self.r_p)  # [none,article_len,hidden_size*2]
        logits = tf.reduce_sum(score, axis=2)  # [none,article_len]
        c = tf.nn.softmax(logits, dim=1)  # [none,article_len]
        # sum of input(context) weighted by attention c
        c_expand = tf.expand_dims(c, axis=2)  # [none,article_len,1]
        m_1 = tf.multiply(c_expand, self.r_p)  # [none,article_len,hidden_size*2]
        m_1 = tf.reduce_sum(m_1, axis=1)  # [none,hidden_size*2]
        return m_1 #[none,hidden_size*2]

    def query_based_similarity_matching(self):
        """
        query-based similarity matching.
        :return: M_2:[none,article_len,hidden_size*2]
        """
        # A_part1=tf.matmul(self.r_p,self.r_q,transpose_b=True) #[none,article_len,sentence_len]
        with tf.variable_scope("query_based_similarity_matching"):
            r_p_expand = tf.expand_dims(self.r_p, axis=2)  # [none,article_len,1,hidden_size*2]
            r_q_expand = tf.expand_dims(self.r_q, axis=1)  # [none,1,sentence_len,hidden_size*2]
            part1 = tf.multiply(r_p_expand,r_q_expand)  # [none,article_len,sentence_len,hidden_size*2].A_part1=tf.reduce_sum(A_part1,axis=-1) #[none,article_len,sentence_len]

            r_p_tile = tf.tile(r_p_expand, [1, 1, self.sentence_len, 1])  # [none,article_len,sentence_len,hidden_size*2]
            r_q_tile = tf.tile(r_q_expand, [1, self.article_len, 1, 1])  # [none,article_len,sentence_len,hidden_size*2]
            # concat
            A = tf.concat([part1, r_p_tile, r_q_tile], axis=-1)  # [none,article_len,sentence_len,hidden_size*6] #TODO you can add more features like |r_p - r_q|, r_pWr_q
            self.A = tf.squeeze(tf.layers.dense(A, 1), axis=-1)  # [none,article_len,sentence_len]=[none,n,m]
            # softmax function is performed across the row vector
            B = tf.nn.softmax(self.A, dim=2) #[none,article_len,sentence_len]. how to understand 'performed across the row vector. and attention is based on query embedding.
            # input:B:[none,article_len,sentence_len];r_q=[none,sentence_len,hidden_size*2];result should be:[none,article_len,hidde_size*2]
            M_2 = tf.matmul(B, self.r_q)  # [none,article_len,hidden_size*2]
        return M_2 #[none,article_len,hidden_size*2]

    def context_based_similarity_matching(self):
        """
        input:A=[none,article_len,sentence_len]
        context-based similarity matching.
        :return:[none,article_len]
        """
        e=tf.reduce_max(self.A,axis=2) # [none,article_len]
        d=tf.nn.softmax(e,dim=1) #[none,article_len]
        d_expand=tf.expand_dims(d,axis=-1) #[none,article_len,1]; r_p=[none,article_len,hidden_size*2]
        m_3=tf.multiply(d_expand,self.r_p) #[none,article_len,hidden_size*2]
        m_3=tf.reduce_sum(m_3,axis=1) #[none,hidden_size*2]
        return m_3 #[none,hidden_size*2]

    def transform_matchings_concat(self):
        """
        transform three differents matching output to get integrated hierarchical matching results.
        :return:
        """
        with tf.variable_scope("transform_matchings_concat"):
            m_1=tf.expand_dims(self.m_1,axis=1)   #[none,1,hidden_size*2]
            M_1=tf.tile(m_1,[1,self.article_len,1]) #[none,article_len,hidden_size*2]

            m_3=tf.expand_dims(self.m_3,axis=1)   #[none,1,hidden_size*2]
            M_3=tf.tile(m_3,[1,self.article_len,1]) #[none,article_len,hidden_size*2]

            M_1=tf.layers.dense(M_1,self.hidden_size*2)     #[none,article_len,hidden_size*2].transform M_1
            M_2=tf.layers.dense(self.M_2,self.hidden_size*2)     #[none,article_len,hidden_size*2].transform M_2
            M_3 = tf.layers.dense(M_3, self.hidden_size * 2)#[none,article_len,hidden_size*2].transform M_3
            M=M_1+M_2+M_3 #[none,article_len,hidden_size*2]
        return M #[none,article_len,hidden_size*2]

    def add_gate(self,M):
        """
        add gate
        M:[none,article_len,hidden_size*2]
        :return:
        """
        with tf.variable_scope("add_gate"):
            g=tf.sigmoid(tf.layers.dense(M,1,use_bias=True))  #[none,article_len,1]
            M=tf.multiply(g,M) #[none,article_len,hidden_size*2]
        return M #[none,article_len,hidden_size*2]

    def bi_directional_LSTM(self,input,scope):
        """
        bi-directional LSTM. input:[none,sequence_length,h]
        :param input:
        :param scope:
        :return: [none,sequence_length,h*2]
        """
        with tf.variable_scope(scope):
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input, dtype=tf.float32,time_major=False, swap_memory=True,scope=scope)
            result = tf.concat([bi_outputs[0], bi_outputs[1]],axis=-1) # bi_outputs_query contain two elements.one for forward part, another for backward part
        return result #[none,sequence_length,h*4]

    def output_layer(self):
        #1.initialize the hidden state of the pointer network by a query-aware representation.
        with tf.variable_scope("output_layer"):
            l_0=self.query_aware_representation() ##[none,hidden_size*2]
            l_k_1=l_k_2=l_0
            #2.predict the indices that represent the answer's location in the passage by using initialized hidden state and passage representation.
            for i in range(self.output_hop_times):
                if i>0:
                    tf.get_variable_scope().reuse_variables()
                start_point_pred,end_point_pred,v_k_1,v_k_2=self.point_network(l_k_1,l_k_2)

                #3.use GRU to update l_k with v_k as input
                l_k_1=self.GRU(1,l_k_1,v_k_1)
                l_k_2 = self.GRU(2,l_k_2, v_k_2)
        return start_point_pred,end_point_pred #[none],[none]

    def point_network(self,l_k_1,l_k_2):
        """
        point network
        :param O: context representation: [none,article_len,hidden_size*4]. it is a output of matching layer.
        :param l_0: query-aware representation:[none,hidden_size*2]
        :return:
        """
        with tf.variable_scope("point_network"):
            p_start,v_k_1=self.point_network_single(1,l_k_1) #[none,]
            p_end, v_k_2 = self.point_network_single(2, l_k_2)
        return p_start,p_end,v_k_1,v_k_2

    def point_network_single(self,k,l_k):
        """
        point network single
        :param k: k=1,2 represent the start point and end point of the answer
        :param l_k: [none,hidden_size*2].
        :return: p:[none,]
        :return: v_k:[none,hidden_size*4]
        """
        with tf.variable_scope("point_network"+str(k)):
            part_1=tf.layers.dense(self.O,self.hidden_size * 2)
            part_2=tf.expand_dims(tf.layers.dense(l_k,self.hidden_size * 2),axis=1)
            z_ =  tf.nn.tanh(part_1+part_2) #[none,article_len,hidden_size*2]
            z=tf.squeeze(tf.layers.dense(z_,1),axis=2) #[none,article_len]
            a=tf.nn.softmax(z,dim=1) #[none,article_len]
            #p=tf.argmax(a,axis=1) #[none,]
            v_k=tf.multiply(tf.expand_dims(a,axis=2),self.O) #[none,article_len,hidden_size*4]
            v_k=tf.reduce_sum(v_k,axis=1) #[none,hidden_size*4]
        return a,v_k

    def query_aware_representation(self):
        """
        query aware representation,in fact, it is a local attention. input is r_q:[none,sentence_len,hidden_size*2]
        :return:
        """
        with tf.variable_scope("query_aware_representation"):
            z_=tf.layers.dense(self.r_q,self.hidden_size*2,activation=tf.nn.tanh,use_bias=True) #[none,sentence_len,hidden_size*2]
            z=tf.layers.dense(z_,1)       #[none,sentence_len,1]
            a=tf.nn.softmax(z,dim=1)     #[none,setence_len,1]
            l_0=tf.multiply(a,self.r_q)   #[none,sentence_len,hidden_size*2]
            l_0=tf.reduce_sum(l_0,axis=1) #[none,hidden_size*2]
        return l_0                    #[none,hidden_size*2]

    def GRU(self,k,l_k,v_k):
        with tf.variable_scope("GRU"+str(k)):
            cell = tf.nn.rnn_cell.GRUCell(self.hidden_size*2)
            l_k=cell(l_k,v_k)
        l_k=l_k[0]
        return l_k

    def loss(self):
        loss_start=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_start, logits=self.logits_start) #[none,]
        loss_start=tf.reduce_mean(loss_start) #scalar
        loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_end, logits=self.logits_end) #[none,]
        loss_end = tf.reduce_mean(loss_end)  # scalar
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('bias' not in v.name)]) * self.l2_lambda #scalar
        loss=loss_start+loss_end+l2_losses
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

l2_lambda=0.0001
vocab_sz=100
embed_sz=128
ner_vocab_sz=100
pos_vocab_sz=100
sentence_len=1
article_len=10
pos_vocab_sz=10
lr=0.0001

def train():
    ckpt_dir='checkpoint/'
    #.create model
    model=MemenNetwork(l2_lambda,article_len,sentence_len,vocab_sz,embed_sz,ner_vocab_sz,pos_vocab_sz,lr=lr)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1500):
            #generate data
            query=random.choice([0,1])
            paragraph, query, start_point, end_point = generate_data(n=article_len,query=query)
            #feed dict
            fetch_dict=[model.loss_val, model.accuracy_start,model.accuracy_end,model.predictions_start,model.predictions_end,model.train_op]
            feed_dict={model.context:paragraph,model.query:query,model.label_start:start_point,model.label_end:end_point}
            #run with session
            loss,accuracy_start,accuracy_end,predictions_start,predictions_end,_=sess.run(fetch_dict,feed_dict=feed_dict)
            #print result and status
            print(i,"paragraph:",paragraph,";query:",query,";start_point:",start_point,";end_point:",end_point)
            print(i,";loss:",loss,";accuracy_start:",accuracy_start,";accuracy_end:",accuracy_end,";predictions_start:",predictions_start,";predictions_end:",predictions_end)
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
                sum=paragraph[i]+paragraph[i+1]+paragraph[i+2]
                if sum>sum_max:
                    max_elements=[]
                    sum_max=sum
                    index_max=i
                    max_elements.append(paragraph[i]);max_elements.append(paragraph[i+1]);max_elements.append(paragraph[i+2])
            else:#query==1: find mini values
                sum=paragraph[i]+paragraph[i+1]+paragraph[i+2]
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