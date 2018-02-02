# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib as tf_contrib
#MEMEN: Multi-layer Embedding with Memory Networks for Machine Comprehension
#https://arxiv.org/pdf/1707.09098.pdf
class MemenNetwork():
    def __init__(self,l2_lambda,article_len,vocab_sz,embed_sz,ner_vocab_sz,pos_vocab_sz,hop_times=3):
        self.l2_lambda=l2_lambda
        self.article_len=article_len
        self.vocab_sz=vocab_sz
        self.embed_sz=embed_sz
        self.ner_vocab_sz=ner_vocab_sz
        self.pos_vocab_sz=pos_vocab_sz
        self.hop_times=hop_times

        self.query = tf.placeholder(tf.int32, [None, self.sentence_len], name="query")
        self.query_ner=tf.placeholder(tf.int32, [None, self.sentence_len], name="query_ner")   #NER tag:PERSON,LOCATION,ORGANIZATION
        self.query_pos = tf.placeholder(tf.int32, [None, self.sentence_len], name="query_pos") #POS tag:part of speech:noun,verb

        self.context = tf.placeholder(tf.int32, [None, self.article_len], name="context")
        self.context_ner=tf.placeholder(tf.int32, [None, self.article_len], name="context_ner")   #NER tag:PERSON,LOCATION,ORGANIZATION
        self.context_pos=tf.placeholder(tf.int32,[None,self.self.article_len],name="context_pos") #POS tag:part of speech:noun,verb

        self.label_start=tf.placeholder(tf.int32,[None],name="start_point") #start point
        self.label_end = tf.placeholder(tf.int32, [None], name="end_point")  # start point

        self.logits_start, self.logits_end = self.inference()
        self.loss_val = self.loss()
        self.train_op = self.train()

    def instantiate_weights(self):
        self.embedding_word = tf.get_variable("embedding_word", [self.vocab_sz, self.embed_sz]) #word embedding matrix      #TODO YOU CAN PRETRAIN THIS, AND LOAD FROM OUTSIDE
        self.embedding_char = tf.get_variable("embedding_char", [self.vocab_sz, self.embed_sz]) #character embedding matrix #TODO YOU CAN PRETRAIN THIS, AND LOAD FROM OUTSIDE
        self.embedding_ner = tf.get_variable("embedding_ner", [self.ner_vocab_sz, self.embed_sz]) #embedding matrix for NER #TODO YOU CAN PRETRAIN THIS, AND LOAD FROM OUTSIDE
        self.embedding_pos = tf.get_variable("embedding_pos", [self.pos_vocab_sz, self.embed_sz]) #embedding matrix for POS #TODO YOU CAN PRETRAIN THIS, AND LOAD FROM OUTSIDE
        self.weight1=tf.get_variable("weight1", [self.hidden_size*6,1])
        self.weight_s = tf.get_variable("weight_s", [self.hidden_size * 2, 1])

    def inference(self):
        self.encode_layer() #1.encode
        self.matching_layer() #2.matching
        start_logits,end_logits=self.output_layer() #3.output layer
        return start_logits,end_logits

    def encode_layer(self):
        # 1.1 embedding of word/character/ner/pos for query
        query_embeddings_word = tf.nn.embedding_lookup(self.embedding_word, self.query)   # [none,self.sentence_len,self.embed_sz]
        query_embeddings_char = tf.nn.embedding_lookup(self.embedding_char, self.query)   # [none,self.sentence_len,self.embed_sz]
        query_embeddings_ner = tf.nn.embedding_lookup(self.embedding_ner,self.query_ner)  # [none,self.sentence_len,self.embed_sz]
        query_embeddings_pos = tf.nn.embedding_lookup(self.embedding_pos,self.query_pos)  # [none,self.sentence_len,self.embed_sz]

        # 1.2.concat embeddings
        query = tf.concat([query_embeddings_word,query_embeddings_char, query_embeddings_ner, query_embeddings_pos],axis=2)  # [none,self.sentence_len,self.embed_sz*4]

        # 1.3.use bi-directional rnn to encoding inputs(query)
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        bi_outputs_query, bi_state_query = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, query, dtype=tf.float32,time_major=False,swap_memory=True,scope="query")
        self.r_q=tf.concat([bi_outputs_query[0],bi_outputs_query[1]],axis=-1) #[none,sentence_len,hidden_size*2] query representation. bi_outputs_query contain two elements.one for forward part, another for backward part

        # 2.1 embedding of word/character/ner/pos for context
        context_embeddings_word = tf.nn.embedding_lookup(self.embedding_word, self.context)   # [none,self.article_len,self.embed_sz]
        context_embeddings_char = tf.nn.embedding_lookup(self.embedding_char, self.context)   # [none,self.article_len,self.embed_sz]
        context_embeddings_ner = tf.nn.embedding_lookup(self.embedding_ner,self.context_ner)  # [none,self.article_len,self.embed_sz]
        context_embeddings_pos = tf.nn.embedding_lookup(self.embedding_pos,self.context_pos)  # [none,self.article_len,self.embed_sz]

        # 1.2.concat embeddings
        context = tf.concat([context_embeddings_word,context_embeddings_char, context_embeddings_ner, context_embeddings_pos],axis=2)  # [none,self.article_len,self.embed_sz*4]

        # 1.3.use bi-directional rnn to encoding inputs(context)
        bi_outputs_context, bi_state_context = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, context, dtype=tf.float32,time_major=False,swap_memory=True,scope="context")
        self.r_p=tf.concat([bi_outputs_context[0],bi_outputs_context[1]],axis=-1) #context representation #[none,article_len,hidden_size*2]

        self.u_q=tf.concat([bi_state_query[0],bi_state_query[1]],axis=-1) #[none,hidden_size*2].the concatenation of both directions' last hidden state.

    def matching_layer(self):
        """
        memory network of full-orientation matching
        :return:
        """
        # 1. full-oriention matching layer
        # 1.1 integral query matching.
        for i in self.hop_times: #each loop is one hop. totally run hop_times times. by default hop_times can be set to 1 or 3.
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
            self.O=self.bi_directional_LSTM(self,M,"matching")    #[none,article_len,hidden_size*4]

            self.r_p=tf.layers.dense(self.O,self.hidden_size*2)   #[none,article_len,hidden_size*2] #IMPORT in multiple layers, the output of directional LSTM O can be regarded as (refined) context r_p


    def integral_query_matching(self):
        """
        integral query matching. input:u_q,r_p. obtain the importance of each word in passage according to the integral query
        #  by means of computing the match between u_q and each rprsnt r_p by taking inner product.
        :return:  m_1:[none,hidden_size*2]
        """
        u_q_expand = tf.expand_dims(self.u_q, axis=1)  # [none,1,hidden_size*2]
        score = tf.multiply(u_q_expand, self.r_p)  # [none,article_len,hidden_size*2]
        logits = tf.reduce_sum(score, axis=1)  # [none,article_len]
        c = tf.nn.softmax(logits, aixs=1)  # [none,article_len]

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

        r_p_expand = tf.expand_dims(self.r_p, axis=2)  # [none,article_len,1,hidden_size*2]
        r_q_expand = tf.expand_dims(self.r_q, axis=1)  # [none,1,sentence_len,hidden_size*2]
        part1 = tf.multiply(r_p_expand,r_q_expand)  # [none,article_len,sentence_len,hidden_size*2].A_part1=tf.reduce_sum(A_part1,axis=-1) #[none,article_len,sentence_len]

        r_p_tile = tf.tile(r_p_expand, [1, 1, self.sentence_len, 1])  # [none,article_len,sentence_len,hidden_size*2]
        r_q_tile = tf.tile(r_q_expand, [1, self.article_len, 1, 1])  # [none,article_len,sentence_len,hidden_size*2]
        # concat
        A = tf.concat([part1, r_p_tile, r_q_tile], axis=-1)  # [none,article_len,sentence_len,hidden_size*6] #TODO you can add more features like |r_p - r_q|, r_pWr_q
        self.A = tf.squeeze(tf.layers.dense(A, 1), axis=-1)  # [none,article_len,sentence_len]=[none,n,m]
        # softmax function is performed across the row vector
        B = tf.nn.softmax(self.A, axis=2) #[none,article_len,sentence_len]. how to understand 'performed across the row vector. and attention is based on query embedding.
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
        d=tf.nn.softmax(e) #[none,article_len]
        d_expand=tf.expand_dims(d,axis=-1) #[none,article_len,1]; r_p=[none,article_len,hidden_size*2]
        m_3=tf.multiply(d_expand,self.r_p) #[none,article_len,hidden_size*2]
        m_3=tf.reduce_sum(m_3,axis=1) #[none,hidden_size*2]
        return m_3 #[none,hidden_size*2]

    def transform_matchings_concat(self):
        """
        transform three differents matching output to get integrated hierarchical matching results.
        :return:
        """
        m_1=tf.expand_dims(self.m_1,axis=1)   #[none,1,hidden_size*2]
        M_1=tf.tile(m_1,[1,self.article_len,1]) #[none,article_len,hidden_size*2]

        m_3=tf.expand_dims(self.m_3,axis=1)   #[none,1,hidden_size*2]
        M_3=tf.tile(m_3,[1,self.article_len,1]) #[none,article_len,hidden_size*2]

        M_1=tf.layers.dense(M_1,self.hidden_size*2)     #[none,article_len,hidden_size*2].transform M_1
        M_2=tf.layers.dense(self.M_2,self.hidden_size*2)     #[none,article_len,hidden_size*2].transform M_2
        M_3 = tf.layers.dense(M_3, self.hidden_size * 2)#[none,article_len,hidden_size*2].transform M_3
        M=M_1+M_2+M_3 #[none,article_len,hidden_size*2]
        return M

    def add_gate(self,M):
        """
        add gate
        :return:
        """
        M=tf.layers.dense(M,1,use_bias=True) #[none,article_len,1]
        g=tf.sigmoid(M)  #[none,article_len,hidden_size*2]
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
        return result #[none,sequence_length,h*2]

    def output_layer(self):
        #1.initialize the hidden state of the pointer network by a query-aware representation.
        self.l_0=self.query_aware_representation() ##[none,hidden_size*2]

        #2.predict the indices that represent the answer's location in the passage by using initialized hidden state and passage representation.
        start_point_pred,end_point_pred=self.point_network()

        #3.GRU
        #TODO
        return start_point_pred,end_point_pred


    def query_aware_representation(self):
        """
        query aware representation,in fact, it is a local attention. input is r_q:[none,sentence_len,hidden_size*2]
        :return:
        """
        z_=tf.layers.dense(self.r_q,self.hidden_size*2,activation=tf.nn.tanh,use_bias=True) #[none,sentence_len,hidden_size*2]
        z=tf.layers.dense(z_,1)       #[none,sentence_len,1]
        a=tf.nn.softmax(z,axis=1)     #[none,setence_len,1]
        l_0=tf.multiply(a,self.r_q)   #[none,sentence_len,hidden_size*2]
        l_0=tf.reduce_sum(l_0,axis=1) #[none,hidden_size*2]
        return l_0                    #[none,hidden_size*2]

    def point_network(self):
        """
        point network
        :param O: context representation: [none,article_len,hidden_size*4]. it is a output of matching layer.
        :param l_0: query-aware representation:[none,hidden_size*2]
        :return:
        """
        p_list=[]
        for i in range(2):
            p=self.point_network_single(i) #[none,]

            p_list.append(p)

        p_start=p_list[0]
        p_end=p_list[1]
        return p_start,p_end

    def point_network_single(self,k):
        """
        point network single
        :param k: k=1,2 represent the start point and end point of the answer
        :return:
        """
        with tf.variable_scope("point_network"+str(k)):
            weight_p = tf.get_variable("weight_p", [self.hidden_size * 4, self.hidden_size * 2])
            weight_h = tf.get_variable("weight_h", [self.hidden_size * 2, self.hidden_size * 2])
            weight_c = tf.get_variable("weight_c", [self.hidden_size * 2, 1])
            z_ =  tf.nn.tanh(tf.multiply(self.O,weight_p)+tf.expand_dims(tf.multiply(self.l_0,weight_h),axis=1)) #[none,article_len,hidden_size*2]
            z=tf.squeeze(tf.multiply(z_,weight_c)) #[none,article_len]
            a=tf.nn.softmax(z,axis=1) #[none,article_len]
            p=tf.argmax(a,axis=1) #[none,]

            ########################################################
            #TODO search multiple times to narrow down predicted answer
            #v=tf.multiply(tf.expand_dims(a,axis=1),self.O) #[none,article_len,hidden_size*4]
            #v=tf.reduce_sum(v,axis=1) #[none,hidden_size*4]
            #self.GRU()
            ########################################################
        return p #[none,]

    def GRU(self):
        pass

    def loss(self):
        loss_start = tf.losses.sparse_softmax_cross_entropy(self.label_start, self.logits_start) #[none,]
        loss_start=tf.reduce_mean(loss_start) #scalar
        loss_end = tf.losses.sparse_softmax_cross_entropy(self.label_end, self.logits_end) #[none,]
        loss_end = tf.reduce_mean(loss_end)  # scalar
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('bias' not in v.name)]) * self.l2_lambda #scalar
        loss=loss_start+loss_end+l2_losses
        return loss #scalar


    def train(self):
        learning_rate = tf.train.exponential_decay(self.lr, self.global_step, self.decay_steps, self.decay_rate,staircase=True)
        train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

#toy task: read a passenge,
#input:
#output:
#train and predict.

def train():
    pass

def predict():
    pass