import os
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
from math import ceil
from sklearn.metrics import roc_auc_score
class BPR(object):

    def __init__(self,
                 user_num,
                 k_num,
                 item_num,
                 learning_rate=0.05,
                 lambda_u = 0.00025,
                 lambda_i = 0.00025,
                 lambda_j = 0.00025,
                 lambda_bias = 0.0
                 ):
        self.user_num = user_num
        self.k_num = k_num
        self.item_num = item_num
        self.learning_rate = learning_rate
        self.lambda_u = lambda_u
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.lambda_bias = lambda_bias

        self.W = tf.get_variable("W_matrix", [self.user_num, self.k_num],
                                 initializer=tf.random_normal_initializer(0, 0.1))
        self.H = tf.get_variable("H_matrix", [self.item_num, self.k_num],
                                 initializer=tf.random_normal_initializer(0, 0.1))
        self.B = tf.get_variable("B", [self.item_num],
                                 initializer=tf.constant_initializer(0, tf.float32))
        self.opt = None
        self.u = None
        self.i = None
        self.j = None
        self._W = None
        self._H = None
        self._B = None

    def random_batch(self,
                       data,
                       batch_size=200):
        #nd--number of data
        nd_user,nd_item = data.shape
        u_ = np.random.choice(nd_user, size=batch_size)
        i_ = np.zeros(batch_size, dtype=np.int)
        j_ = np.zeros(batch_size, dtype=np.int)
        for index, user in enumerate(u_):
            i_items = indices[indptr[user]:indptr[user + 1]]
            i_item = np.random.choice(i_items)
            j_item = np.random.choice(nd_item)

            while j_item in i_items:
                j_item = np.random.choice(nd_item)

            i_[index] = i_item
            j_[index] = j_item

        return u_, i_, j_

    def train(self,
              batch_size = 200):
        u = tf.placeholder(tf.int32, [None])
        i = tf.placeholder(tf.int32, [None])
        j = tf.placeholder(tf.int32, [None])
        W1 = tf.nn.embedding_lookup(self.W, u)
        H1 = tf.nn.embedding_lookup(self.H, i)
        H2 = tf.nn.embedding_lookup(self.H, j)
        B1 = tf.nn.embedding_lookup(self.B, i)
        B2 = tf.nn.embedding_lookup(self.B, j)
        xi = tf.matmul(W1, H1, transpose_b=True)
        xj = tf.matmul(W1, H2, transpose_b=True)
        xij = xi - xj + B1 - B2

        opt = tf.add(-tf.reduce_sum(tf.log(tf.sigmoid(xij))),
                     tf.add_n([self.lambda_u * tf.reduce_sum(tf.multiply(W1, W1)),
                     self.lambda_i * tf.reduce_sum(tf.multiply(H1, H1)),
                     self.lambda_j * tf.reduce_sum(tf.multiply(H2, H2)),
                     self.lambda_bias * tf.reduce_sum(tf.multiply(B1, B1)),
                     self.lambda_bias * tf.reduce_sum(tf.multiply(B2, B2))])
                     )
        # train = tf.train.AdamOptimizer(self.learning_rate).minimize(opt)
        train = tf.train.AdagradOptimizer(self.learning_rate).minimize(opt)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i__ in range(100):
                loss = 0
                u1,i1,j1 = self.random_batch(Train, batch_size)
                opt_val, _,self._W,self._H,self._B= sess.run([opt,train,self.W,self.H,self.B],
                             feed_dict={u: u1, i: i1, j: j1})
                # print("iter %d: loss = %f"% (i__,opt_val))


    def test(self,
             Test):
        auc = 0.0
        nt_user, nt_item = Test.shape
        for index_u, row in enumerate(Test):
            #y_score = self._W[index_u] * trans(self._H)
            y_score = self._W[index_u].dot(self._H.transpose()) + self._B[index_u]
            y_true = np.zeros(n_items)
            y_true[row.indices] = 1
            auc += roc_auc_score(y_true, y_score)
        auc /= nt_user
        print("AUC = ", auc)



# train
# initial
names = ['user_id', 'item_id', 'rating']
input_0 = pd.read_csv("train_data/ratings_0.txt",sep = ' ', names = names)
input_1 = pd.read_csv("train_data/ratings_1.txt",sep = ' ', names = names)
input_2 = pd.read_csv("train_data/ratings_2.txt",sep = ' ', names = names)
input_3 = pd.read_csv("train_data/ratings_3.txt",sep = ' ', names = names)
input_data=pd.concat([input_0,input_1,input_2,input_3],0)
for col in (names[0], names[1], names[2]):
    input_data[col] = input_data[col].astype('category')

ratings = csr_matrix((input_data[names[2]],(input_data[names[0]].cat.codes, input_data[names[1]].cat.codes)))
train = ratings.copy().todok()
test = dok_matrix(train.shape)
rd_state = np.random.RandomState()

test_size = 0.2
for u in range(ratings.shape[0]):
    split_index = ratings[u].indices
    n_splits = ceil(test_size * split_index.shape[0])
    test_index = rd_state.choice(split_index, size=n_splits, replace=False)
    test[u, test_index] = ratings[u, test_index]
    train[u, test_index] = 0

Train = train.tocsr()
Test = test.tocsr()

indptr = ratings.indptr
indices = ratings.indices
n_users, n_items = ratings.shape

# train
bpr = BPR(n_users,30,n_items)
bpr.train(500)
bpr.test(Test)
