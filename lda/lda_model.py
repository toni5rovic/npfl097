import numpy as np
import random
import time

class LDAModel:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

        self.docs_count = 0
        self.words_count = 0
        self.topics_count = 0

        self.z_nd = None
        self.c_d = None
        self.c_w = None
        self.c = None

    def init_znd(self, docs, max_doc_len):
        # numpy.random.uniform(low=1, high=self.topics_count, size=(2,3) )
        #self.z_nd = [np.random.randint(0, K, size=N_D[d]) for d in range(D)]
        self.z_nd = [np.random.randint(0, self.topics_count, size=len(d)) for d in docs]

        #self.z_nd = np.random.random((len(docs), max_doc_len))
        #self.z_nd = self.z_nd * self.topics_count
        #self.z_nd = self.z_nd.astype(int)
        #print(self.z_nd)

    def compute_counts(self, docs, vocabulary, topics_num):
        # init all elements to 0
        self.c_d = [[0 for t in range(self.topics_count)] for doc in docs]
        self.c_w = [[0 for t in range(self.topics_count)] for w in vocabulary]
        self.c = [0 for t in range(self.topics_count)]

        # for each document index, and document list
        for d, document in enumerate(docs):
            # TODO: da se vrati?
            # init the whole list to False. when topic is found 
            # in the document, the value is changed to True
            #topics_of_document = [False for t in range(self.topics_count)]
            
            # for each word with tokenID m, on the index position n in the document
            for n, m in enumerate(document):
                k = self.z_nd[d][n]
                #print("Doc: {}, Word Idx: {}, TokenID: {}, K: {}".format(d, n, m, k))
                self.c_d[d][k] += 1
                self.c_w[m][k] += 1
                self.c[k] += 1
                #topics_of_document[k] = True
            
            #for k, is_found in enumerate(topics_of_document):
            #    if is_found:
            #        self.c[k] += 1
        
        print("sum self.c_w: " + str(np.sum(self.c_w)))
        print("sum self.c_d: " + str(np.sum(self.c_d)))
        print("sum self.c: " + str(np.sum(self.c)))
        # All the counts should sum to each other
        assert np.sum(self.c_w) == np.sum(self.c)
        assert np.sum(self.c_d) == np.sum(self.c)
        assert np.sum(self.c_d) == np.sum(self.c_w)


    # w_nd = docs[d][n]
    def fit(self, docs, vocabulary, topics_num, iterations, alpha, gamma, max_doc_len):
        self.docs_count = len(docs)
        self.words_count = len(vocabulary)
        self.topics_count = topics_num

        self.init_znd(docs, max_doc_len)
        self.compute_counts(docs, vocabulary, topics_num)

        for i in range(iterations):
            start_time = time.time()
            for d, document in enumerate(docs):
                N_d = len(document)
                for n, m in enumerate(document):
                    z_nd_value = self.z_nd[d][n]
                    w_nd = document[n]
                    
                    self.c_d[d][z_nd_value] -= 1
                    self.c_w[w_nd][z_nd_value] -= 1
                    self.c[z_nd_value] -= 1

                    # TODO: where to keep array of probabiliteis - p
                    p = [0 for k in range(self.topics_count)]
                    for k in range(1, self.topics_count):
                        left_side = (alpha + self.c_d[d][k]) / (self.topics_count + N_d - 1)
                        right_side = (gamma + self.c_w[w_nd][k]) / (self.words_count * gamma + self.c[k])
                        p[k] = left_side * right_side

                    # TODO: sample k from probability distribution p[k]
                    # - objasnjeno u svesci kako treba da se radi sample
                    topics = [k for k in range(self.topics_count)]

                    sum_p = sum(p)
                    p = [x / sum_p for x in p]

                    k = np.random.choice(topics, p=p)
                    self.z_nd[d][n] = k
                    
                    self.c_d[d][k] += 1
                    self.c_w[w_nd][k] += 1
                    self.c[k] += 1

            elapsed_time = time.time() - start_time
            print("Iteration: {}, time: {} seconds".format(i, elapsed_time))
        # 1. initialize z_nd
        # 2. compute counts
        # 3. for i in [1, iterations]
        #       for d in [1, docs_count]
        #           for n in [1, Nd]
        #               c_d[d][z_nd]--
        #               c_w[w_nd][z_nd]--
        #               c[z_nd]--
        #               for k in [1, K]
        #                   p[k] := (alpha + c_d[d][k]) / (K*alpha + Nd - 1) * (gamma + c_w[w_nd][k]) / (M*gamma + c[k])
        #               
        #               sample k from probability distribution p[k]
        #               z_nd := k
        #               c_d[d][k]++
        #               c_w[w_nd][k]++
        #               c[k]++

