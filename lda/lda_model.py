import numpy as np
import random
import time
import matplotlib.pyplot as plt
from IPython import display

class LDAModel:
    def __init__(self, 
        seed = None, 
        doc_idx = 12):
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.doc_idx = doc_idx

        self.docs_count = 0
        self.words_count = 0
        self.topics_count = 0

        self.z_nd = None
        self.c_d = None
        self.c_w = None
        self.c = None

        self.iteration_data = []

        # Task 3
        self.entropies_per_iteration = []

        # Task 4
        self.word_freqs_per_iteration = []

    def init_znd(self, docs, max_doc_len):
        self.z_nd = [np.random.randint(0, self.topics_count, size=len(d)) for d in docs]

    def compute_counts(self, docs, vocabulary):
        # init all elements to 0
        start_time = time.time()

        self.c_d = np.zeros((len(docs), self.topics_count), dtype=np.int)
        self.c_w = np.zeros((len(vocabulary), self.topics_count), dtype=np.int)
        self.c = np.zeros((self.topics_count, ), dtype=np.int)

        # for each document index, and document in document list
        for d, document in enumerate(docs):
            # for each word with tokenID m, on the index position n in the document
            for n, m in enumerate(document):
                k = self.z_nd[d][n]

                self.c_d[d, k] += 1
                self.c_w[m, k] += 1
                self.c[k] += 1

        # All the counts should sum to each other
        assert np.sum(self.c_w) == np.sum(self.c)
        assert np.sum(self.c_d) == np.sum(self.c)
        assert np.sum(self.c_d) == np.sum(self.c_w)

        print("Done computing the initial counts. Elapsed time: {}".format(time.time() - start_time))


    def fit(self, docs, vocabulary, topics_num, iterations, alpha, gamma, max_doc_len):
        self.docs_count = len(docs)
        self.words_count = len(vocabulary)
        self.topics_count = topics_num

        self.init_znd(docs, max_doc_len)
        self.compute_counts(docs, vocabulary)

        for i in range(iterations):
            start_time = time.time()
            for d in range(len(docs)):
                document = docs[d]
                N_d = len(document)
                for n in range(len(document)):
                    z_nd_value = self.z_nd[d][n]
                    w_nd = document[n]
                    
                    self.c_d[d][z_nd_value] -= 1
                    self.c_w[w_nd][z_nd_value] -= 1
                    self.c[z_nd_value] -= 1

                    left_side = (alpha + self.c_d[d,:]) / (self.topics_count * alpha + N_d - 1)
                    right_side = (gamma + self.c_w[w_nd, :]) / (self.words_count * gamma + self.c[:])
                    p = np.multiply(left_side, right_side)
                    p /= sum(p)

                    k = np.random.choice(range(self.topics_count), p=p)

                    self.z_nd[d][n] = k
                    
                    self.c_d[d, k] += 1
                    self.c_w[w_nd, k] += 1
                    self.c[k] += 1
            
            elapsed_time = time.time() - start_time

            self.iteration_data.append((i, np.copy(self.c_w), np.copy(self.c_d), np.copy(self.c), np.copy(self.z_nd)))
            
            entropies = self.get_entropies_per_topic(gamma)
            self.entropies_per_iteration.append(entropies)

            self.plot(i)

            print(entropies)
            print("Iteration: {}, time: {} seconds".format(i + 1, elapsed_time))

    def get_entropies_per_topic(self, gamma):
        entropies = np.zeros((self.topics_count, ), dtype=np.float32)

        for k in range(self.topics_count):
            sum_counts = sum(self.c_w[:, k])
            p_k = (gamma + self.c_w[:, k]) / (self.words_count * gamma + sum_counts)
            entropies[k] = -sum(p_k * np.log2(p_k))

        return entropies


    def plot(self, iteration):
        display.clear_output(wait=True)

        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.plot(self.entropies_per_iteration)
        ax1.set_title("Per topic entropy")

        topics_dist = self.c_d[self.doc_idx, :]

        ax2.bar(range(self.topics_count), topics_dist)
        ax2.set_title("Topic distribution: Doc {}, Iteration: {}".format(self.doc_idx, iteration + 1))
        ax2.set_xlabel("Topics")
        ax2.set_ylabel("Word count per topic")
        
        plt.show()