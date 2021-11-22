import numpy as np
import random
import time
import matplotlib.pyplot as plt
from IPython import display

class LDAModel:
    def __init__(self, 
        when_to_plot_topics_dist,
        seed = None, 
        topics_for_histogram = None, 
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

        # Task 2
        self.iterations_for_plotting_topics_dist = when_to_plot_topics_dist
        self.topics_distributions = []

        # Task 3
        self.entropies_per_iteration = []

        # Task 4
        self.topics_for_histogram = topics_for_histogram
        self.word_freqs_per_iteration = []

    def init_znd(self, docs, max_doc_len):
        self.z_nd = [np.random.randint(0, self.topics_count, size=len(d)) for d in docs]

    def compute_counts(self, docs, vocabulary, topics_num):
        # init all elements to 0
        start_time = time.time()

        self.c_d = np.zeros((len(docs), self.topics_count), dtype=np.int)
        self.c_w = np.zeros((len(vocabulary), self.topics_count), dtype=np.int)
        self.c = np.zeros((self.topics_count, ), dtype=np.int)

        # for each document index, and document list
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

                    p = [0 for k in range(self.topics_count)]
                    for k in range(1, self.topics_count):
                        left_side = (alpha + self.c_d[d][k]) / (self.topics_count + N_d - 1)
                        right_side = (gamma + self.c_w[w_nd][k]) / (self.words_count * gamma + self.c[k])
                        p[k] = left_side * right_side

                    # sample k from probability distribution p[k]
                    topics = [k for k in range(self.topics_count)]

                    sum_p = sum(p)
                    p = [x / sum_p for x in p]

                    k = np.random.choice(topics, p=p)
                    self.z_nd[d][n] = k
                    
                    self.c_d[d][k] += 1
                    self.c_w[w_nd][k] += 1
                    self.c[k] += 1

            entropies = self.get_entropies_per_topic()
            self.entropies_per_iteration.append(entropies)
            
            self.plot(i)

            elapsed_time = time.time() - start_time
            print("Iteration: {}, time: {} seconds".format(i + 1, elapsed_time))

    # TODO: izmeni naziv promenljive: p_KM
    def get_entropies_per_topic(self):
        entropies = np.zeros((self.topics_count, ), dtype=np.float32)
        for k in range(self.topics_count):
            p_KM = self.c_w[k][:] / sum(self.c_w[k][:])
            p_KM = p_KM[p_KM > 0]
            entropies[k] = -sum(p_KM * np.log2(p_KM))
        
        return entropies

    def plot(self, iteration):
        display.clear_output(wait=True)

        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.plot(self.entropies_per_iteration)
        ax1.set_title("Per topic entropy")

        topics_dist = self.c_d[self.doc_idx, :]
        if iteration + 1 in self.iterations_for_plotting_topics_dist:
            self.topics_distributions.append(np.copy(topics_dist))

        ax2.bar(range(self.topics_count), topics_dist)
        ax2.set_title("Topic distribution: Doc {}, Iteration: {}".format(self.doc_idx, iteration))
        ax2.set_xlabel("Topics")
        ax2.set_ylabel("Word count per topic")
        
        plt.show()
    
    def plot_topics_distributions(self):
        count = 1
        fig = plt.figure(figsize=(14,8))
        for i in range(0, 2):
            d = len(self.iterations_for_plotting_topics_dist)
            pola = int(d / 2)
            for j in range(pola):
                topics_dist = self.topics_distributions[count - 1]
                iteration = self.iterations_for_plotting_topics_dist[count - 1]

                subplot_code = int("2{}{}".format(pola, count))
                ax = fig.add_subplot(subplot_code)
                
                ax.bar(range(self.topics_count), topics_dist)

                ax.set_title("Topic distribution for document {} in iteration {}".format(self.doc_idx, iteration))
                ax.set_ylabel("Word count per topic")
                ax.set_xlabel("Topics")
                
                count += 1
        
        plt.show()

    def plot_top_words(self, dictionary, topics, top_words=20):
        for topic in topics:
            plt.figure(figsize=(6,3))
            plt.xticks(rotation=90)
            words_in_topic = self.c_w[:, topic] # Counts of words in topic k
            word_tokenIDs = np.argsort(words_in_topic)[-top_words:]

            words_to_plot=[dictionary[tokenID] for tokenID in word_tokenIDs]
            freqs = np.sort(words_in_topic)[-top_words:]
            plt.plot(words_to_plot, freqs, "x")
            print()