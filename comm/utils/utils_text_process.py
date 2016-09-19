# encoding: utf-8
from __future__ import print_function

from datetime import datetime

from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans

from scipy.sparse import hstack

from fuzzywuzzy import utils as utils_fuzz

#ST = LancasterStemmer()
ST = PorterStemmer()
N_COM_DIM_RED = 300  # no. of components after dim reduction

def clean_doc(list_doc):
    list_doc = [u" ".join(utils_fuzz.full_process(x).split()) for x in list_doc]
    return list_doc

def tokenizer_ad(whole_str):
    """Tokenize the string using stemming"""
    list_token = utils_fuzz.full_process(whole_str).split()
    # list_token = [w for w in list_token if not w in stopwords.words('english')]##Filter out stop words:: Decrease accuracy!!!
    list_token = [ST.stem(ele) for ele in list_token]
    return list_token

def feature_transform(list_doc):
    """Transform the raw data using tf-idf"""
    tf_idf_handler = TfidfVectorizer(strip_accents="unicode", ngram_range=(1,3), max_features=1000)
    tf_idf_handler.set_params(analyzer="word", tokenizer=tokenizer_ad)
    tf_idf_handler.fit(list_doc)
    return tf_idf_handler


def reduce_dim(X):
    print("Performing dimensionality reduction using LSA")
    t0 = datetime.now()
    # Taken from http://scikit-learn.org/stable/auto_examples/text/document_clustering.html
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(n_components=N_COM_DIM_RED)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    dim_red_handler = lsa.fit(X)
    print("Complete fit in {}s".format(datetime.now() - t0).seconds)
    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))
    return dim_red_handler

def cluster(X, k_no, use_Kmean=True, use_verbose=True):
    if use_Kmean:
        km = KMeans(n_clusters=k_no, init='k-means++', max_iter=100, n_init=1,
                    verbose=use_verbose)
    else:
        km = MiniBatchKMeans(n_clusters=k_no, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=use_verbose)

    print("Clustering sparse data with %s" % km)
    t0 = datetime.now()
    km.fit(X)
    print("done in {}s".format(datetime.now() - t0))
    print()
    print("Clustering results are: \n{}".format(km.labels_))
    return km.labels_
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    # print("Adjusted Rand-Index: %.3f"
    #       % metrics.adjusted_rand_score(labels, km.labels_))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(X, km.labels_, sample_size=1000))

    # print()






























