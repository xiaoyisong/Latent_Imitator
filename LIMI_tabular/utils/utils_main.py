import sys

sys.path.append("../")
from sklearn.cluster import KMeans
import joblib
import os
import tensorflow as tf

from data.census import census_data
from data.bank import bank_data
from data.credit import credit_data
from .utils_tf import model_loss

# datasets_dict = {"census": census_data, "credit": credit_data, "bank": bank_data}

def cluster_base(dataset_name, X=None, cluster_num=4):
    """
    Construct the K-means clustering model to increase the complexity of discrimination
    :param dataset_name: the name of dataset_name
    :param cluster_num: the number of clusters to form as well as the number of
            centroids to generate
    :return: the K_means clustering model
    """
    datasets_dict = {"census": census_data, "credit": credit_data, "bank": bank_data}
    if X == None:
        print('X is None, use default data')
        X, Y, input_shape, nb_classes = datasets_dict[dataset_name]()
    path = (
        "../exp/clusters/base/"
        + dataset_name
        + ".pkl"
    )
    if os.path.exists(path):
        print(f"load from {path}")
        clf = joblib.load(path)
    else:
        # X, Y, input_shape, nb_classes = datasets_dict[dataset_name]()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        clf = KMeans(n_clusters=cluster_num, random_state=2022).fit(X)
        joblib.dump(clf, path)
        print(f"dump to {path}")
    return clf


def gradient_graph(x, preds, y=None):
    """
    Construct the TF graph of gradient
    :param x: the input placeholder
    :param preds: the model's symbolic output
    :return: the gradient graph
    """
    if y == None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
        y = tf.stop_gradient(y)
    y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Compute loss
    loss = model_loss(y, preds, mean=False)

    # Define gradient of loss wrt input
    (grad,) = tf.gradients(loss, x)

    return grad


if __name__ == "__main__":
    dataset_name = 'credit'
    clusters_num = 4
    cluster_base(dataset_name=dataset_name, cluster_num=clusters_num)
    dataset_name = 'bank'
    clusters_num = 4
    cluster_base(dataset_name=dataset_name, cluster_num=clusters_num)
    dataset_name = 'census'
    clusters_num = 4
    cluster_base(dataset_name=dataset_name, cluster_num=clusters_num)

