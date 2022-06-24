from cv2 import kmeans
from sklearn.cluster import KMeans

def cluster_classification(imgs_feats, text_feats, opt_tags, ret_indices=True):
    k = len(opt_tags)
    cluster_centers = text_feats
    k_means = KMeans(n_clusters=k, init=cluster_centers, n_init=1)
    pred_labels = k_means.fit_predict(imgs_feats)
    
    if ret_indices:
        return pred_labels
    else:
        pred_labels = [opt_tags[idx] for idx in pred_labels]
        return pred_labels
