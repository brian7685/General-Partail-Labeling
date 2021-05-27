from __future__ import division
from __future__ import print_function


def construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero, v_features_nonzero,
                        support, support_t, support_a, support_a_t, labels, cluster_label,cluster_label_n,pair_cluster, u_indices, v_indices, class_values,
                        dropout,cross_labels, u_cross_idx, v_cross_idx, multi_head, u_features_side=None, v_features_side=None):
    """
    Function that creates feed dictionary when running tensorflow sessions.
    """

    feed_dict = dict()
    feed_dict.update({placeholders['u_features']: u_features})
    feed_dict.update({placeholders['v_features']: v_features})
    feed_dict.update({placeholders['u_features_nonzero']: u_features_nonzero})
    feed_dict.update({placeholders['v_features_nonzero']: v_features_nonzero})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['support_t']: support_t})
    feed_dict.update({placeholders['support_a']: support_a})
    feed_dict.update({placeholders['support_a_t']: support_a_t})

    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['cluster_label']: cluster_label})
    feed_dict.update({placeholders['cluster_label_n']: cluster_label_n})
    feed_dict.update({placeholders['pair_cluster']: pair_cluster})
    feed_dict.update({placeholders['user_indices']: u_indices})
    feed_dict.update({placeholders['item_indices']: v_indices})

    feed_dict.update({placeholders['dropout']: dropout})
    feed_dict.update({placeholders['class_values']: class_values})
    feed_dict.update({placeholders['cross_labels']: cross_labels})
    feed_dict.update({placeholders['u_cross_idx']: u_cross_idx})
    feed_dict.update({placeholders['v_cross_idx']: v_cross_idx})

    feed_dict.update({placeholders['multi_head']: multi_head})

    if (u_features_side is not None) and (v_features_side is not None):
        feed_dict.update({placeholders['u_features_side']: u_features_side})
        feed_dict.update({placeholders['v_features_side']: v_features_side})

    return feed_dict
