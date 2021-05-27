""" Experiment runner for the model with knowledge graph attached to interaction data """




import argparse
import datetime
import time

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import sys
import json

from preprocessing import create_trainvaltest_split, \
    sparse_to_tuple, preprocess_user_item_features, globally_normalize_bipartite_adjacency, \
    load_data_monti, normalize_features
from model import RecommenderGAE, RecommenderSideInfoGAE
from utils import construct_feed_dict

import os
def load_official_trainvaltest_split(dataset, testing=False):
    u_features = np.load('name_dataset/face_feature.npy')
    v_features = np.load('name_dataset/name_feature.npy')
    u_features = sp.csr_matrix(u_features)
    v_features = sp.csr_matrix(v_features)
    rating_mx_train = np.load('name_dataset/rating.npy')
    rating_mx_train = sp.csr_matrix(rating_mx_train)
    rating_mx_all = np.load('name_dataset/rating_all.npy')
    rating_mx_all = sp.csr_matrix(rating_mx_all)
    u_train_idx = np.load('name_dataset/u_train_idx.npy')
    v_train_idx = np.load('name_dataset/v_train_idx.npy')
    train_labels = np.load('name_dataset/train_labels.npy')
    u_cross_idx = np.load('name_dataset/u_cross_idx.npy')
    v_cross_idx = np.load('name_dataset/v_cross_idx.npy')
    cross_labels = np.load('name_dataset/cross_labels.npy')
    print("User features shape: " + str(u_features.shape))
    print("Item features shape: " + str(v_features.shape))
    print(u_train_idx.shape)
    print(v_train_idx.shape)
    print(train_labels.shape)
    # *4 if rating =1~5
    train_labels = train_labels * 4
    cross_labels = cross_labels * 4
    rating_mx_train = rating_mx_train * 4
    rating_mx_all = rating_mx_all * 4

    test_labels = np.load('name_dataset/test_labels.npy')
    u_test_idx = np.load('name_dataset/u_test_idx.npy')
    v_test_idx = np.load('name_dataset/v_test_idx.npy')
    cluster_label = np.load('name_dataset/label_cluster.npy')
    cluster_label_n = np.load('name_dataset/label_cluster_n.npy')
    pair_cluster = np.load('name_dataset/pair_cluster.npy')
    class_values = np.array([1, 2, 3, 4, 5])  # np.sort(np.unique(ratings))
    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
           cross_labels, u_cross_idx, v_cross_idx, test_labels, u_test_idx, v_test_idx, class_values, cluster_label, cluster_label_n, pair_cluster, rating_mx_all

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Set random seed
# seed = 123 # use only for unit testing
seed = int(time.time())
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="ml_1m",
                choices=['ml_100k', 'ml_1m', 'ml_10m', 'douban', 'yahoo_music', 'flixster'],
                help="Dataset string.")

ap.add_argument("-lr", "--learning_rate", type=float, default=0.01,
                help="Learning rate")

ap.add_argument("-e", "--epochs", type=int, default=2500,
                help="Number training epochs")

ap.add_argument("-hi", "--hidden", type=int, nargs=2, default=[1000, 75],
                help="Number hidden units in 1st and 2nd layer")

ap.add_argument("-fhi", "--feat_hidden", type=int, default=50,
                help="Number hidden units in the dense layer for features")

ap.add_argument("-ac", "--accumulation", type=str, default="sum", choices=['sum', 'stack'],
                help="Accumulation function: sum or stack.")

ap.add_argument("-do", "--dropout", type=float, default=0.7,
                help="Dropout fraction")

ap.add_argument("-nb", "--num_basis_functions", type=int, default=4,
                help="Number of basis functions for Mixture Model GCN.")

ap.add_argument("-ds", "--data_seed", type=int, default=1234,
                help="""Seed used to shuffle data in data_utils, taken from cf-nade (1234, 2341, 3412, 4123, 1324).
                     Only used for ml_1m and ml_10m datasets. """)

ap.add_argument("-sdir", "--summaries_dir", type=str, default='logs/' + str(datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")).replace(' ', '_'),
                help="Directory for saving tensorflow summaries.")

# Boolean flags
fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-nsym', '--norm_symmetric', dest='norm_symmetric',
                help="Option to turn on symmetric global normalization", action='store_true')
fp.add_argument('-nleft', '--norm_left', dest='norm_symmetric',
                help="Option to turn on left global normalization", action='store_false')
ap.set_defaults(norm_symmetric=True)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-f', '--features', dest='features',
                help="Whether to use features (1) or not (0)", action='store_true')
fp.add_argument('-no_f', '--no_features', dest='features',
                help="Whether to use features (1) or not (0)", action='store_false')
ap.set_defaults(features=False)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-ws', '--write_summary', dest='write_summary',
                help="Option to turn on summary writing", action='store_true')
fp.add_argument('-no_ws', '--no_write_summary', dest='write_summary',
                help="Option to turn off summary writing", action='store_false')
ap.set_defaults(write_summary=True)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-t', '--testing', dest='testing',
                help="Option to turn on test set evaluation", action='store_true')
fp.add_argument('-v', '--validation', dest='testing',
                help="Option to only use validation set evaluation", action='store_false')
ap.set_defaults(testing=False)


args = vars(ap.parse_args())

print('Settings:')
print(args, '\n')

# Define parameters
DATASET = args['dataset']
DATASEED = args['data_seed']
NB_EPOCH = args['epochs']
DO = args['dropout']
HIDDEN = args['hidden']
FEATHIDDEN = args['feat_hidden']
BASES = args['num_basis_functions']
LR = args['learning_rate']
WRITESUMMARY = args['write_summary']
SUMMARIESDIR = args['summaries_dir']
FEATURES = args['features']
SYM = args['norm_symmetric']
TESTING = args['testing']
ACCUM = args['accumulation']

SELFCONNECTIONS = False
SPLITFROMFILE = True
VERBOSE = True

if DATASET == 'ml_1m' or DATASET == 'ml_100k' or DATASET == 'douban':
    NUMCLASSES = 5
elif DATASET == 'ml_10m':
    NUMCLASSES = 10
    print('\n WARNING: this might run out of RAM, consider using train_minibatch.py for dataset %s' % DATASET)
    print('If you want to proceed with this option anyway, uncomment this.\n')
    sys.exit(1)
elif DATASET == 'flixster':
    NUMCLASSES = 10
elif DATASET == 'yahoo_music':
    NUMCLASSES = 71
    if ACCUM == 'sum':
        print('\n WARNING: combining DATASET=%s with ACCUM=%s can cause memory issues due to large number of classes.')
        print('Consider using "--accum stack" as an option for this dataset.')
        print('If you want to proceed with this option anyway, uncomment this.\n')
        sys.exit(1)

# Splitting dataset in training, validation and test set

if DATASET == 'ml_1m' or DATASET == 'ml_10m':
    if FEATURES:
        datasplit_path = 'data/' + DATASET + '/withfeatures_split_seed' + str(DATASEED) + '.pickle'
    else:
        datasplit_path = 'data/' + DATASET + '/split_seed' + str(DATASEED) + '.pickle'
elif FEATURES:
    datasplit_path = 'data/' + DATASET + '/withfeatures.pickle'
else:
    datasplit_path = 'data/' + DATASET + '/nofeatures.pickle'


if DATASET == 'flixster' or DATASET == 'douban' or DATASET == 'yahoo_music':
    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
        val_labels, val_u_indices, val_v_indices, test_labels, \
        test_u_indices, test_v_indices, class_values = load_data_monti(DATASET, TESTING)

elif DATASET == 'ml_100k':
    print("Using official MovieLens dataset split u1.base/u1.test with 20% validation set size...")
    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
        cross_labels, u_cross_idx, v_cross_idx, test_labels, \
        test_u_indices, test_v_indices, class_values, cluster_label, \
        cluster_label_n, pair_cluster, rating_mx_all \
        = load_official_trainvaltest_split(DATASET, TESTING)
else:
    print("Using random dataset split ...")
    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
        val_labels, val_u_indices, val_v_indices, test_labels, \
        test_u_indices, test_v_indices, class_values = create_trainvaltest_split(DATASET, DATASEED, TESTING,
                                                                                 datasplit_path, SPLITFROMFILE,
                                                                                 VERBOSE)
multi_head = 3
num_users, num_items = adj_train.shape

num_side_features = 0

# feature loading
if not FEATURES:
    u_features = sp.identity(num_users, format='csr')
    v_features = sp.identity(num_items, format='csr')

    u_features, v_features = preprocess_user_item_features(u_features, v_features)

elif FEATURES and u_features is not None and v_features is not None:
    # use features as side information and node_id's as node input features

    print("Normalizing feature vectors...")
    u_features_side = normalize_features(u_features)
    v_features_side = normalize_features(v_features)

    u_features_side, v_features_side = preprocess_user_item_features(u_features_side, v_features_side)

    u_features_side = np.array(u_features_side.todense(), dtype=np.float32)
    v_features_side = np.array(v_features_side.todense(), dtype=np.float32)

    num_side_features = u_features_side.shape[1]

    # node id's for node input features
    id_csr_v = sp.identity(num_items, format='csr')
    id_csr_u = sp.identity(num_users, format='csr')
    #u_features, v_features = preprocess_user_item_features(id_csr_u, id_csr_v)
    u_features, v_features = preprocess_user_item_features(u_features, v_features)
    #141*521, 77*521
else:
    raise ValueError('Features flag is set to true but no features are loaded from dataset ' + DATASET)

#change this part to be single support
# global normalization
#"""
support = []
support_t = []
adj_train_int = sp.csr_matrix(adj_train, dtype=np.int32)
NUMCLASSES = 5
for i in range(NUMCLASSES):
    # build individual binary rating matrices (supports) for each rating
    support_unnormalized = sp.csr_matrix(adj_train_int == i, dtype=np.float32)

    #if support_unnormalized.nnz == 0 and DATASET != 'yahoo_music':
        # yahoo music has dataset split with not all ratings types present in training set.
        # this produces empty adjacency matrices for these ratings.
    #    sys.exit('ERROR: normalized bipartite adjacency matrix has only zero entries!!!!!')

    support_unnormalized_transpose = support_unnormalized.T
    support.append(support_unnormalized)
    support_t.append(support_unnormalized_transpose)

#shouldn't normalize in our case
support = globally_normalize_bipartite_adjacency(support, symmetric=SYM)
support_t = globally_normalize_bipartite_adjacency(support_t, symmetric=SYM)

support_a = []
support_a_t = []
adj_all_int = sp.csr_matrix(rating_mx_all, dtype=np.int32)
NUMCLASSES = 5
for i in range(NUMCLASSES):
    # build individual binary rating matrices (supports) for each rating
    support_unnormalized = sp.csr_matrix(adj_all_int == i, dtype=np.float32)

    #if support_unnormalized.nnz == 0 and DATASET != 'yahoo_music':
        # yahoo music has dataset split with not all ratings types present in training set.
        # this produces empty adjacency matrices for these ratings.
    #    sys.exit('ERROR: normalized bipartite adjacency matrix has only zero entries!!!!!')

    support_unnormalized_transpose = support_unnormalized.T
    support_a.append(support_unnormalized)
    support_a_t.append(support_unnormalized_transpose)
#shouldn't normalize in our case
support_a = globally_normalize_bipartite_adjacency(support_a, symmetric=SYM)
support_a_t = globally_normalize_bipartite_adjacency(support_a_t, symmetric=SYM)

"""
support = []
support_t = []
adj_train_int = sp.csr_matrix(adj_train)/4
NUMCLASSES = 5
for i in range(NUMCLASSES):
# build individual binary rating matrices (supports) for each rating
    support_unnormalized = adj_train_int #sp.csr_matrix(adj_train_int == i, dtype=np.float32)

    #if support_unnormalized.nnz == 0 and DATASET != 'yahoo_music':
        # yahoo music has dataset split with not all ratings types present in training set.
        # this produces empty adjacency matrices for these ratings.
    #    sys.exit('ERROR: normalized bipartite adjacency matrix has only zero entries!!!!!')

    support_unnormalized_transpose = support_unnormalized.T
    support.append(support_unnormalized)
    support_t.append(support_unnormalized_transpose)

"""

if SELFCONNECTIONS: #not in
    support.append(sp.identity(u_features.shape[0], format='csr'))
    support_t.append(sp.identity(v_features.shape[0], format='csr'))

num_support = len(support) #5
support = sp.hstack(support, format='csr')
support_t = sp.hstack(support_t, format='csr')

support_a = sp.hstack(support_a, format='csr')
support_a_t = sp.hstack(support_a_t, format='csr')

if ACCUM == 'stack':
    div = HIDDEN[0] // num_support
    if HIDDEN[0] % num_support != 0:
        print("""\nWARNING: HIDDEN[0] (=%d) of stack layer is adjusted to %d such that
                  it can be evenly split in %d splits.\n""" % (HIDDEN[0], num_support * div, num_support))
    HIDDEN[0] = num_support * div



train_support = support
train_support_t = support_t

train_support_a = support_a
train_support_a_t = support_a_t

# features as side info
if FEATURES:


    train_u_features_side = u_features_side
    train_v_features_side = v_features_side

else:
    test_u_features_side = None
    test_v_features_side = None

    val_u_features_side = None
    val_v_features_side = None

    train_u_features_side = None
    train_v_features_side = None

placeholders = {
    'u_features': tf.sparse_placeholder(tf.float32, shape=np.array(u_features.shape, dtype=np.int64)),
    'v_features': tf.sparse_placeholder(tf.float32, shape=np.array(v_features.shape, dtype=np.int64)),
    'u_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'v_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'labels': tf.placeholder(tf.int32, shape=(None,)),
    'cross_labels': tf.placeholder(tf.int32, shape=(None,)),
    'cluster_label': tf.placeholder(tf.int32, shape=(None,)),
    'cluster_label_n': tf.placeholder(tf.int32, shape=(None,)),
    'pair_cluster': tf.placeholder(tf.int32, shape=(None,)),

    'u_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),
    'v_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),

    'user_indices': tf.placeholder(tf.int64, shape=(None,)),
    'item_indices': tf.placeholder(tf.int64, shape=(None,)),
    'u_cross_idx': tf.placeholder(tf.int32, shape=(None,)),
    'v_cross_idx': tf.placeholder(tf.int32, shape=(None,)),

    'class_values': tf.placeholder(tf.float32, shape=class_values.shape),

    'dropout': tf.placeholder_with_default(0., shape=()),
    'weight_decay': tf.placeholder_with_default(0., shape=()),

    'support': tf.sparse_placeholder(tf.float32, shape=(None, None)),
    'support_t': tf.sparse_placeholder(tf.float32, shape=(None, None)),

    'support_a': tf.sparse_placeholder(tf.float32, shape=(None, None)),
    'support_a_t': tf.sparse_placeholder(tf.float32, shape=(None, None)),
    'multi_head': tf.placeholder_with_default(0., shape=()),
}

# create model
if FEATURES:
    model = RecommenderSideInfoGAE(placeholders,
                                   input_dim=u_features.shape[1],
                                   feat_hidden_dim=FEATHIDDEN,
                                   num_classes=NUMCLASSES,
                                   num_support=num_support,
                                   self_connections=SELFCONNECTIONS,
                                   num_basis_functions=BASES,
                                   hidden=HIDDEN,
                                   num_users=num_users,
                                   num_items=num_items,
                                   accum=ACCUM,
                                   learning_rate=LR,
                                   num_side_features=num_side_features,
                                   multi_head=multi_head,
                                   logging=True)
else:
    model = RecommenderGAE(placeholders,
                           input_dim=u_features.shape[1],
                           num_classes=NUMCLASSES,
                           num_support=num_support,
                           self_connections=SELFCONNECTIONS,
                           num_basis_functions=BASES,
                           hidden=HIDDEN,
                           num_users=num_users,
                           num_items=num_items,
                           accum=ACCUM,
                           learning_rate=LR,
                           logging=True)



train_support = sparse_to_tuple(train_support)
train_support_t = sparse_to_tuple(train_support_t)

train_support_a = sparse_to_tuple(train_support_a)
train_support_a_t = sparse_to_tuple(train_support_a_t)

u_features = sparse_to_tuple(u_features)
v_features = sparse_to_tuple(v_features)
assert u_features[2][1] == v_features[2][1], 'Number of features of users and items must be the same!'

num_features = u_features[2][1]
u_features_nonzero = u_features[1].shape[0]
v_features_nonzero = v_features[1].shape[0]

# Feed_dicts for validation and test set stay constant over different update steps
train_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                      v_features_nonzero, train_support, train_support_t, train_support_a, train_support_a_t,
                                      train_labels,cluster_label,cluster_label_n, pair_cluster, train_u_indices, train_v_indices, class_values, DO,
                                      cross_labels, u_cross_idx, v_cross_idx, multi_head, train_u_features_side, train_v_features_side)

test_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                     v_features_nonzero, train_support, train_support_t, train_support_a, train_support_a_t,
                                     test_labels,cluster_label,cluster_label_n,pair_cluster, test_u_indices, test_v_indices, class_values, 0.,
                                     cross_labels, u_cross_idx, v_cross_idx, multi_head, train_u_features_side, train_v_features_side)


# Collect all variables to be logged into summary
merged_summary = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if WRITESUMMARY:
    train_summary_writer = tf.summary.FileWriter(SUMMARIESDIR + '/train', sess.graph)
    val_summary_writer = tf.summary.FileWriter(SUMMARIESDIR + '/val')
else:
    train_summary_writer = None
    val_summary_writer = None

best_val_score = np.inf
best_val_loss = np.inf
best_epoch = 0
wait = 0

print('Training...')
t_all = time.time()
for epoch in range(NB_EPOCH):

    t = time.time()

    # Run single weight update
    # outs = sess.run([model.opt_op, model.loss, model.rmse], feed_dict=train_feed_dict)
    # with exponential moving averages
    outs = sess.run([model.training_op, model.loss, model.rmse, model.w_weight, model.embeddings_u, model.cluster_label, model.outputs, model.u_v,model.row_sum,model.column_sum,model.norm_out,model.out_tensor], feed_dict=train_feed_dict)

    train_avg_loss = outs[1]
    train_rmse = outs[2]
    result_out = outs[3]
    result_out2 = outs[4]
    result_out3 = outs[5]
    result_out4 = outs[6]
    result_out5 = outs[7]
    result_out6 = outs[8]
    result_out7 = outs[9]
    result_out8 = outs[10]
    result_out9 = outs[11]
    a = 0
    #if train_avg_loss<1:
    #    break
    #print(result_out)

    #val_avg_loss, val_rmse = sess.run([model.loss, model.rmse], feed_dict=val_feed_dict)

    if VERBOSE:
        print("[*] Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_avg_loss),
              "train_rmse=", "{:.5f}".format(train_rmse),

              "\t\ttime=", "{:.5f}".format(time.time() - t))



    if epoch % 20 == 0 and WRITESUMMARY:
    #if WRITESUMMARY:
        #print('sum')
        # Train set summary
        summary = sess.run(merged_summary, feed_dict=train_feed_dict)
        train_summary_writer.add_summary(summary, epoch)
        train_summary_writer.flush()


    if epoch % 100 == 0 and epoch > 1000 and not TESTING and False:
        saver = tf.train.Saver()
        save_path = saver.save(sess, "tmp/%s_seed%d.ckpt" % (model.name, DATASEED), global_step=model.global_step)

        # load polyak averages
        variables_to_restore = model.variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, save_path)

        # Load back normal variables
        saver = tf.train.Saver()
        saver.restore(sess, save_path)


# store model including exponential moving averages
saver = tf.train.Saver()
save_path = saver.save(sess, "tmp/%s.ckpt" % model.name, global_step=model.global_step)
outs = sess.run([model.outputs,model.labels], feed_dict=test_feed_dict)


#print(outs[3])
np.savetxt('weight_result.txt',outs[0], fmt='%1.4f')
np.save('name_dataset/weight_result',outs[0])
#print(outs[])
#np.savetxt('labels.txt',outs[1], fmt='%1.4f')

if VERBOSE:
    print("\nOptimization Finished!")
    print('best validation score =', best_val_score, 'at iteration', best_epoch)


if TESTING:

    # restore with polyak averages of parameters
    variables_to_restore = model.variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, save_path)



else:
    # restore with polyak averages of parameters
    variables_to_restore = model.variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, save_path)


print('\nSETTINGS:\n')
for key, val in sorted(vars(ap.parse_args()).items()):
    print(key, val)

print('global seed = ', seed)

# For parsing results from file
results = vars(ap.parse_args()).copy()
results.update({'best_val_score': float(best_val_score), 'best_epoch': best_epoch})
print(json.dumps(results))

print(time.time() - t_all)

sess.close()
