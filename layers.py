from __future__ import print_function


from initializations import *
import tensorflow as tf
import pickle
# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)

    return pre_out * tf.div(1., keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    # Properties
        name: String, defines the variable scope of the layer.
            Layers with common name share variables. (TODO)
        logging: Boolean, switches Tensorflow histogram logging on/off
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer for two types of nodes in a bipartite graph. """

    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, share_user_item_weights=False,
                 bias=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        with tf.variable_scope(self.name + '_vars'):
            if not share_user_item_weights:

                self.vars['weights_u'] = weight_variable_random_uniform(input_dim, output_dim, name="weights_u")
                self.vars['weights_v'] = weight_variable_random_uniform(input_dim, output_dim, name="weights_v")

                if bias:
                    self.vars['user_bias'] = bias_variable_truncated_normal([output_dim], name="bias_u")
                    self.vars['item_bias'] = bias_variable_truncated_normal([output_dim], name="bias_v")


            else: #run this
                self.vars['weights_u'] = weight_variable_random_uniform(input_dim, output_dim, name="weights") #519*10
                self.vars['weights_v'] = self.vars['weights_u']

                if bias: #false
                    self.vars['user_bias'] = bias_variable_truncated_normal([output_dim], name="bias_u")
                    self.vars['item_bias'] = self.vars['user_bias']

        self.bias = bias

        self.dropout = dropout
        self.act = act
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x_u = inputs[0]
        x_u = tf.nn.dropout(x_u, 1 - self.dropout)
        #weight 519*10
        x_u = tf.matmul(x_u, self.vars['weights_u']) #49*10

        x_v = inputs[1]
        x_v = tf.nn.dropout(x_v, 1 - self.dropout)
        x_v = tf.matmul(x_v, self.vars['weights_v'])

        u_outputs = self.act(x_u) #49*10
        v_outputs = self.act(x_v)

        if self.bias:
            u_outputs += self.vars['user_bias']
            v_outputs += self.vars['item_bias']

        return u_outputs, v_outputs,self.vars['weights_u']

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging:
                tf.summary.histogram(self.name + '/inputs_u', inputs[0])
                tf.summary.histogram(self.name + '/inputs_v', inputs[1])
            outputs_u, outputs_v,temp = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs_u', outputs_u)
                tf.summary.histogram(self.name + '/outputs_v', outputs_v)
            return outputs_u, outputs_v,temp


class StackGCN(Layer):
    """Graph convolution layer for bipartite graphs and sparse inputs."""

    def __init__(self, input_dim, output_dim, support, support_t, num_support, u_features_nonzero=None,
                 v_features_nonzero=None, sparse_inputs=False, dropout=0.,
                 act=tf.nn.relu, share_user_item_weights=True, **kwargs):
        super(StackGCN, self).__init__(**kwargs)

        assert output_dim % num_support == 0, 'output_dim must be multiple of num_support for stackGC layer'

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_u'] = weight_variable_random_uniform(input_dim, output_dim, name='weights_u')

            if not share_user_item_weights:
                self.vars['weights_v'] = weight_variable_random_uniform(input_dim, output_dim, name='weights_v')

            else: #run this
                self.vars['weights_v'] = self.vars['weights_u']

        self.weights_u = tf.split(value=self.vars['weights_u'], axis=1, num_or_size_splits=num_support) #1
        self.weights_v = tf.split(value=self.vars['weights_v'], axis=1, num_or_size_splits=num_support) #1

        self.dropout = dropout

        self.sparse_inputs = sparse_inputs
        self.u_features_nonzero = u_features_nonzero
        self.v_features_nonzero = v_features_nonzero
        if sparse_inputs:
            assert u_features_nonzero is not None and v_features_nonzero is not None, \
                'u_features_nonzero and v_features_nonzero can not be None when sparse_inputs is True'
        #support should be only one in the future
        self.support = tf.sparse_split(axis=1, num_split=num_support, sp_input=support)
        self.support_transpose = tf.sparse_split(axis=1, num_split=num_support, sp_input=support_t)

        self.act = act

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x_u = inputs[0]
        x_v = inputs[1]

        if self.sparse_inputs:
            x_u = dropout_sparse(x_u, 1 - self.dropout, self.u_features_nonzero)
            x_v = dropout_sparse(x_v, 1 - self.dropout, self.v_features_nonzero)
        else:
            x_u = tf.nn.dropout(x_u, 1 - self.dropout)
            x_v = tf.nn.dropout(x_v, 1 - self.dropout)

        supports_u = []
        supports_v = []
        #pickle.dump(self.weights_v, open("save.p", "wb"))

        # calculate support in different rating 1~5
        for i in range(len(self.support)):
            tmp_u = dot(x_u, self.weights_u[i], sparse=self.sparse_inputs) #m*125, 49*125
            tmp_v = dot(x_v, self.weights_v[i], sparse=self.sparse_inputs) #n*125, 37*125
            #print(self.weights_u[i])
            #pickle.dump(self.weights_v, open("save.p", "wb"))
            support = self.support[i] #m*n array
            support_transpose = self.support_transpose[i] #n*m array
            """
            #supports_u.append(tf.sparse_tensor_dense_matmul(support, tmp_v))
            #supports_v.append(tf.sparse_tensor_dense_matmul(support_transpose, tmp_u))
            tmp_u2 = tf.expand_dims(tmp_u, axis=0)
            tmp_v2 = tf.expand_dims(tmp_v, axis=0)
            #seq_fts = tf.layers.conv1d(tmp_u, 100, 1, use_bias=False)
            #seq_fts2 = tf.layers.conv1d(tmp_v, 100, 1, use_bias=False)
            f_1 = tf.layers.conv1d(tmp_u2, 1, 1)  # 1*num_node*1, let feature be 1D
            f_2 = tf.layers.conv1d(tmp_v2, 1, 1)

            f_1 = tf.reshape(f_1, (tmp_u2.shape[1], 1))  # 2708,1
            f_2 = tf.reshape(f_2, (tmp_v2.shape[1], 1))

            support = self.support[i]
            coefs = support
            support_transpose = self.support_transpose[i]

            support = tf.SparseTensor(indices=support.indices,
                                      values=support.values,
                                      dense_shape=support.dense_shape)
            f_1 = support * f_1  # (m*n) * (m*1)
            f_2 = support * tf.transpose(f_2, [1, 0])  # (n*n) * (1*n)
            logits = tf.sparse_add(f_1, f_2)
            #lrelu = tf.SparseTensor(indices=logits.indices,
            #                        values=tf.nn.leaky_relu(logits.values),
            #                        dense_shape=logits.dense_shape)
            coefs = tf.sparse_softmax(logits)
            """
            #coefs = lrelu
            #coefs = support
            #coefs = tf.SparseTensor(indices=coefs.indices,
            #                        values=tf.nn.dropout(coefs.values, 1.0 - 0.6),
            #                        dense_shape=coefs.dense_shape)

            # drop out for the feature
            #seq_fts = tf.nn.dropout(seq_fts, 1.0 - 0.6)
            #seq_fts2 = tf.nn.dropout(seq_fts2, 1.0 - 0.6)
            # coefs = tf.sparse_reshape(coefs, [seq_fts.shape[1], seq_fts2.shape[1]])
            #seq_fts = tf.squeeze(seq_fts)
            #seq_fts2 = tf.squeeze(seq_fts2)
            #tf.sparse.transpose(support)
            supports_u.append(tf.sparse_tensor_dense_matmul(support, tmp_v))
            supports_v.append(tf.sparse_tensor_dense_matmul(support_transpose, tmp_u))
        z_u = tf.concat(axis=1, values=supports_u) #5*49*125
        z_v = tf.concat(axis=1, values=supports_v)
        #z_u = supports_u[0]
        #z_v = supports_v[0]
        u_outputs = self.act(z_u)
        v_outputs = self.act(z_v)

        return u_outputs, v_outputs,z_u

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs_u', inputs[0])
                tf.summary.histogram(self.name + '/inputs_v', inputs[1])
            outputs_u, outputs_v,weight_t = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs_u', outputs_u)
                tf.summary.histogram(self.name + '/outputs_v', outputs_v)
            return outputs_u, outputs_v,weight_t



class StackGCN2(Layer):
    """Graph convolution layer for bipartite graphs and sparse inputs."""

    def __init__(self, input_dim, output_dim, support, support_t, num_support, u_features_nonzero=None,
                 v_features_nonzero=None, sparse_inputs=False, dropout=0.,
                 act=tf.nn.relu, share_user_item_weights=True, **kwargs):
        super(StackGCN2, self).__init__(**kwargs)

        assert output_dim % num_support == 0, 'output_dim must be multiple of num_support for stackGC layer'

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_u'] = weight_variable_random_uniform(input_dim, output_dim, name='weights_u')

            if not share_user_item_weights:
                self.vars['weights_v'] = weight_variable_random_uniform(input_dim, output_dim, name='weights_v')

            else: #run this
                self.vars['weights_v'] = self.vars['weights_u']

        self.weights_u = tf.split(value=self.vars['weights_u'], axis=1, num_or_size_splits=1)#num_support)
        self.weights_v = tf.split(value=self.vars['weights_v'], axis=1, num_or_size_splits=1)#num_support)

        self.dropout = dropout

        self.sparse_inputs = sparse_inputs
        self.u_features_nonzero = u_features_nonzero
        self.v_features_nonzero = v_features_nonzero
        if sparse_inputs:
            assert u_features_nonzero is not None and v_features_nonzero is not None, \
                'u_features_nonzero and v_features_nonzero can not be None when sparse_inputs is True'
        #support should be only one in the future
        self.support = tf.sparse_split(axis=1, num_split=num_support, sp_input=support)
        self.support_transpose = tf.sparse_split(axis=1, num_split=num_support, sp_input=support_t)

        self.act = act

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x_u = inputs[0]
        x_v = inputs[1]
        if self.sparse_inputs:
            x_u = dropout_sparse(x_u, 1 - self.dropout, self.u_features_nonzero)
            x_v = dropout_sparse(x_v, 1 - self.dropout, self.v_features_nonzero)
        else:
            x_u = tf.nn.dropout(x_u, 1 - self.dropout)
            x_v = tf.nn.dropout(x_v, 1 - self.dropout)

        supports_u = []
        supports_v = []
        #pickle.dump(self.weights_v, open("save.p", "wb"))

        # calculate support in different rating 1~5
        for i in range(1):#len(self.support)):
            tmp_u = dot(x_u, self.weights_u[i], sparse=self.sparse_inputs) #m*125, 49*125
            tmp_v = dot(x_v, self.weights_v[i], sparse=self.sparse_inputs) #n*125, 37*125
            #print(self.weights_u[i])
            #pickle.dump(self.weights_v, open("save.p", "wb"))
            support = self.support[i] #m*n array
            support_transpose = self.support_transpose[i] #n*m array
            coefs = support
            """
            #supports_u.append(tf.sparse_tensor_dense_matmul(support, tmp_v))
            #supports_v.append(tf.sparse_tensor_dense_matmul(support_transpose, tmp_u))
            tmp_u2 = tf.expand_dims(tmp_u, axis=0)
            tmp_v2 = tf.expand_dims(tmp_v, axis=0)
            #seq_fts = tf.layers.conv1d(tmp_u, 100, 1, use_bias=False)
            #seq_fts2 = tf.layers.conv1d(tmp_v, 100, 1, use_bias=False)
            f_1 = tf.layers.conv1d(tmp_u2, 1, 1)  # 1*num_node*1, let feature be 1D
            f_2 = tf.layers.conv1d(tmp_v2, 1, 1)

            f_1 = tf.reshape(f_1, (tmp_u2.shape[1], 1))  # 2708,1
            f_2 = tf.reshape(f_2, (tmp_v2.shape[1], 1))

            support = self.support[i]
            
            support_transpose = self.support_transpose[i]

            support = tf.SparseTensor(indices=support.indices,
                                      values=support.values,
                                      dense_shape=support.dense_shape)
            f_1 = support * f_1  # (m*n) * (m*1)
            f_2 = support * tf.transpose(f_2, [1, 0])  # (n*n) * (1*n)
            logits = tf.sparse_add(f_1, f_2)
            #lrelu = tf.SparseTensor(indices=logits.indices,
            #                        values=tf.nn.leaky_relu(logits.values),
            #                        dense_shape=logits.dense_shape)
            coefs = tf.sparse_softmax(logits)
            #coefs = lrelu
            #coefs = support
            #coefs = tf.SparseTensor(indices=coefs.indices,
            #                        values=tf.nn.dropout(coefs.values, 1.0 - 0.6),
            #                        dense_shape=coefs.dense_shape)

            # drop out for the feature
            #seq_fts = tf.nn.dropout(seq_fts, 1.0 - 0.6)
            #seq_fts2 = tf.nn.dropout(seq_fts2, 1.0 - 0.6)
            # coefs = tf.sparse_reshape(coefs, [seq_fts.shape[1], seq_fts2.shape[1]])
            #seq_fts = tf.squeeze(seq_fts)
            #seq_fts2 = tf.squeeze(seq_fts2)
            """
            coefs = tf.sparse_to_dense(
                coefs.indices,
                coefs.dense_shape,
                coefs.values)
            supports_u.append(tf.matmul(coefs, tmp_v))
            supports_v.append(tf.matmul(tf.transpose(coefs), tmp_u))
        #z_u = tf.concat(axis=1, values=supports_u) #5*49*125
        #z_v = tf.concat(axis=1, values=supports_v)
        z_u = supports_u[0]
        z_v = supports_v[0]
        u_outputs = self.act(z_u)
        v_outputs = self.act(z_v)

        return u_outputs, v_outputs,z_u

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs_u', inputs[0])
                tf.summary.histogram(self.name + '/inputs_v', inputs[1])
            outputs_u, outputs_v,weight_t = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs_u', outputs_u)
                tf.summary.histogram(self.name + '/outputs_v', outputs_v)
            return outputs_u, outputs_v,weight_t



class OrdinalMixtureGCN(Layer):

    """Graph convolution layer for bipartite graphs and sparse inputs."""

    def __init__(self, input_dim, output_dim, support, support_t, num_support, u_features_nonzero=None,
                 v_features_nonzero=None, sparse_inputs=False, dropout=0.,
                 act=tf.nn.relu, bias=False, share_user_item_weights=False, self_connections=False, **kwargs):
        super(OrdinalMixtureGCN, self).__init__(**kwargs)

        with tf.variable_scope(self.name + '_vars'):

            self.vars['weights_u'] = tf.stack([weight_variable_random_uniform(input_dim, output_dim,
                                                                             name='weights_u_%d' % i)
                                              for i in range(num_support)], axis=0)

            if bias:
                self.vars['bias_u'] = bias_variable_const([output_dim], 0.01, name="bias_u")

            if not share_user_item_weights:
                self.vars['weights_v'] = tf.stack([weight_variable_random_uniform(input_dim, output_dim,
                                                                                 name='weights_v_%d' % i)
                                                  for i in range(num_support)], axis=0)

                if bias:
                    self.vars['bias_v'] = bias_variable_const([output_dim], 0.01, name="bias_v")

            else:
                self.vars['weights_v'] = self.vars['weights_u']
                if bias:
                    self.vars['bias_v'] = self.vars['bias_u']

        self.weights_u = self.vars['weights_u']
        self.weights_v = self.vars['weights_v']

        self.dropout = dropout

        self.sparse_inputs = sparse_inputs
        self.u_features_nonzero = u_features_nonzero
        self.v_features_nonzero = v_features_nonzero
        if sparse_inputs:
            assert u_features_nonzero is not None and v_features_nonzero is not None, \
                'u_features_nonzero and v_features_nonzero can not be None when sparse_inputs is True'

        self.self_connections = self_connections

        self.bias = bias
        support = tf.sparse_split(axis=1, num_split=num_support, sp_input=support)

        support_t = tf.sparse_split(axis=1, num_split=num_support, sp_input=support_t)

        if self_connections:
            self.support = support[:-1]
            self.support_transpose = support_t[:-1]
            self.u_self_connections = support[-1]
            self.v_self_connections = support_t[-1]
            self.weights_u = self.weights_u[:-1]
            self.weights_v = self.weights_v[:-1]
            self.weights_u_self_conn = self.weights_u[-1]
            self.weights_v_self_conn = self.weights_v[-1]

        else:
            self.support = support
            self.support_transpose = support_t
            self.u_self_connections = None
            self.v_self_connections = None
            self.weights_u_self_conn = None
            self.weights_v_self_conn = None

        self.support_nnz = []
        self.support_transpose_nnz = []
        for i in range(len(self.support)):
            nnz = tf.reduce_sum(tf.shape(self.support[i].values))
            self.support_nnz.append(nnz)
            self.support_transpose_nnz.append(nnz)

        self.act = act

        if self.logging:
            self._log_vars()

    def _call(self, inputs):

        if self.sparse_inputs:
            x_u = dropout_sparse(inputs[0], 1 - self.dropout, self.u_features_nonzero)
            x_v = dropout_sparse(inputs[1], 1 - self.dropout, self.v_features_nonzero)
        else:
            x_u = tf.nn.dropout(inputs[0], 1 - self.dropout)
            x_v = tf.nn.dropout(inputs[1], 1 - self.dropout)

        supports_u = []
        supports_v = []

        # self-connections with identity matrix as support
        if self.self_connections:
            uw = dot(x_u, self.weights_u_self_conn, sparse=self.sparse_inputs)
            supports_u.append(tf.sparse_tensor_dense_matmul(self.u_self_connections, uw))

            vw = dot(x_v, self.weights_v_self_conn, sparse=self.sparse_inputs)
            supports_v.append(tf.sparse_tensor_dense_matmul(self.v_self_connections, vw))

        wu = 0.
        wv = 0.
        for i in range(len(self.support)):
            wu += self.weights_u[i]
            wv += self.weights_v[i]

            # multiply feature matrices with weights
            tmp_u = dot(x_u, wu, sparse=self.sparse_inputs)

            tmp_v = dot(x_v, wv, sparse=self.sparse_inputs)

            support = self.support[i]
            support_transpose = self.support_transpose[i]

            # then multiply with rating matrices
            supports_u.append(tf.sparse_tensor_dense_matmul(support, tmp_v))
            supports_v.append(tf.sparse_tensor_dense_matmul(support_transpose, tmp_u))

        z_u = tf.add_n(supports_u)
        z_v = tf.add_n(supports_v)

        if self.bias:
            z_u = tf.nn.bias_add(z_u, self.vars['bias_u'])
            z_v = tf.nn.bias_add(z_v, self.vars['bias_v'])

        u_outputs = self.act(z_u)
        v_outputs = self.act(z_v)

        return u_outputs, v_outputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs_u', inputs[0])
                tf.summary.histogram(self.name + '/inputs_v', inputs[1])
            outputs_u, outputs_v = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs_u', outputs_u)
                tf.summary.histogram(self.name + '/outputs_v', outputs_v)
            return outputs_u, outputs_v


class BilinearMixture(Layer):
    """
    Decoder model layer for link-prediction with ratings
    To use in combination with bipartite layers.
    """

    def __init__(self, num_classes, u_indices, v_indices, input_dim, num_users, num_items, user_item_bias=False,
                 dropout=0., act=tf.nn.softmax, num_weights=3,
                 diagonal=True, **kwargs):
        super(BilinearMixture, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):

            for i in range(num_weights): #2
                if diagonal: #false
                    #  Diagonal weight matrices for each class stored as vectors
                    self.vars['weights_%d' % i] = weight_variable_random_uniform(1, input_dim, name='weights_%d' % i)

                else:
                    self.vars['weights_%d' % i] = orthogonal([input_dim, input_dim], name='weights_%d' % i)

            self.vars['weights_scalars'] = weight_variable_random_uniform(num_weights, num_classes,
                                                                          name='weights_u_scalars')

            if user_item_bias: #false
                self.vars['user_bias'] = bias_variable_zero([num_users, num_classes], name='user_bias')
                self.vars['item_bias'] = bias_variable_zero([num_items, num_classes], name='item_bias')

        self.user_item_bias = user_item_bias

        if diagonal:
            self._multiply_inputs_weights = tf.multiply
        else:
            self._multiply_inputs_weights = tf.matmul

        self.num_classes = num_classes
        self.num_weights = num_weights
        self.u_indices = u_indices
        self.v_indices = v_indices
        self.debug_value = 0
        self.dropout = dropout
        self.act = act
        self.num_users = num_users
        self.num_items = num_items
        if self.logging:
            self._log_vars()

    def _call(self, inputs):

        u_inputs = tf.nn.dropout(inputs[0], 1 - self.dropout) #49*75
        v_inputs = tf.nn.dropout(inputs[1], 1 - self.dropout)
        final_u = u_inputs#inputs[0]#u_inputs
        final_v = v_inputs#inputs[1]#v_inputs
        self.debug_value = u_inputs
        u_inputs = tf.gather(u_inputs, self.u_indices) #53*75
        v_inputs = tf.gather(v_inputs, self.v_indices)

        u_v = tf.concat([u_inputs, v_inputs], axis=1)
        if self.user_item_bias: #false
            u_bias = tf.gather(self.vars['user_bias'], self.u_indices)
            v_bias = tf.gather(self.vars['item_bias'], self.v_indices)
        else:
            u_bias = None
            v_bias = None

        basis_outputs = []
        for i in range(self.num_weights): #2
            #self.debug_value = self.vars['weights_%d' % i] #75*75
            u_w = self._multiply_inputs_weights(u_inputs, self.vars['weights_%d' % i]) #53*75
            x = tf.reduce_sum(tf.multiply(u_w, v_inputs), axis=1) #(165,)
            """
            coords = [self.u_indices, self.v_indices]
            coords = tf.stack((self.u_indices, self.v_indices), axis=1)
            #coefs = tf.sparse_to_dense(coords,[self.num_users,self.num_items],x)#tf.math.exp(x))
            coefs = tf.SparseTensor(indices=coords,
                                    values=x,#tf.math.exp(x),
                                    dense_shape=[self.num_users,self.num_items])

            #coefs = tf.math.exp(coefs)
            row_sum = tf.sparse.reduce_sum(coefs, axis=1)
            column_sum = tf.sparse.reduce_sum(coefs, axis=0)
            row_sum2 = tf.gather(row_sum, self.u_indices)  # 53*75
            column_sum2 = tf.gather(column_sum, self.v_indices)
            contradict = tf.add(row_sum2, column_sum2)#*weight_variable_random_uniform(1, 1, name="weights_con")
            norm_out = tf.subtract(contradict, 2*coefs.values)/10
            res = tf.SparseTensor(indices=coords,
                                    values=coefs.values,#-tf.to_float(norm_out),
                                    dense_shape=[self.num_users,self.num_items])
            """
            """
            row_sum2 = tf.gather(row_sum, self.u_indices)  # 53*75
            column_sum2 = tf.gather(column_sum, self.v_indices)
            
            row_sum = tf.reshape(row_sum, (self.num_users, 1))
            column_sum = tf.reshape(column_sum, (self.num_items, 1))
            one_s = tf.contrib.layers.dense_to_sparse(tf.ones([self.num_users, self.num_items]))
            row_sum2 = one_s * row_sum
            column_sum2 = one_s * tf.transpose(column_sum, [1, 0])

            contradict = tf.sparse.add(row_sum2, column_sum2)
            norm_out = tf.SparseTensor(indices=coords,
                                    values= tf.gather_nd(contradict.values, coords)-coefs.values,
                                    dense_shape=[self.num_users,self.num_items])
            res = tf.SparseTensor(indices=coords,
                                    values=coefs.values/norm_out.values,
                                    dense_shape=[self.num_users,self.num_items])
            """
            #norm_out = contradict-coefs
            #out_tensor = tf.math.divide(coefs,norm_out)

            #res = tf.gather_nd(coefs, coords)

            basis_outputs.append(x)#res.values)


        #u_w = self._multiply_inputs_weights(u_inputs, self.vars['weights_%d' % 0]) #53*75
        #x = tf.reduce_sum(tf.multiply(u_w, v_inputs), axis=1) #(165,)

        #coefs = tf.SparseTensor(indices=coords,
        #                        values=x,
        #                        dense_shape=[self.num_users,self.num_items])
        #norm_out = coefs/(row_sum+)
        #basis_outputs.append(x)


        # Store outputs in (Nu x Nv) x num_classes tensor and apply activation function
        basis_outputs = tf.stack(basis_outputs, axis=1) #53*2
        #self.vars['weights_scalars'] 2*5 this is the implementation trick, should have five weights.
        outputs = tf.matmul(basis_outputs,  self.vars['weights_scalars'], transpose_b=False)

        if self.user_item_bias: #false
            outputs += u_bias
            outputs += v_bias

        outputs = self.act(outputs)

        return outputs,self.debug_value,final_u,final_v,x,x,x,x,x#coefs,row_sum,row_sum2,column_sum2,coords

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs_u', inputs[0])
                tf.summary.histogram(self.name + '/inputs_v', inputs[1])

            outputs,temp,u_w,v_inputs,u_v,row_sum,column_sum,norm_out,out_tensor = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs,temp,u_w,v_inputs,u_v,row_sum,column_sum,norm_out,out_tensor
