from lib2to3.pytree import convert
import tensorflow as tf
from tensorflow.python.ops.math_ops import _bucketize

from ...feature_column import build_input_features, input_from_feature_columns
from ...layers.core import PredictionLayer, DNN
from ...layers.utils import combined_dnn_input, reduce_sum


def AUTOFUSE(user_feature_columns, coarse_feature_columns, fine_feature_columns, pop_feature_columns, label_feature_columns, pop_vocab_size, tower_dnn_hidden_units=(256, 128, 64),
         l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu',
         dnn_use_bn=False, pop_boundaries = [0,2,5,10,12,15,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000]):
    """Instantiates the AutoFuse model architecture.
    :param user_feature_columns: An iterable containing all the user-side features used by the model.
    :param coarse_feature_columns: An iterable containing all the coarse-grained features used by the model.
    :param fine_feature_columns: An iterable containing all the fine-grained features used by the model.
    :param pop_feature_columns: An iterable containing all the popularity features used by the model.
    :param label_feature_columns: An iterable containing all the popularity features used by the model.
    :param pop_vocab_size: Integer, the number of all product IDs (Excluding duplicate items)
    :param tower_dnn_hiddern_units: list, list of positive integer or empty list, the layer number and units in each layer of task-specific DNN.
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN.
    :param seed: integer, to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: String, string of ativation function to use in DNN.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN.
    :param pop_boundaries:  list, split the popularity into several buckets based on numerical ranges and assign a trainable embedding to symbolize each bucket.
    :return: a Keras model instance
    """
 
    fine_features = build_input_features(fine_feature_columns)
    coarse_features = build_input_features(coarse_feature_columns)
    user_features = build_input_features(user_feature_columns)
    pop_features = build_input_features(pop_feature_columns)
    label_features = build_input_features(label_feature_columns)

    inputs_list = list(fine_features.values())+list(coarse_features.values())+list(user_features.values())+list(pop_features.values())

    sparse_fine_embedding_list, dense_value_list = input_from_feature_columns(fine_features, fine_feature_columns,
                                                                         l2_reg_embedding, seed)
    sparse_coarse_embedding_list, dense_coarse_value_list = input_from_feature_columns(coarse_features, coarse_feature_columns,
                                                                         l2_reg_embedding, seed)
    sparse_user_embedding_list, dense_user_value_list = input_from_feature_columns(user_features, user_feature_columns,
                                                                         l2_reg_embedding, seed)
    sparse_pop_embedding_list, dense_pop_value_list = input_from_feature_columns(pop_features, pop_feature_columns,
                                                                         l2_reg_embedding, seed)
    sparse_label_embedding_list, dense_label_value_list = input_from_feature_columns(label_features, label_feature_columns,
                                                                         l2_reg_embedding, seed)

    dnn_input = combined_dnn_input(sparse_fine_embedding_list + sparse_coarse_embedding_list, dense_value_list)
    coarse_input = combined_dnn_input(sparse_coarse_embedding_list, dense_coarse_value_list)
    user_input = combined_dnn_input(sparse_user_embedding_list, dense_user_value_list)
    pop_input = combined_dnn_input(sparse_pop_embedding_list, dense_pop_value_list)
    label_input = combined_dnn_input(sparse_label_embedding_list, dense_label_value_list)

    # pop embedding
    pop_input = tf.cast(pop_input, dtype=tf.int32)
    zeroinit = tf.zeros_initializer()
    cnt_table = tf.compat.v1.get_variable(name = 'cnt_table', shape=[pop_vocab_size, 1], initializer=zeroinit, dtype=tf.int32, trainable=False)
    bucket_table = tf.compat.v1.get_variable(name = 'bucket_table', shape=[pop_vocab_size, 1], initializer=zeroinit, dtype=tf.int32, trainable=False)

    converted_mask = tf.equal(label_input, tf.ones_like(label_input,dtype=tf.float32))
    converted_idx = tf.where(tf.reshape(converted_mask,[-1]))
    converted_idx = tf.reshape(converted_idx,[-1])
    converted = tf.gather(pop_input, converted_idx)
    
    indices_unique, _, indices_cnt = tf.unique_with_counts(tf.reshape(converted,[-1]))
    cnt_update = tf.compat.v1.scatter_add(cnt_table, indices_unique, tf.reshape(indices_cnt,[-1,1]))
    bucketized = _bucketize(cnt_update, boundaries = pop_boundaries)

    bucket_table_update = tf.compat.v1.assign(bucket_table, bucketized)
    indices_bucketized = tf.nn.embedding_lookup(bucket_table_update, pop_input)
    indices_bucketized = tf.squeeze(indices_bucketized, axis=1)

    pop_emb_table = tf.keras.layers.Embedding(len(pop_boundaries)+1, 10)
    pop_emb = pop_emb_table(indices_bucketized)
    pop_emb = tf.squeeze(pop_emb, axis=1) 

    # Group Level
    tower_output_group = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                           name='tower_group')(coarse_input)
    
    # Ad Level, input "coarse_input" and "fine_input" and "popularity_input"
    tower_output_ad = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                           name='tower_ad')(dnn_input)

    # fuse gate
    gate_input_fuse_concat = tf.keras.layers.Lambda(lambda x: tf.concat(x,axis=-1))([coarse_input, pop_emb])
    gate_out_fuse = tf.keras.layers.Dense(2, use_bias=False, activation='softmax',
                                         name='fuse_gate_softmax')(gate_input_fuse_concat)
    gate_out_fuse = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out_fuse)

    # Final ad representation
    tower_output = tf.stack([tower_output_ad, tower_output_group], axis=1)
    tower_output = tf.reduce_sum(tower_output * gate_out_fuse, axis=1)

    # User representation
    tower_output_user = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                           name='tower_user')(user_input)

    estimated_outs = tf.multiply(tower_output, tower_output_user)
    estimated_outs_group = tf.multiply(tower_output_group, tower_output_user)
    estimated_outs_ad = tf.multiply(tower_output_ad, tower_output_user)

    estimated_dot_product = tf.reduce_sum(estimated_outs, axis=-1, keepdims=True)
    estimated_dot_product_group = tf.reduce_sum(estimated_outs_group, axis=-1, keepdims=True)
    estimated_dot_product_ad = tf.reduce_sum(estimated_outs_ad, axis=-1, keepdims=True)

    logit = tf.keras.layers.Dense(1, use_bias=False)(estimated_dot_product)
    logit_group = tf.keras.layers.Dense(1, use_bias=False)(estimated_dot_product_group)
    logit_ad = tf.keras.layers.Dense(1, use_bias=False)(estimated_dot_product_ad)

    output = PredictionLayer('binary', name='ctr1')(logit)
    output_group = PredictionLayer('binary', name='ctr2')(logit_group)
    output_ad = PredictionLayer('binary', name='ctr3')(logit_ad)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=[output, output_group, output_ad])
    return model

# test AUC 0.8761
# 0.9076
