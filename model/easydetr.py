from resnet import resnet

from keras.layers import Input, Conv2D, Embedding, Dropout, add, Reshape, Dense, Lambda, ReLU
from keras.models import Model
import tensorflow as tf
import keras.backend as K
import numpy as np
import math


def MYdetr(input_shape=(512,512,3), pe='sine', n_classes=80, depth=50, dilation=False,
         emb_dim=256, enc_layers=6, dec_layers=6, max_boxes=36, max_target_boxes=20,
         mlp_dim=256, mode='test'):

    # inpt
    feat = Input(input_shape)    # [b,h,w,c]
    #mask = Input((input_shape[0],input_shape[1],1))   # [b,h,w,1]

    # reflect & pe
    x = Conv2D(emb_dim, 1, strides=1, padding='same', name='input_proj')(feat)   # [b,h,w,d]
    x = Reshape((feature_shape[0]*feature_shape[1], emb_dim))(x)                 # [b,hw,d]
    feat_pe = None
    # transformer encoder: parse inputs
    for i in range(enc_layers):
        x = TransformerEncoderBlock(drop_rate=0.1)([x, feat_pe])      # [b,hw,d]
    encoder_feats = x

    # transformer decoder: feed targets x is a zeros-variable initially, get updated through the decoder blocks
    x, target_pe = PrepareDecoderInput(max_boxes, emb_dim)(x)
    print('decoder pe', target_pe.shape)

    for i in range(dec_layers):
        x = TransformerDecoderBlock(drop_rate=0.1)([x, target_pe, encoder_feats, feat_pe])
    x = LayerNormalization()(x)    # norm no matter pre_norm/post_norm

    box = Dense(mlp_dim, activation='relu', name='box_hidden_1')(x)
    box = Dense(mlp_dim, activation='relu', name='box_hidden_2')(box)
    box_output = Dense(4, activation='sigmoid', name='box_pred')(box)    # [b,N2,4]

    model = Model([inpt], [cls_output,box_output])
    
    return model

class PrepareDecoderInput(Model):
    def __init__(self, max_boxes=100, emb_dim=256, name='query_embed'):
        super(PrepareDecoderInput, self).__init__(name=name)
        self.emb = Embedding(max_boxes, emb_dim)
        self.max_boxes = max_boxes
        self.emb_dim = emb_dim

        # create variables
        self.target_indices = tf.expand_dims(tf.range(self.max_boxes), axis=0)  # [1,N1]
        self.target = tf.zeros((1, max_boxes, emb_dim))     # [1,N1,d], zeros

    def call(self, x):
        # x: encoder feats: [b,hw,d]
        b = tf.shape(x)[0]
        target_indices = tf.tile(self.target_indices, [b,1])    # [b,N1]
        target_pe = self.emb(target_indices)     # [b,N1,d], take corresponding word-vecs
        target = tf.tile(self.target, [b,1,1])   # [b,N1,d]
        return [target, target_pe]

    def compute_output_shape(self, input_shape):
        b, N, d = input_shape
        return [(b,self.max_boxes,d), (b,self.max_boxes,d)]

class TransformerEncoderBlock(Model):
    def __init__(self, attn_dim=256, ffn_dim=2048, drop_rate=0.1, norm_before=False):
        super(TransformerEncoderBlock, self).__init__()

        self.ln1 = LayerNormalization()
        self.msa = MultiHeadAttention(num_heads=8, model_size=attn_dim, name='self_att')
        self.drop1 = Dropout(drop_rate)

        self.ln2 = LayerNormalization()
        self.dense1 = Dense(ffn_dim)
        self.act1 = ReLU()
        self.dense2 = Dense(attn_dim)
        self.drop2 = Dropout(drop_rate)
        self.drop3 = Dropout(drop_rate)

        self.norm_before = norm_before

    def call(self, inputs, mask=None):
        if self.norm_before:
            return self.pre_norm(inputs, mask)
        else:
            return self.post_norm(inputs, mask)

    def pre_norm(self, inputs, mask=None):
        x, pe = inputs

        # id path
        inpt = x

        # residual path
        x = self.ln1(x)
        q = k = x + pe
        v = x
        x = self.msa([q,k,v], mask=mask)
        x = self.drop1(x)

        # add
        x = x + inpt

        # id path
        ffn_inpt = x

        # residual path
        x = self.ln2(x)
        x = self.drop2(self.act1(self.dense1(x)))
        x = self.drop3(self.dense2(x))

        # add
        x = x + ffn_inpt

        return x

    def post_norm(self, inputs, mask=None):
        x, pe = inputs

        # id path
        inpt = x

        # residual path
        q = k = x + pe
        v = x
        x = self.msa([q,k,v], mask=mask)
        x = self.drop1(x)

        # add
        x = x + inpt
        x = self.ln1(x)

        # id path
        ffn_inpt = x

        # residual path
        x = self.drop2(self.act1(self.dense1(x)))
        x = self.drop3(self.dense2(x))

        # add
        x = x + ffn_inpt
        x = self.ln2(x)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class TransformerDecoderBlock(Model):
    def __init__(self, attn_dim=256, ffn_dim=2048, drop_rate=0.1, norm_before=False):
        super(TransformerDecoderBlock, self).__init__()

        self.ln1 = LayerNormalization()
        self.msa1 = MultiHeadAttention(num_heads=8, model_size=attn_dim, name='self_att')   # self-att
        self.drop1 = Dropout(drop_rate)

        self.ln2 = LayerNormalization()
        self.msa2 = MultiHeadAttention(num_heads=8, model_size=attn_dim, name='mutual_att')   # mutual-att
        self.drop2 = Dropout(drop_rate)

        self.ln3 = LayerNormalization()
        self.dense1 = Dense(ffn_dim)
        self.act1 = ReLU()
        self.dense2 = Dense(attn_dim)
        self.drop3 = Dropout(drop_rate)
        self.drop4 = Dropout(drop_rate)

        self.norm_before = norm_before

    def call(self, inputs, mask=None, key_mask=None):
        # targets: decoder input
        # inputs: encoder output
        if self.norm_before:
            return self.pre_norm(inputs, key_mask=key_mask)
        else:
            return self.post_norm(inputs, key_mask=key_mask)

    def pre_norm(self, inputs, key_mask=None):
        x, target_pe, inputs, feat_pe = inputs

        # id path
        inpt = x

        # residual path
        x = self.ln1(x)
        q = k = x + target_pe
        v = x
        x = self.msa1([q,k,v])
        x = self.drop1(x)

        # add
        x = x + inpt

        # id path
        inpt = x

        # residual path
        x = self.ln2(x)
        q = x + target_pe
        k = inputs + feat_pe
        v = inputs
        x = self.msa2([q,k,v], key_mask=key_mask)
        x = self.drop2(x)

        # add
        x = x + inpt

        # id path
        ffn_inpt = x

        # residual path
        x = self.ln3(x)
        x = self.drop3(self.act1(self.dense1(x)))
        x = self.drop4(self.dense2(x))

        # add
        x = x + ffn_inpt

        return x

    def post_norm(self, inputs, key_mask=None):
        x, target_pe, inputs, feat_pe = inputs

        # id path
        inpt = x

        # residual path
        q = k = x + target_pe
        v = x
        x = self.msa1([q,k,v])
        x = self.drop1(x)

        # add
        x = x + inpt
        x = self.ln1(x)

        # id path
        inpt = x

        # residual path
        q = x + target_pe
        k = inputs + feat_pe
        v = inputs
        x = self.msa2([q,k,v], key_mask=key_mask)
        x = self.drop2(x)

        # add
        x = x + inpt
        x = self.ln2(x)

        # id path
        ffn_inpt = x

        # residual path
        x = self.drop3(self.act1(self.dense1(x)))
        x = self.drop4(self.dense2(x))

        # add
        x = x + ffn_inpt
        x = self.ln3(x)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class LayerNormalization(Layer):

    # given inputs: [b,(hwd),c], for each sample, compute norm over the c-dim

    def __init__(self,
                 rescale=True,
                 epsilon=1e-5,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        if epsilon is None:
            epsilon = K.epsilon()
        self.rescale=rescale
        self.epsilon = epsilon
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def build(self, input_shape):
        # rescale factor, for each sample, broadcast from last-dim
        shape = (input_shape[-1], )
        if self.rescale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)    # (b,(hwd),1)
        variance = K.var(inputs, axis=-1, keepdims=True)
        # norm
        outputs = (inputs - mean) / K.sqrt(variance + self.epsilon)
        # rescale
        outputs = self.gamma*outputs + self.beta
        return outputs

    def get_config(self):
        config = {
            'epsilon': self.epsilon,
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'beta_constraint': constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

# MSA layer
class MultiHeadAttention(Model):
    def __init__(self, model_size, num_heads, attn_drop=0.1, ffn_drop=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.model_size = model_size
        self.num_heads = num_heads
        self.head_size = model_size // num_heads
        self.WQ = Dense(model_size, name='dense_q')   # [b,Nq,d]
        self.WK = Dense(model_size, name='dense_k')   # [b,Nk,d]
        self.WV = Dense(model_size, name='dense_v')   # [b,Nv,d], Nk=Nv
        self.dense = Dense(model_size)
        self.msa_drop = Dropout(attn_drop)
        self.mlp_drop = Dropout(ffn_drop)

    def call(self, inputs, mask=None, key_mask=None):
        # query: (batch, maxlen, model_size)
        # key  : (batch, maxlen, model_size)
        # value: (batch, maxlen, model_size)

        query, key, value = inputs
        batch_size = tf.shape(query)[0]

        # in_proj: shape: (batch, maxlen, model_size)
        query = self.WQ(query)
        key = self.WK(key)
        value = self.WV(value)

        def _split_heads(x):
            x = tf.reshape(x, shape=[batch_size, x.shape[1], self.num_heads, self.head_size])
            return tf.transpose(x, perm=[0, 2, 1, 3])

        # shape: (batch, num_heads, maxlen, head_size)
        query = _split_heads(query)
        key = _split_heads(key)
        value = _split_heads(value)

        # shape: (batch, num_heads, maxlen, maxlen)
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        # 缩放 matmul_qk
        dk = tf.cast(query.shape[-1], tf.float32)
        score = matmul_qk / tf.math.sqrt(dk)      # [b,h,N1,N2]

        if mask is not None:
            # print('mask', mask, score)
            mask = K.expand_dims(mask, axis=1)
            mask = K.expand_dims(mask, axis=3)
            mask = tf.tile(mask, [1,self.num_heads,1,int(key.shape[2])])
            score += mask * -1e9     # add mask=1 points with -inf, results in 0 in softmax
        if key_mask is not None:
            # print('key_mask', key_mask, score)
            key_mask = K.expand_dims(key_mask, axis=1)
            key_mask = K.expand_dims(key_mask, axis=2)
            key_mask = tf.tile(key_mask, [1,self.num_heads,int(query.shape[2]),1])
            score += key_mask * -1e9

        # softmax & dropout
        alpha = tf.nn.softmax(score)    # [b,Nq,Nk]
        alpha = self.msa_drop(alpha)

        context = tf.matmul(alpha, value)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, query.shape[2], self.model_size))
        output = self.dense(context)
        output = self.mlp_drop(output)

        return output

    def compute_output_shape(self, input_shape):
        B, N, _ = input_shape[0]
        return (B,N,self.model_size)



if __name__ == '__main__':

    pe = PositionalEmbeddingSine(256, (25,38), temp=10000, normalize=True, eps=1e-6)
    print(pe)

    pe_layer = PositionalEmbeddingLearned(128, (8,10))
    x = tf.ones((32,32))
    y = pe_layer(x)
    print(pe_layer.weights)  # The first call will create the weights

    model = detr(input_shape=(512,512,3), pe='learned', n_classes=92, depth=50, dilation=False,
                 emb_dim=256, enc_layers=6, dec_layers=6, max_boxes=100,
                 mode='train')
    model.summary()
    model.load_weights("weights/detr-r50.h5")

    # for l in model.layers:
    #     if 'decoderblock_1' in l.name:
    #         print(l.weights)




