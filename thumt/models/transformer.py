# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import thumt.layers as layers
import thumt.losses as losses
import thumt.utils as utils

from thumt.models.model import NMTModel


def _layer_process(x, mode):
    if not mode or mode == "none":
        return x
    elif mode == "layer_norm":
        return layers.nn.layer_norm(x)
    else:
        raise ValueError("Unknown mode %s" % mode)


def _residual_fn(x, y, keep_prob=None):
    if keep_prob and keep_prob < 1.0:
        y = tf.nn.dropout(y, keep_prob)
    return x + y


def _ffn_layer(inputs, hidden_size, output_size, keep_prob=None,
               dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = layers.nn.linear(inputs, hidden_size, True, True)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output = layers.nn.linear(hidden, output_size, True, True)

        return output


def transformer_encoder(inputs, bias, params, dtype=None, scope=None, get_all_layer=False,
                                memory_src=None, mem_bias_src=None):
    with tf.variable_scope(scope, default_name="encoder", dtype=dtype,
                           values=[inputs, bias]):
        x = inputs
        all_layer_outputs = []
        n = params.sc_num
        for layer in range(params.num_encoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    max_relative_dis = params.max_relative_dis \
                        if params.position_info_type == 'relative' else None

                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        max_relative_dis=max_relative_dis,
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                if scope == "mt_encoder":
                    with tf.variable_scope("srcmt_attention"):
                        y = layers.attention.multihead_attention(
                            _layer_process(x, params.layer_preprocess),
                            tf.tile(memory_src, [n, 1, 1]),
                            tf.tile(mem_bias_src, [n, 1, 1, 1]),
                            params.num_heads,
                            params.attention_key_channels or params.hidden_size,
                            params.attention_value_channels or params.hidden_size,
                            params.hidden_size,
                            1.0 - params.attention_dropout,
                        )
                        y = y["outputs"]
                        x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                        x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)
                    all_layer_outputs.append(x)
        
        if not get_all_layer:
            outputs = _layer_process(x, params.layer_preprocess)
            return outputs
        else:
            all_layer_outputs = [_layer_process(x, params.layer_preprocess) for x in all_layer_outputs]
            return all_layer_outputs


def transformer_decoder(inputs, memory_src, memory_mt, bias, mem_bias_src, mem_bias_mt, similarity, params, state=None,
                        dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="decoder", dtype=dtype,
                           values=[inputs, memory_src, memory_mt, bias, mem_bias_src, mem_bias_mt, similarity]):
        x = inputs
        n = params.sc_num
        bs = tf.shape(x)[0]
        lq = tf.shape(x)[1]
        hs = params.hidden_size
        next_state = {}
        for layer in range(params.num_decoder_layers):
            layer_name = "layer_%d" % layer
            with tf.variable_scope(layer_name):
                layer_state = state[layer_name] if state is not None else None
                max_relative_dis = params.max_relative_dis \
                        if params.position_info_type == 'relative' else None

                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        state=layer_state,
                        max_relative_dis=max_relative_dis,
                    )

                    if layer_state is not None:
                        next_state[layer_name] = y["state"]

                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("srcdec_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        memory_src,
                        mem_bias_src,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("mtdec_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        memory_mt,
                        mem_bias_mt,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        similarity=similarity[:,layer,:,:]
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        y = _ffn_layer(
            _layer_process(x, params.layer_preprocess),
            params.filter_size,
            params.hidden_size,
            1.0 - params.relu_dropout,
            scope="gen_ffn",
        )
        outputs_gen = _residual_fn(x, y, 1.0 - params.residual_dropout)
        outputs_gen = _layer_process(outputs_gen, params.layer_postprocess)
        outputs_gen = _layer_process(outputs_gen, params.layer_preprocess)

        y = _ffn_layer(
            _layer_process(x, params.layer_preprocess),
            params.filter_size,
            params.hidden_size,
            1.0 - params.relu_dropout,
            scope="copy_ffn",
        )
        outputs_copy = _residual_fn(x, y, 1.0 - params.residual_dropout)
        outputs_copy = _layer_process(outputs_copy, params.layer_postprocess)
        outputs_copy = _layer_process(outputs_copy, params.layer_preprocess)

        if state is not None:
            return outputs_gen, outputs_copy, next_state

        return outputs_gen, outputs_copy


def encoding_graph(features, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    dtype = tf.get_variable_scope().dtype
    hidden_size = params.hidden_size
    src_seq = features["source"]
    src_len = features["source_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)

    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        src_embedding = tf.get_variable("weights",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)
    else:
        src_embedding = tf.get_variable("source_embedding",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)

    bias = tf.get_variable("src_bias", [hidden_size])

    inputs = tf.gather(src_embedding, src_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        inputs = inputs * (hidden_size ** 0.5)

    inputs = inputs * tf.expand_dims(src_mask, -1)

    encoder_input = tf.nn.bias_add(inputs, bias)
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking",
                                                    dtype=dtype)
    if params.position_info_type == 'absolute':
        encoder_input = layers.attention.add_timing_signal(encoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)

    encoder_output = transformer_encoder(encoder_input, enc_attn_bias, params, scope="src_encoder")

    return encoder_output


def encoding_graph_mt(encoder_output_src, features, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    n = params.sc_num
    batch_size = tf.shape(features["source"])[0]

    dtype = tf.get_variable_scope().dtype
    hidden_size = params.hidden_size

    src_mask = tf.sequence_mask(features["source_length"],
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)
    src_attn_bias = layers.attention.attention_bias(src_mask, "masking",
                                                    dtype=dtype)

    max_len = 0

    mt_seqs = []
    for i in range(n):
        mt_seqs.append(features["mt_%d" % i])
        max_len = tf.maximum(max_len, tf.shape(mt_seqs[i])[1])

    for i in range(n):
        mt_seqs[i] = tf.concat([mt_seqs[i], tf.zeros([batch_size, max_len-tf.shape(mt_seqs[i])[1]], dtype=tf.int32)], axis=1)

    mt_lens = []
    for i in range(n):
        mt_lens.append(features["mt_length_%d" % i])

    mt_seq = tf.concat(mt_seqs, axis=0)
    mt_len = tf.concat(mt_lens, axis=0)

    mt_mask = tf.sequence_mask(mt_len, maxlen=max_len, dtype=dtype or tf.float32)

    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)

    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("weights",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer)
    else:
        tgt_embedding = tf.get_variable("target_embedding",
                                        [tgt_vocab_size, hidden_size],
                                        initializer=initializer)

    bias = tf.get_variable("mt_bias", [hidden_size])

    inputs = tf.gather(tgt_embedding, mt_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        inputs = inputs * (hidden_size ** 0.5)

    inputs = inputs * tf.expand_dims(mt_mask, -1)

    encoder_input = tf.nn.bias_add(inputs, bias)
    enc_attn_bias = layers.attention.attention_bias(mt_mask, "masking",
                                                    dtype=dtype)
    if params.position_info_type == 'absolute':
        encoder_input = layers.attention.add_timing_signal(encoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)

    all_layer_outputs = transformer_encoder(encoder_input, enc_attn_bias, params, scope="mt_encoder", get_all_layer=True,
                                             memory_src=encoder_output_src, mem_bias_src=src_attn_bias)
    all_layer_outputs = tf.stack(all_layer_outputs, axis=1) # (bs, nl, lk, hs)

    #
    mt_seq = tf.reshape(mt_seq, [n, batch_size, max_len])
    mt_seq = tf.transpose(mt_seq, [1, 0, 2])
    mt_seq = tf.reshape(mt_seq, [batch_size, n*max_len])

    all_layer_outputs = tf.reshape(all_layer_outputs, [n, batch_size, params.num_encoder_layers, max_len, hidden_size])
    all_layer_outputs = tf.transpose(all_layer_outputs, [1, 2, 0, 3, 4])
    all_layer_outputs = tf.reshape(all_layer_outputs, [batch_size, params.num_encoder_layers, n*max_len, hidden_size])
    encoder_output = all_layer_outputs[:,-1,:,:] # [bs, n*lk, hs]
    
    mt_mask = tf.reshape(mt_mask, [n, batch_size, max_len])
    mt_mask = tf.transpose(mt_mask, [1, 0, 2])
    mt_mask = tf.reshape(mt_mask, [batch_size, n*max_len])
    enc_attn_bias = layers.attention.attention_bias(mt_mask, "masking", dtype=dtype)  # (bs, 1, 1, lk)

    dot_product = tf.matmul(all_layer_outputs, all_layer_outputs, transpose_b=True) # (bs, nl, lk, lk)
    dot_sim = dot_product * (hidden_size ** -0.5) # (bs, nl, lk, lk)

    dot_sim = dot_sim + enc_attn_bias
    dot_sim = tf.reshape(dot_sim, [batch_size, params.num_encoder_layers, n*max_len, n, max_len])
    dot_sim = tf.nn.softmax(dot_sim, axis=-1)
    dot_sim = tf.reshape(dot_sim, [batch_size, params.num_encoder_layers, n*max_len, n*max_len]) # (bs, nl, lk, lk)
    similarity = dot_sim

    tag_vector = tf.concat([tf.ones([max_len, 1])*i for i in range(n)], axis=0) # (lk, 1)
    mt_mt_mask = tag_vector - tf.transpose(tag_vector) # (lk, lk)
    mt_mt_mask = tf.cast(tf.not_equal(mt_mt_mask, 0), tf.float32) # (lk, lk)
    mt_mt_mask = tf.expand_dims(mt_mt_mask, axis=0) # (1, lk, lk)
    mt_mt_mask = mt_mt_mask * tf.expand_dims(mt_mask, axis=1) # (bs, lk, lk)
    mt_mt_mask = tf.expand_dims(mt_mt_mask, axis=1) # (bs, 1, lk, lk)

    similarity = similarity * mt_mt_mask # (bs, nl, lk, lk)
    #

    return mt_seq, encoder_output, enc_attn_bias, similarity


def decoding_graph(features, state, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    n = params.sc_num
    dtype = tf.get_variable_scope().dtype
    tgt_seq = features["target"]
    src_len = features["source_length"]
    tgt_len = features["target_length"]
    bs = tf.shape(features["target"])[0]
    lq = tf.shape(features["target"])[1]
    
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)
    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=dtype or tf.float32)

    hidden_size = params.hidden_size
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)

    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("weights",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer)
    else:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("target_embedding",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer)

    if params.shared_embedding_and_softmax_weights:
        weights = tgt_embedding
    else:
        weights = tf.get_variable("softmax", [tgt_vocab_size, hidden_size],
                                  initializer=initializer)

    targets = tf.gather(tgt_embedding, tgt_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        targets = targets * (hidden_size ** 0.5)

    targets = targets * tf.expand_dims(tgt_mask, -1)

    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking",
                                                    dtype=dtype)
    dec_attn_bias = layers.attention.attention_bias(tf.shape(targets)[1],
                                                    "causal", dtype=dtype)
    # Shift left
    decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    if params.position_info_type == 'absolute':
        decoder_input = layers.attention.add_timing_signal(decoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        decoder_input = tf.nn.dropout(decoder_input, keep_prob)

    encoder_output_src = state["encoder_src"]
    encoder_output_mt = state["encoder_mt"]
    mt_attn_bias = state["mt_attn_bias"]
    similarity = state["similarity"]
    mt_seq = state["mt_seq"] # [bs, n*lm]
    mt_seq_one_hot = tf.one_hot(mt_seq, tgt_vocab_size, dtype=tf.float32) # [bs, n*lm, vocab]
    mt_cnt_vocab = tf.reduce_sum(mt_seq_one_hot, axis=1) / n # [bs, vocab]
    pad_mask = tf.pad(tf.ones([1, tgt_vocab_size-1]), [[0, 0], [1, 0]])
    mt_cnt_vocab *= pad_mask # [bs, vocab] set count of <pad> to 0
    mt_vocab = tf.reduce_max(mt_seq_one_hot, axis=1, keepdims=True) # [bs, 1, vocab]
    mt_vocab_bias = (1-mt_vocab) * (-1e14) # [bs, 1, vocab]
    zero_bias = tf.zeros_like(mt_vocab_bias) # [bs, 1, vocab]
    total_bias = tf.concat([zero_bias, mt_vocab_bias], axis=0) # [2*bs, 1, vocab]

    if mode != "infer":
        decoder_output_gen, decoder_output_copy = transformer_decoder(decoder_input, encoder_output_src, encoder_output_mt,
                                             dec_attn_bias, enc_attn_bias, mt_attn_bias, similarity,
                                             params)
    else:
        decoder_input = decoder_input[:, -1:, :]
        dec_attn_bias = dec_attn_bias[:, :, -1:, :]
        decoder_outputs = transformer_decoder(decoder_input, encoder_output_src, encoder_output_mt,
                                              dec_attn_bias, enc_attn_bias, mt_attn_bias, similarity,
                                              params, state=state["decoder"])

        decoder_output_gen, decoder_output_copy, decoder_state = decoder_outputs 
        decoder_output_gen = decoder_output_gen[:, -1, :]
        decoder_output_copy = decoder_output_copy[:, -1, :]
        total_bias = total_bias[:, -1, :]

        decoder_output = tf.concat([decoder_output_gen, decoder_output_copy], axis=0)
        logits = tf.matmul(decoder_output, weights, False, True) # [2*bs, vocab]
        prob = tf.nn.softmax(logits+total_bias)
        full_prob, shrink_prob = tf.split(prob, 2, axis=0)
        
        tgt_seq_one_hot = tf.one_hot(tgt_seq, tgt_vocab_size) # [bs, lq, vocab]
        tgt_cnt_vocab = tf.reduce_sum(tgt_seq_one_hot, axis=1) # [bs, vocab]
        diff_cnt_vocab = mt_cnt_vocab - tgt_cnt_vocab # [bs, vocab]
        diff_cnt_vocab = tf.maximum(diff_cnt_vocab, 0) # [bs, vocab]

        gate = layers.nn.linear(decoder_output_gen, 1, False, True, scope='prob_gate1') +\
               layers.nn.linear(decoder_output_copy, 1, True, True, scope='prob_gate2') # [bs, 1]
        gate = tf.sigmoid(gate)

        prob = gate * full_prob + (1-gate) * shrink_prob

        prob *= tf.log(2+diff_cnt_vocab) / tf.log(2.0)
        prob /= tf.reduce_sum(prob, axis=-1, keepdims=True)

        log_prob = tf.log(prob)

        state = {"mt_seq": mt_seq,
                "encoder_src": encoder_output_src,
                "encoder_mt": encoder_output_mt,
                "mt_attn_bias": mt_attn_bias,
                "similarity": similarity,
                "decoder": decoder_state}

        return log_prob, state

    decoder_output_gen = tf.reshape(decoder_output_gen, [-1, hidden_size])
    decoder_output_copy = tf.reshape(decoder_output_copy, [-1, hidden_size])
    decoder_output = tf.concat([decoder_output_gen, decoder_output_copy], axis=0)
    logits = tf.matmul(decoder_output, weights, False, True) # [2*bs*lq, vocab]

    logits = tf.reshape(logits, [2*bs, lq, tgt_vocab_size]) # [2*bs, lq, vocab]
    prob = tf.nn.softmax(logits + total_bias)
    full_prob, shrink_prob = tf.split(prob, 2, axis=0) # [bs, lq, vocab]
    
    full_prob = tf.reshape(full_prob, [-1, tgt_vocab_size]) # [bs*lq, vocab]
    shrink_prob = tf.reshape(shrink_prob, [-1, tgt_vocab_size]) # [bs*lq, vocab]

    gate = layers.nn.linear(decoder_output_gen, 1, False, True, scope='prob_gate1') +\
           layers.nn.linear(decoder_output_copy, 1, True, True, scope='prob_gate2') # [bs*lq, 1]
    gate = tf.sigmoid(gate)

    prob = gate * full_prob + (1-gate) * shrink_prob
    log_prob = tf.log(prob)

    labels = features["target"]

    # label smoothing
    ce = losses.smoothed_softmax_cross_entropy_with_logits(
        logits=log_prob,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )
    tgt_mask = tf.cast(tgt_mask, ce.dtype)

    ce = tf.reshape(ce, tf.shape(tgt_seq))

    if mode == "eval":
        return -tf.reduce_sum(ce * tgt_mask, axis=1)

    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    return loss


def model_graph(features, mode, params):
    encoder_output_src = encoding_graph(features, mode, params)
    mt_seq, encoder_output_mt, mt_attn_bias, similarity = encoding_graph_mt(encoder_output_src, features, mode, params)
    state = {
        "mt_seq": mt_seq,
        "encoder_src": encoder_output_src,
        "encoder_mt": encoder_output_mt,
        "mt_attn_bias": mt_attn_bias,
        "similarity": similarity,
    }
    output = decoding_graph(features, state, mode, params)
    return output


class Transformer(NMTModel):

    def __init__(self, params, scope="transformer"):
        super(Transformer, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer, regularizer=None, dtype=None):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            custom_getter = utils.custom_getter if dtype else None

            with tf.variable_scope(self._scope, initializer=initializer,
                                   regularizer=regularizer, reuse=reuse,
                                   custom_getter=custom_getter, dtype=dtype):
                loss = model_graph(features, "train", params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                score = model_graph(features, "eval", params)

            return score

        return evaluation_fn

    def get_inference_func(self):
        def encoding_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                encoder_output_src = encoding_graph(features, "infer", params)
                mt_seq, encoder_output_mt, mt_attn_bias, similarity =\
                                encoding_graph_mt(encoder_output_src, features, "infer", params)
                batch = tf.shape(encoder_output_src)[0]

                state = {
                    "mt_seq": mt_seq,
                    "encoder_src": encoder_output_src,
                    "encoder_mt": encoder_output_mt,
                    "mt_attn_bias": mt_attn_bias,
                    "similarity": similarity,
                    "decoder": {
                        "layer_%d" % i: {
                            "key": tf.zeros([batch, 0, params.attention_key_channels or params.hidden_size]),
                            "value": tf.zeros([batch, 0, params.attention_value_channels or params.hidden_size])
                        }
                        for i in range(params.num_decoder_layers)
                    }
                }
            return state

        def decoding_fn(features, state, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                log_prob, new_state = decoding_graph(features, state, "infer",
                                                     params)

            return log_prob, new_state

        return encoding_fn, decoding_fn

    @staticmethod
    def get_name():
        return "transformer"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            append_eos=False,
            hidden_size=512,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            layer_preprocess="none",
            layer_postprocess="layer_norm",
            multiply_embedding_mode="sqrt_depth",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # Override default parameters
            learning_rate_decay="linear_warmup_rsqrt_decay",
            initializer="uniform_unit_scaling",
            initializer_gain=1.0,
            learning_rate=1.0,
            batch_size=4096,
            constant_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0,
            # "absolute" or "relative"
            position_info_type="absolute",
            # 8 for big model, 16 for base model, see (Shaw et al., 2018)
            max_relative_dis=16,
            # Number of system to be combined
            sc_num=3,
        )

        return params
