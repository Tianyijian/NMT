import paddle
import paddle.fluid as fluid

dict_size = 30000  # 词典大小
source_dict_size = target_dict_size = dict_size  # 源/目标语言字典大小
word_dim = 512  # 词向量维度
hidden_dim = 512  # 编码器中的隐层大小
decoder_size = hidden_dim  # 解码器中的隐层大小
max_length = 256 # 解码生成句子的最大长度
beam_size = 4  # beam search的柱宽度
batch_size = 64  # batch 中的样本数

is_sparse = True  # 代表是否用稀疏更新的标志


# 编码器框架
def encoder():
    # 定义源语言id序列的输入数据
    src_word_id = fluid.layers.data(name='src_word_id', shape=[1], dtype='int64', lod_level=1)
    # 将上述编码映射到低维语言空间的词向量
    src_embedding = fluid.layers.embedding(input=src_word_id, size=[source_dict_size, word_dim], dtype='float32',
                                           is_sparse=is_sparse)
    # 用双向GRU编码源语言序列，拼接两个GRU的编码结果得到h
    fc_forward = fluid.layers.fc(input=src_embedding, size=hidden_dim * 3, bias_attr=False)
    src_forward = fluid.layers.dynamic_gru(input=fc_forward, size=hidden_dim)
    fc_backward = fluid.layers.fc(input=src_embedding, size=hidden_dim * 3, bias_attr=False)
    src_backward = fluid.layers.dynamic_gru(input=fc_backward, size=hidden_dim, is_reverse=True)
    encoded_vector = fluid.layers.concat(input=[src_forward, src_backward], axis=1)
    return encoded_vector


# 基于注意力机制的解码器
# 定义RNN中的单步计算
def cell(x, hidden, encoder_out, encoder_out_proj):
    # 定义attention用以计算context，即 c_i，这里使用Bahdanau attention机制
    def simple_attention(encoder_vec, encoder_proj, decoder_state):
        decoder_state_proj = fluid.layers.fc(input=decoder_state, size=decoder_size, bias_attr=False)
        # sequence_expand将单步内容扩展为与encoder输出相同的序列
        decoder_state_expand = fluid.layers.sequence_expand(x=decoder_state_proj, y=encoder_proj)
        mixed_state = fluid.layers.elementwise_add(encoder_proj, decoder_state_expand)
        attention_weights = fluid.layers.fc(input=mixed_state, size=1, bias_attr=False)
        attention_weights = fluid.layers.sequence_softmax(input=attention_weights)
        weigths_reshape = fluid.layers.reshape(x=attention_weights, shape=[-1])
        scaled = fluid.layers.elementwise_mul(x=encoder_vec, y=weigths_reshape, axis=0)
        context = fluid.layers.sequence_pool(input=scaled, pool_type='sum')
        return context

    context = simple_attention(encoder_out, encoder_out_proj, hidden)
    out = fluid.layers.fc(
        input=[x, context], size=decoder_size * 3, bias_attr=False)
    out = fluid.layers.gru_unit(
        input=out, hidden=hidden, size=decoder_size * 3)[0]
    return out, out


# 基于定义的单步计算，使用DynamicRNN实现多步循环的训练模式下解码器
def train_decoder(encoder_out):
    # 获取编码器输出的最后一步并进行非线性映射以构造解码器RNN的初始状态
    encoder_last = fluid.layers.sequence_last_step(input=encoder_out)
    encoder_last_proj = fluid.layers.fc(
        input=encoder_last, size=decoder_size, act='tanh')
    # 编码器输出在attention中计算结果的cache
    encoder_out_proj = fluid.layers.fc(
        input=encoder_out, size=decoder_size, bias_attr=False)
    # 定义目标语言id序列的输入数据，并映射到低维语言空间的词向量
    trg_language_word = fluid.layers.data(
        name="target_language_word", shape=[1], dtype='int64', lod_level=1)
    trg_embedding = fluid.layers.embedding(
        input=trg_language_word,
        size=[target_dict_size, word_dim],
        dtype='float32',
        is_sparse=is_sparse)

    rnn = fluid.layers.DynamicRNN()
    with rnn.block():
        # 获取当前步目标语言输入的词向量
        x = rnn.step_input(trg_embedding)
        # 获取隐层状态
        pre_state = rnn.memory(init=encoder_last_proj, need_reorder=True)
        # 在DynamicRNN中需使用static_input获取encoder相关的内容
        # 对decoder来说这些内容在每个时间步都是固定的
        encoder_out = rnn.static_input(encoder_out)
        encoder_out_proj = rnn.static_input(encoder_out_proj)
        # 执行单步的计算单元
        out, current_state = cell(x, pre_state, encoder_out, encoder_out_proj)
        # 计算归一化的单词预测概率
        prob = fluid.layers.fc(input=out, size=target_dict_size, act='softmax')
        # 更新隐层状态
        rnn.update_memory(pre_state, current_state)
        # 输出预测概率
        rnn.output(prob)

    return rnn()


# 定义训练网络
def train_model():
    encoder_out = encoder()
    rnn_out = train_decoder(encoder_out)
    label = fluid.layers.data(name="target_language_next_word", shape=[1], dtype='int64', lod_level=1)
    # 定义损失函数
    cost = fluid.layers.cross_entropy(input=rnn_out, label=label)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


# 定义优化器
def optimizer_func():
    # 设置梯度裁剪
    fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0))
    # 定义先增后降的学习率策略 Noam 衰减方法
    lr_decay = fluid.layers.learning_rate_scheduler.noam_decay(hidden_dim, 1000)
    return fluid.optimizer.Adam(learning_rate=lr_decay,
                                regularization=fluid.regularizer.L2DecayRegularizer(regularization_coeff=1e-4))


# 预测模式下基于beam search的解码器
def infer_decoder(encoder_out):
    # 获取编码器输出的最后一步并进行非线性映射以构造解码器RNN的初始状态
    encoder_last = fluid.layers.sequence_last_step(input=encoder_out)
    encoder_last_proj = fluid.layers.fc(
        input=encoder_last, size=decoder_size, act='tanh')
    # 编码器输出在attention中计算结果的cache
    encoder_out_proj = fluid.layers.fc(
        input=encoder_out, size=decoder_size, bias_attr=False)

    # 最大解码步数
    max_len = fluid.layers.fill_constant(shape=[1], dtype='int64', value=max_length)
    # 解码步数计算变量
    counter = fluid.layers.zeros(shape=[1], dtype='int64', force_cpu=True)

    # 定义 tensor array 用以保存各个时间步的内容，并写入初始id，score和state
    init_ids = fluid.layers.data(
        name="init_ids", shape=[1], dtype="int64", lod_level=2)
    init_scores = fluid.layers.data(
        name="init_scores", shape=[1], dtype="float32", lod_level=2)
    ids_array = fluid.layers.array_write(init_ids, i=counter)
    scores_array = fluid.layers.array_write(init_scores, i=counter)
    state_array = fluid.layers.array_write(encoder_last_proj, i=counter)

    # 定义循环终止条件变量
    cond = fluid.layers.less_than(x=counter, y=max_len)
    while_op = fluid.layers.While(cond=cond)
    with while_op.block():
        # 获取解码器在当前步的输入，包括上一步选择的id，对应的score和上一步的state
        pre_ids = fluid.layers.array_read(array=ids_array, i=counter)
        pre_score = fluid.layers.array_read(array=scores_array, i=counter)
        pre_state = fluid.layers.array_read(array=state_array, i=counter)

        # 同train_decoder中的内容，进行RNN的单步计算
        pre_ids_emb = fluid.layers.embedding(
            input=pre_ids,
            size=[target_dict_size, word_dim],
            dtype='float32',
            is_sparse=is_sparse)
        out, current_state = cell(pre_ids_emb, pre_state, encoder_out,
                            encoder_out_proj)
        prob = fluid.layers.fc(
            input=current_state, size=target_dict_size, act='softmax')

        # 计算累计得分，进行beam search
        topk_scores, topk_indices = fluid.layers.topk(prob, k=beam_size)
        accu_scores = fluid.layers.elementwise_add(
            x=fluid.layers.log(topk_scores),
            y=fluid.layers.reshape(pre_score, shape=[-1]),
            axis=0)
        accu_scores = fluid.layers.lod_reset(x=accu_scores, y=pre_ids)
        selected_ids, selected_scores = fluid.layers.beam_search(
            pre_ids, pre_score, topk_indices, accu_scores, beam_size, end_id=1)

        fluid.layers.increment(x=counter, value=1, in_place=True)
        # 将 search 结果写入 tensor array 中
        fluid.layers.array_write(selected_ids, array=ids_array, i=counter)
        fluid.layers.array_write(selected_scores, array=scores_array, i=counter)
        # sequence_expand 作为 gather 使用以获取search结果对应的状态，并更新
        current_state = fluid.layers.sequence_expand(current_state,
                                                     selected_ids)
        fluid.layers.array_write(current_state, array=state_array, i=counter)
        current_enc_out = fluid.layers.sequence_expand(encoder_out,
                                                       selected_ids)
        fluid.layers.assign(current_enc_out, encoder_out)
        current_enc_out_proj = fluid.layers.sequence_expand(
            encoder_out_proj, selected_ids)
        fluid.layers.assign(current_enc_out_proj, encoder_out_proj)

        # 更新循环终止条件
        length_cond = fluid.layers.less_than(x=counter, y=max_len)
        finish_cond = fluid.layers.logical_not(
            fluid.layers.is_empty(x=selected_ids))
        fluid.layers.logical_and(x=length_cond, y=finish_cond, out=cond)

    # 根据保存的每一步的结果，回溯生成最终解码结果
    translation_ids, translation_scores = fluid.layers.beam_search_decode(
        ids=ids_array, scores=scores_array, beam_size=beam_size, end_id=1)

    return translation_ids, translation_scores


# 定义预测网络
def infer_model():
    encoder_out = encoder()
    translation_ids, translation_scores = infer_decoder(encoder_out)
    return translation_ids, translation_scores
