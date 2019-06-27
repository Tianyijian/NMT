import paddle
import paddle.fluid as fluid

dict_size = 30000  # 词典大小
source_dict_size = target_dict_size = dict_size  # 源/目标语言字典大小
word_dim = 512  # 词向量维度
hidden_dim = 512  # 编码器中的隐层大小
decoder_size = hidden_dim  # 解码器中的隐层大小

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


