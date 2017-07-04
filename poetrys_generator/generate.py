#coding: utf-8
import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


input_file = 'poetry.txt'

# 诗集
poetrys = []
with open(input_file) as f:
    for line in f:
        try:
            title, content = line.strip().split(':')
            content = content.replace(' ', '')
            if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                continue
            if len(content) < 5 or len(content) > 79:
                continue
            content = '[' + content + ']'
            poetrys.append(content)
        except Exception as e:
            pass

poetrys = sorted(poetrys, key=lambda x: len(x))
all_words = []
for poetry in poetrys:
    all_words += [word for word in poetry]
counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
words, _ = zip(*count_pairs)

# 取前多少个常用字
words = list(words[:len(words)])
words.append(' ')
word_num_map = dict(zip(words, range(len(words))))
# 把诗转换为向量形式，参考TensorFlow练习1
to_num = lambda word: word_num_map.get(word, len(words))
poetrys_vector = [list(map(to_num, poetry)) for poetry in poetrys]


batch_size = 1
n_chunk = len(poetrys_vector) // batch_size
x_batches = []
y_batches = []
for i in range(n_chunk):
    start_index = i * batch_size
    end_index = start_index + batch_size

    batches = poetrys_vector[start_index:end_index]
    length = max(map(len, batches))
    xdata = np.full((batch_size, length), word_num_map[' '], np.int32)
    for row in range(batch_size):
        xdata[row, :len(batches[row])] = batches[row]
    ydata = np.copy(xdata)
    ydata[:, :-1] = xdata[:, 1:]
    """ 
    xdata             ydata 
    [6,2,4,6,9]       [2,4,6,9,9] 
    [1,4,2,8,5]       [4,2,8,5,5] 
    """
    x_batches.append(xdata)
    y_batches.append(ydata)


input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])



def neural_network(model='lstm', rnn_size=128, num_layers=2):
    if model == 'rnn':
        cell_fun = rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = rnn.GRUCell
    elif model == 'lstm':
        cell_fun = rnn.BasicLSTMCell

    cell = rnn.MultiRNNCell([cell_fun(rnn_size, state_is_tuple=True) for _ in range(num_layers)], state_is_tuple=True)

    initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.variable_scope('rnnlm'):
        softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words) + 1])
        softmax_b = tf.get_variable("softmax_b", [len(words) + 1])
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [len(words) + 1, rnn_size])
            inputs = tf.nn.embedding_lookup(embedding, input_data)

    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
    output = tf.reshape(outputs, [-1, rnn_size])

    logits = tf.matmul(output, softmax_w) + softmax_b
    probs = tf.nn.softmax(logits)
    return logits, last_state, probs, cell, initial_state



def gen_poetry(poetry_num=1):
    def to_word(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        sample = int(np.searchsorted(t, np.random.rand(1) * s))
        return words[sample]

    _, last_state, probs, cell, initial_state = neural_network()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess, 'tmp/poetry.module-49')

        state_ = sess.run(cell.zero_state(1, tf.float32))
        count = 0

        #x = np.array([list(map(word_num_map.get, '['))])
        #[probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
        #word = to_word(probs_)
        # word = words[np.argmax(probs_)]
        word = '['
        poem = ''
        while True:
            poem += word
            x = np.zeros((1, 1))
            x[0, 0] = word_num_map[word]
            [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
            word = to_word(probs_)
            # word = words[np.argmax(probs_)]
            if count >= poetry_num:
                break
            if word == ']':
                count = count + 1
                print(poem[1:-1])
                word = '['
                poem = ''
                state_ = sess.run(cell.zero_state(1, tf.float32))

gen_poetry(10)
