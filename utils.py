import tensorflow as tf
import re

def set_rnn_cell(name, hidden_size):
    """
    Choose the encoding unit you want to use
    :param name: encoding unit name
    :param num_units:
    """
    if name.lower() == "lstm":
        return tf.nn.rnn_cell.LSTMCell(hidden_size)
    elif name.lower() == "gru":
        return tf.nn.rnn_cell.GRUCell(hidden_size)
    else:
        return tf.nn.rnn_cell.BasicRNNCell(hidden_size)

def clean_text(sentence, mispell_dict):

    sentence = str(sentence)
    for punct in "/-'":
        sentence = sentence.replace(punct, ' ')
    for punct in '&':
        sentence = sentence.replace(punct, ' {} '.format(punct))
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        sentence = sentence.replace(punct, '')

    sentence = re.sub('[0-9]{5,}', '#####', sentence)
    sentence = re.sub('[0-9]{4}', '####', sentence)
    sentence = re.sub('[0-9]{3}', '###', sentence)
    sentence = re.sub('[0-9]{2}', '##', sentence)
    for mispell_word in mispell_dict.keys():
        sentence = re.sub(mispell_word, mispell_dict[mispell_word], sentence)
    return sentence

MISPELL_DICT = {'colour': 'color',
                'centre': 'center',
                'didnt': 'did not',
                'doesnt': 'does not',
                'isnt': 'is not',
                'shouldnt': 'should not',
                'favourite': 'favorite',
                'travelling': 'traveling',
                'counselling': 'counseling',
                'theatre': 'theater',
                'cancelled': 'canceled',
                'labour': 'labor',
                'organisation': 'organization',
                'wwii': 'world war 2',
                'citicise': 'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium'
                }
