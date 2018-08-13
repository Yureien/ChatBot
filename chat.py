import gzip
import string
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Model, load_model
from keras.layers import Input
from keras.utils import plot_model
from chatterbot_corpus import corpus


# Constants
max_sentence_length = 50
batch_size = 128  # Batch size for training.
batch_data = 5200
epochs = 300  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space
enc_tokenizer = Tokenizer(50000)  # The encoding tokenizer being used.
dec_tokenizer = Tokenizer(50000)  # The decoding tokenizer being used.
to_ask_for_test = ['how are you',
                   'what is your name',
                   'is earth flat']

# Preprocessing
print("Gathering data...")
translator = str.maketrans('', '', string.punctuation)
raw_data = []
end = 16000
counter = 0
# with gzip.open("master_data.txt.gz") as f:
with open('full_conversation.txt') as f:
    while counter < end:
        try:
            l1 = (f.readline().
                  replace("Q::", "").
                  replace("A::", "").
                  replace("\n", "").lower().
                  translate(translator))
            l2 = (f.readline().
                  replace("Q::", "").
                  replace("A::", "").
                  replace("\n", "").lower().
                  translate(translator))
            if len(l1.split()) < max_sentence_length and len(l2.split()) < max_sentence_length:
                raw_data.append(l1)
                raw_data.append(l2)
                counter += 1
        except (IOError, StopIteration, EOFError):
            break
if len(raw_data) % 2 == 1:
    raw_data = raw_data[:-1]
special_corpus = corpus.Corpus()
sp_texts = []
for i in special_corpus.load_corpus('chatterbot.data.english'):
    for j in i:
        xx = [x.translate(translator).lower() for x in j]
        if len(xx[0].split()) < max_sentence_length and len(xx[1].split()) < max_sentence_length:
            sp_texts.append(xx)
input_sp_texts, output_sp_texts = zip(*sp_texts)
input_sp_texts = list(input_sp_texts)
output_sp_texts = list(output_sp_texts)
input_texts = raw_data[::2]
output_texts = raw_data[1::2]
output_texts = ['sossossos ' + t + ' eoseoseos' for t in output_texts]
output_sp_texts = ['sossossos ' + t + ' eoseoseos' for t in output_sp_texts]

print("Tokenizing data...")
enc_tokenizer.fit_on_texts(input_texts + input_sp_texts)
dec_tokenizer.fit_on_texts(output_texts + output_sp_texts)
num_decoder_tokens = len(dec_tokenizer.word_counts) + 1
num_encoder_tokens = len(enc_tokenizer.word_counts) + 1
max_encoder_seq_length = max([len(txt.split()) for txt in input_texts] +
                             [len(txt.split()) for txt in input_sp_texts])
max_decoder_seq_length = max([len(txt.split()) for txt in output_texts] +
                             [len(txt.split()) for txt in output_sp_texts])
print("Dictionary size (encoder):", num_encoder_tokens)
print("Dictionary size (decoder):", num_decoder_tokens)
print("Max sentence length (encoder):", max_encoder_seq_length)
print("Max sentence length (decoder):", max_decoder_seq_length)

print("Vectorising data...")
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
encoder_sp_input_data = np.zeros(
    (len(input_sp_texts), max_encoder_seq_length),
    dtype='float32')
decoder_sp_input_data = np.zeros(
    (len(input_sp_texts), max_decoder_seq_length),
    dtype='float32')
decoder_sp_target_data = np.zeros(
    (len(input_sp_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, output_text) in enumerate(zip(input_texts, output_texts)):
    for t, txt in enumerate(input_text.split()):
        encoder_input_data[i, t] = enc_tokenizer.word_index[txt]
    for t, txt in enumerate(output_text.split()):
        decoder_input_data[i, t] = dec_tokenizer.word_index[txt]
        if t > 0:
            decoder_target_data[i, t - 1, dec_tokenizer.word_index[txt]] = 1.

for i, (input_text, output_text) in enumerate(zip(input_sp_texts, output_sp_texts)):
    for t, txt in enumerate(input_text.split()):
        encoder_sp_input_data[i, t] = enc_tokenizer.word_index[txt]
    for t, txt in enumerate(output_text.split()):
        decoder_sp_input_data[i, t] = dec_tokenizer.word_index[txt]
        if t > 0:
            decoder_sp_target_data[i, t - 1, dec_tokenizer.word_index[txt]] = 1.


# Model
print("Loading the model...")
model = load_model('model.h5')
print("Plotting model...")
plot_model(model, to_file='model_chat.png')

encoder_inputs = model.input[0]  # input_1
encoder_em_inputs = model.layers[2](encoder_inputs)
encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_em_inputs = model.layers[3](decoder_inputs)
decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[5]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_em_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[6]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


# Decoder
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = dec_tokenizer.word_index['sossossos']

    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = dec_tokenizer.index_word[sampled_token_index]
        decoded_sentence.append(sampled_word)

        if (sampled_word == 'eoseoseos' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return decoded_sentence


translator = str.maketrans('', '', string.punctuation)
while True:
    input_sentence = input("> ").strip().translate(translator).lower()
    if len(input_sentence) > max_encoder_seq_length:
        print("Enter smaller sentence! (Less than {} characters)".format(max_encoder_seq_length))
        continue
    input_data = np.zeros((1, max_encoder_seq_length), dtype='float32')
    try:
        for t, w in enumerate(input_sentence.split()):
            input_data[0, t] = enc_tokenizer.word_index[w]
    except KeyError:
        print("Word {} not in dictionary!".format(w))
        continue
    output = decode_sequence(input_data)
    print(' '.join(output).replace('PAD', '').strip().replace(' eoseoseos', '.'))
