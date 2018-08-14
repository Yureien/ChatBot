# import gzip
import string
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.callbacks import LambdaCallback
from keras.utils import plot_model
from chatterbot_corpus import corpus


# Constants
max_sentence_length = 40
batch_size = 128  # Batch size for training.
batch_data = 512
epochs = 200  # Number of epochs to train for.
vocab_size = 10000
latent_dim = 2048  # Latent dimensionality of the encoding space
enc_tokenizer = Tokenizer(vocab_size)  # The encoding tokenizer being used.
dec_tokenizer = Tokenizer(vocab_size)  # The decoding tokenizer being used.
to_ask_for_test = ['how are you',
                   'what is your name',
                   'is earth flat']

# Preprocessing
print("Gathering data...")
translator = str.maketrans('', '', string.punctuation)
raw_data = []
end = 100000
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
num_decoder_tokens = vocab_size
num_encoder_tokens = vocab_size
max_encoder_seq_length = max([len(txt.split()) for txt in input_texts] +
                             [len(txt.split()) for txt in input_sp_texts])
max_decoder_seq_length = max([len(txt.split()) for txt in output_texts] +
                             [len(txt.split()) for txt in output_sp_texts])
print("Dictionary size (encoder):", num_encoder_tokens)
print("Dictionary size (decoder):", num_decoder_tokens)
print("Max sentence length (encoder):", max_encoder_seq_length)
print("Max sentence length (decoder):", max_decoder_seq_length)

print("Vectorising data...")
encoder_input_data = np.array(
    [xi+[0]*(max_encoder_seq_length - len(xi))
     for xi in enc_tokenizer.texts_to_sequences(input_texts)]
)
decoder_input_data = np.array(
    [xi+[0]*(max_decoder_seq_length - len(xi))
     for xi in dec_tokenizer.texts_to_sequences(output_texts)]
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
encoder_sp_input_data = np.array(
    [xi+[0]*(max_encoder_seq_length - len(xi))
     for xi in enc_tokenizer.texts_to_sequences(input_sp_texts)]
)
decoder_sp_input_data = np.array(
    [xi+[0]*(max_decoder_seq_length - len(xi))
     for xi in dec_tokenizer.texts_to_sequences(output_sp_texts)]
)
decoder_sp_target_data = np.zeros(
    (len(input_sp_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
for t, i in enumerate(decoder_input_data):
    for tt, j in enumerate(i):
        if tt > 0:
            decoder_target_data[t, tt - 1, j] = 1.
for t, i in enumerate(decoder_sp_input_data):
    for tt, j in enumerate(i):
        if tt > 0:
            decoder_sp_target_data[t, tt - 1, j] = 1.

# Model
print("Creating the model...")
encoder_inputs = Input(shape=(None,), name='encoder_input')
encoder_em_inputs = Embedding(num_encoder_tokens, latent_dim, mask_zero=True)(encoder_inputs)
encoder = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder(encoder_em_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,), name='decoder_input')
decoder_em_inputs = Embedding(num_decoder_tokens, latent_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_em_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
print("Plotting model...")
plot_model(model, to_file='model_training.png')


def test(zz, xx):
    # Testing
    print("Starting testing...")

    # Model
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,), name='test_input_h')
    decoder_state_input_c = Input(shape=(latent_dim,), name='test_input_c')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_em_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
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

    for cinp, cout in zip(
            encoder_input_data[np.random.choice(encoder_input_data.shape[0], 2), :],
            decoder_input_data[np.random.choice(decoder_input_data.shape[0], 2), :]):  # noqa
        inp = []
        out = []
        for ii, jj in zip(cinp, cout):
            if ii == 0:
                inp.append("PAD")
            else:
                inp.append(enc_tokenizer.index_word[ii])
            if jj == 0:
                out.append("PAD")
            else:
                out.append(dec_tokenizer.index_word[jj])
        output = decode_sequence(cinp)
        print("Input sentence: ")
        print(' '.join(inp).replace('PAD', ''))
        print("Output sentence: ")
        print(' '.join(output).replace('PAD', ''))
        print("Correct sentence: ")
        print(' '.join(out).replace('PAD', ''))

    for s2a in to_ask_for_test:
        test_input_data = np.array(
            [xi+[0]*(max_encoder_seq_length - len(xi))
             for xi in enc_tokenizer.texts_to_sequences([s2a])]
        )
        output = decode_sequence(test_input_data)
        print('Custom input:', s2a)
        print('Output:')
        print(' '.join(output).replace('PAD', ''))


# Training
print("Training the model...")
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
print_callback = LambdaCallback(on_epoch_end=test)
for ep in range(epochs):
    print("Epoch: {0}/{1}".format(ep, epochs-1))
    start_idx = np.random.randint(encoder_input_data.shape[0] - batch_data - 2)
    enc_inp_trimmed = encoder_input_data[start_idx:start_idx + batch_data, :]
    dec_inp_trimmed = decoder_input_data[start_idx:start_idx + batch_data, :]
    dec_tar_trimmed = decoder_target_data[start_idx:start_idx + batch_data, :, :]
    model.fit([enc_inp_trimmed, dec_inp_trimmed], dec_tar_trimmed,
              batch_size=batch_size,
              epochs=4,
              verbose=1, callbacks=[print_callback])
    model.fit([encoder_sp_input_data, decoder_sp_input_data], decoder_sp_target_data,
              batch_size=batch_size,
              epochs=1, verbose=1, callbacks=[print_callback])
    model.save('model.h5')
test(None, None)
