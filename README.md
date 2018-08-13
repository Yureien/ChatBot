# ChatBot
This project was made for learning, using the [official tutorial](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html).

A compiled model done on 50 epochs is attached.

## Theory

* Word-level training
* Uses sequence to sequence model
* Uses teacher forcing

### Corpora used

* [Cornell movie dialogs](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
* [Chatterbot corpus - English](https://github.com/gunthercox/chatterbot-corpus)

## Installation

```
pip install --upgrade -r requirements.txt
python chat.py
```

## Training

```
python train.py  # Adjust the epochs and other stuff in the train.py file. Constants defined at top.
```

## Sample output

(Trained with 50 epochs)

```
> What is your name?
fear is a human emotion with a movie.
> What are you?
i am in the internet.
> Where are you?
i am in the internet.
> Are you a chatbot?
Word chatbot not in dictionary!
> Are you a chatterbot?
yes i am i am as hard.
> How are you?
i am doing well.
> How am I?
i am.
> Do you like pizza?
yes i am not really immortal.
> How old are you?
i am doing well how about you.
```

... Yes, pretty bad. For now.
