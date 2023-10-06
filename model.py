import random
from spacy.util import minibatch
import pandas as pd
import spacy
from spacy.pipeline.textcat import Config, single_label_cnn_config
from spacy.training.example import Example

# Load the spaCy model and create the text classification pipeline
nlp = spacy.blank("en")
config = Config().from_str(single_label_cnn_config)
textcat = nlp.add_pipe("textcat", config=config, last=True)

# Add the labels for your classification task
textcat.add_label("POSITIVE")
textcat.add_label("NEGATIVE")

# Load the reviews dataset
# Extract desired columns and remove rows with missing values
labels = pd.read_csv("labels.csv")
labels = labels[["Review Text", "Recommended IND"]].dropna()


# Prepare the data for training
def prepare_data(df):
    texts = df["Review Text"].tolist()
    labels = [
        {"cats": {"POSITIVE": bool(recommended), "NEGATIVE": not bool(recommended)}}
        for recommended in df["Recommended IND"]
    ]
    return list(zip(texts, labels))


# Create training and evaluation examples in spaCy format
train_data = prepare_data(labels)
train_texts, train_cats = zip(*train_data)
train_examples = []
for i in range(len(train_texts)):
    text, annotation = train_texts[i], train_cats[i]
    example = Example.from_dict(nlp.make_doc(text), annotation)
    train_examples.append(example)

# Initialize the optimizer
optimizer = nlp.begin_training()

# Train for a fixed number of iterations (you can adjust this value)
n_iter = 20
for i in range(n_iter):
    losses = {}
    random.shuffle(train_examples)
    for batch in minibatch(train_examples, size=32):
        nlp.update(batch, drop=0.5, losses=losses)
    print("Losses at iteration {}: {}".format(i, losses))

nlp.to_disk("save_model")
print("Saved to disk!")
