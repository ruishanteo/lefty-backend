import spacy

nlp = spacy.load("save_model")

tests = ["pumpkin", "photography", "garlic", "bear"]
for test in tests:
    doc = nlp(test)
    predicted_label = max(doc.cats, key=doc.cats.get)
    print(test, "is a food?", predicted_label)
