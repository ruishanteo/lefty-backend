from flask import Flask, jsonify, request
import spacy

app = Flask(__name__)
nlp = spacy.load("save_model")


@app.route("/ping", methods=["GET"])
def ping():
    return "pong"


@app.route("/ingredients", methods=["POST"])
def ingredients():
    try:
        input_ingredients = request.get_json(force=True)
        input_ingredients = input_ingredients["ingredients"]

        ingredients = []

        for ingredient in input_ingredients:
            print(ingredient)
            doc = nlp(ingredient.get("name"))
            if max(doc.cats, key=doc.cats.get) == "POSITIVE":
                ingredients.append(ingredient)

        return jsonify(ingredients)
    except Exception as e:
        return jsonify({"error": "no ingredients provided", "exception": str(e)})


app.run(host="0.0.0.0", port=8000)
