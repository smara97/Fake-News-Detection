from flask import request, url_for
from flask_api import FlaskAPI, status, exceptions
from flask import jsonify
from newsplease import NewsPlease
import os,sys,inspect

import fnd_system as fnd


app = FlaskAPI(__name__)

api = {
    "/": {
    "methods":["GET"],
    "description": "Internal FND API Server"
    },
    "/detect": {
    "methods": ["POST"],
    "parameters": {
        "statement":"String",
        "subject":"String",
        "justification":"String"
        },
    "description": "Run FND model and return detection results"
    },
    "/detectURL": {
    "methods": ["POST"],
    "parameters": {
        "url":"String",
        },
    "description": "Run FND model and return detection results from a URL"
    }
}

@app.route("/", methods=['GET'])
def api_home():
    return jsonify(api)

@app.route("/detect", methods=['POST'])
def detect():
    data = request.json
    statement = data['statement']
    subject = data['subject']
    justification = data['justification']
    result = fnd.detect(statement,subject,justification)
    return jsonify(result)

@app.route("/detectURL", methods=['POST'])
def detectURL():
    data = request.json
    url = data['url']
    try:
        article = NewsPlease.from_url(url)
    except:
        return jsonify([None, None])
    statement = article.title
    justification = article.description
    try:
        subject = data['subject'] if data['subject'] != "" else fnd.get_subject(statement)[0]
    except:
        result = (False,statement,justification)
        return jsonify(result)
#    print(subject)

    if justification == None:
        return jsonify([None, None])
    result = fnd.detect(statement,subject,justification)
    result=result+(statement,justification,subject)
#    print(jsonify(article))
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
