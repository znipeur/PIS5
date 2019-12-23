from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from bson.json_util import dumps
from flask_pymongo import PyMongo
from flask_cors import CORS
from bson.objectid import ObjectId
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('C:\\Users\\HP\\Desktop\\PI\\Flask-restfull\\E0.csv')
X = dataset.iloc[:, [3, 4]].values
y = dataset.iloc[:, 7].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
labelencoder_y = LabelEncoder()
labelencoder_X.fit(X[0])
labelencoder_y.fit(y)
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=11)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=4)
classifier.fit(X_train, y_train)

y_testpred = classifier.predict([labelencoder_X.fit_transform(['Man City','Liverpool'])])


class JSONEncoder(json.JSONEncoder):
    ''' extend json-encoder class'''

    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime.datetime):
            return str(o)
        return json.JSONEncoder.default(self, o)

app = Flask(__name__)
api = Api(app)
app.config["MONGO_URI"] = "mongodb://localhost:27017/testdb"
mongo = PyMongo(app)
CORS(app)
app.json_encoder = JSONEncoder

def predictResult(home,away):
   return classifier.predict([labelencoder_X.fit_transform([home, away])])

class foot(Resource):

    def post(self):
        some_json = request.get_json()
        result = predictResult(some_json['homeTeam'] , some_json['awayTeam'])
        if result == 1:
            return {'result': 'Draw'}, 201
        elif result == 0:
            return {'result': some_json['awayTeam']}, 201
        else :
            return {'result': some_json['homeTeam']}, 201

class Multi(Resource):
    def get(self, num):
        return {'result': num * 10}

api.add_resource(foot, '/foot')
api.add_resource(Multi, '/multi/<int:num>')

if __name__ == '__main__':
    app.run(debug=True)
