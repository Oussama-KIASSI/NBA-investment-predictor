import numpy as np
from flask import Flask, render_template, request, jsonify
from src.form import input_form
from utils import *

model = load_model("Models/Pipeline.pkl")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'SECRET_KEY'
app.config['WTF_CSRF_SECRET_KEY'] = "secretkey"


@app.route('/')
def home_page():
    return render_template('home.html')


@app.route('/predict', methods=('GET', 'POST'))
def predict():
    f = input_form()
    if f.is_submitted():
        GP = f.GP.data
        MIN = f.MIN.data
        FGA = f.FGA.data
        FGperc = round((f.FGM.data / f.FGA.data) * 100, 1)
        EFGperc = round((f.FGM.data + 0.5 * f.threePM.data) * 100 / f.FGA.data, 1)
        FTM = f.FTM.data
        FTperc = round((f.FTM.data / f.FTA.data) * 100, 1)
        OREB = f.OREB.data
        DREB = f.DREB.data
        AST = f.AST.data
        STL = f.STL.data
        BLK = f.BLK.data
        TOV = f.TOV.data
        prediction = model.predict(np.array([GP, MIN, FGA, FGperc, EFGperc, FTM, FTperc, OREB,
                                             DREB, AST, STL, BLK, TOV]).reshape(1, -1))[0]
        if prediction == 1:
            return f"Player {f.Name.data} is worth investing."
        else:
            return f"Player {f.Name.data} isn't worth investing."
    else:
        return render_template('predict.html', form=f)


@app.route('/predict_from_request', methods=['POST'])
def predict_from_request():
    data = request.args
    Name = data.get("Name")
    GP = int(data.get("GP"))
    MIN = float(data.get("MIN"))
    FGA = float(data.get("FGA"))
    FGM = float(data.get("FGM"))
    FGperc = round((FGM / FGA) * 100, 1)
    EFGperc = round((FGM + 0.5 * float(data.get("3PM")) * 100) / FGA, 1)
    FTM = float(data.get("FTM"))
    FTperc = round((FTM / float(data.get("FTA"))) * 100, 1)
    OREB = float(data.get("OREB"))
    DREB = float(data.get("DREB"))
    AST = float(data.get("AST"))
    STL = float(data.get("STL"))
    BLK = float(data.get("BLK"))
    TOV = float(data.get("TOV"))
    prediction = int(model.predict(np.array([GP, MIN, FGA, FGperc, EFGperc, FTM, FTperc, OREB,
                                             DREB, AST, STL, BLK, TOV]).reshape(1, -1))[0])

    return jsonify({'result': f"Player {Name} is{[' not', ''][prediction]} worth investing."})


if __name__ == "__main__":
    app.run()
