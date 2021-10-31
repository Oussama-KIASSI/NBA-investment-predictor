import numpy as np
from flask import Flask, render_template, request, jsonify
from src.form import input_form
from utils import *

model = load_model("Models/Pipeline.pkl")  # load pipeline model in the backend
app = Flask(__name__)  # create app
app.config['SECRET_KEY'] = 'SECRET_KEY'  # configure secret key
app.config['WTF_CSRF_SECRET_KEY'] = "secretkey"  # configure wtforms csrf secret key


@app.route('/')  # route for the homepage
def home_page():
    """
    render home page
    :return:
    """
    return render_template('home.html')


@app.route('/predict', methods=('GET', 'POST'))  # route for the prediction form
def predict():
    """
    predict the worthiness of a player from form data
    :return: if the player is worthy of investment of no
    """
    # generate form
    f = input_form()
    if not f.is_submitted():  # render prediction form
        return render_template('predict.html', form=f)
    else:  # run prediction if form is_submitted
        GP = f.GP.data  # number of games played
        MIN = f.MIN.data  # number of minutes played per game
        FGM = f.FGM.data  # number of field goals made per game
        FGA = f.FGA.data  # number of field goals attempted per game
        FGperc = round((FGM / FGA) * 100, 1)  # field goals made and field goals attempted ratio
        EFGperc = round((FGM + 0.5 * f.threePM.data) * 100 / FGA, 1)  # effective field game percentage
        FTM = f.FTM.data  # number of free throws made
        FTperc = round((FTM / f.FTA.data) * 100, 1)  # ratio of free throws made and free throws attempted
        OREB = f.OREB.data  # number of offensive rebounds
        DREB = f.DREB.data  # number of defensive rebounds
        AST = f.AST.data  # number of assists
        STL = f.STL.data  # number of steals
        BLK = f.BLK.data  # number of blocks
        TOV = f.TOV.data  # number of turnovers
        prediction = int(model.predict(np.array([GP, MIN, FGA, FGperc, EFGperc, FTM, FTperc, OREB,
                                                 DREB, AST, STL, BLK, TOV]).reshape(1, -1))[0])  # run prediction
        # return prediction results
        return f"Player {f.Name.data} is{[' not', ''][prediction]} worth investing."


@app.route('/predict_from_request', methods=['POST'])  # route for the prediction request
def predict_from_request():
    """
    predict the worthiness of a player from request data
    :return: if the player is worthy of investment of no
    """
    data = request.args  # data held in the request
    Name = data.get("Name")  # player's name
    GP = int(data.get("GP"))  # number of games played
    MIN = float(data.get("MIN"))  # number of minutes played per game
    FGM = float(data.get("FGM"))  # number of field goals made per game
    FGA = float(data.get("FGA"))  # number of field goals attempted per game
    FGperc = round((FGM / FGA) * 100, 1)  # field goals made and field goals attempted ratio
    EFGperc = round((FGM + 0.5 * float(data.get("3PM")) * 100) / FGA, 1)  # effective field game percentage
    FTM = float(data.get("FTM"))  # number of free throws made
    FTperc = round((FTM / float(data.get("FTA"))) * 100, 1)  # ratio of free throws made and free throws attempted
    OREB = float(data.get("OREB"))  # number of offensive rebounds
    DREB = float(data.get("DREB"))  # number of defensive rebounds
    AST = float(data.get("AST"))  # number of assists
    STL = float(data.get("STL"))  # number of steals
    BLK = float(data.get("BLK"))  # number of blocks
    TOV = float(data.get("TOV"))  # number of turnovers
    prediction = int(model.predict(np.array([GP, MIN, FGA, FGperc, EFGperc, FTM, FTperc, OREB,
                                             DREB, AST, STL, BLK, TOV]).reshape(1, -1))[0])  # run prediction
    # return prediction results
    return jsonify({'result': f"Player {Name} is{[' not', ''][prediction]} worth investing."})


if __name__ == "__main__":
    app.run()
