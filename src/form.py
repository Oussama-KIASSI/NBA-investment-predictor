from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, FloatField
from wtforms.validators import DataRequired


class input_form(FlaskForm):
    """
    Form class that accepts inputs
    """
    Name = StringField("What is the player name? (Name) ",
                       validators=[DataRequired()]) # Input for the player's name
    GP = IntegerField("How many games did he play? (GP) ",
                      validators=[DataRequired()]) # Input for the number of games played
    MIN = FloatField("How many minutes did he play per game? (MIN) ",
                     validators=[DataRequired()]) # Input for the number of minutes played per game
    threePM = FloatField("How many three points did he shoot per game? (3PM) ",
                         validators=[DataRequired()]) # Input for the number of three points shot per game
    FGM = FloatField("How many field goals did he make per game? (FGM) ",
                     validators=[DataRequired()]) # Input for the number of field goals made per game
    FGA = FloatField("How many field goals did he attempted to make per game? (FGA) ",
                     validators=[DataRequired()]) # Input for the number of field goals attempted per game
    FTM = FloatField("How many free throws did he make per game? (FTM) ",
                     validators=[DataRequired()]) # Input for the number of free throws made per game
    FTA = FloatField("How many free throws did he attempted to make per game? (FTA) ",
                     validators=[DataRequired()]) # Input for the number of free throws attempted per game
    OREB = FloatField("How many offensive rebound did he make per game? (OREB) ",
                      validators=[DataRequired()]) # Input for the number of offensive rebounds per game
    DREB = FloatField("How many defensive rebound did he make per game? (DREB) ",
                      validators=[DataRequired()]) # Input for the number of defensive rebounds per game
    AST = FloatField("How many assists did he make per game? (AST) ",
                     validators=[DataRequired()]) # Input for the number of assists per game
    STL = FloatField("How many steals did he make per game? (STL) ",
                     validators=[DataRequired()]) # Input for the number of steals per game
    BLK = FloatField("How many blocks did he make per game? (BLK) ",
                     validators=[DataRequired()]) # Input for the number of blocks per game
    TOV = FloatField("How many turnovers did he make per game? (TOV) ",
                     validators=[DataRequired()]) # Input for the number of turnovers per game
