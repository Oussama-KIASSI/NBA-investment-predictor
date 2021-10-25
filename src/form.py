from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, FloatField
from wtforms.validators import DataRequired


class input_form(FlaskForm):
    Name = StringField("What is the player name? (Name) ", validators=[DataRequired()])
    GP = IntegerField("How many games did he play? (GP) ", validators=[DataRequired()])
    MIN = FloatField("How many minutes did he play per game? (MIN) ", validators=[DataRequired()])
    threePM = FloatField("How many three points did he shoot per game? (3PM) ", validators=[DataRequired()])
    FGM = FloatField("How many field goals did he make per game? (FGM) ", validators=[DataRequired()])
    FGA = FloatField("How many field goals did he attempted to make per game? (FGA) ", validators=[DataRequired()])
    FTM = FloatField("How many free throws did he make per game? (FTM) ", validators=[DataRequired()])
    FTA = FloatField("How many free throws did he attempted to make per game? (FTA) ", validators=[DataRequired()])
    OREB = FloatField("How many offensive rebound did he make per game? (OREB) ", validators=[DataRequired()])
    DREB = FloatField("How many defensive rebound did he make per game? (DREB) ", validators=[DataRequired()])
    AST = FloatField("How many assists did he make per game? (AST) ", validators=[DataRequired()])
    STL = FloatField("How many steals did he make per game? (STL) ", validators=[DataRequired()])
    BLK = FloatField("How many blocks did he make per game? (BLK) ", validators=[DataRequired()])
    TOV = FloatField("How many turnovers did he make per game? (TOV) ", validators=[DataRequired()])
