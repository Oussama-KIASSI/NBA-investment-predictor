To lunch the app on your local machine, do the following:
- run pip install -r requirements.txt
- run app.py

The project is organized as follow:
```
NBA-investment-predictor
│   app.py
│   EDA & Processing.ipynb
│   LICENSE
│   Model development.ipynb
│   README.md
│   requirements.txt
│   utils.py
│
├───Data
│   │   README.md
│   │
│   ├───Processed
│   │       Metadata.md
│   │       nba_logreg.csv
│   │       nba_logreg_selected.csv
│   │
│   └───Raw
│           Metadata.md
│           nba_logreg.csv
│
├───Models
│   │   Pipeline.pkl
│   │   README.md
│   │
│   ├───Baseline
│   │       DecisionTreeClassifier.pkl
│   │       KNeighborsClassifier.pkl
│   │       LGBMClassifier.pkl
│   │       LogisticRegression.pkl
│   │       RandomForestClassifier.pkl
│   │       SGDClassifier.pkl
│   │       SVC.pkl
│   │       XGBClassifier.pkl
│   │
│   ├───Scaler
│   │       MinMaxScaler.pkl
│   │
│   └───Tuned Models
│           SGDClassifier.pkl
│           SVC.pkl
│
└───templates
    │   form.html
    │
    └───helpers
            generate_fields.html
```
- I wrote utils.py that groups useful functions for our notebooks.
- I created notebooks for Exploratory Data Analysis, Preprocessing and Models development.
- I developped app.py for model deployment, you may run it on your local machine for prediction.
