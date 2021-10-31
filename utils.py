import os
import numpy
import pandas
from pandas import read_csv, DataFrame, concat
from matplotlib.pyplot import figure, show, title
from seaborn import heatmap, displot
import pickle
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler


def load_data(data_store_path: str = r"Data",
              data_class: str = r"Raw", data: str = r"nba_logreg.csv"):
    """
    loads data from the data store of this project with regard to the data class
    :param data_store_path: parent folder containing all project data
    :param data_class: data class or data subfolder (Raw or Processed)
    :param data: file containing data
    :return: pandas dataframe of loaded data
    """
    path = os.path.join(data_store_path, data_class, data)  # path where data is stored
    return read_csv(path)


def save_data(data: DataFrame, data_store_path: str = r"Data",
              data_class: str = r"Processed", file: str = r"nba_logreg.csv"):
    """
    save data in the data store path with regards to the data class using file name
    :param data: pandas dataframe
    :param data_store_path: parent folder containing all project data
    :param data_class: data class or data subfolder (Raw or Processed)
    :param file: file name and extension
    """
    path = os.path.join(data_store_path, data_class, file)  # path where data will be stored
    data.to_csv(path, index=False)  # save data to path in csv format


def missing_values_table(data: DataFrame):
    """
    analyze missing values in dataframe
    :param data: pandas dataframe to analyze
    :return: return a pandas Dataframe of columns with missing values
    """
    mis_val = data.isnull().sum()  # count missing values in dataframe
    mis_val_percent = 100 * data.isnull().sum() / len(data)  # count percentage of missing values in dataframe
    mis_val_table = concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    # keep only columns with empty rows and sort using percentage
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(2)
    print("Your selected dataframe has " + str(data.shape[1]) + " columns.\n" +
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")  # print recap message concerning missing values
    return mis_val_table_ren_columns


def distplot_by_target(data: DataFrame, columns: list, target: str = "Target"):
    """
    plot displot of target by dataframe column in columns
    :param data: data to analyze
    :param columns: columns to visualize
    :param target: target or label column
    """
    for c in columns:
        displot(data, x=c, hue=target, multiple="layer", palette="mako")  # distribution plot
        title(target + " Distribution by " + c)  # set title
        show()  # show plot


def correlation_heatmap(data: DataFrame, figsize: tuple = (12, 9), matrix: bool = False):
    """
    plot correlation heatmap
    :param data: data to analyze
    :param figsize: plot figure size
    :param matrix: boolean to indicate whether to show p-value (True) or not (False)
    """
    corrmat = data.corr()  # compute correlation
    figure(figsize=figsize)  # create figure with figsize
    heatmap(corrmat, vmax=.8, annot=matrix, cmap="mako", square=True)  # plot heatmap of correlation
    show()  # show plot


def compare_features(data: pandas.DataFrame, data_select: pandas.DataFrame, models: list):
    """
    compare features impact on models performance
    :param data: original dataframe
    :param data_select: dataframe with feature selection
    :param models: list of models to use in comparison
    :return: dataframe of features selection method comparison
    """
    # Features and target of original data
    X = data.iloc[:, 1:-1].to_numpy()  # features
    Y = data.iloc[:, -1].to_numpy()  # target
    # fit and transform scaler
    X_scaled = MinMaxScaler().fit_transform(X)
    # Features and target of dataframe with selected features
    X_select = data_select.iloc[:, 1:-1].to_numpy()
    Y_select = data_select.iloc[:, -1].to_numpy()
    # fit and transform scaler
    X_select_scaled = MinMaxScaler().fit_transform(X_select)
    comparison = {}
    for model in models:
        # original data evaluation
        model.fit(X_scaled, Y)
        Y_predict = model.predict(X_scaled)
        # new data evaluation
        model.fit(X_select_scaled, Y_select)
        Y_select_predict = model.predict(X_select_scaled)
        # set comparison row
        comparison[model.__class__.__name__] = [balanced_accuracy_score(Y, Y_predict),
                                                balanced_accuracy_score(Y_select, Y_select_predict)]
    return DataFrame.from_dict(comparison, orient="index", columns=["All features", "Selected features"])


def compare_models_kfolds(x_train: numpy.ndarray, y_train: numpy.ndarray, models: list, kfolds: int):
    """
    compare models and metrics
    :param x_train: training features
    :param y_train: training target
    :param models: list of models
    :param kfolds: number of folds used in cross-validation
    :return: dataframe to compare models and metrics
    """
    comparison = {}
    # stratified kfolds cross validation of data
    skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=0)
    for model in models:  # looping over models
        accuracy = precision = recall = f1 = 0  # initializing metrics
        # indices of train and validation using stratified kfolds cross validation
        for train_index, val_index in skf.split(x_train, y_train):
            # training validation split
            x_train_, x_val_ = x_train[train_index], x_train[val_index]  # features
            y_train_, y_val_ = y_train[train_index], y_train[val_index]  # target
            # model fitting
            model.fit(x_train_, y_train_)
            # model prediction
            y_pred = model.predict(x_train_)
            # metrics calculations using actual arrangement of training and validation split
            accuracy += balanced_accuracy_score(y_train_, y_pred)
            precision += precision_score(y_train_, y_pred)
            recall += recall_score(y_train_, y_pred)
            f1 += f1_score(y_train_, y_pred, average="weighted")
        # set comparison row
        comparison[model.__class__.__name__] = [accuracy / kfolds,
                                                precision / kfolds,
                                                recall / kfolds,
                                                f1 / kfolds]
    return DataFrame.from_dict(comparison, orient="index",
                               columns=["Accuracy", "Precision", "Recall", "F1-score"])


def compare_models(x_train: numpy.ndarray, x_test: numpy.ndarray, y_train: numpy.ndarray,
                   y_test: numpy.ndarray, models: list, model_store_path: str):
    """
    compare using F1-score and save models
    :param x_train: training features
    :param x_test: test features
    :param y_train: training target
    :param y_test: test target
    :param models: list of models to compare
    :param model_store_path: model store path to store models after fitting
    :return: dataframe of models evaluation
    """
    comparison = {}
    for model in models:  # looping over models
        model.fit(x_train, y_train)
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        # set comparison row
        comparison[model.__class__.__name__] = [f1_score(y_train, y_train_pred, average="weighted"),
                                                f1_score(y_test, y_test_pred, average="weighted")]
        # model saving
        save_model(model, model_store_path)
    return DataFrame.from_dict(comparison, orient="index", columns=["Training", "Test"])


def tune_model(x_train: numpy.ndarray, y_train: numpy.ndarray, model: object, kfolds: int, grid_param: dict):
    """
    tune model using grid search
    :param x_train: training features
    :param y_train: training target
    :param model: model to be tuned
    :param kfolds: number of folds used in cross-validation
    :param grid_param: grid search parameters
    :return: return grid search best parameters
    """
    skf = StratifiedKFold(n_splits=kfolds)  # stratified kfolds
    grid_search = GridSearchCV(model(), param_grid=grid_param, scoring="f1_weighted",
                               cv=skf)  # create grid search object
    grid_search.fit(x_train, y_train)  # fit to grid search
    print("Best params with F1-score = " +
          str(grid_search.best_score_) + " :\n\n" +
          str(grid_search.best_params_))  # print recap message
    return grid_search.best_params_


def save_scaler(scaler: object, scaler_store_path: str = r"Models\Scaler",
                scaler_type: str = "MinMax"):
    """
    save scaler to scaler store with regards to its type
    :param scaler: scaler to be saved
    :param scaler_store_path: folder to save in scaler
    :param scaler_type: type of scaler (MinMax, Standard, etc.)
    """
    path = os.path.join(scaler_store_path, scaler_type + "Scaler.pkl")
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)


def load_scaler(file_path: str = r"Models\Scaler\MinMaxScaler.pkl"):
    """
    load scaler from path
    :param file_path: path of the scaler
    :return: loaded scaler
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_model(model: object, model_store_path: str = r"Models\Baseline"):
    """
    save model to path
    :param model: model to save
    :param model_store_path: path to store the model
    """
    path = os.path.join(model_store_path, model.__class__.__name__ + ".pkl")
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(file_path: str = r"Models\Baseline\GradientBoostingClassifier.pkl"):
    """
    load model from path
    :rtype: object
    :param file_path: path of the model to load
    :return: loaded model
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)
