import numpy as np
import pandas as pd

from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

#Const
repeats = 50 ## number of repeats in HP tuning
splits = 4 ## number of k folds in HP tuning
cores = -1 ## CPU cores used in HP tuning
verboseTraining = 1 ## training messages
printParams = 1 # decides if output features best performing model parameters
printTrainingDataTestScores = 0 # decides if output features test scores of test performed on training data

#Data input and preprocessing
def preprocessing(df):
    df.drop(columns=["ID"], inplace=True)
    df["CTR - Cardiothoracic Ratio"] = [x.replace(',', '.') for x in df["CTR - Cardiothoracic Ratio"]]
    df["Inscribed circle radius"] = [x.replace(',', '.') for x in df["Inscribed circle radius"]]
    df["Heart perimeter"] = [x.replace(',', '.') for x in df["Heart perimeter"]]
    df['CTR - Cardiothoracic Ratio'] = df['CTR - Cardiothoracic Ratio'].astype(float)
    df["Inscribed circle radius"] = df['Inscribed circle radius'].astype(float)
    df["Heart perimeter"] = df['Heart perimeter'].astype(float)
    return df

data = pd.read_csv("task_data.csv", decimal = '.')
data = preprocessing(pd.DataFrame(data))
#data.info()

#ML preprocessing

x = data.drop(columns=["Cardiomegaly","xx","xy","yy","Lung area","Heart area ","Heart width","Lung width"])
y = data["Cardiomegaly"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)


#Hyperparameter tuning
def KNNtune():
    pGrid = {
        "model__n_neighbors": range(2, 7),
        "model__weights": ["uniform", "distance"],
        "model__metric": ["manhattan", "euclidean", "minkowski"],
        "model__algorithm": ["auto"],
        #"abc": range(1, 11),
    }

    rskf = RepeatedStratifiedKFold(
        n_splits=splits,
        n_repeats=repeats,
        random_state=None
    )

    pipe_knn = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ])

    grid_search = GridSearchCV(
        #error_score='raise',
        estimator=pipe_knn,
        param_grid=pGrid,
        scoring="accuracy",
        cv=rskf,
        verbose=verboseTraining,
        n_jobs=cores
    )

    grid_search.fit(x_train, y_train)

    if (printParams == 1):
        print("KKN predicted accuracy (mean CV)")
        print(grid_search.best_score_)

    return grid_search.best_estimator_

def SVCtune():
    pGrid = {
        "model__kernel": ["linear", "rbf", "sigmoid"],
        "model__gamma": ["auto", "scale"],
        "model__C": [0.1,0.2,0.5,1,2],
        "model__class_weight": [None],
    }

    rskf = RepeatedStratifiedKFold(
        n_splits=splits,
        n_repeats=repeats,
        random_state= None
    )

    pipe_svc = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", SVC())
    ])

    grid_search = GridSearchCV(
        #error_score='raise',
        estimator=pipe_svc,
        param_grid=pGrid,
        scoring="accuracy",
        cv=rskf,
        verbose=verboseTraining,
        n_jobs=cores
    )

    grid_search.fit(x_train, y_train)

    if (printParams == 1):
        print("SVC predicted accuracy (mean CV)")
        print(grid_search.best_score_)

    return grid_search.best_estimator_

def DTCtune():
    pGrid = {
        "model__max_depth": range(1,6),
        "model__min_samples_split": range(2,6),
        "model__min_samples_leaf": range(2,6),
    }

    rskf = RepeatedStratifiedKFold(
        n_splits=splits,  # Number of folds per repetition
        n_repeats=repeats,  # Number of times to repeat the process
        random_state=None
    )

    pipe_dtc = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier())
    ])

    grid_search = GridSearchCV(
        #error_score='raise',
        estimator=pipe_dtc,
        param_grid=pGrid,
        scoring="accuracy",
        cv=rskf,
        verbose=verboseTraining,
        n_jobs=cores
    )

    grid_search.fit(x_train, y_train)

    if (printParams == 1):
        print("DTC predicted accuracy (mean CV)")
        print(grid_search.best_score_)

    return grid_search.best_estimator_

def RFCtune():
    pGrid = {
        "model__n_estimators": [2,4,6],
        "model__max_depth": [2,4,6],
        "model__min_samples_split": [2,4,6],
    }

    rskf = RepeatedStratifiedKFold(
        n_splits=splits,  # Number of folds per repetition
        n_repeats=repeats,  # Number of times to repeat the process
        random_state=None
    )

    pipe_rfc = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier())
    ])

    grid_search = GridSearchCV(
        #error_score='raise',
        estimator=pipe_rfc,
        param_grid=pGrid,
        scoring="accuracy",
        cv=rskf,
        verbose=verboseTraining,
        n_jobs=cores
    )

    grid_search.fit(x_train, y_train)

    if (printParams == 1):
        print("RFC predicted accuracy (mean CV)")
        print(grid_search.best_score_)

    return grid_search.best_estimator_

KNNModel = KNNtune()
print("KNN params tuned\n")
SVCModel = SVCtune()
print("SVC params tuned\n")
DTCModel = DTCtune()
print("DTC params tuned\n")
RFCModel = RFCtune()
print("RFC params tuned\n")


#Model training
KNNModel.fit(x_train, y_train)
print("KNN model fitted")
SVCModel.fit(x_train, y_train)
print("SVC model fitted")
DTCModel.fit(x_train, y_train)
print("DTC model fitted")
RFCModel.fit(x_train, y_train)
print("RFC model fitted")

#Accuracy evaluation
KKN_cv_score = np.round(cross_val_score(KNNModel, x_test, y_test, cv = 4), 3)
SVC_cv_score = np.round(cross_val_score(SVCModel, x_test, y_test, cv = 4), 3)
DTC_cv_score = np.round(cross_val_score(DTCModel, x_test, y_test, cv = 4), 3)
RFC_cv_score = np.round(cross_val_score(RFCModel, x_test, y_test, cv = 4), 3)

KKN_cv_score_train = np.round(cross_val_score(KNNModel, x_train, y_train, cv = 4), 3)
SVC_cv_score_train = np.round(cross_val_score(SVCModel, x_train, y_train, cv = 4), 3)
DTC_cv_score_train = np.round(cross_val_score(DTCModel, x_train, y_train, cv = 4), 3)
RFC_cv_score_train = np.round(cross_val_score(RFCModel, x_train, y_train, cv = 4), 3)

#Output
print("\n\nK closest neighbours params and scores: ")
if (printParams == 1):
    print(KNNModel.get_params())
print(f"\nCross-validation mean score: {np.mean(KKN_cv_score):.3f}")
print(f"Standard deviation of CV score: {np.std(KKN_cv_score):.3f}")
if (printTrainingDataTestScores == 1):
    print(f"\nCross-validation mean score (training data): {np.mean(KKN_cv_score_train):.3f}")
    print(f"Standard deviation of CV score (training data): {np.std(KKN_cv_score_train):.3f}")

print("\n\nSVC params and scores: ")
if (printParams == 1):
    print(SVCModel.get_params())
print(f"\nCross-validation mean score: {np.mean(SVC_cv_score):.3f}")
print(f"Standard deviation of CV score: {np.std(SVC_cv_score):.3f}")
if (printTrainingDataTestScores == 1):
    print(f"\nCross-validation mean score (training data): {np.mean(SVC_cv_score_train):.3f}")
    print(f"Standard deviation of CV score (training data): {np.std(SVC_cv_score_train):.3f}")

print("\n\nDecision params and tree scores: ")
if (printParams == 1):
    print(DTCModel.get_params())
print(f"\nCross-validation mean score: {np.mean(DTC_cv_score):.3f}")
print(f"Standard deviation of CV score: {np.std(DTC_cv_score):.3f}")
if (printTrainingDataTestScores == 1):
    print(f"\nCross-validation mean score (training data): {np.mean(DTC_cv_score_train):.3f}")
    print(f"Standard deviation of CV score (training data): {np.std(DTC_cv_score_train):.3f}")

print("\n\nRandom forest params and scores: ")
if (printParams == 1):
    print(RFCModel.get_params())
print(f"\nCross-validation mean score: {np.mean(RFC_cv_score):.3f}")
print(f"Standard deviation of CV score: {np.std(RFC_cv_score):.3f}")
if (printTrainingDataTestScores == 1):
    print(f"\nCross-validation mean score (training data): {np.mean(RFC_cv_score_train):.3f}")
    print(f"Standard deviation of CV score (training data): {np.std(RFC_cv_score_train):.3f}")
