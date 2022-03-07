# imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import RFE
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
import pandas as pd

# read in all the different datasets
datapath = "../Datafiles/"
outfile = "match_level_model_all_metrics_results.csv"
averaged_match_level_filename = "match_level_dataset.csv"
averaged_match_level_data = pd.read_csv(datapath + averaged_match_level_filename)


# best parameters, previously identified
# random forest
averaged_match_level_rf_best_params = {"class_weight": None,
                                       "criterion": "gini",
                                       "max_depth": None,
                                       "max_features": "log2",
                                       "min_impurity_decrease": 0.0,
                                       "n_estimators": 64
                                      }

# logistic regression
averaged_match_level_lr_best_params = {"C": 0.5,
                                       "class_weight": None
                                      }

# svc
averaged_match_level_svc_best_params = {"C": 0.5,
                                       "class_weight": None
                                      }


# here is the list of most relevant features previously identified
averaged_match_features = ["tentat", "differ", "auxverb", "negate", "CharNames", "drives",
                        "WC", "we", "adverb", "percept", "bio", "achieve", "reward",
                        "focusfuture", "netspeak", "rank", "prep", "friend", "leisure",
                        "nonflu"]

def run_kfold(data, model):
    '''
    data: Pandas dataframe
    model: model to train
    Run repeated kfold cross validation. Note that matches are not separated, i.e.
    clips from the same match might appear in both training and test sets
    :return: list of accuracy results for each fold
    '''

    # set up
    n_splits = 5
    n_repeats = 100

    X = data.drop(["binary_toxicity"], axis=1)
    y = data["binary_toxicity"]
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

    train_accuracies = []
    test_accuracies = []
    train_precisions = []
    test_precisions = []
    train_recalls = []
    test_recalls = []
    train_f1s = []
    test_f1s = []
    train_aucs = []
    test_aucs = []

    # run kfold cross validation, store results for each fold
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        md = model
        md.fit(X_train,y_train)

        md = model
        md.fit(X_train, y_train)
        # get prediction results for training and testing sets
        y_train_pred = md.predict(X_train)
        y_test_pred = md.predict(X_test)
        # results
        # accuracy
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        # precision
        train_precision = precision_score(y_train, y_train_pred)
        test_precision = precision_score(y_test, y_test_pred)
        train_precisions.append(train_precision)
        test_precisions.append(test_precision)
        # recall
        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)
        train_recalls.append(train_recall)
        test_recalls.append(test_recall)
        # f1 score
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        train_f1s.append(train_f1)
        test_f1s.append(test_f1)
        # auc
        train_auc = roc_auc_score(y_train, y_train_pred)
        test_auc = roc_auc_score(y_test, y_test_pred)
        train_aucs.append(train_auc)
        test_aucs.append(test_auc)

    results_dict = {"train_accuracies": train_accuracies,
                    "test_accuracies": test_accuracies,
                    "train_precisions": train_precisions,
                    "test_precisions": test_precisions,
                    "train_recalls": train_recalls,
                    "test_recalls": test_recalls,
                    "train_f1s": train_f1s,
                    "test_f1s": test_f1s,
                    "train_aucs": train_aucs,
                    "test_aucs": test_aucs
                    }

    return results_dict


# main
# run kfold validation on all the models
averaged_match_level_models = [RandomForestClassifier(class_weight=averaged_match_level_rf_best_params["class_weight"],
                                                       criterion=averaged_match_level_rf_best_params["criterion"],
                                                       max_depth=averaged_match_level_rf_best_params["max_depth"],
                                                       max_features=averaged_match_level_rf_best_params["max_features"],
                                                       min_impurity_decrease=averaged_match_level_rf_best_params["min_impurity_decrease"]),
                                LogisticRegression(C=averaged_match_level_lr_best_params["C"],
                                                   class_weight=averaged_match_level_lr_best_params["class_weight"],
                                                   max_iter=10000),
                                LinearSVC(C=averaged_match_level_svc_best_params["C"],
                                          class_weight=averaged_match_level_svc_best_params["class_weight"],
                                          dual=False,
                                          max_iter=10000)]


model_names = ["random forest", "logistic regression", "SVM"]


results = {"model_name": [], "train_accuracy": [], "test_accuracy": [], "test_accuracy_std_dev": [],
           "all_test_accuracies": [], "test_accuracy_confidence_interval": [],
           "train_precision": [], "test_precision": [], "test_precision_std_dev": [],
           "all_test_precisions": [], "test_precision_confidence_interval": [],
           "train_recall": [], "test_recall": [], "test_recall_std_dev": [],
           "all_test_recalls": [], "test_recall_confidence_interval": [],
           "train_f1": [], "test_f1": [], "test_f1_std_dev": [],
           "all_test_f1s": [], "test_f1_confidence_interval": [],
           "train_auc": [], "test_auc": [], "test_auc_std_dev": [],
           "all_test_aucs": [], "test_auc_confidence_interval": []
           }

for i in range(0, len(model_names)):
    model_name = model_names[i]
    model = averaged_match_level_models[i]

    new_data = averaged_match_level_data[averaged_match_features + ["binary_toxicity"]]

    metrics = run_kfold(new_data, model)

    # accuracy results
    results["train_accuracy"].append(sum(metrics["train_accuracies"]) / len(metrics["train_accuracies"]))
    results["test_accuracy"].append(sum(metrics["test_accuracies"]) / len(metrics["test_accuracies"]))
    results["test_accuracy_std_dev"].append(np.std(metrics["test_accuracies"]))
    results["model_name"].append(model_name)
    results["all_test_accuracies"].append(metrics["test_accuracies"])
    # Get upper and lower bounds of the model's accuracy, on the 95% confidence interval
    lower, upper = proportion_confint(sum(metrics["test_accuracies"]), len(metrics["test_accuracies"]), 0.05)
    results["test_accuracy_confidence_interval"].append((lower, upper))
    # precision results
    results["train_precision"].append(sum(metrics["train_precisions"]) / len(metrics["train_precisions"]))
    results["test_precision"].append(sum(metrics["test_precisions"]) / len(metrics["test_precisions"]))
    results["test_precision_std_dev"].append(np.std(metrics["test_precisions"]))
    results["all_test_precisions"].append(metrics["test_precisions"])
    lower, upper = proportion_confint(sum(metrics["test_precisions"]), len(metrics["test_precisions"]), 0.05)
    results["test_precision_confidence_interval"].append((lower, upper))
    # recall results
    results["train_recall"].append(sum(metrics["train_recalls"]) / len(metrics["train_recalls"]))
    results["test_recall"].append(sum(metrics["test_recalls"]) / len(metrics["test_recalls"]))
    results["test_recall_std_dev"].append(np.std(metrics["test_recalls"]))
    results["all_test_recalls"].append(metrics["test_recalls"])
    lower, upper = proportion_confint(sum(metrics["test_recalls"]), len(metrics["test_recalls"]), 0.05)
    results["test_recall_confidence_interval"].append((lower, upper))
    # f1 results
    results["train_f1"].append(sum(metrics["train_f1s"]) / len(metrics["train_f1s"]))
    results["test_f1"].append(sum(metrics["test_f1s"]) / len(metrics["test_f1s"]))
    results["test_f1_std_dev"].append(np.std(metrics["test_f1s"]))
    results["all_test_f1s"].append(metrics["test_f1s"])
    lower, upper = proportion_confint(sum(metrics["test_f1s"]), len(metrics["test_f1s"]), 0.05)
    results["test_f1_confidence_interval"].append((lower, upper))
    # auc results
    results["train_auc"].append(sum(metrics["train_aucs"]) / len(metrics["train_aucs"]))
    results["test_auc"].append(sum(metrics["test_aucs"]) / len(metrics["test_aucs"]))
    results["test_auc_std_dev"].append(np.std(metrics["test_aucs"]))
    results["all_test_aucs"].append(metrics["test_aucs"])
    lower, upper = proportion_confint(sum(metrics["test_aucs"]), len(metrics["test_aucs"]), 0.05)
    results["test_auc_confidence_interval"].append((lower, upper))

# convert to a dataframe and save as a csv
results = pd.DataFrame.from_dict(results)
results.to_csv(datapath + outfile, index=False)