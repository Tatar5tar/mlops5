import os
from Clean_data import clean_data
from Generate_features import generate_features
from Split_data import split_data
from Train_model import train_model
from Eval_model import eval_model


RAW_DATA_PATH = "../data/mall_customers.csv"
CLEAN_DATA_PATH = "../data/clean_mall_customers.csv"
FEATURES_DATA_PATH = "../data/features_mall_customers.csv"
MODEL_PATH = "../model/model.pkl"
MODEL_PATH_TEMP = "../model/model_step_{step}.pkl"
METRICS_PATH = "../metrics/metrics.json"


if __name__ == "__main__":
    clean_data(RAW_DATA_PATH, CLEAN_DATA_PATH)
    generate_features(CLEAN_DATA_PATH, FEATURES_DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(FEATURES_DATA_PATH)

    for step in range(3):
        model_path = train_model(X_train, y_train, MODEL_PATH_TEMP, step)
        eval_model(model_path, X_test, y_test, METRICS_PATH, step)

    print("Работа конвейера завершена.")
