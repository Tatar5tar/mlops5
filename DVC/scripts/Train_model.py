from sklearn.ensemble import RandomForestClassifier
import pickle


def train_model(X_train, y_train, model_path_temp, step):
    hyperparams = [
        {'n_estimators': 10, 'max_depth': 5},
        {'n_estimators': 50, 'max_depth': 7},
        {'n_estimators': 100, 'max_depth': 12}
    ]

    params = hyperparams[step % len(hyperparams)]
    model = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=42)
    model.fit(X_train, y_train)

    model_path = model_path_temp.format(step=step)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Модель обучена с гиперпараметрами: {params}")

    return model_path
