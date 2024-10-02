from sklearn.model_selection import cross_val_score

def cross_validate_model(model, X_train, y_train, cv=5):
    """
    Perform cross-validation on the model.
    """
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
    return cv_scores