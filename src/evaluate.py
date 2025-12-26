def model_evaluate(model,X_test,y_test):

    from sklearn.metrics import mean_squared_error, r2_score

    y_pred = model.predict(X_test)
    
    return {
        "mse": mean_squared_error(y_test, y_pred),
        "r2" : r2_score(y_test, y_pred)
        }

    """
    #plot target vs predicted labels
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()], "r--")
    plt.show()
    """