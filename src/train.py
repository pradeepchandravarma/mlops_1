def train_regression():
    import pandas as pd
    

    #load dataset
    df = pd.read_csv("data/Student_Performance.csv")
    #print(df.head())

    #data cleaning 
    df.isna().sum()
    df.duplicated().sum()
    #drop duplicates
    df = df.drop_duplicates()
    #print("After removal:", df.duplicated().sum())

    #feature engineering
    #df['Extracurricular Activities'] = df['Extracurricular Activities'].replace({"Yes":1,"No":0})
    df['Extracurricular Activities'] = (
    df['Extracurricular Activities']
    .replace({"Yes": 1, "No": 0})
    .infer_objects(copy=False)
    )

    df.infer_objects(copy=False)
    #print(df.head())


    #determine input X and ouput y
    
    y = df.iloc[:,-1]
    X = df.iloc[:,:-1]
    #print(X.head())
    #print(y.head())

    #predict performance index using gradient descent regression
    
    from sklearn.linear_model import SGDRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # model
    model = SGDRegressor(
        loss="squared_error",
        learning_rate="constant",
        eta0=0.01,
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)


    return model, scaler, X_test, y_test
    
    



