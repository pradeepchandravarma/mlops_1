
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def encode_features(df):
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map({
        "Yes": 1,
        "No": 0
    })
    return df


def split_data(df, target, test_size, random_state):
    X = df.drop(target, axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
