
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_model(input_dim: int):
    model_NN = Sequential([
        Dense(32, activation="relu", input_shape=(input_dim,)),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model_NN.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )
    return model_NN
