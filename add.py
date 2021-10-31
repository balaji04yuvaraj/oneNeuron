from utils.model import Perceptron
from utils.all_func import prepare_data,save_model,save_plot

import pandas as pd
import numpy as np

AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1],
}

df = pd.DataFrame(AND)

df

X,y = prepare_data(df)

print("The Value is :",X)

ETA = 0.3 # 0 and 1
EPOCHS = 10

model = Perceptron(eta=ETA, epochs=EPOCHS)
model.fit(X, y)

_ = model.total_loss()

save_model(model,filename="and.model")
save_plot(df,"and.png",model)