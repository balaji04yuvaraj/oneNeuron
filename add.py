from utils.model import Perceptron
from utils.all_func import prepare_data,save_model,save_plot

import pandas as pd
import numpy as np
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "log"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename = os.path.join(log_dir,"running.log"),level=logging.INFO,format=logging_str)

def main(data,ETA,EPOCHS,fileName,plotName):

    df = pd.DataFrame(data)
    logging.info(f"This is the actual dataframe {df}")
    X,y = prepare_data(df)

    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model,filename=fileName)
    save_plot(df,plotName,model)
if __name__ == '__main__':
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }
    ETA = 0.3 # 0 and 1
    EPOCHS = 300
    try:
        logging.info(">>>>>> Starting the training <<<<<<")
        main(data=AND, ETA=ETA, EPOCHS=EPOCHS, fileName="and.model", plotName = "and.png")
        logging.info(">>>>>> Stopping the training <<<<<<")
    except Exception as e:
        logging.exception(e)