import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from sklearn import model_selection, preprocessing, svm, pipeline, compose
import vetiver

load_dotenv(find_dotenv())

# api_key = os.getenv("API_KEY")
# rsc_url = os.getenv("RSC_URL")

np.random.seed(500)
scooby = feather.read_feather('scooby-do.arrow').astype({'monster_real': 'category'})

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    scooby.iloc[:,1:3],
    scooby['monster_real'],
    test_size=0.2
)

scaler = preprocessing.StandardScaler().fit(X_train)
svc = svm.LinearSVC().fit(scaler.transform(X_train), y_train)

svc_pipeline = pipeline.Pipeline([('std_scaler', scaler), ('svc', svc)])

v = vetiver.VetiverModel(svc_pipeline, "isabel.zimmerman/scooby-doo", ptype_data = X_train)

# import pins
# # could be board_s3, board_azure, board_folder, etc
# board = pins.board_rsconnect(api_key=api_key, server_url=rsc_url, allow_pickle_read=True)

#vetiver.vetiver_pin_write(board, v)

api = vetiver.VetiverAPI(v, check_prototype=True)
api.run(port=8000)
