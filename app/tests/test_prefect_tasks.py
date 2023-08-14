

import numpy as np
import pandas as pd
import prefect_tasks


def test_split_data():
    """
    Test the split_data task.
    """
    input_df = prefect_tasks.load_data.fn("~/mlops-project/data/day.csv")

    df_train, df_val = prefect_tasks.split_data.fn(input_df)

    actual_df_train_shape = df_train.shape
    actual_df_val_shape = df_val.shape

    expected_df_train_shape = (584, 11)
    expected_df_val_shape = (147, 11)

    assert actual_df_train_shape == expected_df_train_shape
    assert actual_df_val_shape == expected_df_val_shape



def test_prepare_dictionaries():
    """
    Tests the prepare_dictionaries task.
    """

    input_data = [
        (1,0,1,0,6,0,2,0.344167,0.363625,0.805833,0.160446)
    ]

    input_columns = [
            "season",
            "yr",
            "mnth",
            "holiday",
            "weekday",
            "workingday",
            "weathersit",
            "temp",
            "atemp",
            "hum",
            "windspeed"
    ]
    
    input_df = pd.DataFrame(input_data, columns=input_columns)
    actual_dictionaries = prefect_tasks.prepare_dictionaries.fn(input_df)
    
    expected_dictionaries = [{'season': 1, 'mnth': 1, 'holiday': 0, 'weekday': 6, 'workingday': 0, 'weathersit': 2, 'temp': 0.344167, 'atemp': 0.363625, 'hum': 0.805833, 'windspeed': 0.160446}]

    assert actual_dictionaries==expected_dictionaries