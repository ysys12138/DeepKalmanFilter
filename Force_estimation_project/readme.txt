raw data in folder data/rawdata/S2 should be txt and csv
For training:
python raw_to_np.py S2
python data_preprocessing.py S2
python cnn_causal_newData_FS.py S2 11.01 100
output: training log, figures, model in folder pt, and final result in result folder

For estimation:
python raw_to_np.py S2
python data_preprocessing.py S2
python TCN_estimate.py S2 11.01 0.01
output: predicted force in result folder