     Predict.py
Filtering the input signal in real time. Model: TCN pt10.04
If the sample frequency is 500Hz, it's fine to set parameters delay=20000, refresh=0.2

    Input:
    argv1(filename): name of the row data file, should be csv, this file should change over time.(e.g. test0105)
    argv2(delay): Warmup time before started, e.g. 20000 refers to 40 seconds, which means you need to wait 40
    seconds before started, This is just a single wait at the beginning. Subsequent outputs are in real time and will
    only be delayed by much less than 0.1s.(20000 is the best choice)
    argv3(refresh): Refresh every how many seconds, e.g. 0.5 This determines the true latency you can feel,
    can theoretically be set to 0.
    argv4(offset): key parameter, bias due to differences in environment and sensors.
    This parameter should result in a static measurement of approximately -0.302V at an applied force of 0

    Output:
    An output csv file contains the result after signal processing. This file can keep changing over time.

Predict-KF.py
Filtering the input signal in real time. Model: DLKF pt0.795
If the sample frequency is 500Hz, it's fine to set parameters delay=550, refresh=0.2
