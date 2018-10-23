# Supervised Learning for Road Anomaly Detection

### Prerequisites

This project assumes that you have a working application that collect sensor data and stores it in a dropbox folder.

### Notes

Feel free to tweak the project accordingly to your own needs. This project requires a csv file contain time stamp (Epoch Seconds) Column and Z axis.

Refer to the sample zip file for an example. Do note that data collected do not contain headers and they are defined in the pipeline.

### Input Output

Input would be calling the function below, supplying it with the start and end times in Epoch Seconds
It will group the results in 1 second intervals
```
getTrainTestDF(startTime,endTime)
```

## Authors

* **Richmond Goh** - *Initial work*

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details

