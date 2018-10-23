import dropbox, os, contextlib, time
import csv, math, zipfile, io, datetime, calendar
import numpy as np
import pandas as pd
import config
#import pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

def rounder(a):
    x = int(a//60 * 60)
    #print(x)
    return x

class bumpReader:
    def __init__(self):
        self.filename = 'bump.csv'
        self.data = pd.read_csv(self.filename)
        self.test = self.data.values.ravel()

class ScrappyProcessor():
    def parseFrame(self, orgiDF):
        specDF,labelArr = self.startFrame(orgiDF)
        transformer = RobustScaler().fit(specDF)

        return transformer.transform(specDF), labelArr

    def startFrame(self, frame):
        data = frame
        data['TimeStamp'] = data.apply(lambda row: self.toEpoch(row.TimeStamp) , axis=1)
#
        specDF = data.groupby(['TimeStamp']).agg({'Z': ['skew', 'mean']}).reset_index(col_level=1)

        specDF['kurtosis'] = data.groupby('TimeStamp', as_index=False)[['Z']].apply(pd.DataFrame.kurt)
        
        specDF.columns = specDF.columns.droplevel(0)

        bumpTBL = bumpReader()
        specDF['abnormal'] = specDF.apply(lambda x: self.retrieveLabel(x['TimeStamp'],bumpTBL.test), axis=1)

        specDF = specDF.drop(['TimeStamp'],axis=1)
        specDF.rename(columns={specDF.columns[2]:'kurtosis'}, inplace=True)
        
        specDF = specDF.dropna().reset_index(drop=True)
        label = specDF['abnormal']
        specDF = specDF.drop(['abnormal'],axis=1)
        data = data.iloc[0:0] #clear data frame

        return specDF, label

    def toEpoch(self, timeStamp):
        return int(timeStamp / 1000)

    def retrieveLabel(self, timeStamp,test):
        if timeStamp in test:
            return 1
        return 0

    def Z_Score(self, value):
        z_score = (value - mean)/sd
        return z_score

def download(dbx, folder, subfolder, name):
    """Download a file.
    Return the bytes of the file, or None if it doesn't exist.
    """
    path = '/%s/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'), name)
    while '//' in path:
        path = path.replace('//', '/')
    try:
        md, res = dbx.files_download(path)
    except dropbox.exceptions.HttpError as err:
        print('*** HTTP error', err)
        return None
    data = res.content
    #print(len(data), 'bytes; md:', md)
    return data

def getTrainTestDF(startTime, endTime):
    dbx = dropbox.Dropbox(config.dropbox_config['secretkey'])
    deviceName = config.dropbox_config['deviceid']
    accelPath = config.dropbox_config['pathtype']

    newDF = pd.DataFrame()

    for entry in dbx.files_list_folder('/{}/{}'.format(deviceName, accelPath)).entries:

        tmpNameWExt = entry.name
        currentFileStamp = int(tmpNameWExt.split(".")[0])

        if startTime <= currentFileStamp <= endTime:
            byteData = download(dbx, deviceName, accelPath, entry.name)

            with zipfile.ZipFile(io.BytesIO(byteData)) as zf:
                csv_filename = zf.namelist()[0]
                file_path = zf.open(csv_filename)
                df = pd.read_csv(file_path, usecols=[0,3], names=['TimeStamp', 'Z'])
                newDF = newDF.append(df, ignore_index = True)
                df = df.iloc[0:0] #clear data frame
    
    #print(newDF.info(memory_usage='deep'))  #Float64, Int64 (345kb for 4 files(4,000kb))
    my_processor = ScrappyProcessor()
    final_df = my_processor.parseFrame(newDF)
    return final_df

def supervised():
    if config.dropbox_config['deviceid'] == '':
        print("=====================")
        print("Please Input Device ID to read from Network Path")
        print("=====================")
        return
    if config.dropbox_config['secretkey'] == '':
        print("=====================")
        print("No Valid Dropbox Key Supplied")
        print("=====================")
        return


    # Training
    X,labels = getTrainTestDF(rounder(1539579774),rounder(1539579791)) # Route(1) with labels
    Y = labels.values.ravel()
    my_classifier = tree.DecisionTreeClassifier()
    my_classifier.fit(X, Y)

    # Predictions
    print("=======Normal Results=======")
    X,labels = getTrainTestDF(rounder(1539579998),rounder(1539580080)) # Route(2) with labels
    Y = labels.values.ravel()
    predictions = my_classifier.predict(X)
    #print(predictions)
    #print(Y)
    print(accuracy_score(Y, predictions)*100) #Percentage of Accuracy

    print("=======J5 Results=======")
    X,labels = getTrainTestDF(rounder(1540103640),rounder(1540103820)) # Route(2) with labels
    Y = labels.values.ravel()
    predictions = my_classifier.predict(X)
    #print(predictions)
    #print(Y)
    print(accuracy_score(Y, predictions)*100) #Percentage of Accuracy

    print("=======Silicone Results=======")
    X,labels = getTrainTestDF(rounder(1540103875),rounder(1540103943)) # Route(2) with labels
    Y = labels.values.ravel()
    predictions = my_classifier.predict(X)
    #print(predictions)
    #print(Y)
    print(accuracy_score(Y, predictions)*100) #Percentage of Accuracy

if __name__ == '__main__':
    supervised()
