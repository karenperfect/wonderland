# -*- coding: utf-8-sig -*-

from collections import defaultdict
import csv
import pandas as pd
from matplotlib import pyplot as plt
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance
from sklearn.neural_network import MLPClassifier
import numpy as np


def data_processing():
    rawDataFileList = ['part-00000-e6120af0-10c2-4248-97c4-81baf4304e5c-c000',
                       'part-00001-e6120af0-10c2-4248-97c4-81baf4304e5c-c000',
                       'part-00002-e6120af0-10c2-4248-97c4-81baf4304e5c-c000',
                       'part-00003-e6120af0-10c2-4248-97c4-81baf4304e5c-c000',
                       'part-00004-e6120af0-10c2-4248-97c4-81baf4304e5c-c000',
                       'part-00005-e6120af0-10c2-4248-97c4-81baf4304e5c-c000',
                       'part-00006-e6120af0-10c2-4248-97c4-81baf4304e5c-c000',
                       'part-00007-e6120af0-10c2-4248-97c4-81baf4304e5c-c000',
                       'part-00008-e6120af0-10c2-4248-97c4-81baf4304e5c-c000',
                       'part-00009-e6120af0-10c2-4248-97c4-81baf4304e5c-c000']

    # --------------------------------------------------------------------------- Getting the data needed.
    dataInfo = defaultdict(dict)
    for rawDataFile in rawDataFileList:
        rawData = csv.DictReader(open(rawDataFile + '.csv'))
        for row in rawData:
            bookingID = row['bookingID']
            rowData = [int(float(row['second'])),
                       float(row['Speed']),
                       int(float(row['Bearing'])),
                       float(row['acceleration_x']),
                       float(row['acceleration_y']),
                       float(row['acceleration_z']),
                       float(row['gyro_x']),
                       float(row['gyro_y']),
                       float(row['gyro_z']),
                       int(float(row['Accuracy']))]
            try:
                dataInfo[bookingID].append(rowData)
            except:
                dataInfo[bookingID] = []
                dataInfo[bookingID].append(rowData)
    return dataInfo


def feature_engineering(dataInfo):
    # ----------------------------------------------- Data reading.
    label = csv.DictReader(open('labels.csv'))
    labelInfo = dict()
    for row in label:
        labelInfo[row['bookingID']] = int(row['label'])

    # ----------------------------------------------- Calculate features for each bookingID
    featureInfo = {}
    finalFeature = defaultdict(dict)                # to save the final feature output for each bookingID

    count = 0
    for key, value in dataInfo.items():
        bookingID = key
        highestRiskMark = -10000
        count += 1
        print(count)
        bookingData = sorted(value, key=lambda x: x[0])
        iterations = int(np.floor(len(bookingData) / 10)) + 1

        # Feature calculation for each window period
        for i in range(0, iterations, 1):
            try:
                filteredData = bookingData[i * 10: 10]
            except:
                filteredData = bookingData[i * 10:]

            if len(filteredData) <= 3:
                continue

            # --------------------------------------- Calculations
            speed = [x[1] for x in filteredData]      # 速度危险度
            accX = [abs(x[3]) for x in filteredData]  # 换道危险度
            accY = [abs(x[4]) for x in filteredData]  # 加速/制动危险度
            accZ = [x[5] for x in filteredData]
            gyroX = [abs(x[6]) for x in filteredData] # 车道保持危险度
            gyroY = [abs(x[7]) for x in filteredData]
            gyroZ = [abs(x[8]) for x in filteredData]
            bearing = [x[2] for x in filteredData]    # 轮胎磨损程度
            accuracy = [x[9] for x in filteredData]

            # --------------------------------------- Feature calculation (giving marks based on threshold values for each metric)
            avgSpeed = sum(speed)/len(speed) - 5

            avgAccX = sum(accX)/len(accX) - 3
            sigmaAccX = (sum([(x - sum(accX)/len(accX)) ** 2 for x in accX])**0.5 - 2)/2
            avgAccY = sum(accY)/len(accY) - 8
            sigmaAccY = (sum([(x - sum(accY)/len(accY)) ** 2 for x in accY])**0.5 - 2)/2
            avgAccZ = sum(accZ)/len(accZ) - 4
            sigmaAccZ = (sum([(x - sum(accZ)/len(accZ)) ** 2 for x in accZ])**0.5 - 3)/2

            avgGyroX = sum(gyroX)/len(gyroX)/0.5
            sigmaGyroX = ((sum([(x - sum(gyroX)/len(gyroX)) ** 2 for x in gyroX])**0.5)-0.5)/0.5
            avgGyroY = sum(gyroY)/len(gyroY)/0.5
            sigmaGyroY = (sum([(x - sum(gyroY)/len(gyroY)) ** 2 for x in gyroY])**0.5-0.5)/0.5
            avgGyroZ = sum(gyroZ)/len(gyroZ)/0.5
            sigmaGyroZ = (sum([(x - sum(gyroZ)/len(gyroZ)) ** 2 for x in gyroZ])**0.5-0.5)/0.5

            avgBearing = (sum(bearing)/len(bearing)-200)/10
            avgAccuracy = sum(accuracy)/len(accuracy) - 6

            riskMark = avgSpeed + avgAccX + sigmaAccX + avgAccY + sigmaAccY + avgAccZ + sigmaAccZ   # the most important metrics contribute to the risk mark

            if riskMark > highestRiskMark:
                highestRiskMark = riskMark
                featureInfo[bookingID]=[avgSpeed,
                                       avgAccX,
                                       sigmaAccX,
                                       avgAccY,
                                       sigmaAccY,
                                       avgAccZ,
                                       sigmaAccZ,
                                       avgGyroX,
                                       sigmaGyroX,
                                       avgGyroY,
                                       sigmaGyroY,
                                       avgGyroZ,
                                       sigmaGyroZ,
                                       avgBearing,
                                       avgAccuracy]

        # Final feature output for each bookingID
        featureList = ['avgSpeed',
                       'avgAccX',
                       'sigmaAccX',
                       'avgAccY',
                       'sigmaAccY',
                       'avgAccZ',
                       'sigmaAccZ',
                       'avgGyroX',
                       'sigmaGyroX',
                       'avgGyroY',
                       'sigmaGyroY',
                       'avgGyroZ',
                       'sigmaGyroZ',
                       'avgBearing',
                       'avgAccuracy']

        for i in range(0, len(featureList), 1):
            finalFeature[bookingID][featureList[i]] = featureInfo[bookingID][i]

    # ----------------------------------------------- Write out the final features
    header = ['bookingID'] + featureList + ['label']
    writer = csv.DictWriter(open('Safety_final_feature_mark_final.csv', 'wb'), header)
    writer.writeheader()

    for bookingID, value in finalFeature.items():
        value['bookingID'] = bookingID
        value['label'] = labelInfo[bookingID]
        writer.writerow(value)


def xgboost():
    # -------------------------------- Data preparation
    df = pd.read_csv('Safety_final_feature_mark_final.csv')
    trainSize = int(float(df.shape[0])*0.8)

    train = df[: trainSize]
    test = df[trainSize :]
    featureList = ['avgSpeed',
                   'avgAccX',
                   'sigmaAccX',
                   'avgAccY',
                   'sigmaAccY',
                   'avgAccZ',
                   'sigmaAccZ',
                   'avgGyroX',
                   'sigmaGyroX',
                   'avgGyroY',
                   'sigmaGyroY',
                   'avgGyroZ',
                   'sigmaGyroZ',
                   'avgBearing',
                   'avgAccuracy'
                   ]

    trainX = train[featureList]
    trainY = train['label']
    testX = test[featureList]
    testY = test['label']

    # -------------------------------- CV and para selection
    #xgbModel = xgb.XGBClassifier(objective='rank:pairwise', learning_rate=0.1)
    #clf = GridSearchCV(xgbModel,
    #                   {'max_depth': [4, 5, 6, 7, 8],
    #                    'n_estimators': [300, 500, 800, 1000]},
    #                   verbose=1)
    #clf.fit(trainX, trainY)
    #print(clf.best_score_)
    #print(clf.best_params_)

    # -------------------------------- Train model and print metrics
    gbm = xgb.XGBClassifier(max_depth=5,
                            learning_rate=0.1,
                            silent=1,
                            eval_metric='auc',
                            n_estimators=500,
                            objective='rank:pairwise',
                            reg_lambda=0.01
                            )
    gbm.fit(trainX, trainY)

    predictY = gbm.predict(trainX)
    predictY = (predictY >= 0.5) * 1
    print 'AUC: %.4f' % metrics.roc_auc_score(trainY, predictY)
    print 'ACC: %.4f' % metrics.accuracy_score(trainY, predictY)
    print 'F1-score: %.4f' % metrics.f1_score(trainY, predictY)

    predictY = gbm.predict(testX)
    predictY = (predictY >= 0.5) * 1
    print 'AUC: %.4f' % metrics.roc_auc_score(testY, predictY)
    print 'ACC: %.4f' % metrics.accuracy_score(testY, predictY)
    print 'F1-score: %.4f' % metrics.f1_score(testY, predictY)

    # -------------------------------- Importance
    #plot_importance(gbm)
    #plt.show()




def main():
    #dataInfo = data_processing()
    #print('Finish reading.')
    #feature_engineering(dataInfo)
    xgboost()


if __name__ == '__main__':
    main()