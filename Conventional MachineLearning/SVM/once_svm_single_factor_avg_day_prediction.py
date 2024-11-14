# -*- coding: utf-8 -*-
"""LSTM_MulStep_Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1f53iOkrNGJVJbtltFXufONraFote2HWU
"""

# !wmic path win32_VideoController get Caption, DeviceID, VideoProcessor

# # # 指定GPU训练
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"  ##表示使用GPU编号为0的GPU进行计算
import csv

import matplotlib
import warnings
# 忽略警告
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

import matplotlib
import numpy as np

import pandas as pd
import warnings
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from numpy import array
import os
import glob
from pygame import mixer




def RF_ET0(filename):

    mixer.init()
    mixer.music.load('../../Resources/ExecuteTipAudio.mp3')

    file_name = filename
    df = pd.read_csv(file_name)

    # df = df.iloc[:, 7:]
    # df

    day = os.path.basename(file_name).split('_')[0]

    print("This day: ",day)

    # df = df.iloc[:, 7:]
    # df

    df.shape

    # 数据范围过大，对数据进行归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    sacler_data_len = len(df.columns)

    sacler_data_len

    #数据集划分
    def splitData(var, per_test):
        num_test = int(len(var) * per_test)
        train_size = int(len(var) - num_test)
        train_data = var[0:train_size]
        test_data = var[train_size:train_size + num_test]
        return train_data, test_data


    df_training, df_testing = splitData(scaled_data, 0.2)
    print(df_training.shape)
    print(df_testing.shape)

    def createXY(data,n_past,n_steps_out):
        dataX,dataY = list(),list()
        for i in range(len(data)):
            ## such as: len(data) = 19624
            end_ix = i + n_past ## 0+3=3,1+3=4,...,19618+3=19621
            out_end_ix = end_ix + n_steps_out ## 3+3=6,4+3=7,...,19621+3=19624
            # print('out_end_ix:',out_end_ix)
            '''
            加入此处判断，使得最终的dataX和dataY中每行的数组长度一致，进而可以转换为array(数组),如若不加以限制，
            则导致最后几次循环由于i是在len(data)范围内的，
            但是由于out_end_ix=end_ix + n_steps_out,最终会超出data的数据范围，而导致其无法获取到数据,
            加入dataY之后，最后几行的数据长度是和前面的数据长度不一致，最终导致无法进行array转换
            '''
            if out_end_ix > len(data): ## 6 < len(data),7<len(data),...,19623+3=19626>len(data)=19624
                # print("------------out_end_ix of end",out_end_ix,'---------------')
                break

            dataX.append(data[i:end_ix,0:data.shape[1]]) ## 0:3,0:7;1:4,0:7
            dataY.append(data[end_ix:out_end_ix,sacler_data_len-1]) ##3:6,6;4:7,6
        return array(dataX),array(dataY)

    n_past = 1
    n_output = 1
    trainX,trainY = createXY(df_training,n_past,n_output)
    testX,testY = createXY(df_testing,n_past,n_output)

    print('train Shape---', trainX.shape)
    print('trainY Shape---', trainY.shape)
    print('testX Shape---', testX.shape)
    print('testY Shape---', testY.shape)

    trainX = trainX.reshape(trainX.shape[0], -1)

    trainX.shape

    testX = testX.reshape(testX.shape[0], -1)

    testX.shape

    trainY = trainY.flatten()
    testY = testY.flatten()

    from sklearn.svm import SVR
    svr = SVR(kernel='rbf')
    model = svr.fit(trainX, trainY)

    mixer.music.play()
    time.sleep(1)
    mixer.music.stop()

    model.fit(trainX, trainY)

    y_pred = model.predict(testX)

    y_pred

    y_pred = np.array(y_pred).reshape(-1, 1)

    y_pred

    # prediction_copies_array = np.repeat(y_pred, sacler_data_len, axis=-1)
    #
    # prediction_copies_array

    # pred = scaler.inverse_transform(np.reshape(prediction_copies_array, (len(prediction_copies_array), sacler_data_len)))  #进行逆变换但是，只需要最后一列
    # pred = pred[:, sacler_data_len - 1]
    pred = scaler.inverse_transform(y_pred)


    test_data = np.array(testY).reshape(-1, 1)


    # original_copies_array = np.repeat(test_data, sacler_data_len, axis=-1)
    #
    # original_copies_array
    #
    # # print('IMF_True:',scaler.inverse_transform(np.reshape(original_copies_array, (len(testY), feature_num))))
    # y_true = scaler.inverse_transform(np.reshape(original_copies_array, (len(original_copies_array), sacler_data_len)))[:, sacler_data_len - 1]
    #
    # y_true
    y_true = scaler.inverse_transform(test_data)
    # y_true = y_true[0:len(y_true) - 1]
    # pred = pred[0:]

    y_true

    pred

    ## when n_output=15,LSTM happen error at row>68000 rows that r-square < 0
    y_trues = y_true[0:len(y_true)-n_output]
    preds = pred[n_output:]

    # y_trues = y_true
    # preds = pred

    # preds


    # print(pred)
    plt.plot(y_trues, color='red', label='Real Value')
    plt.plot(preds, color='blue', label='Pred Value')
    plt.title('Prediction Water Level(m)')
    plt.xlabel('Time (h)')
    plt.ylabel('Water Level(m)')
    plt.legend()
    # plt.savefig('../Images/SW_IN_F_Train_'+'n_past='+'{}'.format((trainX[1].shape)[0])+'_epochs={}'.format((grid_search.best_params_)['epochs'])+'.png',dpi=600)
    plt.show()

    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae
    from sklearn.metrics import r2_score as r2

    statistical_indicator = []

    print('MSE:', mse(y_trues, preds))
    print('MAE:', mae(y_trues, preds))
    print('R²:', r2(y_trues, preds))
    print('RMSE:', np.sqrt(mse(y_trues, preds)))

    statistical_indicator = {
        'R2':r2(y_trues, preds),
        'RMSE': np.sqrt(mse(y_trues, preds)),
        'MAE': mae(y_trues, preds)
    }
    # mode = 'w' :覆盖.mode= 'a' ： 追加
    with open('svm_58693_statistical_indicator.csv',mode='a',newline='') as file:
        writer = csv.writer(file)

        # 如果文件是空的，写入表头
        if file.tell() == 0:
            writer.writerow(['Day', 'R2', 'RMSE', 'MAE'])

        writer.writerow([str(day),statistical_indicator['R2'],statistical_indicator['RMSE'],statistical_indicator['MAE']])

    print('pred_finally:', y_trues)
    print('---------------Day is ',day,'----------------')


    result_save_file_true = pd.DataFrame(y_trues)
    # 使用rename方法来重命名列
    result_save_file_true = result_save_file_true.rename(columns={0: "True"})
    result_save_file_true.to_csv('Result/ET0_SingleFactor_20years_Result/58693/' + day + '_day_true.csv', index_label='Time')
    result_save_file_pred = pd.DataFrame(preds)
    result_save_file_pred = result_save_file_pred.rename(columns={0: "Pred"})
    result_save_file_pred.to_csv('Result/ET0_SingleFactor_20years_Result/58693/' + day + '_day_prediction.csv', index_label='Time')


import time
start_time = time.time()

csv_path = '../../Data/ProcessedData/SingleFactorData/58693/'
csv_files = glob.glob(os.path.join(csv_path,'*.csv'))
# moth = [3,6,9,12]
for file in csv_files:
    file_name =file
    day = file_name.split('\\')[1].split('_')[0]
    print(file)
    print('this is ',day,' day')
    print(file)
    RF_ET0(file)
    print("------------This is prediction ",day," day ---------------")

end_time = time.time()

execution_time = end_time - start_time

# 将时间差转换为时：分：秒格式
hours, remainder = divmod(execution_time, 3600)
minutes, seconds = divmod(remainder, 60)

# 格式化输出
formatted_time = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
print(f"代码运行时间: {formatted_time} 时: 分: 秒")