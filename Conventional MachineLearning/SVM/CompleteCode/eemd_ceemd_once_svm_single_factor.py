import csv

import matplotlib
import warnings
# 忽略警告
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import MaxPooling1D, Conv1D,Flatten
from keras.models import Sequential
from keras.layers import Dense
# from pygame import mixer
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
import time
import os
import glob
import matplotlib
import numpy as np
import pandas as pd
import warnings
import math
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score as r2, mean_squared_error
# 忽略警告
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score as r2


def calculate_nse(observed, predicted):
    """
    计算 Nash-Sutcliffe Efficiency (NSE) 指标。

    参数:
    - observed: 实测值数组或列表
    - predicted: 模拟值数组或列表

    返回:
    - NSE 指标值
    """
    observed = np.array(observed)
    predicted = np.array(predicted)

    # 检查两组数据长度是否一致
    if len(observed) != len(predicted):
        raise ValueError("observed 和 predicted 的长度必须一致")

    # 计算 NSE
    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)

    nse = 1 - (numerator / denominator)
    return nse

def Result_Ana(Model, feature_num,testX,testY,scaler):
    print('Result_Ana_Function:',testX.shape)
    y_pred = Model.predict(testX)
    y_pred = np.array(y_pred).reshape(-1, 1)

    pred = scaler.inverse_transform(y_pred) #进行逆变换但是，只需要最后一列

    print('testY of shape :',testY.shape)
    y_true = np.array(testY).reshape(-1, 1)

    y_true = scaler.inverse_transform(y_true)

    y_true = y_true[0:len(y_true) - 1]
    pred = pred[1:]

    plt.plot(y_true, color='red', label='Real Value')
    plt.plot(pred, color='blue', label='Pred Value')
    plt.title('Prediction ETO (mm)')
    plt.xlabel('Time (day)')
    plt.ylabel('ETO (mm)')
    plt.legend()
    plt.show()

    # 计算 NSE
    # nse = 1 - np.sum((y_true - pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    nse_value = calculate_nse(y_true, pred)

    # 计算 CA
    ca = 0.33 * (np.sqrt(mse(y_true,pred)) + mae(y_true,pred) + (1 - r2(y_true,pred)))



    print('NSE:',nse_value)
    print('MSE:',mse(y_true,pred))
    print('MAE:',mae(y_true,pred))
    print('R²:',r2(y_true,pred))
    print('RMSE:',np.sqrt(mse(y_true,pred)))
    print('CA:', ca)

    return y_true,pred


def ceemdan_cnn(day,model,station_id):
    # mixer.init()
    # mixer.music.load('../../../Resources/ExecuteTipAudio.mp3')

    reshape1 = 0
    IMF = pd.read_excel(f'../../../Data/{model}/SingleFactor_20years_ET0/{station_id}/'+f'{model}_{station_id} Station_'+day+'.xlsx',header=None)

    imf_choose = 0
    finally_rsult = []
    finally_y_true = []

    for imf_run in range(int(len(IMF.columns))):

        Single_Factor_IMF = IMF[imf_run]
        print('--------------------------',imf_run,'--------------------------')
        Single_Factor_IMF = np.array(Single_Factor_IMF)
        # IMF_Input = np.transpose(Single_Factor_IMF)

        df_IMF = pd.DataFrame(Single_Factor_IMF)
        df_IMF.columns = ['ET0']

        # print(df_IMF)

        scaler = MinMaxScaler(feature_range=(0, 1))
        df = scaler.fit_transform(df_IMF)

        sacler_data_len = len(df_IMF.columns)

        #数据集划分
        test_split = round(len(df) * 0.20)
        print(test_split)
        def splitData(var, per_test):
            num_test = int(len(var) * per_test)
            train_size = int(len(var) - num_test)
            train_data = var[0:train_size]
            test_data = var[train_size:train_size + num_test]
            return train_data, test_data


        df_training, df_testing = splitData(df, 0.2)
        print('df_training.shape:',df_training.shape)
        print('df_testing.shape:',df_testing.shape)

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

                dataX.append(data[i:end_ix,0]) ## 0:3,0:7;1:4,0:7
                # print('dataX:---------------')
                # print(data[i:end_ix,0])
                # print('dataX:---------------')
                dataY.append(data[end_ix:out_end_ix,data.shape[1]-1]) ##3:6,6;4:7,6
                # print('dataY:----------------------')
                # print(data[end_ix:out_end_ix,data.shape[1]-1])
                # print('dataY:----------------------')
            return np.array(dataX), np.array(dataY)

        n_past=1
        n_steps_out=1

        trainX, trainY = createXY(df_training, n_past, n_steps_out)
        testX, testY = createXY(df_testing, n_past, n_steps_out)

        reshape1 = testY.shape[0]-1

        print('trainX.Shape:----', trainX.shape)
        print('trainY.shape:----', trainY.shape)
        print('testX.shape:----', testX.shape)
        print('testY.shape:----', testY.shape)

        # trainY = trainY.reshape(-1)
        # testY = testY.reshape(-1)
        # print('trainY Shape 2 ---', trainY.shape)
        # print('testY Shape 2 ---', testY.shape)
        # trainX = np.expand_dims(trainX,axis=1)
        # testX = np.expand_dims(testX,axis=1)
        # print('expand_dims_trainX of shape',trainX.shape)
        # print('expand_dims_testX of shape',testX.shape)

        print('----------------------------',imf_run,'------------------------------')

        # trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1] * trainX.shape[2]))
        #
        # testX = np.reshape(testX, (testX.shape[0], testX.shape[1] * testX.shape[2]))

        print('reshape trainX.shape:', trainX.shape)
        print('reshape testX.shape:', testX.shape)

        # # Flatten trainY and testY to 1D arrays
        # trainY = trainY.flatten()
        # testY = testY.flatten()

        from sklearn.svm import SVR
        svr = SVR(kernel='rbf', degree=7, gamma='auto', C=0.1, )
        # Create MultiOutputRegressor with SVR as the base estimator
        model = svr.fit(trainX, trainY)

        y_trues, prediction_result = Result_Ana(model, sacler_data_len, testX, testY, scaler)

        # from xgboost import XGBRegressor
        #
        # xgb = XGBRegressor(objective='reg:squarederror', n_estimators=42)
        #
        # model = xgb.fit(trainX, trainY)
        #
        # y_trues,prediction_result = Result_Ana(model, sacler_data_len,testX,testY,scaler)

        print('prediction_result:',prediction_result)
        finally_rsult.append(prediction_result)
        print('lstm_y_turs:',y_trues)
        finally_y_true.append(y_trues)

        # mixer.music.play()
        # time.sleep(1)
        # mixer.music.stop()
    return reshape1,finally_y_true,finally_rsult,day,n_steps_out

def plot_save_true_prediction(reshape1,finally_y_true,finally_pre_rsult,day,n_steps_out,formatted_time,model,station_id):

    finally_ytrue_r = np.array(finally_y_true)

    reshape2 = len(finally_ytrue_r)

    finally_column_names = []

    for i in range(1, reshape2 + 1):
        if i < (reshape2):
            finally_column_names.append("IMF" + str(i))
        else:
            finally_column_names.append("Residual")

    ALL_Factor_ReIMFS_Yture_Result = np.transpose(finally_ytrue_r)
    reshape1

    ALL_Factor_ReIMFS_Yture_Result = ALL_Factor_ReIMFS_Yture_Result.reshape(reshape1, reshape2)

    all_factor_ceemdan_true = pd.DataFrame(ALL_Factor_ReIMFS_Yture_Result, columns=finally_column_names)
    all_factor_ceemdan_true

    sums = all_factor_ceemdan_true.iloc[:, :].sum(axis=1)

    # 将求和结果添加到DataFrame中作为新的一列
    all_factor_ceemdan_true["True"] = sums
    # 保存为CSV文件
    true_file_path = f"../Result/{model}_Single_Factor_20_years/{station_id}/"+day+f"_day_{model}_IMF" + str(reshape2 - 1) + "-XGBoost_true.csv"

    all_factor_ceemdan_true.to_csv(true_file_path,
                                   index=False)
    # WL(Water Level(m)) IMF1-IMF13-R Prediction Value

    finally_pred = np.array(finally_pre_rsult)
    finally_pred
    ALL_Factor_ReIMFS_Result = np.transpose(finally_pred)
    ALL_Factor_ReIMFS_Result = ALL_Factor_ReIMFS_Result.reshape(reshape1, reshape2)
    finally_column_names
    all_factor_ceemdan_prediction = pd.DataFrame(ALL_Factor_ReIMFS_Result, columns=finally_column_names)
    all_factor_ceemdan_prediction
    sums = all_factor_ceemdan_prediction.iloc[:, :].sum(axis=1)

    # 将求和结果添加到DataFrame中作为新的一列
    all_factor_ceemdan_prediction["Pred"] = sums
    prediction_file_path = f"../Result/{model}_Single_Factor_20_years/{station_id}/"+day+f"_day_{model}_IMF" + str(reshape2 - 1) + "-XGBoost_prediction.csv"
    # 保存为CSV文件
    all_factor_ceemdan_prediction.to_csv(prediction_file_path, index=False)
    ### 验证最终结果
    df = pd.read_csv(prediction_file_path)
    df = df[['Pred']]
    df

    origin = pd.read_csv(true_file_path)
    origin = origin[['True']]
    # y_true = origin.iloc[:,len(origin.columns)-1]
    y_true = origin
    # y_true = y_true[0:len(y_true)-n_steps_out]
    # pred = df[n_steps_out:]
    # y_true
    pred = df
    plt.plot(y_true, color='red', label='Real Value')
    plt.plot(pred, color='blue', label='Pred Value')
    plt.title('Prediction ET0 ')
    plt.xlabel('Time')
    plt.ylabel('Detail Value')
    plt.legend()

    plt.show()

    # 计算 NSE
    # nse = 1 - np.sum((y_true - pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    nse_value = calculate_nse(y_true, pred)

    # 计算 CA
    ca = 0.33 * (np.sqrt(mse(y_true, pred)) + mae(y_true, pred) + (1 - r2(y_true, pred)))

    print('NSE:', nse_value)
    print('MSE:', mse(y_true, pred))
    print('MAE:', mae(y_true, pred))
    print('R²:', r2(y_true, pred))
    print('RMSE:', np.sqrt(mse(y_true, pred)))
    print('CA:', ca)


    statistical_indicator = {
        'R2': r2(y_true, pred),
        'RMSE': np.sqrt(mse(y_true, pred)),
        'MAE': mae(y_true, pred),
        'NSE': nse_value,
        'CA':ca
    }

    with open(f'{model}_xgboost_{station_id}_statistical_indicator.csv',mode='a',newline='') as file:
        writer = csv.writer(file)

        # 如果文件是空的，写入表头
        if file.tell() == 0:
            writer.writerow(['Day', 'R2', 'RMSE', 'MAE','NSE','CA','Execution Time'])

        writer.writerow([str(day),statistical_indicator['R2'],statistical_indicator['RMSE'],statistical_indicator['MAE'],statistical_indicator['NSE'],
                         statistical_indicator['CA'],formatted_time])

start_times = time.time()

if __name__ == '__main__':
    days = ['one','three','five','seven','ten','fifteen']
    model=['CEEMD','EEMD']

    station_id = ['58606','58618','58693']
    for ml in model:
        for sd in station_id:
            for day in days:

               start_time = time.time()

               print('------------------',day ,'--------------------')
               print('This is ',day,'-th for prediction et0')
               reshape1,finally_y_true,finally_pre_rsult,day,n_steps_out = ceemdan_cnn(day,ml,sd)
               print('This is ',day,'-th for prediction et0')
               print('------------------',day,'---------------------')

               end_time = time.time()

               execution_time = end_time - start_time


               # 将时间差转换为时：分：秒格式
               hours, remainder = divmod(execution_time, 3600)
               minutes, seconds = divmod(remainder, 60)

               # 格式化输出
               formatted_time = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
               plot_save_true_prediction(reshape1,finally_y_true,finally_pre_rsult,day,n_steps_out,formatted_time,ml,sd)


end_time = time.time()

execution_time = end_time - start_times

# 将时间差转换为时：分：秒格式
hours, remainder = divmod(execution_time, 3600)
minutes, seconds = divmod(remainder, 60)

# 格式化输出
formatted_time = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
print(f"代码运行时间: {formatted_time} 时: 分: 秒")