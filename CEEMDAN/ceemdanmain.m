%% clear
clc;
clear;
close all;

day = ["one","three","five","seven","ten","fifteen"]
for j=1:6
    file_start = "Data/Need_Des_Data/58618/";
    days = day(j);
    end_file = "_day_20years_data_58618";
    suffix_file = ".csv";
    
    csv_file_path = strcat(file_start,days,end_file,suffix_file);
    disp(csv_file_path)
    %% 数据导入
    data_sw = csvread(csv_file_path);
    % disp(data_sw(2:20,1))
    file_save_name = days;
    file_path = "ET0/Result/CEEMDAN_20_years_ET0/58618_SingleFactor/";
    filename = "CEEMDAN_58618 Station_";
    suffix = ".xlsx";

    %%
    a_index = 0
    for i=1
        data = data_sw(2:end, i);
        disp(data)
        %% 参数设置
        Nstd=0.2; %信噪比,一般0-1
        NR=100;   %添加噪音次数,一般50-100
        Maxlter=10; %内部最大包络次数设定，即分量个数
        ceemdan_imf=ceemdan(data,Nstd,NR,Maxlter); 


        %%  图形绘制
        plotimf(ceemdan_imf,size(ceemdan_imf,1),'r',' CEEMDAN分解结果'); %画图
    %   a_index = a_index + 1
        str = strcat(file_path,filename,days,suffix);
        %% 对结果进行转置，方便存入xlsx或者csv中
        ceemdan_imf_t = ceemdan_imf.'

        %% 
        xlswrite(str, ceemdan_imf_t);
    end
end
