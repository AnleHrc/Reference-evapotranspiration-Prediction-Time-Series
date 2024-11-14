%% clear
clc;
clear;
close all;
%% 数据导入

data_sw = xlsread('data.csv');

% 
data = data_sw(1:end, 7);
%% 参数设置
Nstd=0.2; %信噪比,一般0-1s
NR=100;   %添加噪音次数,一般50-100
Maxlter=10; %内部最大包络次数设定，即分量个数
ceemdan_imf=ceemdan(data,0.2,100,Maxlter); 

%%  图形绘制
plotimf(ceemdan_imf,size(ceemdan_imf,1),'r',' CEEMDAN分解结果'); %画图

%% 对结果进行转置，方便存入xlsx或者csv中
ceemdan_imf_t = ceemdan_imf.'
    
%% 
xlswrite("ET0/Result/CEEMDAN/58606_fifteen_day/CEEMDAN_58606 Station.xlsx", ceemdan_imf_t);
