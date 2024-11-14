function [ ] = plotimf( IMF,num,color,name)
%   绘制imf
%   此处显示详细说明
for k_first=0:num:size(IMF,1)-1
    figure;
    clear k_second;
    for k_second=1:min(num,size(IMF,1)-k_first)
        subplot(num,1,k_second);
        plot(IMF(k_first+k_second,:),color);axis('tight');
        if(k_first==0 && k_second==1)
            title(name);
        end
       if k_first+k_second<size(IMF,1)
        ylabel(['IMF',num2str(k_first+k_second)]);
       else
        ylabel('残差');
        end
    end
end
disp(name)
end