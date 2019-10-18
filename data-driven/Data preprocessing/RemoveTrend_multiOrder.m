% Function: 
% Elimination of trend terms in vibration signals by Least Square method with different orders.
% From the result figure, it can been seen that 1-4 order is mostly used in the elimination, because the vibration
% data is very large and the tred curve will be almost the same after 4 order.
% 
% Author:
% Weichao Luo
% 
% Date: 
% 2019.10.14

clear
clc
close all hidden

rawData = csvread('Train_A_001.csv');% raw data csv file read
fx=rawData(:,1); % X direction cutting force
fy=rawData(:,2); % Y direction cutting force
fz=rawData(:,3); % Z direction cutting force
vx=rawData(:,4); % X direction vibration
vy=rawData(:,5); % Y direction vibration
vz=rawData(:,6); % Z direction vibration
ae=rawData(:,7); % Acoustic Emission Sensor data

% Elimination of Trend Terms in Vibration Signals by Least Square Method

sampleFrequence=50000; %Sampling frequency value
processingData=fx(1:1000);
processingLength=length(processingData);
time=(0:1/sampleFrequence:(processingLength-1)/sampleFrequence)';

for n=1:9
    ANSS=polyfit(time,processingData,n);  %Fitting curve with "polyfit"
    for i=1:n+1           
       answer(i,n)=ANSS(i);
       % answer matrix stores the coefficients of the equation obtained at each time, 
       % and stores them in columns.
   end
    x0=time;
    %Initialization and construction of polynomial equation based on the obtained coefficients
    y0=ANSS(1)*x0.^n; 
    for num=2:1:n+1     
        y0=y0+ANSS(num)*x0.^(n+1-num);
    end
    subplot(3,3,n)
    plot(time,processingData)
    hold on
    plot(x0,y0)
end
suptitle('Trend eliminating with different order polynomials,1 to 9')