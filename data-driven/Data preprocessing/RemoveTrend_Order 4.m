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
order=4; % fitting polynomial order
processingData=fx(1:1000);
processingLength=length(processingData);
time=(0:1/sampleFrequence:(processingLength-1)/sampleFrequence)';

a=polyfit(time,processingData,order);
y=processingData-polyval(a,time); % result after trend elimination

plot(time,processingData);
grid on;
hold on;
plot(time,y);
legend("Raw","Elimination");