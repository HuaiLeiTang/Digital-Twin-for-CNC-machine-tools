%Five_three smoothing method
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

sampleFrequence=50000; %Sampling frequency value
m=300; % smoothing cycles
processingData=fx(1:3000);
processingLength=length(processingData);
time=(0:1/sampleFrequence:(processingLength-1)/sampleFrequence)';

a=processingData;
n=processingLength;

for k=1:m
    b(1)=(69*a(1)+4*(a(2)+a(4))-6*a(3)-a(5))/70;
    b(2)=(2*(a(1)+a(5))+27*a(2)+12*a(3)-8*a(4))/35;
    for j=3:n-2
        b(j)=(-3*(a(j-2)+a(j+2))+12*(a(j-1)+a(j+1))+17*a(j))/35;
    end
    b(n-1)=(2*(a(n)+a(n-4))+27*a(n-1)+12*a(n-2)-8*a(n-3))/35;
    b(n)=(69*a(n)+4*(a(n-1)+a(n-3))-6*a(n-2)-a(n-4))/70;
    a=b;
end

plot(time,processingData);
grid on;
hold on;

plot(time,a,'r');
legend('Raw','Five-three');
grid on;