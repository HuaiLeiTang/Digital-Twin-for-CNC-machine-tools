% Five-point sliding average smoothing method to denoise data
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

processingData=fx(1:10000);
processingLength=length(processingData);
time=(0:1/sampleFrequence:(processingLength-1)/sampleFrequence)';

m=30; %The cycles of sliding average smoothing

a=processingData;
n=processingLength;
for k=1:m
    b(1)=(3*a(1)+2*a(2)+a(3)-a(4))/5;
    b(2)=(4*a(1)+3*a(2)+2*a(3)+a(4))/10;
    for j=3:n-2
        b(j)=(a(j-2)+a(j-1)+a(j)+a(j+1)+a(j+2))/5;
    end
    b(n-1)=(a(n-3)+2*a(n-2)+3*a(n-1)+4*a(n))/10;
    b(n)=(-a(n-3)+a(n-2)+2*a(n-1)+3*a(n))/5;
    a=b;
end
y=a;
figure(1)
subplot(2,1,1);
plot(time,processingData);
xlabel('time(s)');
ylabel('acceleration(g)');
legend("Raw")
grid on;
subplot(2,1,2);
plot(time,y);
xlabel('time(s)');
ylabel('acceleration(g)');
grid on;
legend("Denoised")