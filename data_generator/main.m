clear;
clc;
close;

%% Set Parameter
param = set_AP_parameter(false,false);
%% Aliev Panfilov stimulation model
[Vsav,filepath] = AlievPanfilov2D(param);
%% EGM Calculation
phie = calcphie(param,Vsav,filepath);

%%
figure;
hold on
plot(squeeze(Vsav(5,5,:)))
plot(squeeze(Vsav(50,50,:)))


