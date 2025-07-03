clear;
clc;
close;

%% Set Parameter
param = set_AP_parameter(false,false);
%% Aliev Panfilov stimulation model
[Vsav,filepath] = AlievPanfilov2D(param);
%% EGM Calculation
% clc;
phie = calcphie(param,Vsav,filepath);


