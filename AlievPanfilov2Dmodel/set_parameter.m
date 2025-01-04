function param = set_parameter(cyclelength,numcycle,numcells,numelec)
% create parameter structure for real-time simulation of Aliev-Panfilov
% mofel for EP properties analysis
% Nurin Mohd Amrilshah, 29/10/2024
close all;
clc;
% size of rectangle
param.ncells = numcells;
param.X = numcells+2;
param.Y =numcells+2;

% forward time step in ms
param.dt = 0.01;
param.gathert = round(1/param.dt);
param.displayt = round(10/param.dt);
param.savet = 100*param.dt;
% total time of simulation
param.BCL = cyclelength;
param.ncyc = numcycle;
param.extra = 0;
param.cyclic = 0;
param.tend = (cyclelength*numcycle)+param.extra;

% number of electrodes
param.numelec = numelec;
% electrode positions
% Number of electrodes along x and y axis
numelec_x = numelec;  % Example: 5 electrodes along x
numelec_y = numelec;  % Example: 5 electrodes along y

% Domain dimensions
centerX = param.X / 2;
centerY = param.Y / 2;

% Spacing between electrodes (assuming electrodes should be spaced evenly in the middle)
spacing_x = param.X / (numelec_x + 1);  % Spacing in x-direction
spacing_y = param.Y / (numelec_y + 1);  % Spacing in y-direction

% Define grid coordinates for electrodes in the center of the domain
gridx = linspace(centerX - (spacing_x * (numelec_x - 1)) / 2, centerX + (spacing_x * (numelec_x - 1)) / 2, numelec_x);
gridy = linspace(centerY - (spacing_y * (numelec_y - 1)) / 2, centerY + (spacing_y * (numelec_y - 1)) / 2, numelec_y);

% Create meshgrid of electrode positions
param.elecpos = meshgrid(gridx,gridy);

% diffusion coefficient, D
param.D = 0.1;

% initial simulation details
param.stimloc1 = zeros(param.X,param.Y);
param.stimloc1(1:10,:) = 1; % horizontal-moving 
param.stimdur = 1;

% spatial
param.h = 0.1;



end
