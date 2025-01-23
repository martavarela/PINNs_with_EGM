function param = set_parameter(cyclelength,numcycle,numcells,numelec,cyclic)
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
param.displayt = round(5/param.dt);
param.savet = 100*param.dt;
% total time of simulation
param.BCL = cyclelength;
param.ncyc = numcycle;
param.extra = 0;
param.cyclic = cyclic;
param.heter = 0;
param.tend = (cyclelength*numcycle)+param.extra;

% number of electrodes
param.numelec = numelec;
% Calculate electrode grid properties
numelec_side = sqrt(numelec);
center = [param.X, param.Y] / 2;
spacing = [param.X, param.Y] ./ (numelec_side + 1);

% Generate grid coordinates
grid = arrayfun(@(c, s, n) linspace(c - s * (n - 1) / 2, c + s * (n - 1) / 2, n), ...
                center, spacing, repmat(numelec_side, 1, 2), 'UniformOutput', false);
% Create meshgrid and compute electrode positions
[X, Y] = meshgrid(grid{:});
param.elecpos = [X(:), Y(:)]' + 0.1;

% diffusion coefficient, D
param.D = 0.1;

% initial simulation details
param.stimloc = false(param.X,param.Y);
param.stimloc(1:numcells/10,:) = true; % horizontal-moving 
param.stimdur = 1;
if cyclic
    param.crossloc = false(param.X,param.Y);
    param.crossloc(:,1:floor(param.X/3)) = true; % horizontal-moving 
    param.crosstime = 5;
end

% spatial
param.h = 0.1;
end
