function param = set_parameter(cyclelength,numcycle,numcells)
% create parameter structure for real-time simulation of Aliev-Panfilov
% model for EP properties analysis
% Nurin Mohd Amrilshah, 29/10/2024

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
param.nelec = 5;
% electrode positions
param.elecpos(1,:) = floor(2+linspace(param.X/(param.nelec+2),param.X-(param.X/(param.nelec+2)),param.nelec));
param.elecpos(2,:) = floor(2+linspace(param.Y/(param.nelec+2),param.Y-(param.Y/(param.nelec+2)),param.nelec));

% diffusion coefficient, D
param.D = 0.005;

% initial simulation details
param.stimloc1 = zeros(param.X,param.Y);
param.stimloc1(1:10,:) = 1; % horizontal-moving 
param.stimdur = 1;

% spatial
param.h = 0.1;



end