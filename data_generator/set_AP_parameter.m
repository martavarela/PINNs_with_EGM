function param = set_AP_parameter(cross,heter)
% mode : 0 for plane wave, 1 for cross-stimulation (spiral)
clc;
close all;
param.ncells  = 25;
param.X = param.ncells+2;
param.Y = param.ncells+2;
param.h = 0.1;

param.dt = 0.005;
param.gathert = round(1/param.dt);
param.displayt = round(1/param.dt);
param.savet = 100*param.dt;

param.BCL = 75;
param.ncyc = 1;
param.extra = 0;
param.cyclic = 0;
param.tend = (param.BCL*param.ncyc)+param.extra;
param.heter = heter;
param.cross = cross;

param.numelec = 9;
numelec_side = sqrt(param.numelec);
center = [(param.ncells) + 2, (param.ncells) + 2] / 2;
spacing = [(param.ncells) + 2, (param.ncells) + 2] ./ (numelec_side + 1);
grid = arrayfun(@(c, s, n) linspace(c - s * (n - 1) / 2, c + s * (n - 1) / 2, n), ...
                center, spacing, repmat(numelec_side, 1, 2), 'UniformOutput', false);
[X, Y] = meshgrid(grid{:});
param.elecpos = [X(:), Y(:)]' + 0.1;

if heter
    D0 = 0.1;
    Dfac = 0.2;
    param.D = D0*ones((param.ncells) + 2,(param.ncells) + 2);
    fibloc=[floor(param.X/3) ceil(param.X/3+param.X/5)];
    param.D(fibloc(1):fibloc(2),fibloc(1):fibloc(2))=D0*Dfac;
else
    %param.D = 0.1;
    param.D = 0.1*ones((param.ncells) + 2,(param.ncells) + 2);
end

param.stimgeo = false((param.ncells) + 2,(param.ncells) + 2);
param.stimgeo(1:5,:) = true;
param.stimdur = 5;

if cross
    param.crossgeo = false((param.ncells) + 2,(param.ncells) + 2);
    param.crossgeo(:,1:floor(param.X/3))=true;
    param.crosstime = 30;
end


end