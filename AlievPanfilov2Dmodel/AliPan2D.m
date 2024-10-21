function [Vdata,Wdata] = AliPan2D(pars,nsim)
% nsim : how many action potential/stimulation

X = pars.X;
Y = pars.Y;
dt = pars.dt;
tend = 0.1; %pars.nms;
stimdur = pars.stimdur;
% initial condition of the whole area
V= 0.01*ones(X,Y);
W = 0.01*ones(X,Y);

y = [V;W];

%node properties
pacegeo = pars.pacegeo;
Vstim = 1.0;
tbetween = 10; % for 60bpm

% stimulation
ind = 0;
stimcount = 0;
for t = dt:dt:tend %frame by frame
    ind = ind+1;
    % start stimulation
    if t>=(tbetween*stimcount)
        V(pacegeo==1) = Vstim;
    end
    % when stimulation ends
    if t>=(tbetween*stimcount + stimdur)
        stimcount = stimcount+1;
    end

    %update V and W
    y=AliPanODE(y);
    
    









end
