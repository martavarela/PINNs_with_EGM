function pars=fun_setpars()
% creates pars structure for real-time ablation simulations
% homogeneous D parameters
% adapted on 21/02/2023

% size
pars.X=102; % two extra for boundary condition purposes 
pars.Y=102;
pars.dt=0.010; % time step for forward Euler (ms)
pars.gathert=round(10/pars.dt); % frame display counter (for ini function)
pars.savet=100; % frame saving interval (100*dt = 1 ms)
pars.nms=600; % end time for simulation (for ini function)
n=20; 
pars.nelec=n^2; % number of electrodes

% regular positioning of electrodes
pars.elpos(1,:)=2+linspace(pars.X./(n+2),pars.X-pars.X./(n+2),n);
pars.elpos(2,:)=2+linspace(pars.Y./(n+2),pars.Y-pars.Y./(n+2),n);

pars.model='Goodman1';
if strcmp(pars.model,'Goodman1')
      pars.D= 0.005; % for h = 0.1, lower values below lead to spontaneous break-up
    % pars.D=0.1; % homogeneous isotropic diffusion coefficient (mm2/ms)
%     pars.D=0.15; % homogeneous isotropic diffusion coefficient (mm2/ms)
%     pars.D=0.115; % homogeneous isotropic diffusion coefficient (mm2/ms)
elseif strcmp(pars.model,'Goodman2')
elseif strcmp(pars.model,'Goodman2')
    pars.D=0.12; % homogeneous isotropic diffusion coefficient (mm2/ms)
end

% initialisation details
pars.pacegeo=zeros(pars.X,pars.Y);
pars.pacegeo(1:20,:)=1; % for first stimulus
pars.crossgeo=zeros(pars.X,pars.Y);
pars.crossgeo(:,round(pars.Y/2):end)=1; % for final stimulus
if strcmp(pars.model,'Goodman1')
    % pars.crosstime=102; % timing of last stimulus (ms)
    pars.crosstime=92; % timing of last stimulus (ms)
elseif strcmp(pars.model,'Goodman2')
    pars.crosstime=205; % timing of last stimulus (ms)
end
pars.stimdur=2; % duration of each stimulus (ms)

% display setup
pars.szscreenx=1183; 
pars.szscreeny=821;

pars.diff=1;
pars.iscyclic=[0 0]; % no perodic boundary conditions
pars.iso=1; % isotropic
pars.h=0.1; %mm, spatial resolution 

% uncomment below for display of heterogeneities in diffusion coefficient
% binD=imbinarize(pars.D);
% [B,L,N,A] = bwboundaries(~binD,'holes');
% pars.bnd=B;