function [Vsav,fullPath]=AlievPanfilov2D(param)
% (Ventricular) Aliev-Panfilov model in single-cell with the parameters
% from Goektepe et al, 2010
% Nurin, 29/10/2024
% t is the time in AU - to scale do tms = t *12.9

disp('Simulating...')
function dydt = AlPan(y,Istim)
    % Aliev-Panfilov model parameters 
    a = 0.05;
    k = 8.0;
    mu1 = 0.2;
    mu2 = 0.3;
    epsi = 0.002;
    b  = 0.15;

    h = param.h; 
    D = param.D; %diffusion coefficient (for monodomain equation)
    
    % SA
    V=squeeze(y(1,:,:));
    W=squeeze(y(2,:,:));
    dV=4*D.*del2(V,h);
    dWdt=(epsi + mu1.*W./(mu2+V)).*(-W-k.*V.*(V-b-1));
    dVdt=(-k.*V.*(V-a).*(V-1)-W.*V)+dV+Istim;
    dydt(1,:,:)=dVdt;
    dydt(2,:,:)=dWdt;

    % AV
    
end

Ia=0.1; % AU, value for Istim when cell is stimulated

%initialisation for V and W
V(1:(param.ncells) + 2,1:(param.ncells)+2)=0; 
W(1:(param.ncells) + 2,1:(param.ncells) + 2)=0.01;

% array for saving data
Vsav=zeros(param.ncells,param.ncells,ceil(param.tend/param.gathert)); 
Wsav=zeros(param.ncells,param.ncells,ceil(param.tend/param.gathert));


ind=0; % iterations counter
stimind=0; % counter for number of stimuli applied

y=zeros(2,size(V,1),size(V,2));
t_values =[];
Istim = zeros((param.ncells) + 2,(param.ncells) + 2);

% for loop for explicit RK4 finite differences simulation
for t=param.dt:param.dt:param.tend
    ind=ind+1;
    
    %Istim = zeros(param.X, param.Y);

    % stimulate at every BCL time interval for ncyc times
    if param.cross
        if t<=param.stimdur
            Istim = Ia*param.stimgeo;
        elseif t>=param.crosstime&&t<=(param.crosstime+param.stimdur)
            Istim = Ia*param.crossgeo;
        else
            Istim = zeros((param.ncells) + 2,(param.ncells) + 2);
        end
    else % for plane wave
        if t>=(param.BCL*stimind)&&stimind<param.ncyc
            Istim = Ia*param.stimgeo; % stimulating current
        end
        if t>=(param.BCL*stimind + 2*param.stimdur)
            stimind = stimind + 1;
            Istim = zeros((param.ncells) + 2,(param.ncells) + 2);
        end
    end
   
    % Runge-Kutta calculation of current V and W
    y(1,:,:)=V;
    y(2,:,:)=W;
    k1=AlPan(y,Istim);
    k2=AlPan(y+param.dt/2.*k1,Istim);
    k3=AlPan(y+param.dt/2.*k2,Istim);
    k4=AlPan(y+param.dt.*k3,Istim);
    y=y+param.dt/6.*(k1+2*k2+2*k3+k4);
    V=squeeze(y(1,:,:));
    W=squeeze(y(2,:,:));

    % rectangular boundary conditions: no flux of V
    if  param.cross
        V(1,:)=V(2,:);
        V(end,:)=V(end-1,:);
        V(:,1)=V(:,2);
        V(:,end)=V(:,end-1);
    end
    
    % save V and W 
    if mod(ind,param.gathert)==0
        % save values
        t_values = [t_values, t];
        Vsav(:,:,round(ind/param.gathert))=V(2:end-1,2:end-1)';
        Wsav(:,:,round(ind/param.gathert))=W(2:end-1,2:end-1)';

        % show heatmap
        if mod(ind,param.displayt)==0
            subplot(2,1,1)
            imagesc(V(2:end-1,2:end-1)',[0 1])
            axis image
            set(gca,'FontSize',8)
            elecind = 0;
            for i=1:1:length(param.elecpos(1,:))
                elecind = elecind+1;
                text(param.elecpos(1,i),param.elecpos(2,i),num2str(elecind))
            end
            xlabel('x')
            ylabel('y')
            set(gca,'FontSize',8)
            title(['V (AU) - Time: ' num2str(t,'%.0f') ' ms'])
            colorbar
            
            subplot(2,1,2)
            imagesc(W(2:end-1,2:end-1)',[0 1])
            axis image
            set(gca,'FontSize',8)
            elecind = 0;
            for i=1:1:length(param.elecpos(1,:))
                elecind = elecind+1;
                text(param.elecpos(1,i),param.elecpos(2,i),num2str(elecind))
            end
            xlabel('x')
            ylabel('y')
            set(gca,'FontSize',8)
            title(['V (AU) - Time: ' num2str(t,'%.0f') ' ms'])
            colorbar
            set(gca,'FontSize',8)
            title('W (AU)')
            colorbar
            pause(0.01)
        end    
    end
end

    % Vsav = permute(Vsav, [3,2,1]);
    % Wsav = permute(Wsav, [3,2,1]);
    % to load into python: t, x, y, Vsav, Wsav
    x = 1:param.ncells;
    y = 1:param.ncells;
    t = t_values;
    disp(size(x))


    % correction based on Aliev-Panfilov model
    % Vsav = 100*Vsav - 80;
    % t_values = 12.9*t_values;

    % saving file
    directory = 'data_for_PINNs';  
    if ~exist(directory, 'dir')
        mkdir(directory);
    end

    if param.cross
        stimulation = 'spiral';
    else
        stimulation = 'plane';
    end

    if param.heter
        diffusivity = 'heter';
    else
        diffusivity = 'homo';
    end

    % Format the datetime to avoid invalid characters
    timestamp = datestr(datetime('now'),'dd-mm-yyyy_HH-MM-SS');
    % Construct the file name
    fileName = [timestamp,'_input_', stimulation, '_', diffusivity, '.mat'];

    fullPath = fullfile(directory, fileName);
    save(fullPath,'x','y','t','Vsav','Wsav');
end
