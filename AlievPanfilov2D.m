function [Vsav,Wsav]=AlievPanfilov2D(param)
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
    D = param.D; diffusion coefficient (for monodomain equation)
    
    V=squeeze(y(1,:,:));
    W=squeeze(y(2,:,:));
    dV=4*D.*del2(V,h);
    dWdt=(epsi + mu1.*W./(mu2+V)).*(-W-k.*V.*(V-b-1));
    dVdt=(-k.*V.*(V-a).*(V-1)-W.*V)+dV+Istim;
    dydt(1,:,:)=dVdt;
    dydt(2,:,:)=dWdt;
end

Ia=0.1; % AU, value for Istim when cell is stimulated

%initialisation for V and W
V(1:param.X,1:param.Y)=0; 
W(1:param.X,1:param.Y)=0.01;

% array for saving data
Vsav=zeros(ncells,ncells,ceil(tend/gathert)); 
Wsav=zeros(ncells,ncells,ceil(tend/gathert)); 

ind=0; % iterations counter
stimind=0; % counter for number of stimuli applied

y=zeros(2,size(V,1),size(V,2));
t_values =[];
Istim = zeros(param.X,param.Y);

% for loop for explicit RK4 finite differences simulation
for t=dt:dt:tend
    ind=ind+1;
    t_values = [t_values, t];

    % stimulate at every BCL time interval for ncyc times
    if t<=param.BCL*stimind && stimind<param.ncyc
        Istim=Ia*param.stimgeo; % stimulating current
    end
     
    if param.cross
        if t>=(BCL*stimind+param.crosstime)
            Istim = Ia*param.crossloc;
        end
    end

    % stop stimulating after stimdur
    if t>=BCL*stimind+stimdur*2
        stimind=stimind+1;
        Istim=zeros(param.X,param.Y); % stimulating current
    end
    
    % Runge-Kutta calculation of current V and W
    y(1,:,:)=V;
    y(2,:,:)=W;
    k1=AlPan(y,Istim);
    k2=AlPan(y+dt/2.*k1,Istim);
    k3=AlPan(y+dt/2.*k2,Istim);
    k4=AlPan(y+dt.*k3,Istim);
    y=y+dt/6.*(k1+2*k2+2*k3+k4);
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
    if mod(ind,gathert)==0
        % save values
        Vsav(:,:,round(ind/gathert))=V(2:end-1,2:end-1)';
        Wsav(:,:,round(ind/gathert))=W(2:end-1,2:end-1)';

        % show heatmap
        if flagmovie&&mod(ind,displayt)==0
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
    % to load into python: t, x, y, Vsav, Wsav
    t = t_values;
    x = 1:param.h:param.X;
    y = 1:param.h:param.Y;
 
end

