function [Vsav,Wsav]=AlievPanfilov2D(param)
% (Ventricular) Aliev-Panfilov model in single-cell with the parameters
% from Goektepe et al, 2010
% Nurin, 29/10/2024
% t is the time in AU - to scale do tms = t *12.9

%close all
% V is the electrical potential difference across the cell membrane in AU
function dydt = AlPan(y,Istim)
    % Aliev-Panfilov model parameters 
    a = 0.05;
    k = 8.0;
    mu1 = 0.2;
    mu2 = 0.3;
    epsi = 0.002;
    b  = 0.15;
    h = 0.1; % mm cell length
    D = param.D; % mm^2/UA, diffusion coefficient (for monodomain equation)
    
    V=squeeze(y(1,:,:));
    W=squeeze(y(2,:,:));
    dV=4*D.*del2(V,h);
    dWdt=(epsi + mu1.*W./(mu2+V)).*(-W-k.*V.*(V-b-1));
    dVdt=(-k.*V.*(V-a).*(V-1)-W.*V)+dV+Istim;
    dydt(1,:,:)=dVdt;
    dydt(2,:,:)=dWdt;
end

% stimulation parameters
BCL=param.BCL; % basic cycle length (time between stimuli) in AU;
ncyc=param.ncyc; % how many times stimulation happens;
extra=param.extra; % time after all the cycles during simulation in AU
ncells = param.ncells;
iscyclic=param.cyclic; % 1 for spiral, connecting the ends
flagmovie=1; % to write a movie of the electrical potential propagation%100;

[sel,col] = chooseEGMs(param);

% stimulation
stimgeo=param.stimloc1; % indices of cells where external stimulus is felt
dt=param.dt; % AU, time step for finite differences solver
displayt=param.displayt; % frequency at which V and W heatmap is displayed
gathert=param.gathert; % frequency of which data of V and W is saved

% for plotting, set to correspond to 1 ms, regardless of dt
tend=param.tend; % ms, duration of simulation
stimdur=param.stimdur; % UA, duration of stimulus
Ia=0.1*stimgeo; % AU, value for Istim when cell is stimulated

%initialisation for V and W
V(1:param.X,1:param.Y)=0; 
W(1:param.X,1:param.Y)=0.01;

% phie calculation
phiesav = [];
nsps = length(param.elecpos(1,:))*2;
calcphie = 1;
showphie = 100;


% array for saving data
Vsav=zeros(ncells,ncells,ceil(tend/gathert)); % array where V will be saved during simulation
Wsav=zeros(ncells,ncells,ceil(tend/gathert)); % array where W will be saved during simulation

ind=0; %iterations counter
stimind=0; %counter for number of stimuli applied

y=zeros(2,size(V,1),size(V,2));

% for loop for explicit RK4 finite differences simulation
for t=dt:dt:tend % for every timestep
    ind=ind+1; % count interations
        % stimulate at every BCL time interval for ncyc times
        if t>=BCL*stimind&&stimind<ncyc
            Istim=Ia; % stimulating current
        end
        % stop stimulating after stimdur
        if t>=BCL*stimind+stimdur*2
            stimind=stimind+1;
            Istim=zeros(param.X,param.Y); % stimulating current
        end
        
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
        if  ~iscyclic % 1D cable
            V(1,:)=V(2,:);
            V(end,:)=V(end-1,:);
            V(:,1)=V(:,2);
            V(:,end)=V(:,end-1);
        else % periodic boundary conditions in x, y or both
            % set up later - need to amend derivatives calculation too
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
                xlabel('x')
                ylabel('y')
                set(gca,'FontSize',8)
                title(['V (AU) - Time: ' num2str(t,'%.0f') ' ms'])
                colorbar
                
                subplot(2,1,2)
                imagesc(W(2:end-1,2:end-1)',[0 1])
                axis image
                set(gca,'FontSize',8)
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

        % if t>0
        %     for j=1:(nsps/2)
        %         subplot(nsps,1,j+floor(nsps/2))
        %         plot(max(1,t-showphie+1):t,squeeze(V(sel(1,j),sel(2,j))))
        %         te=ylabel(num2str(j),'Color',col{j});
        %         set(te,'rotation',0);
        %         set(gca,'YTick',[])
        %         xlim([max(1,t-showphie+1) t])
        %         if strcmpi(pars.col(j),'b')
        %             ylim([-1 1])
        %         else
        %             ylim([-0.5 0.5])
        %         end
        %         if j~=nsps/2
        %             set(gca,'XTickLabel',[])
        %         end
        %         grid on
        %     end
        % end

end
end
