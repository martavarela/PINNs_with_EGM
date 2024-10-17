function [iniu,iniv,iniw]=iniFK_2D(pars,savname,moviename)
% Fenton-Karma model in 2D
% same implementation as in FentonKarma2.m, following setup in June3_in2D.m
% Marta, 19/11/2015

close all
% clear all
figure
if exist('moviename','var')
    writerObj = VideoWriter(strcat(moviename,'.avi'));
    open(writerObj);
end

if exist('savname','var')
    savname = 'test';
end

X=pars.X;
Y=pars.Y;
pacegeo=pars.pacegeo;
crossgeo=pars.crossgeo;
D=pars.D;
dt=pars.dt;
gathert=1000; %pars.gathert;
crosstime=pars.crosstime;
stimdur=pars.stimdur;
nms=400;
h=pars.h;

% X=120;
% Y=120;
% pacegeo=zeros(X,Y);
% crossgeo=zeros(X,Y);
% crosstime=102;
% pacegeo(2:10,:)=1;
% crossgeo(:,end-30:end)=1;
% crossgeo(:,1:30)=1;
% nms=300;
% stimdur=2;
% D=1*ones(X,Y);
% % D(x(1):x(2),y(1):y(2))=0;
% dt=0.010; % should be 0.005 ms
% gathert=round(10/dt); % every ms

uv=0.160; % uc for v
uw=0.160; % uc for w
uu=0.160; % uc for u
uvsi=0.040; % uv
ucsi=0.85; %uc_si 
k=10; % k
taud=0.125; % tau_d
tauv2=60; % tauv2-
tauv1=82.5; % tauv1-
tauvplus=5.75; % tauv+
tauo=32.5; 
tauwminus=400; % tauw- 
tauwplus=300; % tauw+ 
taur=70; % taur
tausi=114; % tausi

tend=500;
ua=0.95;
v=0.99*ones(X,Y); % v
w=0.99*ones(X,Y); % w
u=zeros(X,Y);

ind=0;
for ms=dt:dt:nms 
        ind=ind+1;
        if ms<stimdur
            u(pacegeo==1)=ua;
        end
        if abs(ms-crosstime)<dt
            u(crossgeo==1)=0;
            v(crossgeo==1)=0.99;
            w(crossgeo==1)=0.99;        
        end
            
        %fast inward current and gate
        Fu=zeros(X,Y);
        vinf=ones(X,Y);
        tauv=tauvplus*ones(X,Y);
        
        vinf(u>=uv)=0;
        tauv(u<uvsi&u<uv)=tauv1;
        tauv(u>=uvsi&u<uv)=tauv2;
        Fu(u>=uv)=(u(u>=uv)-uv).*(1-u(u>=uv));

        %fast inward current
        Jfi=Fu.*(-v)./taud;

        %update v
        v=v+(vinf-v)./tauv.*dt;

        %ungated outward current
        Uu=ones(X,Y);
        Uu(u<=uu)=u(u<=uu);
        tauu=taur*ones(X,Y);
        tauu(u<=uu)=tauo;

        Jso=Uu./tauu;

        %slow inward current and slow gate
        winf=ones(X,Y);
        winf(u>=uw)=0;
        tauw=tauwminus*ones(X,Y);
        tauw(u>=uw)=tauwplus;
        Jsi=-w./tausi.*0.5.*(1+tanh(k.*(u-ucsi)));

        % update w
        w=w+(winf-w)./tauw*dt;

        Iion=-(Jfi+Jsi+Jso);
        u=u+(Iion+4*D.*del2(u,h)).*dt;
                
        % boundary conditions
        u(1,:)=u(2,:);
        u(end,:)=u(end-1,:);
        u(:,1)=u(:,2);
        u(:,end)=u(:,end-1);
        
        % show images
        if mod(ind,gathert)==0
            imagesc(u(1:end,1:end),[0 1])
            colorbar
            text(5,5,['t:' num2str(ms) ' ms'],'color','k')
            disp(['t:' num2str(ms) ' ms']) 
            pause(0.01);
            if exist('moviename','var')
                fram = getframe;
                writeVideo(writerObj,frame);
            end
        end
        
 end % end of ms loop
 if exist('moviename','var')
    close(writerObj);   
 end
disp('End.');

iniv=v;
iniw=w;
iniu=u;

save(savname,'iniv','iniw','iniu','pars','-v7');
