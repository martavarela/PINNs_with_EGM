function loadFK_EGM(loadnam,vers)
% Fenton-Karma model in 2D
% same implementation as in FentonKarma2.m, following setup in June3_in2D.m
% Marta, 19/11/2015

close all

pars=fun_setpars_forNurin;
iniFK_2D_forNurin(pars,'test');

tolu=0.0001; % tolerance for oldu - u in termination clause
timemouse=0.5; % wait time for mouse click
warning off;
movieflag=1;

% load state variables so that a rotor is present initially
load(loadnam);
pars.savename = strcat(loadnam, 'v1')

if vers==1
    [sel,col]=chooseEGMs(pars);
else
    sel=pars.sel;
    col=pars.col;
end
pars.sel=sel;
pars.col=col;

figure
set(gcf, 'Position', [0, 0, pars.szscreenx, pars.szscreeny]);
if movieflag
    moviename=strcat(pars.savename,'_v', num2str(vers));
    writerObj = VideoWriter(moviename);
    writerObj.FrameRate = 3;
    open(writerObj);
end

kind=0;
phiesav=[];
nsps=length(sel)*2; % 2*number of EGMs shown
nhole=0;
X=pars.X;
Y=pars.Y;
% rad=pars.rad;
D=pars.D;
if isscalar('D')
    D=D*ones(X,Y);
end
h=pars.h;
logD=logical(D);
[FX,FY] = gradient(logD,h);
[xx,yy]=find(D==0&(FX~=0|FY~=0));
dt=pars.dt;
showms=10;
calcphie=1;
showphie=220;
nms=500; % total time of trace in ms
elpos=pars.elpos;
indi=1;

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

% load data
v=iniv;
w=iniw; 
u=iniu; 
oldu=u;

phie=[];
[Dx,Dy]=gradient(D,h);

timind=0;
for ms=0:dt:nms 
        timind=timind+1;
        %fast inward current and gate
        Fu=zeros(X,Y);
        vinf=ones(X,Y);
        tauv=tauvplus.*ones(X,Y);
        
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
        tauu=taur.*ones(X,Y);
        tauu(u<=uu)=tauo;

        Jso=Uu./tauu;

        %slow inward current and slow gate
        winf=ones(X,Y);
        winf(u>=uw)=0;
        tauw=tauwminus.*ones(X,Y);
        tauw(u>=uw)=tauwplus;
        Jsi=-w./tausi.*0.5.*(1+tanh(k.*(u-ucsi)));

        % update w
        w=w+(winf-w)./tauw.*dt;

        % update u
        Iion=-(Jfi+Jsi+Jso);
        [gx,gy]=gradient(u,h);
        du=4*D.*del2(u,h); %+Dx.*gx+Dy.*gy;
        u=u+(Iion+du).*dt;
                      
        % rectangular boundary conditions
        if ~pars.iscyclic(1)
            u(1,:)=u(2,:);
            u(end,:)=u(end-1,:);
        else
            u(1,:)=u(end-1,:);
            u(end,:)=u(2,:);
        end
        if ~pars.iscyclic(2)
            u(:,1)=u(:,2);
            u(:,end)=u(:,end-1);
        else
            u(:,1)=u(:,end-1);
            u(:,end)=u(:,2);
        end
        
        % holes boundary conditions
        if any(D(:)==0)
            for k=1:length(xx)
                if xx(k)==1||xx(k)==X||yy(k)==1||yy(k)==Y
                   u(xx(k),yy(k))=0;
                   continue; 
                end
                u(xx(k),yy(k))=u(xx(k)-logD(xx(k)-1,yy(k))+logD(xx(k)+1,yy(k)),...
                    yy(k)-logD(xx(k),yy(k)-1)+logD(xx(k),yy(k)+1));
            end
        end
        
        if mod(ms,calcphie)==0&&ms>0
            kind=kind+1;
            for k1=sel(1,:) %1:length(elpos)
                for k2=sel(2,:) %1:length(elpos)
                    [matx,maty]=meshgrid((2:X-1)-elpos(1,k1),(2:Y-1)-elpos(2,k2));
                    phie1=-sum(sum(du(2:X-1,2:Y-1)'./sqrt(matx.^2+maty.^2))); 
                    % [matx,maty]=meshgrid((2:X-1)-elpos(1,k1)-1,(2:Y-1)-elpos(2,k2)-1);
                    % phie2=-sum(sum(du(2:X-1,2:Y-1)'./sqrt(matx.^2+maty.^2))); 
                    phie(k1,k2,kind)=phie1; 
                end
            end
        end
        
        % show images
        if mod(ms,showms)==0
            disp(['t:' num2str(ms) ' ms'])
            kind=0;
            if ms>0
                phiesav=cat(3,phiesav,phie);
            end
            % if ms>showphie % keep EGM shown to showphie length
            %     % phiesav(:,:,max(1,end-showphie):end)=[];
            % end
            subplot(nsps,1,1:floor(nsps/2))

            imagesc(u,[0 1])
            colorbar
            text(5,5,['t:' num2str(ms) ' ms'],'color','k','FontSize',16)
            % title(['Ablation lesions: ' ...
            %     num2str(length(find(~D(:)&pars.D(:)))./(X*Y)*100,'%.1f') ' %'])

            if isfield(pars,'bnd')
                    hold all
                    for i=1:length(pars.bnd)
                        B=pars.bnd{i};
                        plot(B(:,2),B(:,1),'k-')
                    end
                    hold off
                    axis off
            end
            % pts = mouseinput_timeout(timemouse, gca); % waits 0.5 s for a click

            if ms>0
                for j=1:nsps/2
                    subplot(nsps,1,j+floor(nsps/2))
                    plot(max(1,ms-showphie+1):ms,...
                        squeeze(phiesav(sel(1,j),sel(2,j),...
                        max(1,ms-showphie+1):ms)),'-',...
                        'LineWidth',2,'Color',col{j})
                    te=ylabel(num2str(j),'Color',col{j});
                    set(te,'rotation',0);
                    set(gca,'YTick',[])
                    xlim([max(1,ms-showphie+1) ms])
                    if strcmpi(pars.col(j),'b')
                        ylim([-1 1])
                    else
                        ylim([-0.5 0.5])
                    end
                    if j~=nsps/2
                        set(gca,'XTickLabel',[])
                    end
                    grid on
                end
                xlabel('Time (ms)')
            end

            % if ~isempty(pts)
            %     xi=pts(:,2);
            %     yi=pts(:,1);
            %     rec(indi).x=xi;
            %     rec(indi).y=yi;
            %     rec(indi).time=ms;
            %     indi=indi+1;
            %     pts=[];
            %     for i=1:length(xi)
            %         if (yi(i)>1&&yi(i)<Y-1&&xi(i)>1&&xi(i)<X-1)
            %         	nhole=nhole+1;
            %             [matx,maty]=meshgrid((1:Y)-round(yi(i)),(1:X)-round(xi(i)));
            %             D((matx.^2+maty.^2)<rad^2)=0;
            %             u((matx.^2+maty.^2)<rad^2)=0;
            %         end
            %     end
            %     logD=logical(D);
            %     [FX,FY] = gradient(logD,h);
            %     [xx,yy]=find(D==0&(FX~=0|FY~=0));
            %     [Dx,Dy] = gradient(D,h);
            % end
            
            if movieflag
                frame = getframe(gcf);
                writeVideo(writerObj,frame);
            end
            
            % termination clause
            if all(abs(oldu-u)<tolu)
                disp(['Terminated at ' num2str(ms) ' ms.'])
                break;
            end
            oldu=u;
        end % end of visualisation loop       
 end % end of ms loop
 if movieflag
    close(writerObj);   
 end

 eval(strcat('phiesav',num2str(vers),'=phiesav;'))
 eval(strcat('Vsav',num2str(vers),'=Vsav;'))
 eval(strcat('rec',num2str(vers),'=rec;'))
 disp('End.');
 save(pars.savename(1:end-3),'rec','-append');
 save('rec','rec','-append');
 save([pars.savename '_ver' num2str(vers)],'pars',...
     strcat('Vsav',num2str(vers)),strcat('phiesav',num2str(vers)),...
    strcat('rec',num2str(vers)),'-v7.3','-nocompression')
