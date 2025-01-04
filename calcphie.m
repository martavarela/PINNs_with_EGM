function phie = calcphie(param,Vsav)

sz = size(Vsav);
tmax = sz(3);
tini = 2;
X = sz(1);
Y = sz(2);
D = param.D;
if isscalar(param.D)
    D = param.D*ones(X,Y);
elseif size(param.D,1) == sz(1)+2&&size(D,2)==sz(2)+2
    D = param.D(2:end-1,1);
end
h = param.h;
[Dx,Dy]=gradient(D,h);

elecpos = param.elecpos;
[xel,yel] = meshgrid(elecpos);

K=1; % phie scalar constant

tt=0;
figure
for t=tini:tmax
    tt = tt+1;
    V = squeeze(Vsav(:,:,t));

    [gx,gy] = gradient(V,h);
    du = 4.*D*del2(V,h)+(Dx.*gx)+(Dy.*gy);
    
    % calculate phie
    for k=1:length(param.elecpos)
        [matx,maty] = meshgrid((2:X-1)-xel(k),(2:Y-1)-yel(k));
        % euclidian distance
        distance = sqrt(matx.^2+maty.^2);
        sum1 = sum((du(2:X-1,2:Y-1)'./distance));
        phie(k,tt) = -sum(sum1);
    end

    % display
    for j = 1:length(param.elecpos)
        subplot(length(param.elecpos),1,j)
        plot(tini:t,squeeze(phie(j,1:tt)),LineWidth=1,Color= "k")
        hold on
        grid on
        ylabel([num2str(j)],'Color','k')
        %set(gca,'YTick',[])
        xlim([tini t+1])
        %set(gca,'YTick',-K*5:K*5:K*5)
        title(['Electrode at (',num2str(round(param.elecpos(1,j))),',',num2str(round(param.elecpos(1,j))),')'])

    end
end
end
