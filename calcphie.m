function phie = calcphie(param,Vsav)
disp('Calculating phie...')
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
    rows_per_column = 6;
    for j = 1:length(param.elecpos)
        row = mod(j-1, rows_per_column) + 1;  % Row index (1 to rows_per_column)
        col = ceil(j / rows_per_column);
        subplot(rows_per_column, ceil(length(param.elecpos) / rows_per_column), (row-1)*ceil(length(param.elecpos) / rows_per_column) + col);
        plot(tini:t,squeeze(phie(j,1:tt)),LineWidth=1,Color= "k")
        hold on
        grid on
        ylabel('E(au)','Color','k')
        xlabel('Time(au)','Color','k')
        xlim([tini t+1])
        title(['Electrode at ', num2str(j),' (',num2str(round(param.elecpos(1,j))),',',num2str(round(param.elecpos(2,j))),')'])
    end
end
end
