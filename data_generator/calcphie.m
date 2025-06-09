function phie=calcphie(param,Vsav,filePath)
disp('Calculating phie...')
sz = size(Vsav);
tmax = sz(3);
tini = 1;
X = sz(1);
Y = sz(2);

D = param.D;
if isscalar(param.D)
    D = param.D*ones(X,Y);
elseif size(param.D,1) == sz(1)+2&&size(D,2)==sz(2)+2
    D = param.D(2:end-1,2:end-1);
end
h = param.h;
[Dx,Dy]=gradient(D,h);

elecpos = round(param.elecpos)+0.1;
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
        [matx,maty] = meshgrid((2:X-1)-elecpos(1,k),(2:Y-1)-elecpos(2,k));
        % euclidian distance
        distance = sqrt(matx.^2+maty.^2);
        mask_radius = 5;
        valid_mask = (distance <= mask_radius) & (distance > 0); % Exclude electrode point
        distance(~valid_mask) = inf;

        sum1 = sum((du(2:X-1,2:Y-1)'./distance));
        phie(k,tt) = -sum(sum1);
        % if t == 1 && k==3
        %     disp('--------elecpos------------')
        %     disp(elecpos(1,k))
        %     disp(elecpos(2,k))
        %     disp(t)
        %     disp('--------du------------')
        %     disp(du(2:X-1,2:Y-1)')
        %     disp(distance)
        % end
    end


    % display
    rows_per_column = sqrt(param.numelec);
    for j = 1:length(param.elecpos)
        row = mod(j-1, rows_per_column) + 1;  % Row index (1 to rows_per_column)
        col = ceil(j / rows_per_column);
        subplot(rows_per_column, ceil(length(param.elecpos) / rows_per_column), (row-1)*ceil(length(param.elecpos) / rows_per_column) + col);
        plot(tini:t,squeeze(phie(j,1:tt)),LineWidth=1.5)
        hold on
        grid on
        ylabel('E(au)','Color','k')
        xlabel('Time(au)','Color','k')
        xlim([tini t+1])
        title(['Electrode at ', num2str(j),' (',num2str(round(param.elecpos(1,j))),',',num2str(round(param.elecpos(2,j))),')'])
    end
    sgtitle('{\phi_e} Traces Measured from Electrodes')
end

elecpos = param.elecpos;
disp(param.elecpos)
save(filePath,'phie','elecpos','-append')
end
