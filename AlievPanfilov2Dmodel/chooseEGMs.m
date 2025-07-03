function [sel,col]=chooseEGMs(pars)
% chooses points in which EGMs are going to be displayed
% Marta, 31/05/2017
maxpts=9;

figure
litD=imresize(pars.D,[length(pars.elpos) length(pars.elpos)],'nearest');
imagesc(litD)
colormap spring
colorbar

disp(['Please click on up to ' num2str(maxpts) ...
    ' points. Press return to terminate selection.'])
[y,x]=ginput(maxpts);
sel(1,:)=round(x);
sel(2,:)=round(y);

for i=1:length(sel)
    if litD(sel(1,i),sel(2,i))==max(pars.D(:))
        col{i}='b';
    else
        col{i}='r';
    end
    text(sel(2,i),sel(1,i),num2str(i),'Color',col{i});
end
title([pars.savename ' - D (mm^2/ms)'])
saveas(gcf,[pars.savename '_elpos.fig'])
saveas(gcf,[pars.savename '_elpos.tif'])