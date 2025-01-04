function y = AliPanODE(y0)
% referred to https://www.ibiblio.org/e-notes/html5/ap.html
% from Aliev & Panfilov, 1995
% V and t are in arbitary units
% for homogenous and isotropic conduction
[t,y] = ode15s(@solveAliPan,[0;100],y0);

    function dydt = solveAliPan(t,y)
        % parameters
        a = 0.05;
        b = 0.15;
        k = 8.0;
        eps0 = 0.002;
        mu1 = 0.2;
        mu2 = 0.3;
        d2V = 0; % isotropic condition 
        V=y(1);
        W=y(2);
        dV = d2V -(k*V)*(V-a)*(V-1)-(V*W);
        dW = (eps0 + ((mu1*W)/V+mu2))*(-W-((k*V)*(V-b-1)));
        dydt = [dV;dW];
    end
    
hold on
plot(t,y(:,1))
plot(t,y(:,2))
legend('V','W')
xlabel('Time(AU)')
ylabel('AU')
title('Electrical Potential and Refractory Parameter of One Cardiac Cell')

end


