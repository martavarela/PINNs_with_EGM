param = set_parameter(75,3,100,8);
%print(param.elecpos)
[V,W]=AlievPanfilov2D(param);
phie = calcphie(param,V);

%param = fun_setpars_forNurin()
% loadFK_EGM_forNurin('test',1)

