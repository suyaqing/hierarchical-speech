function [xc, xt, xa, xr, xp, xm, xsyn] = spm_MDP_VB_ERP_hybrid(MDP)

Ne  = MDP.T;         % number of epochs
D = MDP.D;

% context
Nc = length(D{1}); xc = cell(Ne, Nc);
Nt = length(D{2}); xt = cell(Ne, Nt);
Na = length(D{3}); xa = cell(Ne, Na);
Nr = length(D{4}); xr = cell(Ne, Nr);
Np = length(D{5}); xp = cell(Ne, Np);
Nm = length(D{6}); xm = cell(Ne, Nm);
Nsyn = length(D{7}); xsyn = cell(Ne, Nsyn);

for k = 1:Ne
    for j = 1:Nc
        xc{k, j} = MDP.nxc(:, j, k);
    end
    for j = 1:Nt
        xt{k, j} = MDP.nxt(:, j, k);
    end
    for j = 1:Na
        xa{k, j} = MDP.nxs{1}(:, j, k);
    end
    for j = 1:Nr
        xr{k, j} = MDP.nxs{2}(:, j, k);
    end
    for j = 1:Np
        xp{k, j} = MDP.nxs{3}(:, j, k);
    end
    for j = 1:Nm
        xm{k, j} = MDP.nxs{4}(:, j, k);
    end
    for j = 1:Nsyn
        xsyn{k, j} = MDP.nxsyn{k}(:, j);
    end
end



