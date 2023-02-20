function [MDP] = spm_MDP_VB_X_hybrid(MDP, prediction)
% active inference and learning using variational message passing
% FORMAT [MDP] = spm_MDP_VB_X(MDP,OPTIONS)

global ichunk

% set up and preliminaries
%==========================================================================

% defaults
%--------------------------------------------------------------------------
try, stept   = MDP.stept;   catch, stept = 256;    end % learning rate
try, stepc   = MDP.stepc;   catch, stepc   = 256;    end % update time constant
try, stepsyn   = MDP.stepsyn;   catch, stepsyn   = 8;    end % update time constant
try, steps   = MDP.steps;   catch, steps   = 16*ones(1,4); end % Occam window updates
try, erp   = MDP.erp;   catch, erp   = 2;    end % update reset


T = MDP.T;
D = MDP.D;
L = MDP.L;
A = MDP.A;
Z = MDP.Z;
Nc = length(D{1});
Nt = length(D{2});
Nw = size(A{1}, 1);

W = cell(1, T);
W0 = cell(1, T);
% initialise model-specific variables
%==========================================================================
Ni    = 16;                                % number of VB iterations

% initialize x, X, nx
Xc = zeros(Nc, T);
Xc(:, 1) = D{1};
xc = D{1};
nxc = zeros(Ni, Nc, T);

Xt = zeros(Nt, T);
Xt(:, 1) = D{2};
xt = D{2};
nxt = zeros(Ni, Nt, T);

Xs = cell(1, 4); xs = Xs; nxs = Xs;
for ks = 1:4
    Ns = size(A{ks+1}, 2);
    Xs{ks} = zeros(Ns, T);
    Xs{ks}(:, 1) = D{ks+2};
    xs{ks} = D{ks+2};
    nxs{ks} = zeros(Ni, Ns, T);
end

Nsyn = size(D{end}, 1);
Xsyn = cell(1, T); xsyn = Xsyn; nxsyn = Xsyn;
for tau = 1:T    
    Xsyn{tau} = zeros(Nsyn, 1);
    xsyn{tau} = D{end}(:, tau);
    nxsyn{tau} = zeros(Ni, Nsyn);
end

xc(:) = 1/length(xc);
xt(:) = 1/length(xt);
for ks = 1:4
    xs{ks}(:) = 1/length(xs{ks});
end
for tau = 1:4
    xsyn{tau}(:) = 1/length(xsyn{tau});
end
F = zeros(Ni, T);



% belief updating over successive time points
%==========================================================================
for t = 1:T
    
    % calculate word prediction
    %======================================================================
    if t==1
        W0{1} = A{1};
    else
        W0{t} = zeros(Nw, 1);
        syn = Z{t}*Xt(:, t);
        W0{t} = W0{t}+A{1}*syn(1);
        for ksyn = 2:5
            W0{t} = W0{t}+syn(ksyn)*A{ksyn}*Xs{ksyn-1}(:, t);
        end
    end
    try
        mdp = MDP.mdp(t);
    catch
        try
            mdp     = spm_MDP_update(MDP.MDP(t),MDP.mdp(t - 1));
        catch
            try
                mdp = spm_MDP_update(MDP.MDP(1),MDP.mdp(t - 1));
            catch
                mdp = MDP.MDP(1);
            end
        end
    end
    mdp.factor = 1;
    if prediction
        mdp.D{1} = spm_softmax(spm_log(W0{t})/erp);
        MDP.mdp(t) = spm_MDP_VB_X(mdp);
    else
%         mdp.D{1} = spm_softmax(spm_log(W0{t})/erp);
        mdp.D{1} = ones(Nw, 1)/Nw;
        OPTIONS.pred = 0;
        MDP.mdp(t) = spm_MDP_VB_X(mdp, OPTIONS);
    end
    
    W{t} = MDP.mdp(t).X{1}(:, 1);
               
    for i = 1:Ni
        
        
        % Variational updates 
        %==================================================================
            
            % processing time and reset
            %--------------------------------------------------------------
            tstart = tic;
           
            
            % Variational updates (hidden states) under sequential policies
            %==============================================================
            % context
            v0 = spm_log(xc);
            BU = zeros(length(v0), 1);
            for ks = 1:4
                for kst = 1:2
                    ll = xt(kst)*squeeze(spm_log(L{ks}(:, :, kst)));
                    BU = BU + ll'*xs{ks};
                end
            end
            dFdx = v0 - BU - spm_log(D{1});
            dFdx = dFdx - mean(dFdx);
            sxc = spm_softmax(v0 - dFdx/stepc);
            F(i, t) = F(i, t) + sxc'*(spm_log(sxc) - spm_log(D{1}));
            
            % type
            v0 = spm_log(xt);
            BU1 = zeros(length(v0), 1);
            BU2 = BU1;
            for tau = 1:t
                BU1 = BU1 + spm_log(Z{tau}')*xsyn{tau};
            end
            for ks = 1:4
                for kst = 1:2
                    ll = xs{ks}'*squeeze(spm_log(L{ks}(:, :, kst)));
                    BU2(kst) = BU2(kst) + ll*xc;
                end
            end
            dFdx = v0 - BU1 - BU2 - spm_log(D{2});
            dFdx = dFdx - mean(dFdx);
            sxt = spm_softmax(v0 - dFdx/stept);
            F(i, t) = F(i, t) + sxt'*(spm_log(sxt) - spm_log(D{2}));
            
            % syntax
            for tau = 1:t
                v0 = spm_log(xsyn{tau});
                BU = zeros(length(v0), 1);
                BU(1) = W{tau}'*spm_log(A{1});
                for ksyn = 2:5
                    BU(ksyn) = W{tau}'*(spm_log(A{ksyn})*xs{ksyn-1});
                end
                TD = spm_log(Z{tau})*xt;
                dFdx = v0 - BU - TD;
                dFdx = dFdx - mean(dFdx);
                sxsyn{tau} = spm_softmax(v0 - dFdx/stepsyn);
                F(i, t) = F(i, t) + sxsyn{tau}'*(spm_log(sxsyn{tau}) - spm_log(Z{tau})*sxt);
            end
            
            % semantic
            for ks = 1:4
                v0 = spm_log(xs{ks});
                BU = zeros(length(v0), 1);
                TD = BU;
                ww = zeros(Nw, 1);
                if t>1
                    for tau = 2:t
                        ww = ww + W{tau}*xsyn{tau}(ks+1);
                    end
                end
                BU = spm_log(A{ks+1}')*ww;
                for kst = 1:2
                    ll = xt(kst)*squeeze(spm_log(L{ks}(:, :, kst)));
                    TD = TD + ll*xc;
                end
                dFdx = v0 - BU - TD;
                dFdx = dFdx - mean(dFdx);
                sxs{ks} = spm_softmax(v0 - dFdx/steps(ks));
                F(i, t) = F(i, t) + sxs{ks}'*spm_log(sxs{ks});
                for kst = 1:2
                    ll = sxt(kst)*squeeze(spm_log(L{ks}(:, :, kst)));
                    F(i, t) = F(i, t) - sxs{ks}'*(ll*sxc);
                end
            end
            
            
            
            xc = sxc;
            Xc(:, t) = sxc;
            nxc(i, :, t) = sxc;
            if t<T
                Xc(:, t+1) = sxc;
            end
            
            xt = sxt;
            Xt(:, t) = sxt;
            nxt(i, :, t) = sxt;
            if t<T
                Xt(:, t+1) = sxt;
            end
            
            for ks = 1:4
                xs{ks} = sxs{ks};
                Xs{ks}(:, t) = sxs{ks};
                nxs{ks}(i, :, t) = sxs{ks};
                if t<T
                    Xs{ks}(:, t+1) = Xs{ks}(:, t);
                end
            end
            
            for tau = 1:t
                xsyn{tau} = sxsyn{tau};
                Xsyn{tau}(:) = sxsyn{tau};
                nxsyn{tau}(i, :) = sxsyn{tau};
            end
            % Free energy
            %--------------------------------------------------------------
            for tau = 1:t
                F(i, t) = F(i, t) - xsyn{tau}(1)*W{tau}'*spm_log(A{1});
                for ksyn = 2:5
                    F(i, t) = F(i, t) - xsyn{tau}(ksyn)*W{tau}'*(spm_log(A{ksyn})*xs{ksyn-1});
                end
            end
           
    end
    
    xc(:) = 1/length(xc);
    xt(:) = 1/length(xt);
    for ks = 1:4
        xs{ks}(:) = 1/length(xs{ks});
    end
    for tau = 1:4
        xsyn{tau}(:) = 1/length(xsyn{tau});
    end
   
            
            
            
    % processing (i.e., reaction) time
    %--------------------------------------------------------------
    rt(t)      = toc(tstart);


end % end of loop over time


    
    % assemble results and place in NDP structure
    %----------------------------------------------------------------------
   
MDP.Xc  = Xc;       % Bayesian model averages over T outcomes
MDP.nxc = nxc;
MDP.Xt = Xt;
MDP.nxt = nxt;
MDP.Xs = Xs;
MDP.nxs = nxs;
MDP.Xsyn = Xsyn;
MDP.nxsyn = nxsyn;
MDP.W = W;
MDP.W0 = W0;
MDP.F = F;

MDP.rt = rt;        % simulated reaction time (seconds)
    



% auxillary functions
%==========================================================================

function A  = spm_log(A)
% log of numeric array plus a small constant
%--------------------------------------------------------------------------
A  = log(A + 1e-16);

function A  = spm_norm(A, mode)
% normalisation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
A           = bsxfun(@rdivide,A,sum(A,1));
if nargin<2 || mode==1
    A(isnan(A)) = 1/size(A,1);
else
    A(isnan(A)) = 0;
end


function MDP = spm_MDP_update(MDP,OUT)
% FORMAT MDP = spm_MDP_update(MDP,OUT)
% moves Dirichlet parameters from OUT to MDP
% MDP - structure array (new)
% OUT - structure array (old)
%__________________________________________________________________________

% check for concentration parameters at this level
%--------------------------------------------------------------------------
try,  MDP.a = OUT.a; end
try,  MDP.b = OUT.b; end
try,  MDP.c = OUT.c; end
try,  MDP.d = OUT.d; end
try,  MDP.e = OUT.e; end

% check for concentration parameters at nested levels
%--------------------------------------------------------------------------
try,  MDP.MDP(1).a = OUT.mdp(end).a; end
try,  MDP.MDP(1).b = OUT.mdp(end).b; end
try,  MDP.MDP(1).c = OUT.mdp(end).c; end
try,  MDP.MDP(1).d = OUT.mdp(end).d; end
try,  MDP.MDP(1).e = OUT.mdp(end).e; end

return




