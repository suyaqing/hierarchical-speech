function [DEM, demi] = spm_MDP_DEM_speech_gamma(sig, file)

% set up and preliminaries: first level
%--------------------------------------------------------------------------
rng('default')
SP = load(file);

%==========================================================================
% same as spm_MDP_DEM_speech, but the I vector is calculated from only one
% index encoding both the syllable identity and the gamma unit

% get spectrotemporal patterns for all possible syllables
%--------------------------------------------------------------------------
% global STIM
% global sig
global ichunk
% slist = {'a', 'c', 'g', 'k', 'm', 'mo', 'n', 'o', 'p', 'po', 'r', ...
%     's', 't', 'th', 'w', ''};
% STIM.S{i} stores the spectrotemporal pattern for syllable i, each an 6x8
% matrix
for i = 1:numel(SP.slist)
    STIM.S{i} = SP.I{i};
end
                      

% and gamma units
%--------------------------------------------------------------------------
Gamma = cell(1, 8);
for ig = 1:8
    % alternatively, use softmax function here
    Gamma{ig} = sparse(ig, 1, 1, 8, 1);
end
STIM.G = Gamma;


% mapping from outputs of higher (discrete) level to (hidden) causes
%==========================================================================

% true causes (U) and priors (C) for every combination of discrete states
%--------------------------------------------------------------------------
N     = 25;                                  % length of data sequence
ns    = length(STIM.S);                      % number of syllables
ng    = length(STIM.G);                      % number of gammas
for i = 1:ns
    for j = 1:ng
        c           = STIM.S{i}*STIM.G{j};
        u           = STIM.S{i}*STIM.G{j};
        demi.U{(i-1)*ng+j, 1} = u*ones(1,N);
        demi.C{(i-1)*ng+j, 1} = c*ones(1,N);
    end
end

Nchunk = floor(size(sig, 2)/N);
for ichunk = 1:Nchunk
    ind = ((ichunk-1)*N+1):ichunk*N;
    demi.Y{ichunk} = sig(:, ind);
end
ichunk = 0;

% evaluate true and priors over causes given discrete states
%--------------------------------------------------------------------------
o     = [4];
O{1}  = full(spm_softmax(sparse(1:(ns-1)*ng,1,1,ns*ng,1)));
% O{2}  = spm_softmax(sparse(1,1,1,ng,1));

% generative model
%==========================================================================
M(1).E.s = 1/2;                               % smoothness
M(1).E.n = 2;                                 % order of
M(1).E.d = 1;                                 % generalised motion

% hidden states
%--------------------------------------------------------------------------
x      = sparse(6, 1);                               % ST pattern
v      = sparse(6, 1);

% level 1: Displacement dynamics and mapping to sensory/proprioception
%--------------------------------------------------------------------------
M(1).f = @(x,v,P) Mf1(x, v, P);
M(1).g = @(x,v,P) x;
M(1).x = x;                                   % hidden states
M(1).V = exp(6);                                   % error precision (g): input
M(1).W = exp(6);                                   % error precision (f): state

% level 2:
%--------------------------------------------------------------------------
M(2).v = v;                                   % priors
M(2).V = ones(1, 6)*exp(8);


% generative process
%==========================================================================

% first level
%--------------------------------------------------------------------------
G(1).f = @(x,v,a,P) Gf1(x, v, a, P);
G(1).g = @(x,v,a,P) x;
G(1).x = x;                                  % hidden states
G(1).V = exp(16);                            % error precision
G(1).W = exp(16);                            % error precision
G(1).U = ones(1,ns);                  % gain

% second level
%--------------------------------------------------------------------------
G(2).v = v;                                  % exogenous forces
G(2).a = [0;0];                              % action forces
G(2).V = exp(8);

% generate and invert
%==========================================================================
DEM.G  = G;
DEM.M  = M;

% initialize
%--------------------------------------------------------------------------
DEM    = spm_MDP_DEM_model(DEM,demi,full(O),o);
% DEM    = spm_MDP_DEM(DEM,demi,full(O),o);
ichunk = 0;

% function f for generative model and process
function dx = Mf1(x, v, P)
D = 0.2 * diag(ones(1,6));
    
W = [-0.8881    0.4397    0.2279    0.2280   -0.0147    0.4345;
0.1931   -0.9626   -0.0836    0.1892    0.3324    0.0405;
0.4909   -0.1355   -0.7123   -0.5790   -0.0435   -0.5619;
0.0119    0.0580   -0.6032   -1.0000   -0.2894   -0.0376;
-0.4133    0.0856   -0.0541   -0.1186   -0.3464    0.1709;
0.5559    0.1764   -0.3075   -0.0122    0.4482   -0.9253];
% Hopfield network
kappa1 = 2;
x1 = x;
dx1 = kappa1.*(-D*x1 + W*tanh(x1) + v);
dx = dx1;

function dx = Gf1(x, v, a, P)
D = 0.2 * diag(ones(1,6));
    
W = [-0.8881    0.4397    0.2279    0.2280   -0.0147    0.4345;
0.1931   -0.9626   -0.0836    0.1892    0.3324    0.0405;
0.4909   -0.1355   -0.7123   -0.5790   -0.0435   -0.5619;
0.0119    0.0580   -0.6032   -1.0000   -0.2894   -0.0376;
-0.4133    0.0856   -0.0541   -0.1186   -0.3464    0.1709;
0.5559    0.1764   -0.3075   -0.0122    0.4482   -0.9253];

kappa1 = 2;
x1 = x;
dx1 = kappa1.*(-D*x1 + W*tanh(x1) + v);
dx = dx1;

