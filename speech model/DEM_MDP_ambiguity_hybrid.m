function MDP = DEM_MDP_ambiguity_hybrid(sen)
% syntax at the same level as semantics, and represented as transiion
% matrices. semantic+syntax directly output words/phrases
%__________________________________________________________________________
%
%
%--------------------------------------------------------------------------
rng('default')
% close all
% clear
% sen = 1;
global ichunk
ichunk = 0;
prediction = 0;
d = load('Knowledge.mat');
dict = d.dict2;
dict = dict(1:20);
clear d;
fname = 'speech_expand2.mat';

f = load(fname);
% if nargin
    input = f.sentences{sen};
% else
%     input = f.sig_fixsylb_extend{1};
% end
clear f

alist = {'card A', 'serve', 'run', 'card J', 'neckband', 'score', 'buzz'};
rlist = {'win', 'ruin', 'be_s'};
plist = {'tennis', 'poker', 'game', 'evening', 'null'};
adlist = {'sufficient', 'unexpected', 'not pretty', 'not fair', 'high volumn', 'high freq', 'null'};
wlist = {dict.Word};

slist = {'aa', 'eis', 'eve', 'fair', 'geim', 'gli', 'in', 'ind', ...
    'is', 'jo', 'ker', 'laod', 'more', 'naf', 'ning', 'nis', 'noiz', ...
    'po', 'prai', 'ru', 'sbrint', 'shaap', 'sur', 'tai', 'te', 'the', 'un', 'wan', ...
    'wins', 'zat', 'zing', '-'};  

%------Level zero, syllable to spectrotemporal stripe (attracor)-----------
D{1} = ones(numel(slist), 1); % which syllable
D{2} = [1 0 0 0 0 0 0 0]'; % where (one of the 8 gamma period)


Nf = numel(D);
for f = 1:Nf
    Ns(f) = numel(D{f});
end
for f1 = 1:Ns(1)
    for f2 = 1:Ns(2) 
%         for f3 = 1:Ns(3)
        % index for the I vector
%             if f3==1
                A{1}((f1-1)*8+f2, f1, f2) = 1;
%             else
%                 A{1}(15*8+f2, f1, f2) = 1;
%             end
%         end
    end
end

for f = 1:Nf
    B{f} = eye(Ns(f));
end
 
% controllable gamma location: move to the next location
%--------------------------------------------------------------------------
B{2}(:,:,1) = spm_speye(Ns(2),Ns(2),-1); 
B{2}(end,end,1) = 1;

% % add inprecision
B{1} = B{1};

[DEM, demi] = spm_MDP_DEM_speech_gamma(input, fname);

% MDP Structure
%--------------------------------------------------------------------------
mdp.T = 8;                      % number of updates
mdp.A = A;                      % observation model
mdp.B = B;                      % transition probabilities
mdp.D = D;                      % prior over initial states

mdp.Aname = {'gamma'};
mdp.Bname = {'Sylb','gamma'};
% mdp.chi   = 1/16;
mdp.tau   = 4;
mdp.DEM   = DEM;
mdp.demi  = demi;
mdp.label.name{1} = slist;


MDP = spm_MDP_check(mdp);
clear mdp

clear A B D
%------Level one, syllable to word-----------------------------------------
D{1} = ones(numel(wlist), 1); % what word, excluding the null 
D{2} = [1 0 0]';
Nf = numel(D);
for f = 1:Nf
    Ns(f) = numel(D{f});
end
for f1 = 1:Ns(1)
    s1 = dict(f1).Sylb1;
    s2 = dict(f1).Sylb2;
    s3 = dict(f1).Sylb3;
    
    idx1 = find(strcmp(slist, s1));
    A{1}(idx1, f1, 1) = 1;
    
    idx2 = find(strcmp(slist, s2));
    A{1}(idx2, f1, 2) = 1;
    
    idx3 = find(strcmp(slist, s3));
    A{1}(idx3, f1, 3) = 1;

end
Ng    = numel(A);
for f = 1:Nf
    B{f} = eye(Ns(f));
end

B{1}(:, :, 1) = B{1}(:, :, 1)+0.005;

B{2}(:,:,1) = spm_speye(Ns(2),Ns(2),-1); 
B{2}(end,end,1) = 1;
 
 
% MDP Structure
%--------------------------------------------------------------------------
mdp.T = 3;                      % number of updates
mdp.A = A;                      % observation model
mdp.B = B;                      % transition probabilities
mdp.D = D;                      % prior over initial states
mdp.Aname = {'what sylb'};
mdp.Bname = {'Lemma', 'where'};
% mdp.chi   = 1/16;
mdp.tau   = 4;
mdp.MDP  = MDP;
mdp.label.name{1} = wlist(1:end);
mdp.link = sparse([1], [1], 1,numel(MDP(1).D),Ng);
 
MDP = spm_MDP_check(mdp);
clear mdp


clear A B D

% level three: association--possible combinations and their probability
%==========================================================================
 
% prior beliefs about initial states (in terms of counts_: D and d
%--------------------------------------------------------------------------
context{1} = 'poker game'; context{2} = 'tennis game'; %context{3} = 'run game';
context{3} = 'night party'; context{4} = 'racing game'; 
% context{5} = 'novel'; 

D{1} = [1.5 1 1 1]';
D{2} = [1 1]'; % type of sentence, event or property


% probabilistic mapping from hidden states to outcomes: A
%--------------------------------------------------------------------------
Nf    = numel(D);
na = length(alist);
nr = length(rlist);
np = length(plist);
nm = length(adlist);
for f = 1:Nf
    Ns(f) = numel(D{f}); 
end
for f1 = 1:Ns(1)
    for f2 = 1:Ns(2)
        if f1==1
            L{1}(1, f1, f2) = 0.6; L{1}(4, f1, f2) = 0.4; 
            if f2==1
                L{2}(1, f1, f2) = 0.8; L{2}(2, f1, f2) = 0.2; 
                L{3}(2, f1, f2) = 1;
                L{4}(7, f1, f2) = 1;
%                     L{4}(1, f1, f2) = 0.2; L{4}(2, f1, f2) = 0.2;
            else
                L{2}(3, f1, f2) = 1;
                L{3}(5, f1, f2) = 1; 
%                     L{3}(2, f1, f2, f3) = .4;
                L{4}(1, f1, f2) = 0.5; L{4}(2, f1, f2) = 0.5;
            end

        end
        if f1==2
            L{1}(2, f1, f2) = 0.6; L{1}(3, f1, f2) = 0.4; 
            if f2==1
                L{2}(1, f1, f2) = 0.8; L{2}(2, f1, f2) = 0.2; 
                L{3}(1, f1, f2) = 1;
                L{4}(7, f1, f2) = 1;
%                     L{4}(1, f1, f2, f3) = 0.2; L{4}(2, f1, f2, f3) = 0.2;
            else
                L{2}(3, f1, f2) = 1;
                L{3}(5, f1, f2) = 1; %L{3}(1, f1, f2) = .4;
                L{4}(1, f1, f2) = 0.5; L{4}(2, f1, f2) = 0.5;
            end               
        end
        if f1==3
            L{1}(5, f1, f2) = 0.6; L{1}(7, f1, f2) = 0.4; 
            if f2==1
                L{2}(1, f1, f2) = 0.2; L{2}(2, f1, f2) = 0.8;  
                L{3}(4, f1, f2) = 1;
                L{4}(7, f1, f2) = 1;

            else
                L{2}(3, f1, f2) = 1;
                L{3}(5, f1, f2) = 1;
                L{4}(3, f1, f2) = 0.3;
                L{4}(5, f1, f2) = 0.3; L{4}(6, f1, f2) = 0.3;
            end               
        end
         if f1==4
            L{1}(6, f1, f2) = 0.6; %L{1}(7, f1, f2) = 0.4; 
            if f2==1
                L{2}(1, f1, f2) = 0.2; L{2}(2, f1, f2) = 0.8;  
                L{3}(3, f1, f2) = 1;
                L{4}(7, f1, f2) = 1;
            else
                L{2}(3, f1, f2) = 1;
                L{3}(5, f1, f2) = 1; 
                L{4}(4, f1, f2) = 0.3;
                L{4}(5, f1, f2) = 0.3; L{4}(6, f1, f2) = 0.3;
            end    
         end
            
    end
            
end
% define Z
Z = cell(1, 4);
Z{1}(:, 1) = [1 0 0 0 0]'; Z{1}(:, 2) = [1 0 0 0 0]';
Z{2}(:, 1) = [0 1 0 0 0]'; Z{2}(:, 2) = [0 1 0 0 0]';
Z{3}(:, 1) = [0 0 1 0 0]'; Z{3}(:, 2) = [0 0 1 0 0]';
Z{4}(:, 1) = [0 0 0 1 0]'; Z{4}(:, 2) = [0 0 0 0 1]';
% calculate and normalize initial distributions for the top level
D{1} = spm_norm_exp(D{1}, 1);
D{2} = spm_norm_exp(D{2}, 1);
for ks = 1:4
    D{ks+2} = zeros(size(L{ks}, 1), 1);
    L{ks} = spm_norm_exp(L{ks});
    if prediction
        for kt = 1:2
            ll = squeeze(L{ks}(:, :, kt));
            D{ks+2} = D{ks+2} + D{2}(kt)*ll*D{1};
        end
    else
        D{ks+2} = ones(length(D{ks+2}), 1);
    end
    D{ks+2} = spm_norm_exp(D{ks+2});
end

D{7} = zeros(5, 4);
for tau = 1:4
    Z{tau} = spm_norm_exp(Z{tau});
    D{7}(:, tau) = spm_norm_exp(Z{tau}*D{2});
end
    
Nw = length(dict);
A = cell(1, 5);
A{1} = zeros(Nw, 1); A{1}(1:2) = 1;
A{2} = zeros(Nw, na);
A{3} = zeros(Nw, nr);
A{4} = zeros(Nw, np);
A{5} = zeros(Nw, nm);
m1 = {dict(:).Meaning1};
m2 = {dict(:).Meaning2};
m3 = {dict(:).Meaning3};
for f = 1:Nf
    Ns(f) = numel(D{f}); % number of total possible states under each factor--YS
end

for ns = 1:na
    sem = alist{ns};
    idx_a = find(strcmp(m1, sem));
    if ~isempty(idx_a)
        A{2}(idx_a, ns) = 1; % can we set bias in this way? need to verify
    else
        idx_a = find(strcmp(m2, sem));
        A{2}(idx_a, ns) = 1;
    end
end

for ns = 1:nr
    sem = rlist{ns};
    idx_r = find(strcmp(m1, sem));
    A{3}(idx_r, ns) = 1;
end

for ns = 1:np
    sem = plist{ns};
    idx_p1 = find(strcmp(m1, sem));
    idx_p2 = find(strcmp(m2, sem));
    idx_p3 = find(strcmp(m3, sem));
    if ~isempty(idx_p1)
        if ~isempty(idx_p2)
            A{4}(idx_p1, ns) = 0.8;
            A{4}(idx_p2, ns) = 0.2;
        else
            if ~isempty(idx_p3)
                A{4}(idx_p1, ns) = 0.8;
                A{4}(idx_p3, ns) = 0.2;
            else
                A{4}(idx_p1, ns) = 1;
            end
        end            
    end
end

for ns = 1:nm
    sem = adlist{ns};
    idx_ad = find(strcmp(m1, sem));
    if ~isempty(idx_ad)
        A{5}(idx_ad, ns) = 1;
    end
end

for ksyn = 1:5
    A{ksyn} = spm_norm_exp(A{ksyn}, 2);
end
% MDP Structure
%--------------------------------------------------------------------------
mdp.MDP  = MDP;
% mdp.link = sparse([1 2 3 4 5],[1 2 3 4 5],[1 1 1 1 1], numel(MDP.D),Ng); % link function is a Ng1*Ng2 matrix with the (1,1) entry equals to 1
% because the first factors of both levels are linked (sentence and word)
 
mdp.T = 4;                      % number of moves

mdp.A = A;                      % observation model
mdp.L = L;
mdp.Z = Z;
mdp.D = D;                      % prior over initial states

mdp.stepc = 128;
mdp.stept = 256;
mdp.stepsyn = 8;
mdp.steps = [16 32 4 4]; %[a r p m]


mdp.label.name{1} = context;
mdp.label.name{2} = {'Event', 'Property'};
mdp.label.name{3}   = alist;
mdp.label.name{4}   = rlist;
mdp.label.name{5}   = plist;
mdp.label.name{6}   = adlist;
mdp.label.name{7} = {'Attribute', 'Subject', 'Verb', 'Object', 'Adjective'};
mdp.label.factor   = {'Context', 'Type', 'Agent', 'Relation', 'Patient', 'Modifier', 'Syntax'};
% mdp         = spm_MDP_check(mdp);
%%
% illustrate a single trial
%==========================================================================
% prediction = 1;

MDP  = spm_MDP_VB_X_hybrid(mdp, prediction);
if nargin
    return;
end
 

spm_MDP_VB_ERP_ALL_hybrid(MDP)

% figure;
% spm_MDP_VB_ERP_YS(MDP.mdp(4).mdp, 2)

% spm_figure('GetWin','Figure 2'); clf
% spm_MDP_VB_LFP(MDP.mdp(4).mdp.mdp,[], 1); 
% 
