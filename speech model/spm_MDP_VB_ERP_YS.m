function [x,y,u,v] = spm_MDP_VB_ERP_YS(MDP,FACTOR,TT, T)
% auxiliary routine for hierarchical electrophysiological responses
% FORMAT [x,y] = spm_MDP_VB_ERP(MDP,FACTOR,T)
%
% MDP    - structure (see spm_MDP_VB)
% FACTOR - hidden factors (at high and low level) to plot
% T      - flag to return cell of expectations (at time T; usually 1)
%
% x      - simulated ERPs (high-level) (full lines)
% y      - simulated ERPs (low level)  (dotted lines)
% ind    - indices or bins at the end of each (synchronised) epoch
%
% This routine combines first and second level hidden expectations by
% synchronising them; such that first level updating is followed by an
% epoch of second level updating - during which updating is suspended
% (and expectations are held constant). The ensuing spike rates can be
% regarded as showing delay period activity. In this routine, simulated
% local field potentials are band pass filtered spike rates (between eight
% and 32 Hz).
%
% Graphics are provided for first and second levels, in terms of simulated
% spike rates (posterior expectations), which are then combined to show
% simulated local field potentials for both levels (superimposed).
%
% At the lower level, only expectations about hidden states in the first
% epoch are returned (because the number of epochs can differ from trial
% to trial).
%
% see also: spm_MDP_VB_LFP (for single level belief updating)
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_MDP_VB_ERP.m 7656 2019-08-26 14:00:36Z karl $


% defaults: assume the first factor is of interest
%==========================================================================
try, f1 = FACTOR(1); catch, f1 = 1; end
try, f2 = FACTOR(2); catch, f2 = 1; end

% and T = 1
%--------------------------------------------------------------------------
% if nargin < 3, T = 1; end
if nargin < 4
    T = 1; 
    if nargin < 3
        TT = 1; 
    end
end

for m = 1:numel(MDP)

    % dimensions
    %----------------------------------------------------------------------
    xn  = MDP(m).xn{f1};      % neuronal responses
    Nb  = size(xn,1);         % number of time bins per epochs
    Nx  = size(xn,2);         % number of states
    Ne  = size(xn,3);         % number of epochs
    
    
    % expected hidden states
    %======================================================================
    x     = cell(Ne,Nx);
    y     = cell(Ne);
    for k = 1:Ne
        for j = 1:Nx
            x{k,j} = xn(:,j, k ,k); 

%             x{k,j} = xn(:,j, min(k+1, Ne), k); 
%             for i = 1:size(xn, 1)
%                 if ~any(any(xn(i, :, k, k)))
% %                 if sum(xn(i, :, T, k))==0
%                     x{k, j}(i:end) = [];
%                     break
%                 end
%             end
        end
        
        if isfield(MDP,'mdp')
            y{k}   = spm_MDP_VB_ERP_YS(MDP(m).mdp(k),f2,[], 1);
        else
            y{k}   = [];
        end
    end
    
    if nargin > 3, return, end
    
    % synchronise responses
    %----------------------------------------------------------------------
    u   = {};
    v   = {};
    uu  = spm_cat(x(1,:));
    for k = 1:Ne
        if isfield(MDP,'mdp')
            % low-level
            %------------------------------------------------------------------
            v{end + 1,1} = spm_cat(y{k}); % new epoch
            if k > 1
                prev = spm_cat(x(k-1,:));
                u{k,1} = ones((size(v{end,:},1)-size(spm_cat(x(k-1,:)),1)),1) ...
                    *prev(end, :); % start from repeating the previous epoch
            else
    %             u{1,1} = ones(size(v{end,:},1),1)*uu(1,:);
                d = MDP.D{f1};
                u{1,1} = ones(size(v{end,:},1),1)*(d'/sum(d));
    %             disp('one')
            end

            % time bin indices
            %------------------------------------------------------------------
            ind(k) = size(u{end},1);

            % high-level
            %------------------------------------------------------------------
            u{k,1} = [u{k,1}; spm_cat(x(k,:))]; % the real epoch at the end of the epoch
            if k==Ne
                v{end + 1,1} = ones(size(spm_cat(x(k,:)),1),1)*v{end,1}(end,:); % repeat
            end

            % time bin indices
            %------------------------------------------------------------------
            ind(k) = ind(k) + size(u{end},1);
        else
            if k>1
                u{k,1} = spm_cat(x(k,:)); 
            else
                u{k,1} = uu;
            end
        end

    end
    
    % accumulate over trials
    %----------------------------------------------------------------------
    U{m,1} = u;
    V{m,1} = v;
    
end

if nargout > 1, return, end
% time bin (seconds)
%--------------------------------------------------------------------------
u  = spm_cat(U);
v  = spm_cat(V);
dt = TT*1/(128*5);
t  = (1:size(u,1))*dt;

% bandpass filter between 8 and 32 Hz
%--------------------------------------------------------------------------
c  = 1/32;
x  = log(u + c);
y  = log(v + c);
x  = spm_conv(x,1,0) - spm_conv(x,128/TT,0);
y  = spm_conv(y,1,0) - spm_conv(y,128/TT,0);



% simulated firing rates and the local field potentials
%==========================================================================

% higher-level unit responses
%--------------------------------------------------------------------------
factor = MDP(1).label.factor{f1};
name   = MDP(1).label.name{f1};

subplot(4,1,1), image(t,1:(size(u,2)),64*(1 - u')), ylabel('Unit')
title(sprintf('Unit reponses : %s',factor),'FontSize',16)
if numel(name) < 16
    grid on, set(gca,'YTick',1:numel(name))
    set(gca,'YTickLabel',name)
end

% lower-level unit responses
%--------------------------------------------------------------------------
factor = MDP(1).MDP(1).label.factor{f2};
name   = MDP(1).MDP(1).label.name{f2};

subplot(4,1,2), image(t,1:(size(v,2)),64*(1 - v')), ylabel('Unit')
title(sprintf('Unit reponses : %s',factor),'FontSize',16)
if numel(factor) < 16
    grid on, set(gca,'YTick',1:numel(name))
    set(gca,'YTickLabel',name)
end

% event related responses at higher level
%--------------------------------------------------------------------------
subplot(4,1,3), plot(t,x','-.')
title(['LFP, ' MDP(1).label.factor{f1}],'FontSize',16)
ylabel('Depolarisation'),spm_axis tight
grid on, xlabel('time (seconds)')

% event related responses at lower level
%--------------------------------------------------------------------------
subplot(4,1,4), plot(t,y','-.')
title(['LFP, ' MDP(1).MDP(1).label.factor{f2}],'FontSize',16)
ylabel('Depolarisation'),spm_axis tight
grid on, xlabel('time (seconds)')


fs = 1/dt;
L = length(t);
% fs = 2000;
f = fs*(0:floor(L/2))/L;
meanx = mean(x, 2);
meany = mean(y, 2);
meanxy = mean([x y], 2);

fx = abs(fft(meanx'));
fx = fx(1:floor(L/2)+1);
fx(2:end-1) = 2*fx(2:end-1);

fy = abs(fft(meany'));
fy = fy(1:floor(L/2)+1);
fy(2:end-1) = 2*fy(2:end-1);

fxy = abs(fft(meanxy'));
fxy = fxy(1:floor(L/2)+1);
fxy(2:end-1) = 2*fxy(2:end-1);

figure;
subplot(321)
plot(t, meanx);
title(['Sum LFP, ' MDP(1).label.factor{f1}],'FontSize',12)
ylabel('Depolarisation'),spm_axis tight
grid on, xlabel('time (seconds)')

subplot(323)
plot(t, meany)
title(['Sum LFP, ' MDP(1).MDP(1).label.factor{f2}],'FontSize',12)
ylabel('Depolarisation'),spm_axis tight
grid on, xlabel('time (seconds)')

subplot(325)
plot(t, meanxy)
title(['Grand sum LFP, ' MDP(1).label.factor{f1} '+' ...
    MDP(1).MDP(1).label.factor{f2}],'FontSize',12)
ylabel('Depolarisation'),spm_axis tight
grid on, xlabel('time (seconds)')


subplot(322)
plot(f, fx)
title(['Spectrum, ' MDP(1).label.factor{f1}],'FontSize',12)
xlabel('Frequency (Hz)') 
spm_axis tight

subplot(324)
plot(f, fy)
title(['Spectrum, ' MDP(1).MDP(1).label.factor{f2}],'FontSize',12)
xlabel('Frequency (Hz)')
spm_axis tight

subplot(326)
plot(f, fxy)
title(['Sum spectrum, ' MDP(1).label.factor{f1} '+' ...
    MDP(1).MDP(1).label.factor{f2}],'FontSize',12);
xlabel('Frequency (Hz)')
spm_axis tight
