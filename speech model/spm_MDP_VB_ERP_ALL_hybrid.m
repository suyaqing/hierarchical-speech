function [U, W] = spm_MDP_VB_ERP_ALL_hybrid(MDP)
% clearvars -except MDP
xc = {}; % context
xt = {}; % pos

xa = {}; % agent
xr = {}; % relation
xp = {}; % patient
xm = {}; % modifier
xsyn = {}; % syntax
x21 = {}; % word
x22 = {}; % where (sylb)
x31 = {}; % syllable
% x42 = {}; % where (gamma)

% y1 = {}; % subject
% y2 = {}; % word
y31 = {}; % syllable
% y32 = {}; % gamma
% y4 = {}; % gamma

pc = [];
pt = [];
pa = [];
pr = [];
pp = [];
pm = [];
psyn = [];

p21 = [];
p22 = [];

p31 = [];
p32 = [];
p41 = [];

for m = 1:numel(MDP)
    [xc, xt, xa, xr, xp, xm, xsyn] = spm_MDP_VB_ERP_hybrid(MDP);
%     [xt, ~] = spm_MDP_VB_ERP_hybrid(MDP, [2 1]);
    pc = [pc, MDP.D{1}];
    pt = [pt, MDP.D{2}];
    pa = [pa, MDP.D{3}];
    pr = [pr, MDP.D{4}];
    pp = [pp, MDP.D{5}];
    pm = [pm, MDP.D{6}];
    psyn = [psyn, MDP.D{7}(:, 1)];
%     d21 = [d21, mdp1.D{1}];
    Ne1 = size(xc, 1);
   
    for k1 = 1:Ne1 % lemma, where
        mdp1 = MDP.mdp(k1);
        [s1, ~] = spm_MDP_VB_ERP_YS(mdp1, [1 1]);
        x21 = [x21; s1];
        p21 = [p21, mdp1.D{1}];
        [s1, ~] = spm_MDP_VB_ERP_YS(mdp1, [2 1]);
        x22 = [x22; s1];
        p22 = [p22, mdp1.D{2}];
        Ne2 = size(s1, 1);
        
        for k2 = 1:Ne2 % syllable, where
            mdp2 = mdp1.mdp(k2);
            [s1, s2] = spm_MDP_VB_ERP_YS(mdp2, [1 1], 1, 1);
            x31 = [x31; s1];
            y31 = [y31; s2];
            p31 = [p31, mdp2.D{1}];
        end
    end   
end

% align response across different levels in time, because each level is in
% its own timescale
% [u4, v4, w4] = spm_MDP_VB_ERP_align(x4, y4, MDP.mdp(1).mdp(1).mdp(1).D{1}, 1);
[u31, v31, w31] = spm_MDP_VB_ERP_align(x31, y31, MDP.mdp(1).mdp(1).D{1}, 1);
[u21, ~, w21] = spm_MDP_VB_ERP_align(x21, u31, MDP.mdp(1).D{1}, 2);
[u22, ~, w22] = spm_MDP_VB_ERP_align(x22, u31, MDP.mdp(1).D{2}, 2);
% [u32, v32, w32] = spm_MDP_VB_ERP_align(x32, u4, MDP.mdp(1).mdp(1).D{2}, 2);
[ua, ~, wa] = spm_MDP_VB_ERP_align(xa, u21, MDP.D{3}, 2);
[ur, ~, wr] = spm_MDP_VB_ERP_align(xr, u21, MDP.D{4}, 2);
[up, ~, wp] = spm_MDP_VB_ERP_align(xp, u21, MDP.D{5}, 2);
[um, ~, wm] = spm_MDP_VB_ERP_align(xm, u21, MDP.D{6}, 2);
[usyn, ~, wsyn] = spm_MDP_VB_ERP_align(xsyn, u21, MDP.D{7}(:, 1), 2);
[uc, ~, wc] = spm_MDP_VB_ERP_align(xc, u21, MDP.D{1}, 2);
[ut, ~, wt] = spm_MDP_VB_ERP_align(xt, u21, MDP.D{2}, 2);


Uc = spm_cat(uc); Wc = spm_cat(wc);
Ut = spm_cat(ut); Wt = spm_cat(wt);

Ua = spm_cat(ua); Wa = spm_cat(wa);
Ur = spm_cat(ur); Wr = spm_cat(wr);
Up = spm_cat(up); Wp = spm_cat(wp);
Um = spm_cat(um); Wm = spm_cat(wm);
Usyn = spm_cat(usyn); Wsyn = spm_cat(wsyn);
U21 = spm_cat(u21); W21 = spm_cat(w21);
U22 = spm_cat(u22); W22 = spm_cat(w22);
U31 = spm_cat(u31); W31 = spm_cat(w31);
% U32 = spm_cat(u32); W32 = spm_cat(w32);
% U41 = spm_cat(v31); W41 = spm_cat(w4);
% U42 = spm_cat(v32); 
%%
treal = size(U31, 1);
tmax = floor(size(Uc, 1)*5/4);
% posterior probabilities in different levels
Uc = [Uc; ones(tmax - size(Uc, 1), 1)*Uc(end, :)];
Ut = [Ut; ones(tmax - size(Ut, 1), 1)*Ut(end, :)];
Ua = [Ua; ones(tmax - size(Ua, 1), 1)*Ua(end, :)];
Ur = [Ur; ones(tmax - size(Ur, 1), 1)*Ur(end, :)];
Up = [Up; ones(tmax - size(Up, 1), 1)*Up(end, :)];
Um = [Um; ones(tmax - size(Um, 1), 1)*Um(end, :)];
Usyn = [Usyn; ones(tmax - size(Usyn, 1), 1)*Usyn(end, :)];
U21 = [U21; ones(tmax-size(U21, 1), 1)*U21(end, :)];
U22 = [U22; ones(tmax-size(U22, 1), 1)*U22(end, :)];
U31 = [U31; ones(tmax-size(U31, 1), 1)*U31(end, :)];

% posterior probability but ignore the gradient descend process
Wc = [Wc; ones(tmax - size(Wc, 1), 1)*Uc(end, :)];
Wt = [Wt; ones(tmax - size(Wt, 1), 1)*Ut(end, :)];
Wa = [Wa; ones(tmax - size(Wa, 1), 1)*Ua(end, :)];
Wr = [Wr; ones(tmax - size(Wr, 1), 1)*Ur(end, :)];
Wp = [Wp; ones(tmax - size(Wp, 1), 1)*Up(end, :)];
Wm = [Wm; ones(tmax - size(Wm, 1), 1)*Um(end, :)];
Wsyn = [Wsyn; ones(tmax - size(Wsyn, 1), 1)*Usyn(end, :)];
W21 = [W21; ones(tmax-size(W21, 1), 1)*U21(end, :)];
W22 = [W22; ones(tmax-size(W22, 1), 1)*U22(end, :)];
W31 = [W31; ones(tmax-size(W31, 1), 1)*U31(end, :)];
% W42_ex = [W42; ones(tmax-size(U42, 1), 1)*W42(end, :)];

% predictions sent to lemma and syllable levels
D21 = span(p21, floor(treal/size(p21, 2)));

D31 = span(p31, floor(treal/size(p31, 2)));


U = [];
U.uc = Uc; U.ut = Ut;
U.ua = Ua; U.ur = Ur; U.up = Up; U.um = Um; U.usyn = Usyn;
U.u21 = U21; U.u22 = U22;
U.u31 = U31; 

W = [];
W.wc = Wc; W.wt = Wt;
W.wa = Wa; W.wr = Wr; W.wp = Wp; W.wm = Wm; W.wsyn = Wsyn;
W.w21 = W21; W.w22 = W22; 
W.w31 = W31; 


dt = 1/(128*5);
t  = (1:size(Uc,1))*dt;
td = (1:size(D21,1))*dt;
L = length(t);
idxt = find(t<=3.6);
step = 1;

%--------------------------------------------------------------------------
c  = 1/32; %l = 128; h = 2;


if nargout >=1
    return;
end


%--------------------------------------------------------------------------


factorc = MDP(1).label.factor{1};
namec   = MDP(1).label.name{1};

factort = MDP(1).label.factor{2};
namet   = MDP(1).label.name{2};

factora = MDP(1).label.factor{3};
namea   = MDP(1).label.name{3};
 
factorr = MDP(1).label.factor{4};
namer   = MDP(1).label.name{4};

factorp = MDP(1).label.factor{5};
namep   = MDP(1).label.name{5};

factorm = MDP(1).label.factor{6};
namem   = MDP(1).label.name{6};

factorsyn = MDP(1).label.factor{7};
namesyn   = MDP(1).label.name{7};

factor21 = MDP(1).MDP(1).label.factor{1};
name21   = MDP(1).MDP(1).label.name{1};

factor22 = MDP(1).MDP(1).label.factor{2};
name22   = MDP(1).MDP(1).label.name{2};

factor31 = MDP(1).MDP(1).MDP(1).label.factor{1};
name31   = MDP(1).MDP(1).MDP(1).label.name{1};

rp1 = rpalette('blue');
rp2 = rpalette('teal');

figure;
% set(gcf, 'Position', [1000 200 250 300]);
h = subplot(6,1,1); plot_raster(h, t(idxt), Uc(idxt, :), factorc, namec, 1, rp1);
set(gca,'XTickLabel',[])

h = subplot(6,1,2); plot_raster(h, t(idxt), Ut(idxt, :), factort, namet, 1, rp1);
set(gca,'XTickLabel',[])

h = subplot(6,1,3); plot_raster(h, t(idxt), Ua(idxt, :), factora, namea, 1, rp1);
set(gca,'XTickLabel',[])

h = subplot(6,1,4); plot_raster(h, t(idxt), Ur(idxt, :), factorr, namer, 1, rp1);
set(gca,'XTickLabel',[])

h = subplot(6,1,5); plot_raster(h, t(idxt), Up(idxt, :), factorp, namep, 1, rp1);
set(gca,'XTickLabel',[])

h = subplot(6,1,6); plot_raster(h, t(idxt), Um(idxt, :), factorm, namem, 1, rp1);
% set(gca,'XTickLabel',[])

% h = subplot(6,1,7); plot_raster(h, t(idxt), Usyn(idxt, :), factorsyn, namesyn, 1);
% set(gca,'XTickLabel',[])

xlabel('Time (s)')


% xlabel('Time (s)')


figure
% 
% h = subplot(3,1,1); plot_raster(h, t(idxt), Usyn(idxt, :), factorsyn, namesyn, 1);
% set(gca,'XTickLabel',[])
% set(gcf, 'Position', [1000 200 250 300]);
h = subplot(2,1,1); plot_raster(h, t(idxt), U21(idxt, :), factor21, name21, 1, rp1);
set(gca,'XTickLabel',[])

h = subplot(2,1,2); plot_raster(h, t(idxt), U31(idxt, :), factor31, name31, 1, rp1);
xlabel('Time (s)')

%--------------------------------------------------------------------
figure;
% set(gcf, 'Position', [1000 200 250 300]);
h = subplot(2,1,1); plot_raster(h, td, D21, factor21, name21, 2, rp2);
set(gca,'XTickLabel',[])

h = subplot(2,1,2); plot_raster(h, td, D31, factor31, name31, 2, rp2);
% xlabel('Time (s)')
% set(gca,'XTickLabel',[])
xlabel('Time (s)')
%----------------------------------------------------------------------------------------------------


end


function plot_raster(h, t, data, factor, name, mode, RGB)
% set(gca, h);
axes(h);
RGB = 1-RGB;
% RGB = RGB*255;
if nargin>6
    C = zeros(size(data, 2), size(data, 1), 3);
    C(:, :, 1) = 1-RGB(1)*data';
    C(:, :, 2) = 1-RGB(2)*data';
    C(:, :, 3) = 1-RGB(3)*data';
    C_RGB8 = uint8(round(C*255));
    image(t,1:(size(data,2)),C_RGB8);
else
    image(t,1:(size(data,2)),64*(1 - data'))
end
% ylabel('Unit');
if mode==1
    title(sprintf('Estimation : %s',factor),'FontSize',12)
else
    title(sprintf('Prediction : %s',factor),'FontSize',12)
end
grid on;
set(gca,'GridColor',uint8(round((1-RGB)*255)))
set(gca,'YTick',1:numel(name))
set(gca,'YTickLabel',name, 'FontSize',14)
pos = get(gcf, 'Position')
% colormap(gray(64));
end

function plot_LFP(h, t, data, factor)
% set(gca, h)
axes(h);
plot(t,data','-.')
title(['LFP, ' factor],'FontSize',16)
ylabel('Depolarisation');spm_axis tight
grid on
end

function plot_DIV(h, t, data, factor, met)
% set(gca, h)
axes(h);
plot(t,data, 'linewidth', 1.5)

yl = ylim;
ticks = [0.6, 1.2, 1.8, 2.4, 3]; line_x = [ticks;ticks];
line_y = [min(yl(1), 0);1.2*yl(2)]*ones(1, 5);
ylim([min(yl(1), 0), 1.1*yl(2)]);
line(line_x, line_y, 'LineStyle', '--', 'color', 'r');

title(sprintf('%s : %s',met,factor),'FontSize',16)
set(gca,'FontSize',14)
grid on
axis tight
end

function D = span(d, lblock)
D = [];
for Ne = 1:size(d, 2)
    dd = d(:, Ne)/sum(d(:, Ne));
    blk = dd*ones(1, lblock);
    D = [D, blk];
end
D = D';
end

