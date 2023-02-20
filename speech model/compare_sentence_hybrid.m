% compare semantic level response
clear
% close all
folder = '';
s1 = 'sprint';
s2 = 'ace';
o1 = 'tennis';
o2 = 'tennis';
sfx1 = '_context1_5_new_step.mat';
sfx2 = '_context1_5_new_step.mat';
sen1 = load([folder s1 '_' o1 sfx1]);
sen2 = load([folder s2 '_' o2 sfx2]);
%%
MDP1 = sen1.MDP; MDP2 = sen2.MDP;
[U1, W1] = spm_MDP_VB_ERP_ALL_hybrid(MDP1);
[U2, W2] = spm_MDP_VB_ERP_ALL_hybrid(MDP2);
% [R1, ~, ~] = spm_MDP_VB_PE_ALL(MDP1, 0, 'vn');
% [R2, ~, ~] = spm_MDP_VB_PE_ALL(MDP2, 0, 'vn');

% get entropy, cross entropy (surprise), KL divergence for story and
% semantic levels
step = 1;
[dcon1, ~, econ1] = KLDiv(W1.wc, step);
[dtyp1, ~, etyp1] = KLDiv(W1.wt, step);
[dag1, ~, eag1] = KLDiv(W1.wa, step);
[drel1, ~, erel1] = KLDiv(W1.wr, step);
[dpat1, ~, epat1] = KLDiv(W1.wp, step);
[dmod1, ~, emod1] = KLDiv(W1.wm, step);
[dsyn1, ~, esyn1] = KLDiv(W1.wsyn, step);

[dcon2, ~, econ2] = KLDiv(W2.wc, step);
[dtyp2, ~, etyp2] = KLDiv(W2.wt, step);
[dag2, ~, eag2] = KLDiv(W2.wa, step);
[drel2, ~, erel2] = KLDiv(W2.wr, step);
[dpat2, ~, epat2] = KLDiv(W2.wp, step);
[dmod2, ~, emod2] = KLDiv(W2.wm, step);
[dsyn2, ~, esyn2] = KLDiv(W2.wsyn, step);

dt = 1/(128*5);
t  = (1:size(econ1,1))*dt;
L = length(t);
idxt = find(t<=3.6);
%%
plot_comparison(t, econ1, econ2, eag1, eag2, epat1, epat2, s1, s2, o1, o2, 'Entropy', idxt)
plot_comparison(t, dcon1, dcon2, dag1, dag2, dpat1, dpat2, s1, s2, o1, o2, 'Divergence', idxt)


% plot_comparison_all(t, econ1, econ2, eag1, eag2, epat1, epat2, s1, s2, o1, o2, 'Entropy', idxt)
% plot_comparison_all(t, dcon1, dcon2, dag1, dag2, dpat1, dpat2, s1, s2, o1, o2, 'Divergence', idxt)

% add vertical dashed lines at critital time points
%% 

function plot_comparison(t, st1, st2, sub1, sub2, ob1, ob2, s1, s2, o1, o2, tt, idxt)
figure;
% subplot(3,2,1)
% plot(t(idxt), st1(idxt), 'linewidth', 1.5); hold on
% yl = ylim;
% ticks = [1.2, 2.4, 3]; line_x = [ticks;ticks];
% line_y = [yl(1);yl(2)]*ones(1, 3);
% line(line_x, line_y, 'LineStyle', '--', 'color', 'r');
% title([s1 '-' o1]);
% ylabel('Context');
% set(gca, 'fontsize', 12);
% set(gca,'XTickLabel',[]);
% spm_axis tight
% grid on


subplot(3,1,1)
plot(t(idxt), st2(idxt)-st1(idxt), 'linewidth', 2, 'color', rpalette('blue')); hold on
y1 = ylim;
yl = y1;

subplot(3,1,2)
plot(t(idxt), sub2(idxt)-sub1(idxt), 'linewidth', 2, 'color', rpalette('blue')); hold on
y2 = ylim;

subplot(3,1,3)
plot(t(idxt), ob2(idxt)-ob1(idxt), 'linewidth', 2, 'color', rpalette('blue')); hold on
y3 = ylim;

% yl(1) = min([y1(1), y2(1), y3(1)]);
% yl(2) = max([y1(2), y2(2), y3(2)]);
if strcmp(tt,'Entropy')
    yl(1) = -0.0625;
    yl(2) = 1.0625;
else
    yl(1) = -6.8682;
    yl(2) = 5.6981;
end

subplot(311)
set(gca, 'ylim', yl);
ticks = [1.2, 2.4]; line_x = [ticks;ticks];
line_y = [yl(1);yl(2)]*ones(1, 2);
line(line_x, line_y, 'LineStyle', '--', 'color', 'r');
title(['(' s2 '-' o2 ')-(' s1 '-' o1 ')']);
set(gca, 'fontsize', 12);
set(gca,'XTickLabel',[]);
ylabel('Context');
spm_axis tight
grid on


subplot(312)
set(gca, 'ylim', yl);
ticks = [1.2, 2.4]; line_x = [ticks;ticks];
line_y = [yl(1);yl(2)]*ones(1, 2);
line(line_x, line_y, 'LineStyle', '--', 'color', 'r');
% title('Difference');
ylabel('Agent');
set(gca, 'fontsize', 12);
set(gca,'XTickLabel',[]);
spm_axis tight
grid on

subplot(313)
set(gca, 'ylim', yl);
ticks = [1.2, 2.4]; line_x = [ticks;ticks];
line_y = [yl(1);yl(2)]*ones(1, 2);
line(line_x, line_y, 'LineStyle', '--', 'color', 'r');
% title('Difference (sen2 - sen1)');
ylabel('Patient');
spm_axis tight
set(gca, 'fontsize', 12);
xlabel('Time (s)')
grid on
sgtitle(tt)
% 
end

function plot_comparison_all(t, st1, st2, sub1, sub2, ob1, ob2, s1, s2, o1, o2, tt, idxt)
figure;
subplot(1,2,1)
% plot(t(idxt), st1(idxt), 'linewidth', 1); hold on
% plot(t(idxt)*0.98, sub1(idxt), 'linewidth', 1);
% plot(t(idxt)*1.02, ob1(idxt), 'linewidth', 1); 
plot(t(idxt), st1(idxt)+sub1(idxt)+ob1(idxt), 'linewidth', 1.5, 'linestyle', '-'); hold on

yl = ylim;
ticks = [1.2, 2.4, 3]; line_x = [ticks;ticks];
line_y = [yl(1);yl(2)]*ones(1, 3);
line(line_x, line_y, 'LineStyle', '--', 'color', 'r');
% legend('Story', 'Subject', 'Object', 'Sum');
title([s1 '-' o1]);
% ylabel('Story', 'Rotation', 0);
set(gca, 'fontsize', 14);
% set(gca,'XTickLabel',[]);
spm_axis tight
grid on
hold off

subplot(1,2,2)
% plot(t(idxt), st2(idxt)-st1(idxt), 'linewidth', 1); hold on
% plot(t(idxt)*0.98, sub2(idxt)-sub1(idxt), 'linewidth', 1); 
% plot(t(idxt)*1.02, ob2(idxt)-ob1(idxt), 'linewidth', 1); 
plot(t(idxt), st2(idxt)+sub2(idxt)+ob2(idxt)-(st1(idxt)+sub1(idxt)+ob1(idxt)), 'linewidth', 1.5, 'linestyle', '-'); hold on

yl = ylim;
ticks = [1.2, 2.4, 3]; line_x = [ticks;ticks];
line_y = [yl(1);yl(2)]*ones(1, 3);
line(line_x, line_y, 'LineStyle', '--', 'color', 'r');
% legend('Story', 'Subject', 'Object', 'Sum');
title('difference ace-tennis');
set(gca, 'fontsize', 14);
% set(gca,'XTickLabel',[]);
xlabel('Time (s)')
spm_axis tight
grid on
hold off

suptitle(tt)
% 
end

%%
% figure;
% subplot(2,3,1)
% plot(t(idxt), es1(idxt),'-.', 'linewidth', 1); hold on
% yl = ylim;
% ticks = [1.2, 2.4, 3]; line_x = [ticks;ticks];
% line_y = [yl(1);yl(2)]*ones(1, 3);
% line(line_x, line_y, 'LineStyle', '--', 'color', 'r');
% title([s1 '-' o1]);
% ylabel('Subject', 'Rotation', 0);
% spm_axis tight
% grid on
% 
% subplot(2,3,2)
% plot(t(idxt), es2(idxt), '-.', 'linewidth', 1); hold on
% yl = ylim;
% ticks = [1.2, 2.4, 3]; line_x = [ticks;ticks];
% line_y = [yl(1);yl(2)]*ones(1, 3);
% line(line_x, line_y, 'LineStyle', '--', 'color', 'r');
% title([s2 '-' o2]);
% spm_axis tight
% grid on
% 
% subplot(2,3,3)
% plot(t(idxt), es2(idxt)-es1(idxt), '-.', 'linewidth', 1); hold on
% yl = ylim;
% ticks = [1.2, 2.4, 3]; line_x = [ticks;ticks];
% line_y = [yl(1);yl(2)]*ones(1, 3);
% line(line_x, line_y, 'LineStyle', '--', 'color', 'r');
% title('Difference');
% spm_axis tight
% grid on
% 
% subplot(2,3,4)
% plot(t(idxt), etot1(idxt),'-.', 'linewidth', 1)
% ylabel('Total', 'Rotation', 0);
% spm_axis tight
% grid on
% 
% subplot(2,3,5)
% plot(t(idxt), etot2(idxt), '-.', 'linewidth', 1)
% xlabel('Time (s)')
% spm_axis tight
% grid on
% 
% subplot(2,3,6)
% plot(t(idxt), etot2(idxt)-etot1(idxt), '-.', 'linewidth', 1)
% spm_axis tight
% grid on
% 
% suptitle('Entropy')
% 
% figure;
% subplot(2,3,1)
% plot(t(idxt), pws1(idxt),'-.', 'linewidth', 1); hold on
% yl = ylim;
% ticks = [1.2, 2.4, 3]; line_x = [ticks;ticks];
% line_y = [yl(1);yl(2)]*ones(1, 3);
% line(line_x, line_y, 'LineStyle', '--', 'color', 'r');
% title([s1 '-' o1]);
% ylabel('Subject', 'Rotation', 0);
% spm_axis tight
% grid on
% 
% subplot(2,3,2)
% plot(t(idxt), pws2(idxt), '-.', 'linewidth', 1); hold on
% yl = ylim;
% ticks = [1.2, 2.4, 3]; line_x = [ticks;ticks];
% line_y = [yl(1);yl(2)]*ones(1, 3);
% line(line_x, line_y, 'LineStyle', '--', 'color', 'r');
% title([s2 '-' o2]);
% spm_axis tight
% grid on
% 
% subplot(2,3,3)
% plot(t(idxt), pws2(idxt)-pws1(idxt), '-.', 'linewidth', 1); hold on
% yl = ylim;
% ticks = [1.2, 2.4, 3]; line_x = [ticks;ticks];
% line_y = [yl(1);yl(2)]*ones(1, 3);
% line(line_x, line_y, 'LineStyle', '--', 'color', 'r');
% title('Difference');
% spm_axis tight
% grid on
% 
% subplot(2,3,4)
% plot(t(idxt), ptot1(idxt),'-.', 'linewidth', 1)
% ylabel('Total', 'Rotation', 0);
% spm_axis tight
% grid on
% 
% subplot(2,3,5)
% plot(t(idxt), ptot2(idxt), '-.', 'linewidth', 1)
% xlabel('Time (s)')
% spm_axis tight
% grid on
% 
% subplot(2,3,6)
% plot(t(idxt), ptot2(idxt)-ptot1(idxt), '-.', 'linewidth', 1)
% spm_axis tight
% grid on
% suptitle('Change Power')
% 
% figure;
% subplot(2,3,1)
% plot(t(idxt), pes1(idxt),'-.', 'linewidth', 1); hold on
% yl = ylim;
% ticks = [1.2, 2.4, 3]; line_x = [ticks;ticks];
% line_y = [yl(1);yl(2)]*ones(1, 3);
% line(line_x, line_y, 'LineStyle', '--', 'color', 'r');
% title([s1 '-' o1]);
% ylabel('Subject', 'Rotation', 0);
% spm_axis tight
% grid on
% 
% subplot(2,3,2)
% plot(t(idxt), pes2(idxt), '-.', 'linewidth', 1); hold on
% yl = ylim;
% ticks = [1.2, 2.4, 3]; line_x = [ticks;ticks];
% line_y = [yl(1);yl(2)]*ones(1, 3);
% line(line_x, line_y, 'LineStyle', '--', 'color', 'r');
% title([s2 '-' o2]);
% spm_axis tight
% grid on
% 
% subplot(2,3,3)
% plot(t(idxt), pes2(idxt)-pes1(idxt), '-.', 'linewidth', 1); hold on
% yl = ylim;
% ticks = [1.2, 2.4, 3]; line_x = [ticks;ticks];
% line_y = [yl(1);yl(2)]*ones(1, 3);
% line(line_x, line_y, 'LineStyle', '--', 'color', 'r');
% title('Difference');
% spm_axis tight
% grid on
% 
% subplot(2,3,4)
% plot(t(idxt), petot1(idxt),'-.', 'linewidth', 1)
% ylabel('Total', 'Rotation', 0);
% spm_axis tight
% grid on
% 
% subplot(2,3,5)
% plot(t(idxt), petot2(idxt), '-.', 'linewidth', 1)
% xlabel('Time (s)')
% spm_axis tight
% grid on
% 
% subplot(2,3,6)
% plot(t(idxt), petot2(idxt)-petot1(idxt), '-.', 'linewidth', 1)
% spm_axis tight
% grid on
% suptitle('Prediction Error')
