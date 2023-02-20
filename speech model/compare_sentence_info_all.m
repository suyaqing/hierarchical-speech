% compare context semantic level response
clear
% close all
folder = '';
s1 = 'ace';
s2 = 'ace';
o1 = 'tennis';
o2 = 'tennis';
sfx1 = '_context1_5_new_step_P_pre_-4_8.mat';
sfx2 = '_context1_5_new_step_NNP_pre_-4_8.mat';
sen1 = load([folder s1 '_' o1 sfx1]);
sen2 = load([folder s2 '_' o2 sfx2]);
%%
MDP1 = sen1.MDP; MDP2 = sen2.MDP;
[U1, W1] = spm_MDP_VB_ERP_ALL_hybrid(MDP1);
[U2, W2] = spm_MDP_VB_ERP_ALL_hybrid(MDP2);

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
[dw1, ~, ew1] = KLDiv(W1.w21, step);
[dsyl1, ~, esyl1] = KLDiv(W1.w31, step);

[dcon2, ~, econ2] = KLDiv(W2.wc, step);
[dtyp2, ~, etyp2] = KLDiv(W2.wt, step);
[dag2, ~, eag2] = KLDiv(W2.wa, step);
[drel2, ~, erel2] = KLDiv(W2.wr, step);
[dpat2, ~, epat2] = KLDiv(W2.wp, step);
[dmod2, ~, emod2] = KLDiv(W2.wm, step);
[dsyn2, ~, esyn2] = KLDiv(W2.wsyn, step);
[dw2, ~, ew2] = KLDiv(W2.w21, step);
[dsyl2, ~, esyl2] = KLDiv(W2.w31, step);



dt = 1/(128*5);
t  = (1:size(econ1,1))*dt;
L = length(t);
idxt = find((t<=2.5)&(t>=0));
%%
% plot_comparison_top(t, econ1, econ2, eag1, eag2, epat1, epat2, s1, s2, o1, o2, 'Entropy', idxt)
% plot_comparison_top(t, dcon1, dcon2, dag1, dag2, dpat1, dpat2, s1, s2, o1, o2, 'Divergence', idxt)
plot_comparison_lower(t, ew1, ew2, esyl1, esyl2, s1, s2, o1, o2, 'Entropy', idxt)
plot_comparison_lower(t, dw1, dw2, dsyl1, dsyl2, s1, s2, o1, o2, 'Divergence', idxt)


function plot_comparison_top(t, st1, st2, sub1, sub2, ob1, ob2, s1, s2, o1, o2, tt, idxt)
figure;
subplot(3,1,1)
plot(t(idxt), st1(idxt), 'linewidth', 1.5); hold on
plot(t(idxt), st2(idxt), 'linewidth', 1.5); hold on
yl = ylim;
ticks = [1.2, 2.4]; line_x = [ticks;ticks];
line_y = [yl(1);yl(2)]*ones(1, 2);
line(line_x, line_y, 'LineStyle', '--', 'color', 'r');
legend('Prediction', 'No Prediction')
title([s2 '-' o2]);
set(gca, 'fontsize', 10);
set(gca,'XTickLabel',[]);
ylabel('Context');
spm_axis tight
grid on

subplot(3,1,2)
plot(t(idxt), sub1(idxt), 'linewidth', 2); hold on
plot(t(idxt), sub2(idxt), 'linewidth', 2); hold on
yl = ylim;
ticks = [1.2, 2.4]; line_x = [ticks;ticks];
line_y = [yl(1);yl(2)]*ones(1, 2);
line(line_x, line_y, 'LineStyle', '--', 'color', 'r');
% title('Difference');
ylabel('Agent');
set(gca, 'fontsize', 10);
set(gca,'XTickLabel',[]);
spm_axis tight
grid on

subplot(3,1,3)
plot(t(idxt), ob1(idxt), 'linewidth', 1.5); hold on
plot(t(idxt), ob2(idxt), 'linewidth', 1.5); hold on
yl = ylim;
ticks = [1.2, 2.4]; line_x = [ticks;ticks];
line_y = [yl(1);yl(2)]*ones(1, 2);
line(line_x, line_y, 'LineStyle', '--', 'color', 'r');
% title('Difference (sen2 - sen1)');
ylabel('Patient');
spm_axis tight
set(gca, 'fontsize', 10);
xlabel('Time (s)')
grid on
sgtitle(tt)
end

function plot_comparison_lower(t, w1, w2, syl1, syl2, s1, s2, o1, o2, tt, idxt)
figure;
subplot(2,1,1)
plot(t(idxt), w1(idxt)-w2(idxt), 'linewidth', 2, 'color', rpalette('blue'));
% plot(t(idxt), w1(idxt), 'linewidth', 2, 'color', rpalette('blue')); hold on
% plot(t(idxt), w2(idxt), 'linewidth', 2, 'color', rpalette('scarlet')); hold on
% if strcmp(tt, 'Entropy')
%     set(gca, 'ylim', [-0.25, 4.25]);
% else
%     set(gca, 'ylim', [-12.5, 212.5]);
% end
if strcmp(tt, 'Entropy')
    set(gca, 'ylim', [0, 0.4]);
else
    set(gca, 'ylim', [-0.1, 2]);
end
yl = ylim;
% ticks = [2.4]; line_x = [ticks;ticks];
% line_y = [yl(1);yl(2)]*ones(1, 2);
% line(line_x, line_y, 'LineStyle', '--', 'color', 'r');

ticks = [1.8, 2, 2.2]; line_x = [ticks;ticks];
line_y = [yl(1);yl(2)]*ones(1, 3);
line(line_x, line_y, 'LineStyle', ':', 'color', 'r', 'Linewidth', 1.5);
% legend('Ace', 'Sprint')
% title('Low Precision');
set(gca, 'fontsize', 14);
set(gca,'XTickLabel',[]);
ylabel('Lemma');
spm_axis tight
grid on

subplot(2,1,2)
plot(t(idxt), syl1(idxt)-syl2(idxt), 'linewidth', 2, 'color', rpalette('blue'));
% plot(t(idxt), syl1(idxt), 'linewidth', 2, 'color', rpalette('blue')); hold on
% plot(t(idxt), syl2(idxt), 'linewidth', 2, 'color', rpalette('scarlet')); hold on
% if strcmp(tt, 'Entropy')
%     set(gca, 'ylim', [-0.25, 4.25]);
% else
%     set(gca, 'ylim', [-12.5, 212.5]);
% end
if strcmp(tt, 'Entropy')
    set(gca, 'ylim', [0, 0.4]);
else
    set(gca, 'ylim', [-0.1, 2]);
end
yl = ylim;
% ticks = [2.4]; line_x = [ticks;ticks];
% line_y = [yl(1);yl(2)]*ones(1, 2);
% line(line_x, line_y, 'LineStyle', '--', 'color', 'r');
ticks = [1.8, 2, 2.2]; line_x = [ticks;ticks];
line_y = [yl(1);yl(2)]*ones(1, 3);
line(line_x, line_y, 'LineStyle', ':', 'color', 'r', 'Linewidth', 1.5);
% title('Difference');
ylabel('Syllable');
set(gca, 'fontsize', 14);
% set(gca,'XTickLabel',[]);
spm_axis tight
xlabel('Time (s)')
grid on
sgtitle(tt, 'fontsize', 14)
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

% suptitle(tt)
% 
end

