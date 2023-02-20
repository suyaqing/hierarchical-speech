% compare entropy and divergence at key time points for different contexts
clear all
folder = '';
s = 'ace_game_context';
sfx = '_new_step.mat';

% simulations with different contexts
ctx = {'5_0r', '4_0r', '3_0r', '2_5r', '2_0r', '1_5r', '1_0', ...
    '1_5', '2_0', '2_5', '2_5', '3_0', '4_0', '5_0'};
entropy = zeros(length(ctx), 2);
diverge = entropy;
context = zeros(length(ctx), 1);
agent = context;

bias = zeros(length(ctx), 1);
dt = 1/(128*5);
step = 1;

for k = 1:length(ctx)
    sen = load([folder s ctx{k} sfx]);
    MDP = sen.MDP;
    context_temp = sen.context;
    bias(k) = context_temp(1)/context_temp(2);
    
    [U, W] = spm_MDP_VB_ERP_ALL_hybrid(MDP);
    step = 1;
    [dcon, ~, econ] = KLDiv(W.wc, step);
    [dtyp, ~, etyp] = KLDiv(W.wt, step);
    [dag, ~, eag] = KLDiv(W.wa, step);
    [drel, ~, erel] = KLDiv(W.wr, step);
    [dpat, ~, epat] = KLDiv(W.wp, step);
    [dmod, ~, emod] = KLDiv(W.wm, step);
    [dsyn, ~, esyn] = KLDiv(W.wsyn, step);   
    
    t  = (1:size(econ,1))*dt;
    L = length(t);
    ent = [econ, eag, epat];
%     ent = [econ, etyp, eag, erel, epat, emod, esyn];
    ent1 = mode(ent(t<1.4&t>1.3, :));
    ent2 = mode(ent(t<2.6&t>2.5, :));
%     ent3 = mode(ent(t<3.2&t>3.1, :));
    entropy(k, :) = [sum(ent1) sum(ent2)];
    div = [dcon, dag, dpat];
%     div = [dcon, dtyp, dag, drel, dpat, dmod, dsyn];
    div1 = max(div(t<1.3&t>1.2, :));
    div2 = max(div(t<2.5&t>2.4, :));
%     div3 = max(div(t<3.1&t>3.0, :));
    diverge(k, :) = [sum(div1) sum(div2)];
    
    con = U.uc(t<3&t>2.4, :); [~, icon] = max(con'); context(k) = mode(icon);
    ag = U.ua(t<3&t>2.4, :); [~, iag] = max(ag'); agent(k) = mode(iag);
end
%%

figure;
subplot(132)
plot(bias, entropy, 'marker', 'o', 'linewidth', 2, 'markersize', 6); hold on
title('Entropy')
set(gca, 'fontsize', 14);
xlim([0 5.2])
ylim([0 3.1]);
yl = ylim;
ticks = [1]; line_x = [ticks;ticks];
line_y = [yl(1);1.2*yl(2)]*ones(1);
line(line_x, line_y, 'LineStyle', '--', 'color', 'k');
legend('sub offset', 'sen offset');
xlabel('Bias Poker/Tennis')

subplot(133)
plot(bias, diverge, 'marker', 'o', 'linewidth', 2, 'markersize', 6); hold on
xlim([0 5.2])
ylim([0 16]);
yl = ylim;
ticks = [1]; line_x = [ticks;ticks];
line_y = [yl(1);1.2*yl(2)]*ones(1);
line(line_x, line_y, 'LineStyle', '--', 'color', 'k');
title('Divergence')
legend('sub offset', 'sen offset');
set(gca, 'fontsize', 14);

subplot(131)
plot(bias, context-0.8, 'marker', 'o', 'linewidth', 2, 'markersize', 6); hold on
plot(bias, agent+0.8, 'marker', '+', 'linewidth', 2, 'markersize', 6);
xlim([0 5.2])
ylim([0 3.1]);
yl = ylim;
ticks = [1]; line_x = [ticks;ticks];
line_y = [yl(1);1.2*yl(2)]*ones(1);
line(line_x, line_y, 'LineStyle', '--', 'color', 'k');
title('Inferred States')
set(gca, 'fontsize', 14);
legend('Context', 'Agent');