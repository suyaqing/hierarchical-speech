clear
f = load('speech_expand2.mat');
d = load('Knowledge.mat');
stemplate = f.ST_noise_25;
dict = d.dict2;
% stemplate = rand(6, 192, 19);
% stemplate(:, :, 16) = stemplate(:, :, 16)/8;
alist = {'card A', 'serve', 'run', 'card J', 'neckband', 'score', 'buzz'};
rlist = {'win', 'ruin', 'be_s'};
plist = {'tennis', 'poker', 'game', 'evening', 'null'};
adlist = {'enough', 'surprising', 'ugly', 'unfair', 'loud', 'sharp', 'null'};
wlist = {dict.Word};
% vocab = {'one more', 'ace', 'sprint', 'wins', 'the tennis', 'the poker', 'the game', ....
%      'this', 'tie', 'noise', 'ruined', 'the look', 'the evening', ...
%      'these', 'fans', 'drains', 'are', 'faulty', 'noisy', ...
%     'note', 'sign', 'is', 'high', 'long', 'silence'};
slist = {'aa', 'eis', 'eve', 'fair', 'geim', 'gli', 'in', 'ind', ...
    'is', 'jo', 'ker', 'laod', 'more', 'naf', 'ning', 'nis', 'noiz', ...
    'po', 'prai', 'ru', 'sbrint', 'shaap', 'sur', 'tai', 'te', 'the', 'un', 'wan', ...
    'wins', 'zat', 'zing', '-'};    
S = cell(numel(wlist)-1, 3);
for kw = 1:length(S)
    S(kw, :) = {dict(kw).Sylb1, dict(kw).Sylb2, dict(kw).Sylb3};
end

sentence = [1 3 8 11; % ace-tennis
            1 3 8 12; % ace-poker
            1 3 8 13; % ace-game
            1 4 8 11; % sprint-tennis
            1 4 8 13; % sprint-game
            1 5 8 12; % joker-poker
            1 5 8 13; % joker-game
            1 3 10 15; % ace-enough
            1 3 10 16; % ace-surprising
            1 4 10 15; % sprint-enough
            1 4 10 16; % sprint-surprsing
            1 5 10 15; % joker-enough
            1 5 10 16; % joker-surprising
            2 6 9 13; % tie-game
            2 6 9 14; % tie-evening
            2 7 9 13; % noise-game
            2 7 9 14; % noise-evening
            2 6 10 17; % tie-ugly
            2 6 10 18; % tie-unfair
            2 7 10 19; % noise-loud
            2 7 10 20]; % noise-sharp
sen = cell(1, 21);
% sig_fixed2_extend = f.sig_fixed2_extend;
sig_fixsylb_extend = cell(1, 21);
% syllables = zeros(size(sentence, 2), 4);
for isen = 1:21
    sen{isen} = wlist(sentence(isen, :));
    sen_temp = [];
    for iw = 1:size(sentence, 2)
        word = S(sentence(isen, iw), :);
%         syllables = length(word);
        for is = 1:length(word)
            inds = find(strcmp(word{is}, slist));
            sen_temp = [sen_temp, stemplate(:, :, inds)];
%             if strcmp(word{is}, '')
%                 break;
%                 continue;
%             else
%                 inds = find(strcmp(word{is}, slist));
%                 sen_temp = [sen_temp, stemplate(:, :, inds)];
%             end
        end
    end
    for iw = 1:size(sentence, 2)
        word = S(sentence(isen, iw), :);
        for is = 1:length(word)
%             inds = 12;
            sen_temp = [sen_temp, stemplate(:, :, end)];

        end
    end
        
    sig_fixsylb_extend{isen} = sen_temp;
end

%%
sentences_noise_25 = sig_fixsylb_extend;
save('speech_expand2.mat', 'sentences_noise_25', '-append')
%%
N = 24;
ST_copy = zeros(6, 8, numel(slist));
% ST_copy = f.ST;
% stemplate(:, :, 16) = stemplate(:, :, 16)/100; % silence
for ig = 1:8
    ind = (ig-1)*N+1:ig*N;
    gamma = stemplate(:, ind, :);
    ST_copy(:, ig, :) = mean(gamma, 2);
end
%%
D = 0.2 * diag(ones(1,6));
    
W = [-0.8881    0.4397    0.2279    0.2280   -0.0147    0.4345;
0.1931   -0.9626   -0.0836    0.1892    0.3324    0.0405;
0.4909   -0.1355   -0.7123   -0.5790   -0.0435   -0.5619;
0.0119    0.0580   -0.6032   -1.0000   -0.2894   -0.0376;
-0.4133    0.0856   -0.0541   -0.1186   -0.3464    0.1709;
0.5559    0.1764   -0.3075   -0.0122    0.4482   -0.9253];

ST = ST_copy;
I_copy = f.I;
for is = 1:numel(slist)
    st = ST(:, :, is);
    P = D*st - W*tanh(st);
    I_copy{is} = P;
end   
I = I_copy;
 save('toy_speech_varsylb_extend.mat', 'ST', 'I', '-append');