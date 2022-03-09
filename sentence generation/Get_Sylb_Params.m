% extract spectrotemporal pattern of each syllable
addr = 'speech signal';
% sylb_ID = 'ace';
clear STo STa I
% slist = {'ar', 'dreins', 'eis', 'eve', 'fal', 'fans', 'geim', 'hai', 'ind', ...
%     'is', 'ker', 'krei', 'long', 'luk', 'me', 'more', 'ning', 'nis', 'noi', 'noiz', 'nout', ...
%     'po', 'ru', 'sain', 'sbrint', 'si', 'tai', 'te', 'ti', 'the', 'wan', ...
%     'wins', 'zat', 'zi', 'ziis', '-'};    
slist = {'aa', 'eis', 'eve', 'fair', 'geim', 'gli', 'in', 'ind', ...
    'is', 'jo', 'ker', 'laod', 'more', 'naf', 'ning', 'nis', 'noiz', ...
    'po', 'prai', 'ru', 'sbrint', 'shaap', 'sur', 'tai', 'te', 'the', 'un', 'wan', ...
    'wins', 'zat', 'zing', '-'};
Ns = length(slist);
win = 25*8; % samples per gamma unit
I = cell(1, Ns);
ST = zeros(6, 8*25, Ns);
for ks = 1:Ns-1
    sylb_ID = slist{ks};
    STo = get_aud_spectrogram(sylb_ID, addr);
%     [y, fs] = audioread([addr '\' sylb_ID '.wav']);
    
    Lchunk = floor(size(STo, 2))/win;
    STa = zeros(6, win);
    for k = 1:win
        STa(:, k) = mean(STo(:, (k-1)*Lchunk+1: k*Lchunk), 2);
    end   
    sbound = [1; length(STo)];
    I_temp = get_syl_parameters(STo, sbound);
    I{ks} = I_temp{1};
    ST(:, :, ks) = STa;
end



%% for visualization
ks = 6;
sylb_ID = slist{ks};
[y, fs] = audioread([addr '\' sylb_ID '.wav']);
STo = get_aud_spectrogram(sylb_ID, addr);
figure;
colormap('parula')
subplot(2, 2, 1)
plot(y(1:16:end)); axis tight
subplot(2, 2, 3)
imagesc(STo)
subplot(2, 2, 2)
imagesc(ST(:, :, ks))
subplot(2, 2, 4)
imagesc(I{ks})
%%
null = zeros(6, 8);
I(:, :, 12) = null;
ST(:, :, 12) = zeros(6, 200);
