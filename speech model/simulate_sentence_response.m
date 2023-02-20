clear

sentences = {'ace_tennis', 'ace_poker', 'ace_game', 'sprint_tennis', 'sprint_game', ...
    'joker_poker', 'joker_game', 'ace_enough', 'ace_surprising', 'sprint_enough', ...
    'sprint_surprising', 'joker_enough', 'joker_surprising', ...
    'tie_game', 'tie_eve', 'noise_game', 'noise_eve', ...
    'tie_ugly', 'tie_unfair', 'noise_loud', 'noise_sharp'};

suffix = '_context1_5_new_step_NNP_pre_6_8';

for j = 1
    MDP = DEM_MDP_ambiguity_hybrid(j);
    context = MDP.D{1};
    save([sentences{j} suffix '.mat'], 'context', 'MDP');
    clear MDP
end

