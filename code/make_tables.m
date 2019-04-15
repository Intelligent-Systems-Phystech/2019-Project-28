num_K = size(K_all, 2);
header = 'Данные / K ';
for index=1:size(K_all, 2)
    header = strcat(header, ' & ', num2str(K_all(1, index)));
end
header = strcat(header, '\\');
rows = cell(11, 1);
rows{1} = header;

cur_row = strcat('$\text{AUC}_{s}$ & \multicolumn{', num2str(num_K), '}{|c|}{', get_round(mean_AUC_common_joint(1), 4), '} \\');
rows{2} = cur_row;

names = {'$\text{AUC}_{m}$', '$\text{AUC}_{m}^{\text{adeq}}$', '$\text{AUC}_{\text{mix}}$', '$\text{AUC}_{\text{mix}}^{\text{adeq}}$',...
    '$t_{m}$', '$t_{\text{mix}}$', '$K_{m}^{\text{adeq}}$', '$K_{\text{mix}}$', '$K_{\text{mix}}^{\text{adeq}}$'};

digits = {4, 4, 4, 4, 2, 2, 2, 2, 2};
matrices = {'mean_AUC_joint', 'mean_AUC_adequate_joint', 'mean_AUC_mix_joint', 'mean_AUC_mix_adequate_joint',...
    't_stat', 't_stat_mix', 'mean_multilevel_K', 'mean_K_mix_init', 'mean_K_mix'};

for pos=1:size(names, 2)
    cur_row = names{pos};
    cur_matrix = eval(matrices{pos});
    for index=1:size(K_all, 2)
       cur_row = strcat(cur_row, ' & ', get_round(cur_matrix(index), digits{pos})); 
    end
    cur_row = strcat(cur_row, '\\');
    rows{2 + pos} = cur_row;
end

for index=1:size(rows, 1)
   disp('\hline')
   disp(rows{index})
end
disp('\hline')