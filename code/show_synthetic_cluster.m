% Demonstrating multilevel problems with clustering

m = [100, 350, 1000, 2500];
div = [1, 5, 10, 50];
w_scale = [0.1, 0.2, 0.3, 0.5, 1];
K = 10;

AUC_adequate_mean = zeros(size(m, 2), size(div, 2), size(w_scale, 2));
AUC_adequate_min = zeros(size(m, 2), size(div, 2), size(w_scale, 2));
AUC_adequate_std = zeros(size(m, 2), size(div, 2), size(w_scale, 2));
AUC_multilevel_mean = zeros(size(m, 2), size(div, 2), size(w_scale, 2));
AUC_multilevel_min = zeros(size(m, 2), size(div, 2), size(w_scale, 2));
AUC_multilevel_std = zeros(size(m, 2), size(div, 2), size(w_scale, 2));
num_groups = zeros(size(m, 2), size(div, 2), size(w_scale, 2));
prob_one_group = zeros(size(m, 2), size(div, 2), size(w_scale, 2));

max_iter = 10;

for iter=1:max_iter
    for i=1:size(m, 2)
        i
        for j=1:size(div, 2)
            for index=1:size(w_scale, 2)
                to_plot = 0;
                if (and(index == 1, iter==1))
                   to_plot = 1; 
                end
                to_plot = 0;
                [X, y, idx, AUC_adequate_mean_cur, AUC_adequate_min_cur, AUC_adequate_std_cur, AUC_multilevel_mean_cur, ...
                    AUC_multilevel_min_cur, AUC_multilevel_std_cur, num_groups_cur] = generate_syntethic_cluster(m(1, i), K, div(1, j), w_scale(1, index), 1, to_plot);
                AUC_adequate_mean(i, j, index) = AUC_adequate_mean(i, j, index) + AUC_adequate_mean_cur / max_iter;
                AUC_adequate_min(i, j, index) = AUC_adequate_min(i, j, index) + AUC_adequate_min_cur / max_iter;
                AUC_adequate_std(i, j, index) = AUC_adequate_std(i, j, index) + AUC_adequate_std_cur / max_iter;
                
                AUC_multilevel_mean(i, j, index) = AUC_multilevel_mean(i, j, index) + AUC_multilevel_mean_cur / max_iter;
                AUC_multilevel_min(i, j, index) = AUC_multilevel_min(i, j, index) + AUC_multilevel_min_cur / max_iter;
                AUC_multilevel_std(i, j, index) = AUC_multilevel_std(i, j, index) + AUC_multilevel_std_cur / max_iter;
                num_groups(i, j, index) = num_groups(i, j, index) + num_groups_cur / max_iter;
                prob_one_group(i, j, index) = prob_one_group(i, j, index) + (num_groups_cur == 1) / max_iter;
                close 'all'
            end
        end
    end
end

for i=1:size(m, 2)
    h=figure;
    hold('on');

    imagesc(squeeze(AUC_adequate_mean(i, :, :)));
    set(gca, 'FontSize', 24, 'FontName', 'Times');
    
    xlabel('$w_0$','FontSize',24, 'Interpreter', 'latex');
    ylabel('$\delta$','FontSize',24, 'Interpreter', 'latex');
    

    axis 'tight' 'square'
    bar = colorbar('Location','eastoutside');
    set(gca, 'FontSize', 24, 'FontName', 'Times');
    set(gca,'YTick',1:size(div, 2),'YTickLabel', div)
    set(gca,'XTick',1:size(w_scale, 2),'XTickLabel', w_scale)

    fig_name = strcat('figures\cluster_synth_auc_dep_m_', num2str(m(1, i)));
    saveas(h, strcat(fig_name, '.png'), 'png');
    saveas(h, strcat(fig_name, '.eps'), 'psc2');
    close 'all'
    
    h1=figure;
    hold('on');

    imagesc(squeeze(AUC_adequate_mean(i, :, :)) - squeeze(AUC_multilevel_mean(i, :, :)));
    set(gca, 'FontSize', 24, 'FontName', 'Times');
    
    xlabel('$w_0$','FontSize',24, 'Interpreter', 'latex');
    ylabel('$\delta$','FontSize',24, 'Interpreter', 'latex');
    

    axis 'tight' 'square'
    bar = colorbar('Location','eastoutside');
    set(gca, 'FontSize', 24, 'FontName', 'Times');
    set(gca,'YTick',1:size(div, 2),'YTickLabel', div)
    set(gca,'XTick',1:size(w_scale, 2),'XTickLabel', w_scale)

    fig_name = strcat('figures\cluster_synth_diff_auc_dep_m_', num2str(m(1, i)));
    saveas(h1, strcat(fig_name, '.png'), 'png');
    saveas(h1, strcat(fig_name, '.eps'), 'psc2');
    close 'all'
    
    h2=figure;
    hold('on');

    imagesc(squeeze(AUC_adequate_min(i, :, :)));
    set(gca, 'FontSize', 24, 'FontName', 'Times');
    
    xlabel('$w_0$','FontSize',24, 'Interpreter', 'latex');
    ylabel('$\delta$','FontSize',24, 'Interpreter', 'latex');
    

    axis 'tight' 'square'
    bar = colorbar('Location','eastoutside');
    set(gca, 'FontSize', 24, 'FontName', 'Times');
    set(gca,'YTick',1:size(div, 2),'YTickLabel', div)
    set(gca,'XTick',1:size(w_scale, 2),'XTickLabel', w_scale)

    fig_name = strcat('figures\cluster_synth_min_auc_dep_m_', num2str(m(1, i)));
    saveas(h2, strcat(fig_name, '.png'), 'png');
    saveas(h2, strcat(fig_name, '.eps'), 'psc2');
    close 'all'
    
    h3=figure;
    hold('on');

    imagesc(squeeze(AUC_adequate_min(i, :, :)) - squeeze(AUC_multilevel_min(i, :, :)));
    set(gca, 'FontSize', 24, 'FontName', 'Times');
    
    xlabel('$w_0$','FontSize',24, 'Interpreter', 'latex');
    ylabel('$\delta$','FontSize',24, 'Interpreter', 'latex');
    

    axis 'tight' 'square'
    bar = colorbar('Location','eastoutside');
    set(gca, 'FontSize', 24, 'FontName', 'Times');
    set(gca,'YTick',1:size(div, 2),'YTickLabel', div)
    set(gca,'XTick',1:size(w_scale, 2),'XTickLabel', w_scale)

    fig_name = strcat('figures\cluster_synth_diff_min_auc_dep_m_', num2str(m(1, i)));
    saveas(h3, strcat(fig_name, '.png'), 'png');
    saveas(h3, strcat(fig_name, '.eps'), 'psc2');
    close 'all'

end