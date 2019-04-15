w = [-1, 0.5, 0.5; 0, 0.5 * sqrt(3), -0.5 * sqrt(3)];
x = -5:.01:5;
y = -5:.01:5;
[X, Y] = meshgrid(y,x);
X = X(:);
Y = Y(:);
prob_init = X * w(1, :) + Y * w(2, :);
idx = ones(size(prob_init, 1), 1);
sin_phi = Y ./ sqrt(X .* X + Y .* Y);
idx_sec = and(X <= 0, sin_phi >= -0.5);
idx(idx_sec) = 2;
idx_third = and(X > 0, sin_phi >= -0.5);
idx(idx_third) = 3;

[~, idx] = max(abs(prob_init), [], 2);
prob = prob_init(:, 1);
prob((idx == 2)) = prob_init((idx == 2), 2);
prob((idx == 3)) = prob_init((idx == 3), 3);
prob = ones(size(prob, 1), 1) ./ (1 + exp(-prob));

X = vec2mat(X, 1001);
Y = vec2mat(X, 1001);
prob = vec2mat(prob, 1001);

h1=figure;
hold('on');

imagesc(x, y, prob'); colorbar
plot([0, 0], [0, 5], 'k--', 'LineWidth', 3);
plot([0, 5], [0, -5  / sqrt(3)], 'k--', 'LineWidth', 3);
plot([0, -5], [0, -5 / sqrt(3)], 'k--', 'LineWidth', 3);

plot([0, 0], [0, -5], 'k-', 'LineWidth', 3);
plot([0, -5], [0, 5 / sqrt(3)], 'k-', 'LineWidth', 3);
plot([0, 5], [0, 5 / sqrt(3)], 'k-', 'LineWidth', 3);
axis 'tight'

%legend('ROC-curve','sample set','random guessing');
set(gca, 'FontSize', 24, 'FontName', 'Times');
%legend('$g_1(w)$', '$g_2(w)$', 'Location', 'North');
%set(legend,'FontSize',20,'FontName','Times', 'Interpreter', 'latex', 'Location', 'NorthEast');
axis('square');
%axis([-2.5, 2.5, 0, 6.5])

xlabel('$x_1$','FontSize',24, 'Interpreter', 'latex');
ylabel('$x_2$','FontSize',24, 'Interpreter', 'latex');

fig_name = strcat('figures\multilevel_prob_plot');
saveas(h1, strcat(fig_name, '.png'), 'png');
saveas(h1, strcat(fig_name, '.eps'), 'psc2');

% For mixture
prob = prob_init;
prob = ones(size(prob, 1), 3) ./ (1 + exp(-prob));
prob = sum(prob, 2)/ 3;

X = vec2mat(X, 1001);
Y = vec2mat(X, 1001);
prob = vec2mat(prob, 1001);

h=figure;
hold('on');

imagesc(x, y, prob'); colorbar
plot([0, 0], [0, 5], 'k--', 'LineWidth', 2);
plot([0, 5], [0, -5  / sqrt(3)], 'k--', 'LineWidth', 2);
plot([0, -5], [0, -5 / sqrt(3)], 'k--', 'LineWidth', 2);

plot([0, 0], [0, -5], 'k-', 'LineWidth', 2);
plot([0, -5], [0, 5 / sqrt(3)], 'k-', 'LineWidth', 2);
plot([0, 5], [0, 5 / sqrt(3)], 'k-', 'LineWidth', 2);
axis 'tight'

%legend('ROC-curve','sample set','random guessing');
set(gca, 'FontSize', 24, 'FontName', 'Times');
%legend('$g_1(w)$', '$g_2(w)$', 'Location', 'North');
%set(legend,'FontSize',20,'FontName','Times', 'Interpreter', 'latex', 'Location', 'NorthEast');
axis('square');
%axis([-2.5, 2.5, 0, 6.5])

xlabel('$x_1$','FontSize',24, 'Interpreter', 'latex');
ylabel('$x_2$','FontSize',24, 'Interpreter', 'latex');

fig_name = strcat('figures\mixture_prob_plot');
saveas(h, strcat(fig_name, '.png'), 'png');
saveas(h, strcat(fig_name, '.eps'), 'psc2');