m = 20000;
n = 2;
X = rand(m, n);
mul = 2.;

ratio = ((X(:, 1) - 0.5).*(X(:, 1) - 0.5) + (X(:, 2) - 0.5).*(X(:, 2) - 0.5)) / 0.16;
prob_include = 2 ./ (1 + exp(mul * ratio));
idx_ok = (rand(m, 1) < prob_include);
X = X(idx_ok, :);
m = size(X, 1);

X = [ones(m, 1), X];
w_0 = 100;
w = w_0 * [-1, 1, 1]';
prob = 1 ./ (1 + exp(-X * w));
y = 2 * (rand(m, 1) < prob) - 1;

idx_pos = (y == 1);
idx_neg = (y == -1);


h=figure;
hold('on');

plot([0, 1], [1, 0], 'k--', 'LineWidth', 3);
plot(X(idx_pos, 3), X(idx_pos, 2), 'b.', 'MarkerSize', 20);
plot(X(idx_neg, 3), X(idx_neg, 2), 'r.', 'MarkerSize', 20);

axis 'tight'

%legend('ROC-curve','sample set','random guessing');
set(gca, 'FontSize', 24, 'FontName', 'Times');
%legend('$g_1(w)$', '$g_2(w)$', 'Location', 'North');
%set(legend,'FontSize',20,'FontName','Times', 'Interpreter', 'latex', 'Location', 'NorthEast');
axis('square');
%axis([-2.5, 2.5, 0, 6.5])

xlabel('Income','FontSize', 24, 'Interpreter', 'latex');
ylabel('Age','FontSize', 24, 'Interpreter', 'latex');

fig_name = strcat('figures\single_model_plot');
saveas(h, strcat(fig_name, '.png'), 'png');
saveas(h, strcat(fig_name, '.eps'), 'psc2');

w_mul = 100;
w_age = ones(m, 1);
w_income = 3 / 0.6^3 * max(0.8 - X(:, 3), 0.) .* max(0.8 - X(:, 3), 0.);

y_surface = 0.2 + 0.8 * max(0.8 - X(:, 3), 0.) .* max(0.8 - X(:, 3), 0.) .* max(0.8 - X(:, 3), 0.) / 0.6^3;
threshold = w_age .* y_surface + w_income .* X(:, 3);

prob = 1 ./ (1 + exp(w_mul * (-w_age .* X(:, 2) - w_income .* X(:, 3) + threshold)));

y = 2 * (rand(m, 1) < prob) - 1;

idx_pos = (y == 1);
idx_neg = (y == -1);

[sorted_income, idx] = sort(X(:, 3));

h=figure;
hold('on');

plot(X(idx, 3), y_surface(idx), 'k--', 'LineWidth', 3);
plot(X(idx_pos, 3), X(idx_pos, 2), 'b.', 'MarkerSize', 20);
plot(X(idx_neg, 3), X(idx_neg, 2), 'r.', 'MarkerSize', 20);

axis 'tight'

%legend('ROC-curve','sample set','random guessing');
set(gca, 'FontSize', 24, 'FontName', 'Times');
%legend('$g_1(w)$', '$g_2(w)$', 'Location', 'North');
%set(legend,'FontSize',20,'FontName','Times', 'Interpreter', 'latex', 'Location', 'NorthEast');
axis('square');
axis([0, 1, 0, 1])

xlabel('Income','FontSize', 24, 'Interpreter', 'latex');
ylabel('Age','FontSize', 24, 'Interpreter', 'latex');

fig_name = strcat('figures\inhomegeneity_plot');
saveas(h, strcat(fig_name, '.png'), 'png');
saveas(h, strcat(fig_name, '.eps'), 'psc2');


% Plotting densities

x = 0:.001:1;
y = -3:.001:10;

w_income = 3 / 0.6^3 * max(0.8 - x, 0.) .* max(0.8 - x, 0.);
w_income = 3 * w_income / mean(w_income);

prob = normpdf(y, 3, 1);
%prob_new = normpdf(y, 3, 1);
prob = ones(size(x, 1), 1) * prob;

h1=figure;
hold('on');

imagesc(x, y, prob'); colorbar
plot([0, 1], [3, 3], 'k-', 'LineWidth', 3);

axis 'tight'

%legend('ROC-curve','sample set','random guessing');
set(gca, 'FontSize', 24, 'FontName', 'Times');
%legend('$w_{\text{income}}$', 'Location', 'North');
%set(legend,'FontSize',20,'FontName','Times', 'Interpreter', 'latex', 'Location', 'North');
axis('square');
axis([0, 1, 0, 6])

xlabel('Income','FontSize',24, 'Interpreter', 'latex');
ylabel('$w_{\mathrm{income}}$','FontSize',24, 'Interpreter', 'latex');

fig_name = strcat('figures\single_model_density');
saveas(h1, strcat(fig_name, '.png'), 'png');
saveas(h1, strcat(fig_name, '.eps'), 'psc2');

prob_new = zeros(size(x, 2), size(y, 2));
for index=1:size(x, 2)
   prob_new(index, :) = normpdf(y, w_income(index), 1); 
end

h2=figure;
hold('on');

imagesc(x, y, prob_new'); colorbar
plot(x, w_income, 'k-', 'LineWidth', 3);

axis 'tight'

%legend('ROC-curve','sample set','random guessing');
set(gca, 'FontSize', 24, 'FontName', 'Times');
%legend('$w_{\text{income}}$', 'Location', 'North');
%set(legend,'FontSize',20,'FontName','Times', 'Interpreter', 'latex', 'Location', 'North');
axis('square');
axis([0, 1, -3, 10])

xlabel('Income','FontSize',24, 'Interpreter', 'latex');
ylabel('$w_{\mathrm{income}}$','FontSize',24, 'Interpreter', 'latex');

fig_name = strcat('figures\inhomegeneous_model_density');
saveas(h2, strcat(fig_name, '.png'), 'png');
saveas(h2, strcat(fig_name, '.eps'), 'psc2');


% Illustration of presence of similar models in a multimodel
m1 = 1000;
m2 = 100;
scale_1 = 0.1;
scale_2 = 0.07;
X1 = scale_1 * randn(m1, 2);
X1(:, 1) = X1(:, 1) + 0.25;
X1(:, 2) = X1(:, 2) + 0.75;
idx_ok = and(X1(:, 1) >= 0, and(X1(:, 1) <=1, and(X1(:, 2) >= 0, X1(:, 2) <= 1)));
X1 = X1(idx_ok, :);

X2 = scale_2 * randn(m2, 2);
X2(:, 1) = X2(:, 1) + 0.75;
X2(:, 2) = X2(:, 2) + 0.25;

idx_ok = and(X2(:, 1) >= 0, and(X2(:, 1) <=1, and(X2(:, 2) >= 0, X2(:, 2) <= 1)));
X2 = X2(idx_ok, :);

X = [X1; X2];
X = [ones(size(X, 1), 1) X];
w_0 = 10;
w = w_0 * [-1, 1, 1]';

prob = 1 ./ (1 + exp(-X * w));
y = 2 * (rand(size(X, 1), 1) < prob) - 1;
y1 = y(1:size(X1, 1));
y2 = y((size(X1, 1) + 1):size(X, 1));

X1 = X(1:size(X1, 1), :);
X2 = X((size(X1, 1) + 1):size(X, 1), :);

idx_pos = (y == 1);
idx_neg = (y == -1);

w2 = learn_single_logistic(X2, y2, zeros(3, 3));
w_all = learn_single_logistic(X, y, zeros(3, 3));

x_surf = [0.6, 0.9];
y_surf = -(w2(1) / w2(3) + w2(2) / w2(3) * x_surf); 

x_surf_all = [0., 1.];
y_surf_all = -(w_all(1) / w_all(3) + w_all(2) / w_all(3) * x_surf_all); 

h3=figure;
hold('on');

plot(x_surf, y_surf, 'k--', 'LineWidth', 3);
plot(x_surf_all, y_surf_all, 'k-', 'LineWidth', 3);
plot(X(idx_pos, 2), X(idx_pos, 3), 'b.', 'MarkerSize', 20);
plot(X(idx_neg, 2), X(idx_neg, 3), 'r.', 'MarkerSize', 20);

axis 'tight'

%legend('ROC-curve','sample set','random guessing');
set(gca, 'FontSize', 24, 'FontName', 'Times');
legend('$\mathbf{w}_{2}$', '$\mathbf{w}_{\mathrm{all}}$');
set(legend,'FontSize',20,'FontName','Times', 'Interpreter', 'latex', 'Location', 'NorthEast');
axis('square');
axis([0, 1, 0, 1])

xlabel('Income','FontSize',24, 'Interpreter', 'latex');
ylabel('Age','FontSize',24, 'Interpreter', 'latex');

fig_name = strcat('figures\multilevel_similar_illustration');
saveas(h3, strcat(fig_name, '.png'), 'png');
saveas(h3, strcat(fig_name, '.eps'), 'psc2');

% Mixture of models for a single model data

m = 5000;
n = 2;
X = rand(m, n);
mul = 2.;

ratio = ((X(:, 1) - 0.5).*(X(:, 1) - 0.5) + (X(:, 2) - 0.5).*(X(:, 2) - 0.5)) / 0.16;
prob_include = 2 ./ (1 + exp(mul * ratio));
idx_ok = (rand(m, 1) < prob_include);
X = X(idx_ok, :);
m = size(X, 1);

X = [ones(m, 1), X];
w_0 = 2;
w = w_0 * [-1, 1, 1]';
prob = 1 ./ (1 + exp(-X * w));
y = 2 * (rand(m, 1) < prob) - 1;

w_single = learn_single_logistic(X, y, zeros(3, 3));

A = cell(5, 1);
for index=1:5
   A{index} = zeros(3, 3); 
end
[w_mix, pi_mix, hessian_mix] = learn_mixture_logistic(X, y, A, 1.);

idx_pos = (y == 1);
idx_neg = (y == -1);

y_value_0 = -w_mix(1, :) ./ w_mix(2, :);
y_value_1 = (-w_mix(1, :) - w_mix(3, :)) ./ w_mix(2, :);

y_values = [y_value_0; y_value_1];
x_values = [zeros(1, 5); ones(1, 5)];

colors = cell(5,1);
colors{1} = [0, 1, 0];
colors{2} = [1, 0, 1];
colors{3} = [1, 1, 0];
colors{4} = [0, 1, 0.8];
colors{5} = [1, 0.5, 0];

h=figure;
hold('on');

plot([0, 1], [1, 0], 'k--', 'LineWidth', 3);

for index=1:5
    plot(x_values(:, index), y_values(:, index), 'LineWidth', 3, 'Color', colors{index}); 
end

%plot(x_values, y_values, 'LineWidth', 3);

plot(X(idx_pos, 3), X(idx_pos, 2), 'b.', 'MarkerSize', 20);
plot(X(idx_neg, 3), X(idx_neg, 2), 'r.', 'MarkerSize', 20);

axis 'tight'

%legend('ROC-curve','sample set','random guessing');
set(gca, 'FontSize', 24, 'FontName', 'Times');
legend('Initial', '$\pi = 0.263$', '$\pi = 0.258$', '$\pi = 0.202$', '$\pi = 0.153$', '$\pi = 0.124$');
set(legend,'FontSize',20,'FontName','Times', 'Interpreter', 'latex', 'Location', 'East');
axis('square');
axis([0, 1, 0, 1])

xlabel('Income','FontSize', 24, 'Interpreter', 'latex');
ylabel('Age','FontSize', 24, 'Interpreter', 'latex');

fig_name = strcat('figures\single_model_plot_second');
saveas(h, strcat(fig_name, '.png'), 'png');
saveas(h, strcat(fig_name, '.eps'), 'psc2');

% Illustrating evidence origin

x_values = linspace(-2, 2, 4000);
likely_0 = 0.1 * normpdf(x_values, 0, 0.005);
likely_1 = normpdf(x_values, 0, 0.5);

h=figure;
hold('on');

plot(x_values, likely_0, 'r-', 'LineWidth', 3);
plot(x_values, likely_1, 'b-', 'LineWidth', 3);

axis 'tight'

%legend('ROC-curve','sample set','random guessing');
set(gca, 'FontSize', 24, 'FontName', 'Times');
legend('Model 1 likelihood', 'Model 2 likelihood');
set(legend,'FontSize',20,'FontName','Times', 'Interpreter', 'latex', 'Location', 'East');
%axis('square');
%axis([0, 1, 0, 1])

xlabel('$\mathbf{w}$','FontSize', 24, 'Interpreter', 'latex');
ylabel('$p(\mathbf{y}|\mathbf{X},\:\mathbf{w})$','FontSize', 24, 'Interpreter', 'latex');

fig_name = strcat('figures\evidence_origin_plot');
saveas(h, strcat(fig_name, '.png'), 'png');
saveas(h, strcat(fig_name, '.eps'), 'psc2');


