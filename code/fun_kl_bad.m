x = linspace(-5, 5, 10000);
eps = 0.02;
y1 = 0.5 * (abs(x) <= 1);
y2 = eps * (abs(x) <= 1) + (0.5 - eps) * (abs(x) >= 2) .* exp(2 - abs(x));

h=figure;
hold('on');

plot(x, y1,'r-','LineWidth',3);
plot(x, y2,'b-','LineWidth',3);

%legend('ROC-curve','sample set','random guessing');
set(gca, 'FontSize', 24, 'FontName', 'Times');
legend('$g_1(w)$', '$g_2(w)$', 'Location', 'North');
set(legend,'FontSize',20,'FontName','Times', 'Interpreter', 'latex', 'Location', 'NorthEast');
axis('tight');
%axis([-2.5, 2.5, 0, 6.5])

xlabel('w','FontSize',24, 'Interpreter', 'latex');
ylabel('$g(w)$','FontSize',24, 'Interpreter', 'latex');

saveas(h, 'kl_pair_example.png', 'png');
saveas(h, 'kl_pair_example.eps', 'psc2');