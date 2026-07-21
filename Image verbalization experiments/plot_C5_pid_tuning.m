%% plot_C5_pid_tuning.m
% C5 — LLM-driven PID self-tuning: RMSE before vs after per model
% Data hardcoded from CSV files. Grouped bar chart with 95% CI error bars.

clear; clc; close all;
set(groot,'DefaultAxesFontSize',22);

model_lbl = {'Claude','GPT-4o','GPT-4o-mini','Gemini'};
n = 4;

%% ── hardcoded data from C5_summary_*_guardrail_on.csv ───────────────────────
rmse_before   = [0.14906,  0.14287,  0.15084,  0.13754];
rmse_after    = [0.03578,  0.03203,  0.03861,  0.04298];

% err = (ci_hi - ci_lo) / 2
err_before    = [0.02243,  0.04485,  0.017835, 0.02441];
err_after     = [0.004725, 0.002885, 0.00486,  0.007885];

reduction_pct = [75.6,     74.5,     74.1,     67.8];
cycles        = [1.8,      1.6,      4.6,      3.4];

%% ── Figure ───────────────────────────────────────────────────────────────────
figure('Name','C5 — PID Self-Tuning','Position',[60 60 1000 900]);
ax  = axes;
hold on;

cBefore = [0.85 0.25 0.15];
cAfter  = [0.20 0.65 0.30];

x      = 1:n;
bWidth = 0.32;
x_b    = x - bWidth/2;
x_a    = x + bWidth/2;

bar(ax, x_b, rmse_before, bWidth, 'FaceColor',cBefore, 'EdgeColor','none', ...
    'DisplayName','RMSE  before tuning (K_p = 1.5)');
bar(ax, x_a, rmse_after,  bWidth, 'FaceColor',cAfter,  'EdgeColor','none', ...
    'DisplayName','RMSE  after tuning');

errorbar(ax, x_b, rmse_before, err_before, 'k.', 'LineWidth',1.8, 'HandleVisibility','off');
errorbar(ax, x_a, rmse_after,  err_after,  'k.', 'LineWidth',1.8, 'HandleVisibility','off');

for k = 1:n
    text(ax, x_a(k), rmse_after(k) + err_after(k) + 0.004, ...
         sprintf('%.1f%%', reduction_pct(k)), ...
         'HorizontalAlignment','center','FontSize',18,'Color',cAfter,'FontWeight','bold');
    text(ax, x_b(k), rmse_before(k) + err_before(k) + 0.004, ...
         sprintf('%.1f cyc', cycles(k)), ...
         'HorizontalAlignment','center','FontSize',15,'Color',[0.4 0.4 0.4]);
end

set(ax,'XTick',1:n,'XTickLabel',model_lbl,'Box','off', ...
       'YGrid','on','GridAlpha',.25,'GridLineStyle','--','XGrid','off');
xlabel(ax,'Model','FontSize',24);
ylabel(ax,'Roll RMSE  (°)','FontSize',24);
title(ax,{'C5 — LLM-Driven PID Self-Tuning'; ...
          'Roll oscillation reduced by 68–76%  (K_p injected = 1.5, nominal = 0.3)'},'FontSize',22);
legend(ax,'FontSize',20,'NumColumns',1,'Location','northeast');
ylim(ax,[0 0.25]);
hold off;
