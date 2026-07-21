%% plot_C8_three_mode.m
% C8 — Three-mode comparison: altitude RMSE per waypoint (Mode A / B / C).
% Data hardcoded from CSV files. Rows = models, Cols = WP1-WP4.

clear; clc; close all;
set(groot,'DefaultAxesFontSize',22);

%% Mode A (scripted, 1 run) — from thesis Table 3.C8
modeA_wp = [2.92, 2.98, 2.99, 3.00];

%% ── per-model mean RMSE per waypoint (cm) ────────────────────────────────────
%  Rows: [Claude; GPT-4o; GPT-4o-mini; Gemini]  Cols: [WP1, WP2, WP3, WP4]

B_per_model = [ ...
    1.0606,  0.7422,  0.6778,  0.8752; ...   % Claude
    1.1188,  0.7594,  0.6266,  0.8510; ...   % GPT-4o
    1.1350,  0.7370,  0.6298,  0.8128; ...   % GPT-4o-mini
    1.0980,  0.7364,  0.6232,  0.8302  ];    % Gemini

C_per_model = [ ...
    1.1604,  0.7492,  0.6292,  0.8536; ...   % Claude
    1.0902,  0.7536,  0.5642,  0.8412; ...   % GPT-4o
    1.1208,  0.7172,  0.6480,  0.8178; ...   % GPT-4o-mini
    1.0992,  0.7092,  0.6510,  0.8198  ];    % Gemini

B_mean = mean(B_per_model, 1);
C_mean = mean(C_per_model, 1);
B_std  = std(B_per_model,  0, 1);
C_std  = std(C_per_model,  0, 1);

%% ── Figure ───────────────────────────────────────────────────────────────────
figure('Name','C8 — Three-Mode Comparison','Position',[60 60 1200 900]);
ax = axes;
hold on;

cA = [0.55 0.55 0.55];
cB = [0.20 0.55 0.85];
cC = [0.25 0.72 0.38];

x      = 1:4;
bWidth = 0.24;

bar(ax, x - bWidth, modeA_wp, bWidth, 'FaceColor',cA,'EdgeColor','none', ...
    'DisplayName','Mode A — Scripted (no LLM)');
bar(ax, x,          B_mean,   bWidth, 'FaceColor',cB,'EdgeColor','none', ...
    'DisplayName','Mode B — NL-commanded (mean ± sd across models)');
bar(ax, x + bWidth, C_mean,   bWidth, 'FaceColor',cC,'EdgeColor','none', ...
    'DisplayName','Mode C — Full-auto (mean ± sd across models)');

errorbar(ax, x,          B_mean, B_std, 'k.','LineWidth',1.8,'HandleVisibility','off');
errorbar(ax, x + bWidth, C_mean, C_std, 'k.','LineWidth',1.8,'HandleVisibility','off');

model_markers = {'o','s','^','d'};
model_colors  = { [0.14 0.39 0.60]; [1.00 0.60 0.10]; [0.60 0.20 0.70]; [0.85 0.18 0.18] };

for m = 1:4
    scatter(ax, x,          B_per_model(m,:), 60, model_colors{m}, model_markers{m}, ...
            'filled','HandleVisibility','off','MarkerEdgeColor','none','MarkerFaceAlpha',0.7);
    scatter(ax, x + bWidth, C_per_model(m,:), 60, model_colors{m}, model_markers{m}, ...
            'filled','HandleVisibility','off','MarkerEdgeColor','none','MarkerFaceAlpha',0.7);
end

improvement = mean(modeA_wp) / mean([B_mean, C_mean]);
text(ax, 2.5, 2.60, sprintf('LLM modes: %.1f\\times lower RMSE than Mode A', improvement), ...
     'HorizontalAlignment','center','FontSize',19,'Color',[0.2 0.2 0.2],'FontWeight','bold');

yline(ax, mean(modeA_wp), '--', 'Color',cA,'LineWidth',1.2,'HandleVisibility','off');

set(ax,'XTick',1:4,'XTickLabel',{'WP1  (1.0→1.3 m)','WP2  (1.3→1.5 m)','WP3  (1.5→1.0 m)','WP4  (hold)'}, ...
       'Box','off','YGrid','on','GridAlpha',.25,'GridLineStyle','--','XGrid','off');
xlabel(ax,'Waypoint','FontSize',24);
ylabel(ax,'Altitude RMSE  (cm)','FontSize',24);
title(ax,{'C8 — Three-Mode Comparison: Altitude RMSE per Waypoint'; ...
          'Dots = individual models (Claude / GPT-4o / Mini / Gemini), bars = mean across models'}, ...
     'FontSize',20);
ylim(ax,[0 3.5]);
legend(ax,'FontSize',18,'NumColumns',1,'Location','northeast');
hold off;
