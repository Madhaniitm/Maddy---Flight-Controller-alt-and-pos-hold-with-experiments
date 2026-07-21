%% plot_ND1_quality_cost.m  →  fig4_2
% ND1 — Camera-as-tool: quality vs cost scatter per orchestrator.
% Data hardcoded from ND1_summary_20260528_133550.csv + ND1_scored_20260528_150152.csv.

clear; clc; close all;
set(groot,'DefaultAxesFontSize',20);

%% ── hardcoded data ───────────────────────────────────────────────────────────
model_lbl = {'Claude 3.5 Sonnet','GPT-4o','GPT-4o-mini','Gemini 2.5 Flash'};
quality   = [3.00,  4.38,  4.42,  4.40];
cost      = [0.04425, 0.02452, 0.00158, 0.00043];
flip_pct  = [55.2,  18.8,  12.5,  18.8];

colors = {[0.20 0.55 0.85]; [1.00 0.55 0.10]; [0.55 0.20 0.70]; [0.20 0.70 0.35]};
markers = {'o','s','^','d'};

%% ── Figure ───────────────────────────────────────────────────────────────────
figure('Name','ND1 — Quality vs Cost','Position',[60 60 900 750]);
ax = axes;
hold on;

for k = 1:4
    scatter(ax, cost(k), quality(k), 280, colors{k}, markers{k}, 'filled', ...
            'MarkerEdgeColor','k','LineWidth',1.2,'DisplayName',model_lbl{k});
end

% bubble hint for flip rate
for k = 1:4
    text(ax, cost(k), quality(k) - 0.15, sprintf('flip %.0f%%', flip_pct(k)), ...
         'HorizontalAlignment','center','FontSize',14,'Color',[0.4 0.4 0.4]);
end

set(ax,'XScale','log','Box','off','XGrid','on','YGrid','on', ...
       'GridAlpha',.25,'GridLineStyle','--');
xlabel(ax,'Cost per run  (USD, log scale)','FontSize',22);
ylabel(ax,'Report quality  (0–5)','FontSize',22);
title(ax,{'ND1 — Image Verbalization as Tool'; ...
          '100% analyze\_scene call rate across all 4 orchestrators × 8 images × 5 runs'}, ...
     'FontSize',20);
ylim(ax,[2.5 5.0]);
xlim(ax,[0.0002 0.1]);
legend(ax,'FontSize',18,'Location','northeast','NumColumns',1);

% annotate cost ratio
text(ax, 0.0003, 2.65, sprintf('Gemini is %.0f× cheaper than Claude', ...
     cost(1)/cost(4)), 'FontSize',15,'Color',[0.3 0.3 0.3]);
hold off;
