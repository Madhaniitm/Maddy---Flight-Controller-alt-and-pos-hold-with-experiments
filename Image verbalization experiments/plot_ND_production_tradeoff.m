%% plot_ND_production_tradeoff.m  →  fig4_9
% ND series production tradeoff: quality/5 vs cost/run.
% Pareto frontier annotated. Data hardcoded from ND3 consolidated results.

clear; clc; close all;
set(groot,'DefaultAxesFontSize',20);

%% ── hardcoded production configuration data ──────────────────────────────────
% From chapter table and consolidated ND3 results.
model_lbl = {'Claude 3.5 Sonnet', 'GPT-4o', 'Gemini 2.5 Flash', 'GPT-4o-mini + M1–M6'};
quality   = [3.20,  4.10,  3.50,  4.60];
cost      = [1.91,  0.32,  0.027, 0.016];

colors  = {[0.20 0.55 0.85]; [1.00 0.55 0.10]; [0.20 0.70 0.35]; [0.55 0.20 0.70]};
markers = {'o','s','d','^'};
sizes   = [350, 350, 350, 350];

%% ── Figure ───────────────────────────────────────────────────────────────────
figure('Name','ND — Production Tradeoff','Position',[60 60 1000 800]);
ax = axes;
hold on;

% Pre-fix frontier: Gemini → GPT-4o  (without GPT-4o-mini+M1-M6)
% GPT-4o-mini+M1-M6 strictly dominates both: lower cost ($0.016 < $0.027)
% AND higher quality (4.60 > 3.50) — so mini is NOT on this tradeoff line, it beats it.
pareto_cost    = [cost(3), cost(2)];   % Gemini → GPT-4o
pareto_quality = [quality(3), quality(2)];
plot(ax, pareto_cost, pareto_quality, '--', 'Color',[0.60 0.60 0.60], ...
     'LineWidth',1.8,'DisplayName','Pre-fix frontier (Gemini → GPT-4o)');

for k = 1:4
    scatter(ax, cost(k), quality(k), sizes(k), colors{k}, markers{k}, 'filled', ...
            'MarkerEdgeColor','k','LineWidth',1.5,'DisplayName',model_lbl{k});
end

% label offsets
offsets = {[0.05, -0.12]; [0.015, 0.12]; [0.0015, 0.12]; [0.0015, -0.16]};
for k = 1:4
    text(ax, cost(k) + offsets{k}(1), quality(k) + offsets{k}(2), model_lbl{k}, ...
         'FontSize',16,'Color',cell2mat(colors(k)),'FontWeight','bold');
end

% "off frontier" annotations
text(ax, cost(1)*0.5, quality(1)+0.12, '✗ Off frontier', ...
     'FontSize',15,'Color',cell2mat(colors(1))*0.7,'FontAngle','italic');
text(ax, cost(2)*1.05, quality(2)-0.17, '✗ Off frontier', ...
     'FontSize',13,'Color',cell2mat(colors(2))*0.7,'FontAngle','italic');

% mini+M1-M6 breakout annotation
text(ax, cost(4)*1.5, quality(4)-0.18, ...
     {'← dominates entire'; 'pre-fix frontier'}, ...
     'FontSize',14,'Color',cell2mat(colors(4)),'FontWeight','bold');

% cost ratio annotation
text(ax, 0.014, 2.85, sprintf('Claude is %.0f× more expensive than GPT-4o-mini+M1–M6', ...
     cost(1)/cost(4)), 'FontSize',14,'Color',[0.4 0.4 0.4]);

set(ax,'XScale','log','Box','off','XGrid','on','YGrid','on', ...
       'GridAlpha',.25,'GridLineStyle','--');
xlabel(ax,'Cost per run  (USD, log scale)','FontSize',22);
ylabel(ax,'Report quality  (0–5)','FontSize',22);
title(ax,{'ND Series — Production Cost–Quality Tradeoff'; ...
          'GPT-4o-mini+M1–M6 breaks out above the pre-fix frontier (lower cost, higher quality)'}, ...
     'FontSize',20,'FontWeight','bold');
legend(ax,'FontSize',17,'Location','northwest','NumColumns',1);
ylim(ax,[2.5 5.2]);
xlim(ax,[0.010 3.5]);
hold off;
