%% plot_ND3_results.m  →  fig4_7  (4 individual figures)
% ND3 consolidated HITL results — all orchestrators, both scenarios.
% Shows BEFORE/AFTER the M1-M6 fix for GPT-4o-mini alert_room explicitly.
%
% Three bar groups per model:
%   Blue  = clear_room  (same for both states)
%   Red   = alert_room ND3B  (before fix — mini alert = LITM loop)
%   Green = alert_room ND3C  (after fix  — mini alert = M1-M6+HITL)
%   For Claude / GPT-4o / Gemini: red and green bars are identical (no change).
%   For GPT-4o-mini: dramatic difference shows the fix effect.
%
% Sources (all from CSVs, verified against ND3_observations_20260530.md):
%   Claude, GPT-4o     : ND3_summary_20260529_194556.csv
%   Gemini             : ND3B_gemini_summary_20260530_033733.csv
%   GPT-4o-mini clear  : ND3B_mini_summary_20260530_044156.csv
%   GPT-4o-mini alert ND3B : ND3B_mini_summary (patrol=0, quality=0, cost=$1.70)
%   GPT-4o-mini alert ND3C : ND3C_summary_20260530_053658.csv (M1-M6+HITL)

clear; clc; close all;
set(groot,'DefaultAxesFontSize',20);

%% ── data (all from CSVs) ─────────────────────────────────────────────────────
%                          Claude   GPT-4o  GPT-4o-mini  Gemini
patrol_clear   = [ 60,    100,    100,      80 ];   % %
patrol_alert_b = [ 80,     20,      0,      60 ];   % ND3B (before fix)
patrol_alert_c = [ 80,     20,    100,      60 ];   % ND3C (after fix — mini only changes)

quality_clear   = [ 3.60,  4.00,  4.00,   3.00 ];
quality_alert_b = [ 2.80,  4.20,  0.00,   4.00 ];
quality_alert_c = [ 2.80,  4.20,  4.60,   4.00 ];

cost_clear   = [ 1.6885, 0.2764, 0.0654, 0.0233 ];
cost_alert_b = [ 2.1317, 0.3672, 1.6996, 0.0309 ];  % mini: $1.70 ND3B
cost_alert_c = [ 2.1317, 0.3672, 0.0158, 0.0309 ];  % mini: $0.016 ND3C

model_lbl = {'Claude','GPT-4o','GPT-4o-mini','Gemini'};
mcol = {[0.20 0.55 0.85]; [1.00 0.55 0.10]; [0.55 0.20 0.70]; [0.20 0.70 0.35]};

cBlue  = [0.25 0.60 0.90];   % clear_room
cRed   = [0.85 0.20 0.15];   % alert ND3B — before fix
cGreen = [0.15 0.65 0.30];   % alert ND3C — after fix

n  = 4;
x  = 1:n;
bw = 0.22;
xc = x - bw;   % clear bar x-positions
xb = x;        % alert-before (ND3B) x-positions
xa = x + bw;   % alert-after  (ND3C) x-positions

pos = {[50 580 850 550]; [920 580 850 550]; [50 20 850 550]; [920 20 950 560]};

%% ══════════════════════════════════════════════════════════════════════════════
%% Fig 1 — Patrol completion %
%% ══════════════════════════════════════════════════════════════════════════════
figure('Name','ND3 — Patrol Completion','Position',pos{1});
ax1 = axes; hold(ax1,'on');

bar(ax1, xc, patrol_clear,   bw, 'FaceColor',cBlue, 'EdgeColor','none','DisplayName','clear\_room');
bar(ax1, xb, patrol_alert_b, bw, 'FaceColor',cRed,  'EdgeColor','none','DisplayName','alert\_room  (before fix)');
bar(ax1, xa, patrol_alert_c, bw, 'FaceColor',cGreen,'EdgeColor','none','DisplayName','alert\_room  (after fix — M1–M6)');

for k = 1:n
    if patrol_clear(k) > 0
        text(ax1, xc(k), patrol_clear(k)+2.5,   sprintf('%d%%',patrol_clear(k)),   'HorizontalAlignment','center','FontSize',14,'FontWeight','bold','Color',cBlue*0.7);
    end
    if patrol_alert_b(k) > 0
        text(ax1, xb(k), patrol_alert_b(k)+2.5, sprintf('%d%%',patrol_alert_b(k)), 'HorizontalAlignment','center','FontSize',14,'FontWeight','bold','Color',cRed*0.7);
    end
    text(ax1, xa(k), patrol_alert_c(k)+2.5, sprintf('%d%%',patrol_alert_c(k)), 'HorizontalAlignment','center','FontSize',14,'FontWeight','bold','Color',cGreen*0.7);
end

% LITM loop label on mini alert ND3B bar (height=0)
text(ax1, xb(3), 5, {'0%'; 'LITM'; 'loop'}, 'HorizontalAlignment','center','FontSize',12,'Color',cRed*0.8,'FontWeight','bold','VerticalAlignment','bottom');
% fix arrow annotation
text(ax1, xa(3), patrol_alert_c(3)+8, {'+100 pp'; 'vs before fix'}, 'HorizontalAlignment','center','FontSize',13,'Color',cGreen,'FontWeight','bold');
% GPT-4o hazard abort
text(ax1, xb(2), patrol_alert_b(2)+7, {'hazard'; 'abort ✓'}, 'HorizontalAlignment','center','FontSize',11,'Color',[0.70 0.40 0.05]);

set(ax1,'XTick',x,'XTickLabel',model_lbl,'Box','off','YGrid','on', ...
        'GridAlpha',.25,'GridLineStyle','--','XGrid','off','FontSize',19);
ylabel(ax1,'Patrol completion  (%)','FontSize',22);
title(ax1,{'Patrol Completion'; 'Blue: clear\_room  |  Red: alert before fix  |  Green: alert after M1–M6'}, ...
     'FontSize',20,'FontWeight','bold');
legend(ax1,'FontSize',15,'Location','northeast');
ylim(ax1,[0 138]);
hold(ax1,'off');

%% ══════════════════════════════════════════════════════════════════════════════
%% Fig 2 — Report quality
%% ══════════════════════════════════════════════════════════════════════════════
figure('Name','ND3 — Report Quality','Position',pos{2});
ax2 = axes; hold(ax2,'on');

bar(ax2, xc, quality_clear,   bw, 'FaceColor',cBlue, 'EdgeColor','none','DisplayName','clear\_room');
bar(ax2, xb, quality_alert_b, bw, 'FaceColor',cRed,  'EdgeColor','none','DisplayName','alert\_room  (before fix)');
bar(ax2, xa, quality_alert_c, bw, 'FaceColor',cGreen,'EdgeColor','none','DisplayName','alert\_room  (after fix — M1–M6)');

for k = 1:n
    text(ax2, xc(k), quality_clear(k)+0.08,   sprintf('%.1f',quality_clear(k)),   'HorizontalAlignment','center','FontSize',14,'FontWeight','bold','Color',cBlue*0.7);
    if quality_alert_b(k) > 0
        text(ax2, xb(k), quality_alert_b(k)+0.08, sprintf('%.1f',quality_alert_b(k)), 'HorizontalAlignment','center','FontSize',14,'FontWeight','bold','Color',cRed*0.7);
    end
    text(ax2, xa(k), quality_alert_c(k)+0.08, sprintf('%.1f',quality_alert_c(k)), 'HorizontalAlignment','center','FontSize',14,'FontWeight','bold','Color',cGreen*0.7);
end

text(ax2, xb(3), 0.12, {'0.0'; 'LITM'; 'loop'}, 'HorizontalAlignment','center','FontSize',12,'Color',cRed*0.8,'FontWeight','bold','VerticalAlignment','bottom');
text(ax2, xa(3), quality_alert_c(3)+0.22, {'+4.6 pts'; 'vs before fix'}, 'HorizontalAlignment','center','FontSize',13,'Color',cGreen,'FontWeight','bold');

set(ax2,'XTick',x,'XTickLabel',model_lbl,'Box','off','YGrid','on', ...
        'GridAlpha',.25,'GridLineStyle','--','XGrid','off','FontSize',19);
ylabel(ax2,'Report quality  (0–5)','FontSize',22);
title(ax2,{'Report Quality'; 'Blue: clear\_room  |  Red: alert before fix  |  Green: alert after M1–M6'}, ...
     'FontSize',20,'FontWeight','bold');
legend(ax2,'FontSize',15,'Location','northeast');
ylim(ax2,[0 5.8]);
hold(ax2,'off');

%% ══════════════════════════════════════════════════════════════════════════════
%% Fig 3 — Cost per run  (log scale — range spans $0.016 to $2.13)
%% ══════════════════════════════════════════════════════════════════════════════
figure('Name','ND3 — Cost per Run','Position',pos{3});
ax3 = axes; hold(ax3,'on');

bar(ax3, xc, cost_clear,   bw, 'FaceColor',cBlue, 'EdgeColor','none','DisplayName','clear\_room');
bar(ax3, xb, cost_alert_b, bw, 'FaceColor',cRed,  'EdgeColor','none','DisplayName','alert\_room  (before fix)');
bar(ax3, xa, cost_alert_c, bw, 'FaceColor',cGreen,'EdgeColor','none','DisplayName','alert\_room  (after fix — M1–M6)');

for k = 1:n
    text(ax3, xc(k), cost_clear(k)*1.25,   sprintf('$%.3f',cost_clear(k)),   'HorizontalAlignment','center','FontSize',12,'FontWeight','bold','Color',cBlue*0.7);
    text(ax3, xb(k), cost_alert_b(k)*1.25, sprintf('$%.3f',cost_alert_b(k)), 'HorizontalAlignment','center','FontSize',12,'FontWeight','bold','Color',cRed*0.7);
    text(ax3, xa(k), cost_alert_c(k)*1.25, sprintf('$%.3f',cost_alert_c(k)), 'HorizontalAlignment','center','FontSize',12,'FontWeight','bold','Color',cGreen*0.7);
end

% mini alert ND3B → ND3C annotation
text(ax3, xb(3), cost_alert_b(3)*0.50, {'$1.70 (hover loop)'; '↓ −99%'; '$0.016 (M1–M6 fix)'}, ...
     'HorizontalAlignment','center','FontSize',12,'Color',cGreen,'FontWeight','bold');

set(ax3,'YScale','log','XTick',x,'XTickLabel',model_lbl,'Box','off', ...
        'YGrid','on','GridAlpha',.25,'GridLineStyle','--','XGrid','off','FontSize',19);
ylabel(ax3,'Cost per run  (USD, log scale)','FontSize',22);
title(ax3,{'Cost per Run'; 'Blue: clear\_room  |  Red: alert before fix  |  Green: alert after M1–M6'}, ...
     'FontSize',20,'FontWeight','bold');
legend(ax3,'FontSize',15,'Location','northwest');
hold(ax3,'off');

%% ══════════════════════════════════════════════════════════════════════════════
%% Fig 4 — Quality vs Cost scatter  (final / ND3C state)
%% ══════════════════════════════════════════════════════════════════════════════
% Visual encoding:
%   Shape: circle (o) = clear_room | triangle (^) = alert_room (ND3C final)
%   Color: per model (matches Figs 1–3)
%   Star:  GPT-4o-mini alert ND3C = best quality/cost point
%
% ND3B mini-alert (quality=0, $1.70) is not plotted here — shown in Figs 1–3.

figure('Name','ND3 — Quality vs Cost','Position',pos{4});
ax4 = axes; hold(ax4,'on');

for k = 1:n
    % dotted line connecting clear → alert for same model
    plot(ax4, [cost_clear(k), cost_alert_c(k)], [quality_clear(k), quality_alert_c(k)], ...
         ':','Color',mcol{k},'LineWidth',1.8,'HandleVisibility','off');
    % clear_room: filled circle
    scatter(ax4, cost_clear(k), quality_clear(k), 280, mcol{k}, 'o', 'filled', ...
            'MarkerEdgeColor','k','LineWidth',1.2,'HandleVisibility','off');
    % alert_room ND3C: filled triangle
    scatter(ax4, cost_alert_c(k), quality_alert_c(k), 220, mcol{k}, '^', 'filled', ...
            'MarkerEdgeColor','k','LineWidth',1.2,'HandleVisibility','off');
end

% Star overlay on GPT-4o-mini alert ND3C (best value)
scatter(ax4, cost_alert_c(3), quality_alert_c(3), 320, 'y', 'p', 'filled', ...
        'MarkerEdgeColor',mcol{3},'LineWidth',2.2,'HandleVisibility','off');

% ── legend: shape encoding + model colors ────────────────────────────────────
% Shape entries (gray = encoding only, not model)
scatter(ax4, NaN, NaN, 280, [0.45 0.45 0.45], 'o', 'filled', ...
        'MarkerEdgeColor','k','DisplayName','clear\_room  (circle)');
scatter(ax4, NaN, NaN, 220, [0.45 0.45 0.45], '^', 'filled', ...
        'MarkerEdgeColor','k','DisplayName','alert\_room  (triangle)');
plot(ax4, NaN, NaN, ':', 'Color',[0.45 0.45 0.45], 'LineWidth',1.8, ...
     'DisplayName','clear \rightarrow alert  (same model)');
scatter(ax4, NaN, NaN, 320, 'y', 'p', 'filled', ...
        'MarkerEdgeColor',[0.45 0.45 0.45],'DisplayName','best quality/cost  (\bigstar)');
% Model colour entries
for k = 1:n
    scatter(ax4, NaN, NaN, 200, mcol{k}, 's', 'filled', ...
            'MarkerEdgeColor','none','DisplayName',model_lbl{k});
end

% Model labels (next to clear_room circle)
lbl_off = {[1.12, 0.04]; [1.12, 0.04]; [0.82, 0.04]; [0.82, 0.04]};
halign  = {'left','left','right','right'};
for k = 1:n
    text(ax4, cost_clear(k)*lbl_off{k}(1), quality_clear(k)+lbl_off{k}(2), ...
         model_lbl{k}, 'FontSize',14,'Color',mcol{k},'FontWeight','bold', ...
         'HorizontalAlignment',halign{k});
end

% ND3C callout
text(ax4, cost_alert_c(3)*1.30, quality_alert_c(3)-0.14, ...
     {'← M1–M6 fix'; 'before fix: quality 0, $1.70'}, ...
     'FontSize',13,'Color',mcol{3},'FontWeight','bold');

set(ax4,'XScale','log','Box','off','XGrid','on','YGrid','on', ...
        'GridAlpha',.25,'GridLineStyle','--','FontSize',20);
xlabel(ax4,'Cost per run  (USD, log scale)','FontSize',22);
ylabel(ax4,'Report quality  (0–5)','FontSize',22);
title(ax4,{'Quality vs Cost  (after M1–M6 fix)'; ...
           'Circle = clear\_room  |  Triangle = alert\_room  |  Colour = model'}, ...
     'FontSize',20,'FontWeight','bold');
legend(ax4,'FontSize',14,'Location','southwest','NumColumns',1);
ylim(ax4,[2.3 5.4]);
xlim(ax4,[0.010 3.5]);
hold(ax4,'off');
