%% plot_ND3C_progression.m  →  fig4_8  (5 individual figures)
% ND3C — Three-stage progression for GPT-4o-mini alert_room HITL.
%
% Stage 1: ND3 original — Azure 500 API outage, runs silently swallowed (Bug B).
%          0–1 turns, ~$0 cost, 0 patrol, 0 quality, 0 hover denials.
% Stage 2: ND3B — Bug B fixed, HITL present, but NO M1-M6 semantic loop fixes.
%          Hover-navigate cycle runs to max_turns=50. 10 hover denials/run.
%          34.2× context explosion (12k → 440k tokens). $1.70/run.
% Stage 3: ND3C — M1-M6 + HITL. Loop never forms. 8–9 turns. 0 denials.
%          1.45× context growth (10k → 15k tokens). $0.016/run. Quality 4.6.
%
% Data: ND3_observations_20260530.md — Findings 10, 11, 12.
% One figure per metric as requested.

clear; clc; close all;
set(groot,'DefaultAxesFontSize',20);

%% ── stage data ───────────────────────────────────────────────────────────────
stage_x   = [1, 2, 3];
stage_lbl = {'Bug B (API outage)', 'HITL only (hover loop)', 'HITL + M1–M6 fix'};

patrol    = [0,    0,     100  ];   % %
quality   = [0,    0,     4.60 ];   % /5
cost      = [0.001, 1.700, 0.016];  % USD / run (stage 1 ≈ $0 — retry overhead only)
turns     = [0.5,  50,    8.6  ];   % turns/run (stage 1: 0–1 turn avg)
denials   = [0,    10,    0    ];   % hover denials / run (ND3-specific HITL metric)
ctx_growth= [NaN,  34.2,  1.45 ];   % context growth ×  (stage 1: no valid run)

cS = {[0.70 0.70 0.70]; [0.85 0.20 0.15]; [0.15 0.65 0.30]};
bw = 0.45;

% screen positions: 3 cols × 2 rows, non-overlapping
pos = {[50  560 680 520]; [760 560 680 520]; [1420 560 680 520]; ...
       [50   30 680 520]; [760  30 680 520]};

%% ══════════════════════════════════════════════════════════════════════════════
%% Fig 1 — Patrol completion %
%% ══════════════════════════════════════════════════════════════════════════════
figure('Name','ND3C Stage — Patrol','Position',pos{1});
ax = axes; hold(ax,'on');
for s = 1:3
    bar(ax, s, patrol(s), bw, 'FaceColor',cS{s},'EdgeColor','none');
    text(ax, s, patrol(s) + 3.5, sprintf('%d%%', patrol(s)), ...
         'HorizontalAlignment','center','FontSize',22,'FontWeight','bold','Color',cS{s}*0.7);
end
text(ax, 3, patrol(3)+11, '+100 pp vs hover loop', ...
     'HorizontalAlignment','center','FontSize',17,'Color',cS{3},'FontWeight','bold');
set(ax,'XTick',1:3,'XTickLabel',stage_lbl,'Box','off', ...
       'YGrid','on','GridAlpha',.25,'GridLineStyle','--','XGrid','off', ...
       'XTickLabelRotation',12,'FontSize',19);
ylabel(ax,'Patrol completion  (%)','FontSize',22);
title(ax,{'Patrol Completion'; 'GPT-4o-mini alert\_room — three-stage progression'}, ...
     'FontSize',21,'FontWeight','bold');
ylim(ax,[0 130]);
hold(ax,'off');

%% ══════════════════════════════════════════════════════════════════════════════
%% Fig 2 — Report quality
%% ══════════════════════════════════════════════════════════════════════════════
figure('Name','ND3C Stage — Quality','Position',pos{2});
ax = axes; hold(ax,'on');
for s = 1:3
    bar(ax, s, quality(s), bw, 'FaceColor',cS{s},'EdgeColor','none');
    text(ax, s, quality(s) + 0.12, sprintf('%.1f', quality(s)), ...
         'HorizontalAlignment','center','FontSize',22,'FontWeight','bold','Color',cS{s}*0.7);
end
text(ax, 3, quality(3)+0.35, '+4.6 pts vs hover loop', ...
     'HorizontalAlignment','center','FontSize',17,'Color',cS{3},'FontWeight','bold');
set(ax,'XTick',1:3,'XTickLabel',stage_lbl,'Box','off', ...
       'YGrid','on','GridAlpha',.25,'GridLineStyle','--','XGrid','off', ...
       'XTickLabelRotation',12,'FontSize',19);
ylabel(ax,'Report quality  (0–5)','FontSize',22);
title(ax,{'Report Quality'; 'GPT-4o-mini alert\_room — three-stage progression'}, ...
     'FontSize',21,'FontWeight','bold');
ylim(ax,[0 5.8]);
hold(ax,'off');

%% ══════════════════════════════════════════════════════════════════════════════
%% Fig 3 — Cost per run
%% ══════════════════════════════════════════════════════════════════════════════
figure('Name','ND3C Stage — Cost','Position',pos{3});
ax = axes; hold(ax,'on');
for s = 1:3
    bar(ax, s, cost(s), bw, 'FaceColor',cS{s},'EdgeColor','none');
    if s == 1
        lbl = '≈ $0';
    else
        lbl = sprintf('$%.3f', cost(s));
    end
    text(ax, s, cost(s) + 0.025, lbl, ...
         'HorizontalAlignment','center','FontSize',21,'FontWeight','bold','Color',cS{s}*0.7);
end
text(ax, 3, cost(3)+0.07, '−99% vs hover loop', ...
     'HorizontalAlignment','center','FontSize',17,'Color',cS{3},'FontWeight','bold');
% explain stage 2 cost
text(ax, 2, cost(2)-0.08, {'34.2× context growth'; '440k tokens/run'}, ...
     'HorizontalAlignment','center','FontSize',14,'Color',[0.9 0.9 0.9],'FontWeight','bold');
set(ax,'XTick',1:3,'XTickLabel',stage_lbl,'Box','off', ...
       'YGrid','on','GridAlpha',.25,'GridLineStyle','--','XGrid','off', ...
       'XTickLabelRotation',12,'FontSize',19);
ylabel(ax,'Cost per run  (USD)','FontSize',22);
title(ax,{'Cost per Run'; 'GPT-4o-mini alert\_room — three-stage progression'}, ...
     'FontSize',21,'FontWeight','bold');
ylim(ax,[0 2.1]);
hold(ax,'off');

%% ══════════════════════════════════════════════════════════════════════════════
%% Fig 4 — Hover denials per run  (ND3-specific HITL metric)
%% ══════════════════════════════════════════════════════════════════════════════
figure('Name','ND3C Stage — Hover Denials','Position',pos{4});
ax = axes; hold(ax,'on');
for s = 1:3
    bar(ax, s, denials(s), bw, 'FaceColor',cS{s},'EdgeColor','none');
    text(ax, s, denials(s) + 0.3, sprintf('%d', denials(s)), ...
         'HorizontalAlignment','center','FontSize',22,'FontWeight','bold','Color',cS{s}*0.7);
end
text(ax, 2, denials(2)+0.9, {'operator denied hover 10×/run'; 'model looped, never wrote report'}, ...
     'HorizontalAlignment','center','FontSize',14,'Color',[0.65 0.10 0.10],'FontWeight','bold');
text(ax, 3, denials(3)+0.9, {'0 denials — loop never formed'; 'model self-terminated (M4+M6)'}, ...
     'HorizontalAlignment','center','FontSize',14,'Color',cS{3},'FontWeight','bold');
set(ax,'XTick',1:3,'XTickLabel',stage_lbl,'Box','off', ...
       'YGrid','on','GridAlpha',.25,'GridLineStyle','--','XGrid','off', ...
       'XTickLabelRotation',12,'FontSize',19);
ylabel(ax,'Hover denials per run  (HITL gate)','FontSize',22);
title(ax,{'Hover Denials per Run'; 'HITL gate denied hover — no fix: ineffective | M1–M6: not needed'}, ...
     'FontSize',21,'FontWeight','bold');
ylim(ax,[0 14]);
hold(ax,'off');

%% ══════════════════════════════════════════════════════════════════════════════
%% Fig 5 — Context window growth ×  (ND3B vs ND3C — stage 1 invalid)
%% ══════════════════════════════════════════════════════════════════════════════
figure('Name','ND3C Stage — Context Growth','Position',pos{5});
ax = axes; hold(ax,'on');
% only stages 2 and 3 have valid context data
bar(ax, 2, ctx_growth(2), bw, 'FaceColor',cS{2},'EdgeColor','none');
bar(ax, 3, ctx_growth(3), bw, 'FaceColor',cS{3},'EdgeColor','none');
text(ax, 1, 1.5, {'N/A'; '(API outage)'}, ...
     'HorizontalAlignment','center','FontSize',18,'Color',[0.5 0.5 0.5],'FontWeight','bold');
text(ax, 2, ctx_growth(2)+0.6, sprintf('%.1f×', ctx_growth(2)), ...
     'HorizontalAlignment','center','FontSize',22,'FontWeight','bold','Color',cS{2}*0.7);
text(ax, 3, ctx_growth(3)+0.6, sprintf('%.2f×', ctx_growth(3)), ...
     'HorizontalAlignment','center','FontSize',22,'FontWeight','bold','Color',cS{3}*0.7);
text(ax, 2, ctx_growth(2)-2, {'12k → 440k tokens'; '(hover-navigate loop)'}, ...
     'HorizontalAlignment','center','FontSize',14,'Color',[0.9 0.9 0.9],'FontWeight','bold');
text(ax, 3, ctx_growth(3)+1.6, {'10k → 15k tokens'; '(M2 compact state only)'}, ...
     'HorizontalAlignment','center','FontSize',14,'Color',cS{3},'FontWeight','bold');
text(ax, 2.5, 30, '−96% context growth', ...
     'HorizontalAlignment','center','FontSize',17,'Color',cS{3},'FontWeight','bold');
set(ax,'XTick',1:3,'XTickLabel',stage_lbl,'Box','off', ...
       'YGrid','on','GridAlpha',.25,'GridLineStyle','--','XGrid','off', ...
       'XTickLabelRotation',12,'FontSize',19);
ylabel(ax,'Context window growth  (× turn-1 baseline)','FontSize',22);
title(ax,{'Context Window Growth'; 'LITM loop: 34.2× | M1–M6 fix: 1.45× (96% reduction)'}, ...
     'FontSize',21,'FontWeight','bold');
ylim(ax,[0 40]);
hold(ax,'off');
