%% plot_ND2_fix_comparison.m  →  fig4_5 (4 individual figures)
% ND2 — M1-M6 mitigation effect on GPT-4o-mini alert_room.
%
% Fig 1: Patrol completion % before vs after
% Fig 2: Report quality before vs after
% Fig 3: Cost per run before vs after  (with M1-M6 list)
% Fig 4: Per-turn context window — LITM 50-turn explosion vs M1-M6 8-turn completion
%
% Data hardcoded from:
%   ND2_summary_20260528_224548.csv     (before, N=5 runs)
%   ND2_fix_litm_gpt4omini_alert_*csv   (after M1-M6, N=5 runs)

clear; clc; close all;
set(groot,'DefaultAxesFontSize',20);

cBefore = [0.85 0.20 0.15];
cAfter  = [0.15 0.65 0.30];
bw = 0.40;

%% ── per-turn context tokens ──────────────────────────────────────────────────
turns_b = 1:50;
tok_b = [ ...
     7204,  10426,  13509,  16519,  19547,  22630,  25640,  28668,  31751,  34761, ...
    37789,  40872,  43882,  46910,  49993,  53003,  56031,  59114,  62124,  65152, ...
    68235,  71245,  74273,  77356,  80366,  83394,  86477,  89487,  92515,  95598, ...
    98608, 101636, 104719, 107729, 110757, 113840, 116850, 119878, 122961, 125971, ...
   128999, 132082, 135092, 138120, 141203, 144213, 147241, 150324, 153334, 156362 ];

turns_a = 1:8;
tok_a   = [7265, 10549, 10769, 11032, 11262, 11527, 14791, 15033];

%% ══════════════════════════════════════════════════════════════════════════════
%% Fig 1 — Patrol completion %
%% ══════════════════════════════════════════════════════════════════════════════
figure('Name','ND2 Fix — Patrol Completion','Position',[50 600 700 560]);
ax1 = axes;
hold(ax1,'on');

bar(ax1, 1,   0, bw, 'FaceColor',cBefore,'EdgeColor','none','DisplayName','Before (no mitigations)');
bar(ax1, 2, 100, bw, 'FaceColor',cAfter,  'EdgeColor','none','DisplayName','After (M1–M6)');
text(ax1, 1,   4,  '0%',   'HorizontalAlignment','center','FontSize',22,'FontWeight','bold','Color',cBefore);
text(ax1, 2, 104, '100%', 'HorizontalAlignment','center','FontSize',22,'FontWeight','bold','Color',cAfter);
text(ax1, 1.5, 117, '+100 percentage points', ...
     'HorizontalAlignment','center','FontSize',18,'Color',cAfter,'FontWeight','bold');

set(ax1,'XTick',[1 2],'XTickLabel',{'Before','After M1–M6'},'Box','off', ...
        'YGrid','on','GridAlpha',.25,'GridLineStyle','--','XGrid','off','FontSize',20);
ylabel(ax1,'Patrol completion  (%)','FontSize',22);
title(ax1,{'Patrol Completion'; 'GPT-4o-mini alert\_room before vs after M1–M6'}, ...
     'FontSize',21,'FontWeight','bold');
legend(ax1,'FontSize',18,'Location','northeast');
ylim(ax1,[0 135]);
hold(ax1,'off');

%% ══════════════════════════════════════════════════════════════════════════════
%% Fig 2 — Report quality
%% ══════════════════════════════════════════════════════════════════════════════
figure('Name','ND2 Fix — Report Quality','Position',[780 600 700 560]);
ax2 = axes;
hold(ax2,'on');

bar(ax2, 1, 0.0, bw, 'FaceColor',cBefore,'EdgeColor','none','DisplayName','Before (no mitigations)');
bar(ax2, 2, 4.0, bw, 'FaceColor',cAfter,  'EdgeColor','none','DisplayName','After (M1–M6)');
text(ax2, 1, 0.12, '0.0', 'HorizontalAlignment','center','FontSize',22,'FontWeight','bold','Color',cBefore);
text(ax2, 2, 4.12, '4.0', 'HorizontalAlignment','center','FontSize',22,'FontWeight','bold','Color',cAfter);
text(ax2, 1.5, 5.1, 'zero output  →  quality 4/5', ...
     'HorizontalAlignment','center','FontSize',18,'Color',cAfter,'FontWeight','bold');

set(ax2,'XTick',[1 2],'XTickLabel',{'Before','After M1–M6'},'Box','off', ...
        'YGrid','on','GridAlpha',.25,'GridLineStyle','--','XGrid','off','FontSize',20);
ylabel(ax2,'Report quality  (0–5)','FontSize',22);
title(ax2,{'Report Quality'; 'GPT-4o-mini alert\_room before vs after M1–M6'}, ...
     'FontSize',21,'FontWeight','bold');
legend(ax2,'FontSize',18,'Location','northeast');
ylim(ax2,[0 5.8]);
hold(ax2,'off');

%% ══════════════════════════════════════════════════════════════════════════════
%% Fig 3 — Cost per run  +  M1-M6 list
%% ══════════════════════════════════════════════════════════════════════════════
figure('Name','ND2 Fix — Cost per Run','Position',[50 30 700 560]);
ax3 = axes;
hold(ax3,'on');

bar(ax3, 1, 0.615, bw, 'FaceColor',cBefore,'EdgeColor','none','DisplayName','Before (no mitigations)');
bar(ax3, 2, 0.027, bw, 'FaceColor',cAfter,  'EdgeColor','none','DisplayName','After (M1–M6)');
text(ax3, 1, 0.615+0.020, '$0.615', 'HorizontalAlignment','center','FontSize',21,'FontWeight','bold','Color',cBefore);
text(ax3, 2, 0.027+0.020, '$0.027', 'HorizontalAlignment','center','FontSize',21,'FontWeight','bold','Color',cAfter);
text(ax3, 1.5, 0.72, '−96% cost', ...
     'HorizontalAlignment','center','FontSize',18,'Color',cAfter,'FontWeight','bold');

% M1-M6 summary inside the plot
text(ax3, 2.28, 0.02, ...
     {'M1: advisory rate-limit (every N turns)'; ...
      'M2: structured memory injection'; ...
      'M3: final\_text channel fix'; ...
      'M4: loop-detection nudge'; ...
      'M5: turn-budget warning'; ...
      'M6: explicit quit condition'}, ...
     'FontSize',12,'Color',[0.20 0.20 0.20],'FontWeight','bold', ...
     'VerticalAlignment','bottom','HorizontalAlignment','left');

set(ax3,'XTick',[1 2],'XTickLabel',{'Before','After M1–M6'},'Box','off', ...
        'YGrid','on','GridAlpha',.25,'GridLineStyle','--','XGrid','off','FontSize',20);
ylabel(ax3,'Cost per run  (USD)','FontSize',22);
title(ax3,{'Cost per Run'; 'GPT-4o-mini alert\_room before vs after M1–M6'}, ...
     'FontSize',21,'FontWeight','bold');
legend(ax3,'FontSize',18,'Location','northwest');
ylim(ax3,[0 0.82]);
xlim(ax3,[0.5 3.8]);
hold(ax3,'off');

%% ══════════════════════════════════════════════════════════════════════════════
%% Fig 4 — Per-turn context window: LITM vs M1-M6
%% ══════════════════════════════════════════════════════════════════════════════
figure('Name','ND2 Fix — Context Window','Position',[780 30 900 560]);
ax4 = axes;
hold(ax4,'on');

plot(ax4, turns_b, tok_b/1e3, '-', 'Color',cBefore,'LineWidth',2.8, ...
     'DisplayName',sprintf('Before: LITM, %d turns, %.0fk tokens', numel(turns_b), tok_b(end)/1e3));
plot(ax4, turns_a, tok_a/1e3, '-o','Color',cAfter,'LineWidth',2.8, ...
     'MarkerSize',9,'MarkerFaceColor',cAfter, ...
     'DisplayName',sprintf('After M1–M6: %d turns, %.0fk tokens', numel(turns_a), tok_a(end)/1e3));

% endpoint labels
text(ax4, 49, tok_b(end)/1e3 + 5, sprintf('%.0fk', tok_b(end)/1e3), ...
     'HorizontalAlignment','right','FontSize',16,'Color',cBefore,'FontWeight','bold');
text(ax4, numel(turns_a)+0.5, tok_a(end)/1e3 + 5, sprintf('%.0fk', tok_a(end)/1e3), ...
     'FontSize',16,'Color',cAfter,'FontWeight','bold');

% vertical dashed line at turn 8
plot(ax4, [numel(turns_a) numel(turns_a)],[0 tok_a(end)/1e3+3],':', ...
     'Color',cAfter,'LineWidth',1.8,'HandleVisibility','off');
text(ax4, numel(turns_a)+0.6, 8, {'mission'; 'complete'}, ...
     'FontSize',14,'Color',cAfter,'FontWeight','bold','VerticalAlignment','bottom');

xlabel(ax4,'Turn number','FontSize',22);
ylabel(ax4,'Input tokens  (k)','FontSize',22);
title(ax4,{'Per-Turn Context Window'; 'LITM explosion vs M1–M6 early completion'}, ...
     'FontSize',21,'FontWeight','bold');
legend(ax4,'FontSize',17,'Location','northwest');
set(ax4,'Box','off','XGrid','on','YGrid','on','GridAlpha',.25,'GridLineStyle','--','FontSize',20);
xlim(ax4,[0 52]);
ylim(ax4,[0 175]);
hold(ax4,'off');
