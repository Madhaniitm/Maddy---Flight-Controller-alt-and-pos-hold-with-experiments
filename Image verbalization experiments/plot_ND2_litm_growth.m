%% plot_ND2_litm_growth.m  →  fig4_4
% ND2 — LITM mechanism visualised for GPT-4o-mini alert_room run 1 (50 turns).
%
% LEFT PANEL : Cumulative context window growth (k tokens vs turn).
%   Shows: mission objective starts at 7k tokens; by T50 it is buried under 156k tokens.
%   "Lost in the Middle" = the model cannot attend to instructions 149k tokens ago.
%
% RIGHT PANEL: Per-turn token delta (tokens added each turn).
%   Each turn adds a nearly CONSTANT ≈ 3k tokens (one analyze_scene tool loop).
%   This proves the mechanical loop: same call every turn, no mission progress.
%
% Data: ND2_api_stats_20260528_224548.csv (run 1, gpt4o_mini, alert_room)

clear; clc; close all;
set(groot,'DefaultAxesFontSize',20);

%% ── per-turn cumulative input tokens (50 turns) ─────────────────────────────
tok = [ ...
     7204,  10426,  13509,  16519,  19547,  22630,  25640,  28668,  31751,  34761, ...
    37789,  40872,  43882,  46910,  49993,  53003,  56031,  59114,  62124,  65152, ...
    68235,  71245,  74273,  77356,  80366,  83394,  86477,  89487,  92515,  95598, ...
    98608, 101636, 104719, 107729, 110757, 113840, 116850, 119878, 122961, 125971, ...
   128999, 132082, 135092, 138120, 141203, 144213, 147241, 150324, 153334, 156362 ];

deltas = diff(tok);   % 49 values: tokens added each turn (T2..T50)
turns  = 1:50;
cRed   = [0.85 0.18 0.18];

%% ── Figure ───────────────────────────────────────────────────────────────────
figure('Name','ND2 — LITM Growth','Position',[60 60 1600 740]);

%% ─── LEFT: cumulative token growth ──────────────────────────────────────────
ax1 = subplot(1,2,1);
hold(ax1,'on');

plot(ax1, turns, tok/1e3, '-', 'Color',cRed,'LineWidth',2.8, ...
     'DisplayName','Input tokens (k)');

% linear fit overlay
p = polyfit(turns, tok/1e3, 1);
plot(ax1, turns, polyval(p,turns), '--','Color',[0.50 0.50 0.50],'LineWidth',1.5, ...
     'DisplayName',sprintf('Linear fit  (+%.0fk/turn)', p(1)));

% T1 annotation
text(ax1, 2, tok(1)/1e3 + 9, ...
     {sprintf('T1: %.0fk tokens', tok(1)/1e3); 'mission + advisory here'}, ...
     'FontSize',14,'Color',cRed*0.8,'FontWeight','bold','VerticalAlignment','bottom');

% T50 annotation showing burial percentage
burial_pct = tok(1)/tok(end)*100;
text(ax1, 28, tok(end)/1e3 - 18, ...
     {sprintf('T50: %.0fk tokens', tok(end)/1e3); ...
      sprintf('T1 advisory = %.1f%% of context', burial_pct); ...
      '"Lost in the Middle"'}, ...
     'FontSize',14,'Color',[0.50 0.06 0.06],'FontWeight','bold');

% vertical bracket at turn 1 showing advisory depth visually
plot(ax1,[2 2],[tok(1)/1e3 tok(end)/1e3],'Color',[0.65 0.10 0.10], ...
     'LineWidth',1.5,'LineStyle',':','HandleVisibility','off');
text(ax1, 3, (tok(1)/1e3 + tok(end)/1e3)/2, ...
     {'buried', 'under', sprintf('%.0fk', (tok(end)-tok(1))/1e3)}, ...
     'FontSize',13,'Color',[0.55 0.08 0.08],'FontWeight','bold');

xlabel(ax1,'Turn number','FontSize',22);
ylabel(ax1,'Context window  (k tokens)','FontSize',22);
title(ax1,{'Context Window Growth'; 'GPT-4o-mini alert\_room — LITM run'}, ...
     'FontSize',20,'FontWeight','bold');
legend(ax1,'FontSize',17,'Location','northwest');
ylim(ax1,[0 175]);
xlim(ax1,[1 50]);
set(ax1,'Box','off','XGrid','on','YGrid','on','GridAlpha',.2,'GridLineStyle','--');
hold(ax1,'off');

%% ─── RIGHT: per-turn delta — flat line proves mechanical loop ───────────────
ax2 = subplot(1,2,2);
hold(ax2,'on');

bar(ax2, 2:50, deltas/1e3, 0.85, 'FaceColor',cRed,'EdgeColor','none', ...
    'DisplayName','Tokens added per turn');

% mean reference line
mean_delta = mean(deltas)/1e3;
plot(ax2, [1 50], [mean_delta mean_delta], '--k','LineWidth',2.0, ...
     'DisplayName',sprintf('Mean: %.0fk tokens/turn', mean_delta));

% annotation explaining what it means
text(ax2, 26, mean_delta + 0.22, ...
     {sprintf('Mean = %.0fk tokens/turn = one analyze\\_scene loop', mean(deltas)); ...
      sprintf('std = %.0f tokens — nearly zero variance', std(deltas)); ...
      'Model repeats same tool call every turn, no mission progress'}, ...
     'HorizontalAlignment','center','FontSize',14,'Color',[0.40 0.05 0.05], ...
     'FontWeight','bold','VerticalAlignment','bottom');

xlabel(ax2,'Turn number','FontSize',22);
ylabel(ax2,'New tokens added per turn  (k)','FontSize',22);
title(ax2,{'Per-Turn Token Delta'; 'Constant ≈ 3k → mechanical repetition proved'}, ...
     'FontSize',20,'FontWeight','bold');
legend(ax2,'FontSize',17,'Location','northeast');
ylim(ax2,[0 4.0]);
xlim(ax2,[1 50]);
set(ax2,'Box','off','XGrid','on','YGrid','on','GridAlpha',.2,'GridLineStyle','--');
hold(ax2,'off');

sgtitle({'ND2 — Lost-in-the-Middle Failure: GPT-4o-mini alert\_room'; ...
         'Left: mission advisory buried as context grows to 156k tokens (T1 advisory = 4.6% of T50 context)'; ...
         'Right: every turn adds ≈3k tokens (looping analyze\_scene) — no mission progress across 50 turns'}, ...
        'FontSize',18,'FontWeight','bold');
