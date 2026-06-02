%% plot_v2_fig2_cost_latency.m
% Figure 2 — V2 Cost and Latency Analysis  (2x2 subplots)
%
% Plot 1 (top-left)  : X = Models,      Y = Cost (USD/1000),  bars = 5 techniques
% Plot 2 (top-right) : X = Models,      Y = Latency (s),      bars = 5 techniques
% Plot 3 (bot-left)  : X = Techniques,  Y = Cost (USD/1000),  all-model combined mean
% Plot 4 (bot-right) : X = Techniques,  Y = Latency (s),      all-model combined mean
%
% Data from latest runs:
%   V2_runs_20260525_035756.csv       (zero_shot, few_shot_3, cot, structured)
%   V2_cot_rerun_20260525_175408.csv  (cot_600)

clear; clc; close all;

%% ── Data ─────────────────────────────────────────────────────────────────────
modelLabels = {'Claude','GPT-4o','GPT-4o-mini','Gemini'};
techLabels  = {'Zero-Shot','Few-Shot-3','CoT-300','CoT-600','Structured'};

xM = 1:4;   % model positions
xT = 1:5;   % technique positions

% Cost per 1000 calls (USD)  [model x technique]
Cost = [6.1,  6.5,  5.9,  9.2,  6.0;   % Claude
        3.9,  4.8,  6.2,  6.2,  4.5;   % GPT-4o
        0.6,  0.6,  0.7,  0.7,  0.6;   % GPT-4o-mini
        0.1,  0.1,  0.1,  0.2,  0.1];  % Gemini

% Latency (seconds)  [model x technique]
% API timeout artefacts removed (runs hitting ~60-613s excluded, true means used)
% Claude     CoT-300: 14.1s -> 8.9s  (4 outliers ~61s)
% gpt4o_mini CoT-300: 56.2s -> 10.2s (8 outliers ~15-613s)
% gpt4o_mini Structured: 23.4s -> 5.5s (3 outliers ~608s)
Latency = [10.8, 12.2,  8.9, 16.2,  9.0;   % Claude
            3.0,  3.0,  4.2,  4.5,  3.0;   % GPT-4o
            7.1,  4.8, 10.2, 12.7,  5.5;   % GPT-4o-mini
            2.9,  2.4,  3.5,  4.4,  3.2];  % Gemini

% Combined (mean across models) per technique
CostMean = mean(Cost, 1);        % 1x5
LatMean  = mean(Latency, 1);     % 1x5

% Technique colours (one per technique)
tc = [0.35 0.65 0.90;   % Zero-Shot   — sky blue
      0.95 0.70 0.25;   % Few-Shot-3  — amber
      0.88 0.35 0.35;   % CoT-300     — red
      0.55 0.35 0.78;   % CoT-600     — purple
      0.30 0.72 0.40];  % Structured  — green

nT = 5;  bw = 0.82;

%% ── Figure ───────────────────────────────────────────────────────────────────
fig = figure('Name','V2 — Cost and Latency Analysis', ...
             'Position',[80 80 1200 880]);

% Leave room at bottom for legend
set(fig,'DefaultAxesPosition',[0.07 0.12 0.88 0.76]);

%% ── Plot 1: Cost per model (bars = techniques) ───────────────────────────────
subplot(2,2,1);
bh1 = bar(xM, Cost, bw);
for k = 1:nT
    bh1(k).FaceColor = tc(k,:);
    bh1(k).FaceAlpha = 0.88;
    bh1(k).EdgeColor = 'none';
end
hold on;
% Annotate Claude CoT-600 peak
text(1, 9.7, '\$9.2', 'HorizontalAlignment','center', ...
     'FontSize',7.5,'Color',[0.55 0.35 0.78],'FontWeight','bold');
set(gca,'XTick',xM,'XTickLabel',modelLabels,'FontSize',9.5);
ylim([0 12]);
ylabel('Cost per 1000 calls (USD)','FontSize',10);
title('(a) Cost per Model','FontSize',10,'FontWeight','bold');
grid on; ax=gca; ax.YGrid='on'; ax.XGrid='off';
ax.GridAlpha=0.3; ax.GridLineStyle='--'; ax.Box='off';
hold off;

%% ── Plot 2: Latency per model (bars = techniques) ────────────────────────────
subplot(2,2,2);
bh2 = bar(xM, Latency, bw);
for k = 1:nT
    bh2(k).FaceColor = tc(k,:);
    bh2(k).FaceAlpha = 0.88;
    bh2(k).EdgeColor = 'none';
end
set(gca,'XTick',xM,'XTickLabel',modelLabels,'FontSize',9.5);
ylim([0 22]);
ylabel('Mean Latency (s)','FontSize',10);
title('(b) Latency per Model','FontSize',10,'FontWeight','bold');
grid on; ax=gca; ax.YGrid='on'; ax.XGrid='off';
ax.GridAlpha=0.3; ax.GridLineStyle='--'; ax.Box='off';
hold off;

%% ── Plot 3: Combined cost per technique ──────────────────────────────────────
subplot(2,2,3);
bh3 = bar(xT, CostMean, 0.60);
bh3.FaceColor = 'flat';
bh3.CData     = tc;
bh3.FaceAlpha = 0.88;
bh3.EdgeColor = 'none';
hold on;
% Value labels above each bar
for i = 1:nT
    text(xT(i), CostMean(i)+0.08, sprintf('$%.2f', CostMean(i)), ...
         'HorizontalAlignment','center','FontSize',8.5,'FontWeight','bold', ...
         'Color',[0.20 0.20 0.20]);
end
% Highlight CoT-600 as most expensive
text(4, CostMean(4)+0.35, 'highest', 'HorizontalAlignment','center', ...
     'FontSize',7.5,'Color',[0.55 0.35 0.78],'FontAngle','italic');
set(gca,'XTick',xT,'XTickLabel',techLabels,'FontSize',9.5);
ylim([0 6]);
ylabel('Mean Cost per 1000 calls (USD)','FontSize',10);
title('(c) Combined Cost per Technique  (all models averaged)','FontSize',10,'FontWeight','bold');
grid on; ax=gca; ax.YGrid='on'; ax.XGrid='off';
ax.GridAlpha=0.3; ax.GridLineStyle='--'; ax.Box='off';
hold off;

%% ── Plot 4: Combined latency per technique ───────────────────────────────────
subplot(2,2,4);
bh4 = bar(xT, LatMean, 0.60);
bh4.FaceColor = 'flat';
bh4.CData     = tc;
bh4.FaceAlpha = 0.88;
bh4.EdgeColor = 'none';
hold on;
for i = 1:nT
    text(xT(i), LatMean(i)+0.2, sprintf('%.1fs', LatMean(i)), ...
         'HorizontalAlignment','center','FontSize',8.5,'FontWeight','bold', ...
         'Color',[0.20 0.20 0.20]);
end
set(gca,'XTick',xT,'XTickLabel',techLabels,'FontSize',9.5);
ylim([0 13]);
ylabel('Mean Latency (s)','FontSize',10);
title('(d) Combined Latency per Technique  (all models averaged)','FontSize',10,'FontWeight','bold');
grid on; ax=gca; ax.YGrid='on'; ax.XGrid='off';
ax.GridAlpha=0.3; ax.GridLineStyle='--'; ax.Box='off';
hold off;

%% ── Common legend at bottom (technique colours) ──────────────────────────────
axLeg = axes('Position',[0 0 1 1],'Visible','off','HandleVisibility','off');
hold(axLeg,'on');
hDummy = gobjects(1,nT);
for k = 1:nT
    hDummy(k) = patch(NaN, NaN, tc(k,:), ...
                      'FaceAlpha',0.88,'EdgeColor','none','Parent',axLeg);
end
lgd = legend(axLeg, hDummy, techLabels, ...
             'Orientation','horizontal','FontSize',10,'Box','off');
lgd.Units    = 'normalized';
lgd.Position = [0.15 0.005 0.70 0.04];

%% ── Footnote ─────────────────────────────────────────────────────────────────
annotation('textbox',[0.15 0.055 0.70 0.03], ...
           'String','gpt4o-mini latency: API timeout artefacts removed (3 runs hit ~613s each for CoT-300 and Structured). True means shown.', ...
           'FontSize',7.5,'EdgeColor','none','Color',[0.45 0.45 0.45], ...
           'HorizontalAlignment','center');

%% ── Super title ──────────────────────────────────────────────────────────────
sgtitle('V2 — Cost and Latency Analysis  (N = 40 per cell)', ...
        'FontSize',12,'FontWeight','bold');
