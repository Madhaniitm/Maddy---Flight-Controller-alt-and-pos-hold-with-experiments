%% plot_G1_fig1_fig2.m
% G1 — Two-panel figure (2 × 1)
%
% Top    (a): Action Safety (%) by condition and model — grouped bar
% Bottom (b): Dangerous cases heatmap — condition × scene
%
% Key findings:
%   Tier 2 alone has a fatal blind spot on blocked_lens (87.5% safe)
%   Tier 3 LLM-only: gpt4o_mini also fails on wall_close (87.5%)
%   Full pipeline: Gemini achieves 100%; GPT-4o worst at 95%
%
% Data source: G1_runs_20260525_021111.csv  (latest run)

clear; clc; close all;

%% ── Data ─────────────────────────────────────────────────────────────────────

% (a) Action Safety (%) — rows=models, cols=conditions
%     NaN = model not active in that condition
%                          T1.5-only  T2-only  T3-only  Full
SafetyData = [100.0, 87.5,  NaN,   NaN;    % Rule-Based
              NaN,   NaN,  100.0,  97.4;   % Claude
              NaN,   NaN,  100.0,  95.0;   % GPT-4o
              NaN,   NaN,   87.5,  97.5;   % GPT-4o-mini
              NaN,   NaN,  100.0, 100.0];  % Gemini

condLabels  = {'Tier 1.5 Only','Tier 2 Only','Tier 3 Only (LLM)','Full Pipeline'};
modelNames  = {'Rule-Based','Claude','GPT-4o','GPT-4o-mini','Gemini'};

% Model colours
mc = [0.58 0.65 0.65;   % Rule-Based  — grey
      0.88 0.48 0.33;   % Claude      — orange-red
      0.29 0.56 0.85;   % GPT-4o      — blue
      0.42 0.75 0.42;   % GPT-4o-mini — green
      0.69 0.49 0.85];  % Gemini      — purple

% (b) Dangerous count matrix [condition × scene]  (aggregated across all models)
%     tier1_5_only: all zero
%     tier2_only  : blocked_lens = 5 (sensor blind spot)
%     tier3_only  : wall_close   = 5 (depth sensor / LLM error)
%     full_pipeline: wall_close  = 4
DangerMat = [0, 0, 0, 0, 0, 0, 0, 0;   % Tier 1.5 Only
             0, 0, 5, 0, 0, 0, 0, 0;   % Tier 2 Only
             0, 5, 0, 0, 0, 0, 0, 0;   % Tier 3 Only
             0, 4, 0, 0, 0, 0, 0, 0];  % Full Pipeline

sceneLabels = {'person near','wall close','blocked lens','person far', ...
               'dim light','cluttered','object table','door open'};
condLabelsShort = {'Tier 1.5\nOnly','Tier 2\nOnly','Tier 3\nOnly','Full\nPipeline'};

%% ── Layout constants  [left  bottom  width  height] ─────────────────────────
% Both plot areas identical width & height; colorbar placed manually.
LFT  = 0.13;   % wider left margin — fits 'Tier 1.5 Only' y-axis label
WID  = 0.73;   % plot width  (colorbar in remaining ~0.14)
PHGT = 0.32;   % height of each subplot
BOT2 = 0.14;   % bottom of heatmap — room for 30° rotated x-labels
GAP  = 0.18;   % gap between plots — (b) title + whitespace + legend + whitespace
BOT1 = BOT2 + PHGT + GAP;   % bottom of bar chart = 0.64

%% ── Figure ───────────────────────────────────────────────────────────────────
fig = figure('Name','G1 — Tier Comparison', ...
             'Position',[80 60 1150 920]);

%% ══════════════════════════════════════════════════════════════════════════════
%% Subplot (a) — Action Safety grouped bar
%% ══════════════════════════════════════════════════════════════════════════════
ax1 = axes('Position',[LFT BOT1 WID PHGT]);

% bar() with NaN produces no bar for that group entry
bh = bar(1:4, SafetyData', 0.75);   % transpose: rows=conditions, cols=models
for k = 1:5
    bh(k).FaceColor = mc(k,:);
    bh(k).FaceAlpha = 0.87;
    bh(k).EdgeColor = 'none';
end
hold on;

% 100% reference line
yline(100,'--','Color',[0.55 0.55 0.55],'LineWidth',1.0,'HandleVisibility','off');

% Annotate non-100 values
% Need bar x positions — use XOffset from bar handles
nBars  = 5;
bWidth = 0.75;
offs   = linspace(-(nBars-1)/2, (nBars-1)/2, nBars) * (bWidth/nBars);

annots = {
    % {cond_idx, model_idx, value, label}
    2, 1, 87.5, '87.5%';   % Tier2 / Rule-Based
    3, 4, 87.5, '87.5%';   % Tier3 / GPT-4o-mini
    4, 2, 97.4, '97.4%';   % Full  / Claude
    4, 3, 95.0, '95.0%';   % Full  / GPT-4o
    4, 4, 97.5, '97.5%';   % Full  / GPT-4o-mini
};
for i = 1:size(annots,1)
    ci  = annots{i,1};
    mi  = annots{i,2};
    val = annots{i,3};
    lbl = annots{i,4};
    text(ci + offs(mi), val + 0.4, lbl, ...
         'HorizontalAlignment','center','FontSize',8.5, ...
         'Color',[0.75 0.10 0.10],'FontWeight','bold');
end

% Callout: Tier 2 blind spot
text(2 + offs(1), 85.5, {'{\leftarrow} Blocked lens'; 'blind spot'}, ...
     'FontSize',8.5,'Color',[0.75 0.10 0.10], ...
     'HorizontalAlignment','left');

set(ax1,'XTick',1:4,'XTickLabel',condLabels,'FontSize',10);
ylim([70 108]);
ax1.YTick = 70:5:105;
ax1.YTickLabel = {'70%','75%','80%','85%','90%','95%','100%','105%'};
ylabel('Action Safety (%)','FontSize',11);
% Title via annotation so it is pinned to the same centre-x as plot (b)
annotation('textbox', [LFT, BOT1+PHGT+0.005, WID, 0.055], ...
    'String', {'\bf(a)  Action Safety by System Condition'; ...
               'Tier 2 alone has a fatal blind spot (blocked lens) — LLMs eliminate it'}, ...
    'FontSize',11, 'EdgeColor','none', ...
    'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
    'FitBoxToText','off');
ax1.YGrid='on'; ax1.XGrid='off';
ax1.GridAlpha=0.28; ax1.GridLineStyle='--'; ax1.Box='off';

% Legend — sits in the gap between the two plots
axLeg = axes('Position',[0 0 1 1],'Visible','off','HandleVisibility','off');
hold(axLeg,'on');
hD = gobjects(1,5);
for k = 1:5
    hD(k) = patch(NaN,NaN,mc(k,:),'FaceAlpha',0.87,'EdgeColor','none','Parent',axLeg);
end
lgd = legend(axLeg, hD, modelNames, ...
             'Orientation','horizontal','FontSize',9.5,'Box','off');
lgd.Units    = 'normalized';
lgd.Position = [0.15  BOT2+PHGT+GAP*0.62  0.68  0.04];  % upper 2/3 of gap
hold off;

%% ══════════════════════════════════════════════════════════════════════════════
%% Subplot (b) — Dangerous cases heatmap  (pcolor for clean cell edges)
%% ══════════════════════════════════════════════════════════════════════════════
ax2 = axes('Position',[LFT BOT2 WID PHGT]);

% pcolor needs (nRow+1) × (nCol+1) matrix — pad with zeros
[nR, nC] = size(DangerMat);
P = [DangerMat, zeros(nR,1); zeros(1,nC+1)];

h = pcolor(ax2, P);
h.EdgeColor = [1 1 1];   % white grid lines between cells
h.LineWidth = 1.8;

% Colormap: light grey (0=safe) → yellow → orange → red (5=most dangerous)
cmap6 = [0.94 0.97 0.94;   % 0  — near-white green  (safe)
         0.99 0.98 0.82;   % 1  — pale yellow
         0.99 0.85 0.46;   % 2  — amber
         0.98 0.60 0.22;   % 3  — orange
         0.90 0.25 0.12;   % 4  — red-orange
         0.65 0.05 0.05];  % 5  — deep red
nStep = 64;
cmapFull = [];
for s = 1:5
    cmapFull = [cmapFull; ...
                interp1([0 1], [cmap6(s,:); cmap6(s+1,:)], linspace(0,1,nStep))];
end
colormap(ax2, cmapFull);
clim([0 5]);

% Colorbar — placed manually so it does NOT resize ax2
cb = colorbar(ax2,'Location','manual');
cb.Position      = [LFT+WID+0.015  BOT2  0.025  PHGT];
cb.Label.String  = 'Dangerous cases (count)';
cb.Label.FontSize = 9;
cb.Ticks         = 0:5;
cb.TickLabels    = {'0 (safe)','1','2','3','4','5'};
% Restore ax2 to exact size (colorbar creation can still nudge it)
ax2.Position     = [LFT BOT2 WID PHGT];

hold(ax2,'on');

% Cell text: number for all cells; red bold for non-zero, dark green for zero
for i = 1:nR
    for j = 1:nC
        val = DangerMat(i,j);
        if val == 0
            clr = [0.15 0.50 0.20];   % dark green
            fsz = 13;
            lbl = char(10003);        % Unicode ✓ — renders without LaTeX interpreter
        else
            clr = [1 1 1];            % white
            fsz = 14;
            lbl = num2str(val);
        end
        text(ax2, j + 0.5, i + 0.5, lbl, ...
             'HorizontalAlignment','center','VerticalAlignment','middle', ...
             'FontSize',fsz,'Color',clr,'FontWeight','bold');
    end
end

% ── Axes formatting ───────────────────────────────────────────────────────────
set(ax2, ...
    'XTick',                1.5:8.5, ...
    'XTickLabel',           sceneLabels, ...
    'YTick',                1.5:4.5, ...
    'YTickLabel',           {'Tier 1.5 Only','Tier 2 Only','Tier 3 Only','Full Pipeline'}, ...
    'YDir',                 'reverse', ...    % row 1 at top, row 4 at bottom
    'FontSize',             10, ...
    'TickLength',           [0 0], ...
    'XLim',                 [0.5 nC+0.5], ...
    'YLim',                 [0.5 nR+0.5], ...
    'DataAspectRatioMode',  'auto', ...
    'PlotBoxAspectRatioMode','auto', ...
    'Layer',                'top');   % draw axis lines on top of pcolor patches
xtickangle(ax2, 30);

annotation('textbox', [LFT, BOT2+PHGT+0.005, WID, 0.055], ...
    'String', {'\bf(b)  Dangerous Recommendations: Condition \times Scene'; ...
               '(truth = hazard,  action = PITCH\_FORWARD — aggregated across all models)'}, ...
    'FontSize',11, 'EdgeColor','none', ...
    'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
    'FitBoxToText','off', 'Interpreter','tex');
ax2.Box = 'off';
hold(ax2,'off');
