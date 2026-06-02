%% plot_v2_fig1_models.m
% Figure 1 — V2 Prompt Technique Comparison
% 2x2 subplots, one per model. Grouped bar chart.
% X-axis: prompt technique  |  Y-axis: score (%)
% Bars: LblAcc, DescAcc, RsnRisk, RsnAct, ActSafe
% Common legend placed outside subplots at bottom.

clear; clc; close all;

%% ── Data ─────────────────────────────────────────────────────────────────────
techLabels  = {'Zero-Shot','Few-Shot-3','CoT-300','CoT-600','Structured'};
modelLabels = {'Claude','GPT-4o','GPT-4o-mini','Gemini'};
x = 1:5;

% [model x technique]
LblAcc = [40.0,  35.0,   5.0,  57.5,  42.5;
          50.0,  67.5,  52.5,  60.0,  70.0;
          52.5,  47.5,  17.5,  30.0,  62.5;
          25.0,  55.0,   0.0,  37.5,  50.0];

DescAcc = [97.5,  95.0,  90.0,  95.0,  97.5;
          100.0, 100.0, 100.0, 100.0, 100.0;
          100.0, 100.0, 100.0, 100.0, 100.0;
          100.0, 100.0, 100.0, 100.0, 100.0];

RsnRisk = [97.5,  95.0,  90.0,  95.0,  97.5;
           92.5,  92.5, 100.0, 100.0, 100.0;
          100.0,  90.0, 100.0, 100.0, 100.0;
           92.5, 100.0, 100.0, 100.0, 100.0];

RsnAct = [20.0,  30.0,   0.0,  95.0,  62.5;
         100.0, 100.0, 100.0, 100.0, 100.0;
         100.0, 100.0,  30.0, 100.0, 100.0;
          87.5, 100.0,   0.0,  65.0,  95.0];

ActSafe = [100.0, 100.0, 100.0, 100.0, 100.0;
           100.0, 100.0, 100.0, 100.0, 100.0;
           100.0,  87.5, 100.0,  95.0, 100.0;
           100.0, 100.0, 100.0, 100.0, 100.0];

% Stack all metrics: rows=techniques, cols=metrics  (per model)
metricColors = [0.20 0.47 0.75;   % LblAcc  — blue
                0.93 0.65 0.20;   % DescAcc — amber
                0.33 0.68 0.34;   % RsnRisk — green
                0.83 0.24 0.24;   % RsnAct  — red
                0.58 0.38 0.75];  % ActSafe — purple
metricNames  = {'LblAcc','DescAcc','RsnRisk','RsnAct','ActSafe'};
nM = 5;   % number of metrics

%% ── Figure ───────────────────────────────────────────────────────────────────
fig = figure('Name','V2 — Per-Model Metric Profiles', ...
             'Position',[80 80 1150 880]);

% Leave room at bottom for the legend
set(fig,'DefaultAxesPosition',[0.08 0.12 0.86 0.78]);  % [left bottom width height]

bhStore = gobjects(1, nM);   % store one bar handle per metric for legend

for m = 1:4
    % Build data matrix: [nTech x nMetrics]
    D = [LblAcc(m,:); DescAcc(m,:); RsnRisk(m,:); RsnAct(m,:); ActSafe(m,:)]';

    ax = subplot(2, 2, m);
    bh = bar(x, D, 0.80);

    % Apply metric colours
    for k = 1:nM
        bh(k).FaceColor = metricColors(k,:);
        bh(k).FaceAlpha = 0.88;
        bh(k).EdgeColor = 'none';
        if m == 1
            bhStore(k) = bh(k);   % save handles for legend (first subplot only)
        end
    end

    hold on;

    % Shade CoT-300 only for models that were actually truncated
    % GPT-4o (m=2) was NOT truncated (avg 220 tokens, never hit 300 limit)
    if m ~= 2
        patch([2.5 3.5 3.5 2.5],[0 0 108 108],[1 0.82 0.82], ...
              'FaceAlpha',0.20,'EdgeColor','none','HandleVisibility','off');
        text(3, 4, 'trunc.', 'HorizontalAlignment','center', ...
             'FontSize',7,'Color',[0.65 0.10 0.10],'FontAngle','italic');
    end

    set(ax,'XTick',x,'XTickLabel',techLabels,'FontSize',9);
    ylim([0 112]);
    ylabel('Score (%)','FontSize',10);
    title(modelLabels{m},'FontSize',11,'FontWeight','bold');
    grid on;
    ax.YGrid = 'on'; ax.XGrid = 'off';
    ax.GridAlpha = 0.3; ax.GridLineStyle = '--';
    ax.Box = 'off';
    hold off;
end

%% ── Common legend outside subplots at bottom ─────────────────────────────────
% Use an invisible axes spanning the full figure so legend is figure-level
axLeg = axes('Position',[0 0 1 1],'Visible','off','HandleVisibility','off');
hold(axLeg,'on');

% Dummy patches just for legend handles
hDummy = gobjects(1, nM);
for k = 1:nM
    hDummy(k) = patch(NaN, NaN, metricColors(k,:), ...
                      'FaceAlpha',0.88, 'EdgeColor','none', ...
                      'Parent', axLeg);
end

lgd = legend(axLeg, hDummy, metricNames, ...
             'Orientation', 'horizontal', ...
             'FontSize',    10, ...
             'Box',         'off');

lgd.Units    = 'normalized';
lgd.Position = [0.20 0.005 0.60 0.04];

%% ── Super title ──────────────────────────────────────────────────────────────
sgtitle('V2 Prompt Technique Comparison — All Metrics per Model  (N = 40 per cell)', ...
        'FontSize',12,'FontWeight','bold');
