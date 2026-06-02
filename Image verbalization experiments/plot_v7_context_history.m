%% plot_v7_context_history.m
% V7 — Context History Mode Comparison  (1 × 2 figure)
%
% Left  (a): Risk Accuracy (%)         — was the risk label correct?
% Right (b): Change Detection Rate (%) — among change_event==1 frames,
%            did the model correctly flag the transition? (n=10 per cell)
%
% Data source: V7_runs_20260525_221631.csv  (latest run)
%
% Key findings:
%   Risk Accuracy  : short history best for 3/4 models; GPT-4o-mini 4% stateless
%   Change Detect  : Gemini gains most from short history (60%→90%);
%                    GPT-4o surprisingly high stateless (90%)

clear; clc; close all;

%% ── Data ─────────────────────────────────────────────────────────────────────
modelLabels = {'GPT-4o','Gemini','Claude','GPT-4o-mini'};
modeLabels  = {'Stateless','Short History','Full History'};

% Rows = models, Cols = [Stateless, Short, Full]
% Source: latest CSV V7_runs_20260525_221631.csv
RiskAcc = [88.0,  76.0,  60.0;   % GPT-4o       — best: Stateless
           44.0,  68.0,  60.0;   % Gemini        — best: Short
           36.0,  40.0,  36.0;   % Claude         — best: Short (marginal)
            4.0,  48.0,  44.0];  % GPT-4o-mini   — best: Short

% Change Detection Rate (%) — only among change_event==1 frames (n=10/cell)
% Measures: did the model flag a real scene transition correctly?
ChgDet = [90.0,  70.0,  60.0;   % GPT-4o       — best: Stateless (strong base)
          60.0,  90.0,  70.0;   % Gemini        — best: Short (biggest gain)
          50.0,  50.0,  40.0;   % Claude         — flat; slight drop w/ full
          60.0,  60.0,  50.0];  % GPT-4o-mini   — marginal history benefit

xM = 1:4;

% History mode colours
mc = [0.35 0.62 0.90;   % Stateless  — blue
      0.30 0.72 0.40;   % Short      — green
      0.90 0.45 0.35];  % Full       — red/orange

nBars  = 3;
bWidth = 0.75;
offsets = linspace(-(nBars-1)/2, (nBars-1)/2, nBars) * (bWidth/nBars);

%% ── Figure ───────────────────────────────────────────────────────────────────
fig = figure('Name','V7 — Context History Mode Comparison', ...
             'Position',[80 80 1200 480]);

% Leave room at bottom for legend
set(fig,'DefaultAxesPosition',[0.07 0.18 0.87 0.70]);

%% ══════════════════════════════════════════════════════════════════════════════
%% Subplot (a) — Risk Accuracy
%% ══════════════════════════════════════════════════════════════════════════════
ax1 = subplot(1,2,1);
bh1 = bar(xM, RiskAcc, bWidth);
for k = 1:3
    bh1(k).FaceColor = mc(k,:);
    bh1(k).FaceAlpha = 0.88;
    bh1(k).EdgeColor = 'none';
end
hold on;

% GPT-4o-mini stateless callout
text(xM(4) + offsets(1), RiskAcc(4,1) + 5, '4%!', ...
     'HorizontalAlignment','center','FontSize',9, ...
     'Color',[0.80 0.10 0.10],'FontWeight','bold');

% Value labels inside bars (val > 12)
for m = 1:4
    for k = 1:3
        val = RiskAcc(m,k);
        bx  = xM(m) + offsets(k);
        if val > 12
            text(bx, val/2, sprintf('%.0f%%',val), ...
                 'HorizontalAlignment','center','FontSize',7.5, ...
                 'Color','w','FontWeight','bold');
        end
    end
end

% Cross-model mean lines — the system-level story
meanRisk = mean(RiskAcc, 1);   % [43.0, 58.0, 50.0]
yline(meanRisk(1),':', 'Color', mc(1,:), 'LineWidth', 1.4, 'HandleVisibility','off');
yline(meanRisk(2),':', 'Color', mc(2,:), 'LineWidth', 1.8, 'HandleVisibility','off');
yline(meanRisk(3),':', 'Color', mc(3,:), 'LineWidth', 1.4, 'HandleVisibility','off');
text(4.62, meanRisk(1)+1, sprintf('%.0f%%', meanRisk(1)), 'FontSize',7.5, 'Color', mc(1,:), 'FontWeight','bold');
text(4.62, meanRisk(2)+1, sprintf('%.0f%%', meanRisk(2)), 'FontSize',7.5, 'Color', mc(2,:), 'FontWeight','bold');
text(4.62, meanRisk(3)+1, sprintf('%.0f%%', meanRisk(3)), 'FontSize',7.5, 'Color', mc(3,:), 'FontWeight','bold');
text(4.62, meanRisk(2)+4.5, '\mu', 'FontSize',7,'Color',[0.3 0.3 0.3]);

set(ax1,'XTick',xM,'XTickLabel',modelLabels,'FontSize',10.5);
ylim([0 100]);
ylabel('Risk Accuracy (%)','FontSize',11);
title({'(a) Risk Accuracy by History Mode'; ...
       '(dotted lines = cross-model mean per mode)'},'FontSize',10.5);
grid on;
ax1.YGrid='on'; ax1.XGrid='off';
ax1.GridAlpha=0.3; ax1.GridLineStyle='--'; ax1.Box='off';
hold off;

%% ══════════════════════════════════════════════════════════════════════════════
%% Subplot (b) — Change Detection Rate
%% ══════════════════════════════════════════════════════════════════════════════
ax2 = subplot(1,2,2);
bh2 = bar(xM, ChgDet, bWidth);
for k = 1:3
    bh2(k).FaceColor = mc(k,:);
    bh2(k).FaceAlpha = 0.88;
    bh2(k).EdgeColor = 'none';
end
hold on;

% Gemini short callout — biggest gain from history
text(xM(2) + offsets(2), ChgDet(2,2) + 5, '90%!', ...
     'HorizontalAlignment','center','FontSize',9, ...
     'Color',[0.12 0.55 0.18],'FontWeight','bold');

% Value labels inside bars
for m = 1:4
    for k = 1:3
        val = ChgDet(m,k);
        bx  = xM(m) + offsets(k);
        if val > 12
            text(bx, val/2, sprintf('%.0f%%',val), ...
                 'HorizontalAlignment','center','FontSize',7.5, ...
                 'Color','w','FontWeight','bold');
        end
    end
end

% Cross-model mean lines
meanChg = mean(ChgDet, 1);   % [65.0, 67.5, 55.0]
yline(meanChg(1),':', 'Color', mc(1,:), 'LineWidth', 1.4, 'HandleVisibility','off');
yline(meanChg(2),':', 'Color', mc(2,:), 'LineWidth', 1.8, 'HandleVisibility','off');
yline(meanChg(3),':', 'Color', mc(3,:), 'LineWidth', 1.4, 'HandleVisibility','off');
text(4.62, meanChg(1)-3,  sprintf('%.0f%%', meanChg(1)), 'FontSize',7.5, 'Color', mc(1,:), 'FontWeight','bold');
text(4.62, meanChg(2)+1,  sprintf('%.0f%%', meanChg(2)), 'FontSize',7.5, 'Color', mc(2,:), 'FontWeight','bold');
text(4.62, meanChg(3)-3,  sprintf('%.0f%%', meanChg(3)), 'FontSize',7.5, 'Color', mc(3,:), 'FontWeight','bold');
text(4.62, meanChg(2)+4.5, '\mu', 'FontSize',7,'Color',[0.3 0.3 0.3]);

set(ax2,'XTick',xM,'XTickLabel',modelLabels,'FontSize',10.5);
ylim([0 100]);
ylabel('Change Detection Rate (%)','FontSize',11);
title({'(b) Change Detection Rate by History Mode'; ...
       '(scene-transition frames only, n = 10/cell; dotted = cross-model mean)'},'FontSize',10.5);
grid on;
ax2.YGrid='on'; ax2.XGrid='off';
ax2.GridAlpha=0.3; ax2.GridLineStyle='--'; ax2.Box='off';
hold off;

%% ── Common legend at bottom ──────────────────────────────────────────────────
axLeg = axes('Position',[0 0 1 1],'Visible','off','HandleVisibility','off');
hold(axLeg,'on');
hDummy = gobjects(1,3);
for k = 1:3
    hDummy(k) = patch(NaN, NaN, mc(k,:), ...
                      'FaceAlpha',0.88,'EdgeColor','none','Parent',axLeg);
end
lgd = legend(axLeg, hDummy, modeLabels, ...
             'Orientation','horizontal','FontSize',10,'Box','off');
lgd.Units    = 'normalized';
lgd.Position = [0.22 0.01 0.56 0.05];

hold off;
