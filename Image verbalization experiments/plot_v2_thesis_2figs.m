%% plot_v2_thesis_2figs.m
% Three thesis-quality figures for V2 prompt technique comparison.
%
% Figure 1 — LblAcc and RsnAct per model x technique
% Figure 2 — DescAcc, RsnRisk, ActSafe, Danger per model x technique
% Figure 3 — Output tokens, Cost, Latency per model x technique
%
% Data from latest runs:
%   V2_runs_20260525_035756.csv       (zero_shot, few_shot_3, cot, structured)
%   V2_cot_rerun_20260525_175408.csv  (cot_600)

clear; clc; close all;

%% ── Data (from latest CSV files) ─────────────────────────────────────────────
% Rows = models [Claude; GPT-4o; GPT-4o-mini; Gemini]
% Cols = techniques [Zero-Shot, Few-Shot-3, CoT-300, CoT-600, Structured]

techLabels  = {'Zero-Shot','Few-Shot-3','CoT-300','CoT-600','Structured'};
modelLabels = {'Claude','GPT-4o','GPT-4o-mini','Gemini'};
x = 1:5;  nT = 5;  nM = 4;

LblAcc = [40.0,  35.0,   5.0,  57.5,  42.5;
          50.0,  67.5,  52.5,  60.0,  70.0;
          52.5,  47.5,  17.5,  30.0,  62.5;
          25.0,  55.0,   0.0,  37.5,  50.0];

RsnAct = [20.0,  30.0,   0.0,  95.0,  62.5;
         100.0, 100.0, 100.0, 100.0, 100.0;
         100.0, 100.0,  30.0, 100.0, 100.0;
          87.5, 100.0,   0.0,  65.0,  95.0];

DescAcc = [97.5,  95.0,  90.0,  95.0,  97.5;
          100.0, 100.0, 100.0, 100.0, 100.0;
          100.0, 100.0, 100.0, 100.0, 100.0;
          100.0, 100.0, 100.0, 100.0, 100.0];

RsnRisk = [97.5,  95.0,  90.0,  95.0,  97.5;
           92.5,  92.5, 100.0, 100.0, 100.0;
          100.0,  90.0, 100.0, 100.0, 100.0;
           92.5, 100.0, 100.0, 100.0, 100.0];

ActSafe = [100.0, 100.0, 100.0, 100.0, 100.0;
           100.0, 100.0, 100.0, 100.0, 100.0;
           100.0,  87.5, 100.0,  95.0, 100.0;
           100.0, 100.0, 100.0, 100.0, 100.0];

Danger = [0, 0, 0, 0, 0;
          0, 0, 0, 0, 0;
          0, 5, 0, 2, 0;
          0, 0, 0, 0, 0];

OutputTokens = [292.3, 281.8, 269.9, 478.8, 263.0;
                101.5,  99.0, 220.6, 220.6, 109.8;
                124.4,  91.3, 291.4, 342.6, 119.2;
                181.0, 144.6, 299.9, 527.8, 205.2];

Latency_s = [10.8,  12.2,  14.1,  16.2,   9.0;
              3.0,   3.0,   4.2,   4.5,   3.0;
              7.1,   4.8,  56.2,  12.7,  23.4;
              2.9,   2.4,   3.5,   4.4,   3.2];

Cost = [6.1,  6.5,  5.9,  9.2,  6.0;
        3.9,  4.8,  6.2,  6.2,  4.5;
        0.6,  0.6,  0.7,  0.7,  0.6;
        0.1,  0.1,  0.1,  0.2,  0.1];

% Model colours
mc = [0.87 0.42 0.36;   % Claude      — red
      0.29 0.56 0.85;   % GPT-4o      — blue
      0.36 0.75 0.42;   % GPT-4o-mini — green
      0.80 0.55 0.85];  % Gemini      — purple

bw = 0.78;  % bar width


%% ══════════════════════════════════════════════════════════════════════════════
%  FIGURE 1 — LblAcc and RsnAct
%% ══════════════════════════════════════════════════════════════════════════════
figure('Name','Fig 1 - LblAcc and RsnAct','Position',[60 60 1100 460]);

% ── 1a: LblAcc ───────────────────────────────────────────────────────────────
subplot(1,2,1);
bh = bar(x, LblAcc', bw);
for k=1:nM; bh(k).FaceColor=mc(k,:); bh(k).FaceAlpha=0.88; bh(k).EdgeColor='none'; end
hold on;
patch([4.62 5.38 5.38 4.62],[0 0 77 77],[0.29 0.56 0.85], ...
      'FaceAlpha',0.12,'EdgeColor',[0.15 0.40 0.75],'LineWidth',1.2,'HandleVisibility','off');
text(5,73,'70% best','HorizontalAlignment','center', ...
     'FontSize',8,'Color',[0.10 0.35 0.70],'FontWeight','bold');
text(3,3,'Gemini 0%','HorizontalAlignment','center', ...
     'FontSize',7.5,'Color',[0.60 0.10 0.60],'FontAngle','italic');
set(gca,'XTick',x,'XTickLabel',techLabels,'FontSize',9);
ylim([0 82]); ylabel('Label Accuracy (%)','FontSize',10);
title('(a) LblAcc per Model \times Technique','FontSize',10);
legend(modelLabels,'Location','northwest','FontSize',8.5);
grid on; ax=gca; ax.YGrid='on'; ax.XGrid='off';
ax.GridAlpha=0.3; ax.GridLineStyle='--'; ax.Box='off'; hold off;

% ── 1b: RsnAct ───────────────────────────────────────────────────────────────
subplot(1,2,2);
bh = bar(x, RsnAct', bw);
for k=1:nM; bh(k).FaceColor=mc(k,:); bh(k).FaceAlpha=0.88; bh(k).EdgeColor='none'; end
hold on;
text(3,6,'Claude &','HorizontalAlignment','center', ...
     'FontSize',7.5,'Color',[0.70 0.10 0.10],'FontAngle','italic');
text(3,1,'Gemini 0%','HorizontalAlignment','center', ...
     'FontSize',7.5,'Color',[0.70 0.10 0.10],'FontAngle','italic');
text(4,108,'Truncation fixed \rightarrow','HorizontalAlignment','center', ...
     'FontSize',8,'Color',[0.75 0.10 0.10],'FontWeight','bold');
set(gca,'XTick',x,'XTickLabel',techLabels,'FontSize',9);
ylim([0 118]); ylabel('RsnAct — Action Consistency (%)','FontSize',10);
title('(b) RsnAct per Model \times Technique','FontSize',10);
legend(modelLabels,'Location','southwest','FontSize',8.5);
grid on; ax=gca; ax.YGrid='on'; ax.XGrid='off';
ax.GridAlpha=0.3; ax.GridLineStyle='--'; ax.Box='off'; hold off;

sgtitle('V2 — Label Accuracy and Action Consistency  (N = 40 per cell)', ...
        'FontSize',11,'FontWeight','bold');


%% ══════════════════════════════════════════════════════════════════════════════
%  FIGURE 2 — DescAcc, RsnRisk, ActSafe, Danger
%% ══════════════════════════════════════════════════════════════════════════════
figure('Name','Fig 2 - DescAcc RsnRisk ActSafe Danger','Position',[60 580 1300 460]);

% ── 2a: DescAcc ──────────────────────────────────────────────────────────────
subplot(1,4,1);
bh = bar(x, DescAcc', bw);
for k=1:nM; bh(k).FaceColor=mc(k,:); bh(k).FaceAlpha=0.88; bh(k).EdgeColor='none'; end
hold on;
text(3,91.5,'Claude 90%','HorizontalAlignment','center', ...
     'FontSize',7,'Color',[0.70 0.10 0.10],'FontAngle','italic');
set(gca,'XTick',x,'XTickLabel',techLabels,'FontSize',7.5,'XTickLabelRotation',20);
ylim([80 103]); ylabel('DescAcc (%)','FontSize',9.5);
title('(a) DescAcc','FontSize',9.5);
legend(modelLabels,'Location','south','FontSize',7);
grid on; ax=gca; ax.YGrid='on'; ax.XGrid='off';
ax.GridAlpha=0.3; ax.GridLineStyle='--'; ax.Box='off'; hold off;

% ── 2b: RsnRisk ──────────────────────────────────────────────────────────────
subplot(1,4,2);
bh = bar(x, RsnRisk', bw);
for k=1:nM; bh(k).FaceColor=mc(k,:); bh(k).FaceAlpha=0.88; bh(k).EdgeColor='none'; end
hold on;
text(2,91,'GPT-4o 92.5%','HorizontalAlignment','center', ...
     'FontSize',7,'Color',[0.15 0.40 0.75],'FontAngle','italic');
set(gca,'XTick',x,'XTickLabel',techLabels,'FontSize',7.5,'XTickLabelRotation',20);
ylim([80 103]); ylabel('RsnRisk (%)','FontSize',9.5);
title('(b) RsnRisk','FontSize',9.5);
legend(modelLabels,'Location','south','FontSize',7);
grid on; ax=gca; ax.YGrid='on'; ax.XGrid='off';
ax.GridAlpha=0.3; ax.GridLineStyle='--'; ax.Box='off'; hold off;

% ── 2c: ActSafe ──────────────────────────────────────────────────────────────
subplot(1,4,3);
bh = bar(x, ActSafe', bw);
for k=1:nM; bh(k).FaceColor=mc(k,:); bh(k).FaceAlpha=0.88; bh(k).EdgeColor='none'; end
hold on;
text(2,88.5,'Mini 87.5%','HorizontalAlignment','center', ...
     'FontSize',7,'Color',[0.15 0.55 0.20],'FontAngle','italic');
set(gca,'XTick',x,'XTickLabel',techLabels,'FontSize',7.5,'XTickLabelRotation',20);
ylim([82 103]); ylabel('ActSafe (%)','FontSize',9.5);
title('(c) ActSafe','FontSize',9.5);
legend(modelLabels,'Location','south','FontSize',7);
grid on; ax=gca; ax.YGrid='on'; ax.XGrid='off';
ax.GridAlpha=0.3; ax.GridLineStyle='--'; ax.Box='off'; hold off;

% ── 2d: Danger count (stacked bar — only gpt4o_mini has non-zero) ─────────────
subplot(1,4,4);
bh = bar(x, Danger', bw);
for k=1:nM; bh(k).FaceColor=mc(k,:); bh(k).FaceAlpha=0.88; bh(k).EdgeColor='none'; end
hold on;
text(2,5.4,'5 cases','HorizontalAlignment','center', ...
     'FontSize',8,'Color',[0.15 0.55 0.20],'FontWeight','bold');
text(4,2.4,'2 cases','HorizontalAlignment','center', ...
     'FontSize',8,'Color',[0.15 0.55 0.20],'FontWeight','bold');
text(2.5,7.0,'GPT-4o-mini only','HorizontalAlignment','center', ...
     'FontSize',7.5,'Color',[0.15 0.55 0.20],'FontAngle','italic');
set(gca,'XTick',x,'XTickLabel',techLabels,'FontSize',7.5,'XTickLabelRotation',20);
ylim([0 9]); ylabel('Dangerous Cases (count)','FontSize',9.5);
title('(d) Danger  (truth=hazard, action=PITCH\_FWD)','FontSize',9.5);
legend(modelLabels,'Location','northeast','FontSize',7);
grid on; ax=gca; ax.YGrid='on'; ax.XGrid='off';
ax.GridAlpha=0.3; ax.GridLineStyle='--'; ax.Box='off'; hold off;

sgtitle('V2 — Scene Understanding, Reasoning and Safety  (N = 40 per cell)', ...
        'FontSize',11,'FontWeight','bold');


%% ══════════════════════════════════════════════════════════════════════════════
%  FIGURE 3 — Output Tokens, Cost, Latency
%% ══════════════════════════════════════════════════════════════════════════════
figure('Name','Fig 3 - Tokens Cost Latency','Position',[60 60 1300 460]);

% ── 3a: Output Tokens ────────────────────────────────────────────────────────
subplot(1,3,1);
bh = bar(x, OutputTokens', bw);
for k=1:nM; bh(k).FaceColor=mc(k,:); bh(k).FaceAlpha=0.88; bh(k).EdgeColor='none'; end
hold on;
text(4,540,'Claude 479','HorizontalAlignment','center', ...
     'FontSize',7.5,'Color',[0.70 0.10 0.10],'FontWeight','bold');
text(4,500,'Gemini 528','HorizontalAlignment','center', ...
     'FontSize',7,'Color',[0.60 0.30 0.70],'FontAngle','italic');
set(gca,'XTick',x,'XTickLabel',techLabels,'FontSize',8,'XTickLabelRotation',15);
ylim([0 600]); ylabel('Mean Output Tokens','FontSize',10);
title('(a) Output Tokens per Model \times Technique','FontSize',9.5);
legend(modelLabels,'Location','northwest','FontSize',8);
grid on; ax=gca; ax.YGrid='on'; ax.XGrid='off';
ax.GridAlpha=0.3; ax.GridLineStyle='--'; ax.Box='off'; hold off;

% ── 3b: Cost ─────────────────────────────────────────────────────────────────
subplot(1,3,2);
bh = bar(x, Cost', bw);
for k=1:nM; bh(k).FaceColor=mc(k,:); bh(k).FaceAlpha=0.88; bh(k).EdgeColor='none'; end
hold on;
text(4,9.8,'$9.2 peak','HorizontalAlignment','center', ...
     'FontSize',8,'Color',[0.70 0.10 0.10],'FontWeight','bold');
set(gca,'XTick',x,'XTickLabel',techLabels,'FontSize',8,'XTickLabelRotation',15);
ylim([0 12]); ylabel('Cost per 1000 calls (USD)','FontSize',10);
title('(b) Cost per Model \times Technique','FontSize',9.5);
legend(modelLabels,'Location','northeast','FontSize',8);
grid on; ax=gca; ax.YGrid='on'; ax.XGrid='off';
ax.GridAlpha=0.3; ax.GridLineStyle='--'; ax.Box='off'; hold off;

% ── 3c: Latency ──────────────────────────────────────────────────────────────
subplot(1,3,3);
LatPlot = Latency_s;
outlierVal = LatPlot(3,3);
LatPlot(3,3) = NaN;

bh = bar(x, LatPlot', bw);
for k=1:nM; bh(k).FaceColor=mc(k,:); bh(k).FaceAlpha=0.88; bh(k).EdgeColor='none'; end
hold on;
plot(3, 21.0, 'v','Color',mc(3,:),'MarkerSize',9, ...
     'MarkerFaceColor',mc(3,:),'HandleVisibility','off');
text(3,23.0,sprintf('%.0fs*',outlierVal),'HorizontalAlignment','center', ...
     'FontSize',8,'Color',[0.15 0.55 0.20],'FontWeight','bold');
set(gca,'XTick',x,'XTickLabel',techLabels,'FontSize',8,'XTickLabelRotation',15);
ylim([0 27]); ylabel('Mean Latency (s)','FontSize',10);
title('(c) Latency per Model \times Technique','FontSize',9.5);
legend(modelLabels,'Location','northeast','FontSize',8);
grid on; ax=gca; ax.YGrid='on'; ax.XGrid='off';
ax.GridAlpha=0.3; ax.GridLineStyle='--'; ax.Box='off'; hold off;

annotation('textbox',[0.65 0.01 0.34 0.04], ...
           'String','* gpt4o-mini CoT-300: 56s (token budget exceeded)', ...
           'FontSize',7.5,'EdgeColor','none','Color',[0.45 0.45 0.45]);

sgtitle('V2 — Output Tokens, Cost and Latency  (N = 40 per cell)', ...
        'FontSize',11,'FontWeight','bold');
