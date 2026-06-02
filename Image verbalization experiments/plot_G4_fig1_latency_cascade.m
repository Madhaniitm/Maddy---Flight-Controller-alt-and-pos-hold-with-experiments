%% plot_G4_fig1_latency_cascade.m
% G4 — Two-panel figure (2 × 1)
%
% Top    (a): Latency cascade — all tiers on log scale (PID to LLM)
% Bottom (b): Tier 3 LLM latency comparison with 95% CI
%
% Data source: G4_runs_20260524_234054.csv + G4_calls_20260524_234054.csv (latest)

clear; clc; close all;

%% ── Data ─────────────────────────────────────────────────────────────────────

% (a) All-tier cascade
%  YOLO: stable mean from runs 2-5 (run 1 is warm-up)
tierValues = [0.5; 16.21; 263.37; 2332.6; 2626.8; 4333.0; 7218.3];   % ms
freqLabels = {'2000/s'; '~60/s'; '~3-4/s'; '~0.4/s'; '~0.4/s'; '~0.2/s'; '~0.1/s'};
valStrs    = {'0.5ms','16ms','263ms','2,333ms','2,627ms','4,333ms','7,218ms'};

cascadeColors = [0.18 0.80 0.44;   % PID
                 0.15 0.68 0.38;   % MediaPipe
                 0.95 0.61 0.07;   % YOLO
                 0.69 0.49 0.85;   % Gemini
                 0.29 0.56 0.85;   % GPT-4o
                 0.42 0.75 0.42;   % GPT-4o-mini
                 0.88 0.48 0.33];  % Claude

% (b) LLM comparison
modelLabels = {'Claude','GPT-4o','GPT-4o-mini','Gemini'};
means_ms    = [7218.3,  2626.8,  4333.0,  2332.6];
ci95_ms     = [ 414.8,   149.6,   283.7,   196.4];   % 1.96 × SEM, n=40

llmColors = [0.88 0.48 0.33;   % Claude
             0.29 0.56 0.85;   % GPT-4o
             0.42 0.75 0.42;   % GPT-4o-mini
             0.69 0.49 0.85];  % Gemini

%% ── Figure ───────────────────────────────────────────────────────────────────
fig = figure('Name','G4 — Latency Analysis', ...
             'Position',[80 60 1100 900]);

%% ══════════════════════════════════════════════════════════════════════════════
%% Subplot (a) — Latency cascade (log scale)
%% ══════════════════════════════════════════════════════════════════════════════
ax1 = subplot(2,1,1);
x7  = 1:7;
bh1 = bar(x7, tierValues, 0.65);
bh1.FaceColor = 'flat';
for k = 1:7
    bh1.CData(k,:) = cascadeColors(k,:);
end
bh1.FaceAlpha = 0.87;
bh1.EdgeColor = 'none';
set(ax1, 'YScale', 'log');
hold on;

% Tier separator lines
xline(1.5,'--','Color',[0.65 0.65 0.65],'LineWidth',0.9,'HandleVisibility','off');
xline(2.5,'--','Color',[0.65 0.65 0.65],'LineWidth',0.9,'HandleVisibility','off');
xline(3.5,'--','Color',[0.65 0.65 0.65],'LineWidth',0.9,'HandleVisibility','off');

% Value labels above bars
for k = 1:7
    text(x7(k), tierValues(k)*2.5, valStrs{k}, ...
         'HorizontalAlignment','center','FontSize',8.5,'FontWeight','bold', ...
         'Color',[0.15 0.15 0.15]);
end

% Frequency labels inside bars
for k = 1:7
    text(x7(k), tierValues(k)*0.3, freqLabels{k}, ...
         'HorizontalAlignment','center','FontSize',7.5,'FontWeight','bold', ...
         'Color','w');
end

% ×N ratio between adjacent tiers
ratioX  = [1.5, 2.5, 3.5];
ratioY  = [3,   65,  800];
ratioV  = [tierValues(2)/tierValues(1), tierValues(3)/tierValues(2), tierValues(4)/tierValues(3)];
for i = 1:3
    text(ratioX(i), ratioY(i), sprintf('\\times%.0f', ratioV(i)), ...
         'HorizontalAlignment','center','FontSize',9, ...
         'Color',[0.45 0.45 0.45],'FontAngle','italic');
end

% X-tick labels: component name only (multiline via cell array rows)
set(ax1,'XTick',x7, ...
    'XTickLabel',{'PID','MediaPipe','YOLO+Depth','Gemini','GPT-4o','GPT-4o-mini','Claude'}, ...
    'FontSize',9.5);
ylim([0.08 80000]);
ax1.YTick      = [0.1 1 10 100 1000 10000];
ax1.YTickLabel = {'0.1ms','1ms','10ms','100ms','1s','10s'};
ylabel('Latency  (log scale)','FontSize',11);
title({'(a)  Four-Tier Latency Cascade'; ...
       '4 orders of magnitude: PID (0.5ms, 2kHz) \rightarrow LLM (2,333–7,218ms)'},'FontSize',11);
ax1.YGrid='on'; ax1.XGrid='off';
ax1.GridAlpha=0.25; ax1.GridLineStyle='--'; ax1.Box='off';
hold off;

%% ══════════════════════════════════════════════════════════════════════════════
%% Subplot (b) — LLM latency comparison with 95% CI
%% ══════════════════════════════════════════════════════════════════════════════
ax2 = subplot(2,1,2);
x4  = 1:4;
bh2 = bar(x4, means_ms, 0.52);
bh2.FaceColor = 'flat';
for k = 1:4
    bh2.CData(k,:) = llmColors(k,:);
end
bh2.FaceAlpha = 0.87;
bh2.EdgeColor = 'none';
hold on;

% 95% CI error bars
errorbar(x4, means_ms, ci95_ms, ...
         'LineStyle','none','Color',[0.30 0.30 0.30], ...
         'LineWidth',1.6,'CapSize',8,'HandleVisibility','off');

% Value + CI labels above bars
for k = 1:4
    text(x4(k), means_ms(k) + ci95_ms(k) + 180, ...
         sprintf('%.0fms \\pm%.0fms', means_ms(k), ci95_ms(k)), ...
         'HorizontalAlignment','center','FontSize',9,'FontWeight','bold', ...
         'Color',[0.15 0.15 0.15]);
end

% Claude vs Gemini ratio label
ratio = means_ms(1) / means_ms(4);
ymid  = (means_ms(1) + means_ms(4)) / 2;
plot([x4(1) x4(4)], [means_ms(1) means_ms(4)], ...
     'Color',[0.60 0.60 0.60],'LineStyle',':','LineWidth',1.2,'HandleVisibility','off');
text(2.5, ymid + 500, sprintf('Claude is %.1f\\times slower than Gemini', ratio), ...
     'HorizontalAlignment','center','FontSize',9.5, ...
     'Color',[0.45 0.45 0.45],'FontAngle','italic');

set(ax2,'XTick',x4,'XTickLabel',modelLabels,'FontSize',11);
ylim([0 10500]);
ax2.YTick      = 0:1000:10000;
ax2.YTickLabel = {'0','1s','2s','3s','4s','5s','6s','7s','8s','9s','10s'};
ylabel('Mean Latency','FontSize',11);
title({'(b)  Tier 3 LLM Latency Comparison  (95% CI,  n = 40 per model)'; ...
       'Claude 3.1\times slower than Gemini — directly informs model selection'},'FontSize',11);
ax2.YGrid='on'; ax2.XGrid='off';
ax2.GridAlpha=0.28; ax2.GridLineStyle='--'; ax2.Box='off';
hold off;
