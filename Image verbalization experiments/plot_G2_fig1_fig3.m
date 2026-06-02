%% plot_G2_fig1_fig3.m
% G2 — Three-panel figure  (2 × 1, with bottom half split into (b) and (c))
%
% Top     (a): Detection Timeline — 10-frame conceptual diagram
%              Scheduled fires at F1 & F6; Hybrid fires at F3
%              3-frame undetected exposure window highlighted
% Bottom-L (b): Frames Late to Detect Hazard  (grouped bar, per model)
% Bottom-R (c): LLM Calls per Sequence        (grouped bar, per model)
%
% Data source: G2_runs_20260524_232234.csv  (latest run)
%
% Key finding: Hybrid eliminates 3-frame (100ms) detection lag at cost of +1 LLM call

clear; clc; close all;

%% ── Data ─────────────────────────────────────────────────────────────────────
% Means from G2_runs_20260524_232234.csv — deterministic, identical across models
%   Scheduled: frames_late = 3, llm_calls = 2
%   Hybrid   : frames_late = 0, llm_calls = 3

modelLabels = {'Claude','GPT-4o','GPT-4o-mini','Gemini'};
stratLabels = {'Scheduled','Hybrid'};

%         [scheduled, hybrid]  — same for all 4 models
FramesLate = [3, 0;   % Claude
              3, 0;   % GPT-4o
              3, 0;   % GPT-4o-mini
              3, 0];  % Gemini

LLMCalls   = [2, 3;   % Claude
              2, 3;   % GPT-4o
              2, 3;   % GPT-4o-mini
              2, 3];  % Gemini

stratColors = [0.58 0.65 0.65;   % Scheduled — grey
               0.30 0.72 0.40];  % Hybrid    — green

%% ── Layout constants  [left  bottom  width  height] ─────────────────────────
LFT  = 0.10;
WID  = 0.82;
PHGT = 0.30;
BOT2 = 0.11;
GAP  = 0.18;
BOT1 = BOT2 + PHGT + GAP;   % = 0.59  (bottom of panel a)

HWID = (WID - 0.03) / 2;    % = 0.395 (half-width for b / c)
LGAP = 0.03;                 % gap between b and c

%% ── Figure ───────────────────────────────────────────────────────────────────
fig = figure('Name','G2 — Detection Strategy Analysis', ...
             'Position',[80 60 1150 880]);

%% ══════════════════════════════════════════════════════════════════════════════
%% Panel (a) — Detection Timeline  (axis off, conceptual diagram)
%% ══════════════════════════════════════════════════════════════════════════════
ax1 = axes('Position',[LFT BOT1 WID PHGT]);
hold on;

safeColor   = [0.92 0.98 0.95];   % pale green
hazardColor = [0.98 0.86 0.85];   % pale red
SCENE = [1 1 2 2 2 2 2 1 1 1];   % 1 = safe, 2 = hazard

% ── Frame boxes ──────────────────────────────────────────────────────────────
for f = 1:10
    if SCENE(f) == 1
        fc = safeColor;   tc = [0.12 0.52 0.22];
    else
        fc = hazardColor; tc = [0.72 0.10 0.10];
    end
    rectangle('Position',[f - 0.45, 0.10, 0.90, 0.80], ...
              'Curvature',0.15,'FaceColor',fc, ...
              'EdgeColor',[0.70 0.74 0.77],'LineWidth',1.2);
    text(f, 0.50, sprintf('F%d',f), ...
         'HorizontalAlignment','center','VerticalAlignment','middle', ...
         'FontSize',10,'FontWeight','bold','Color',tc);
end

% ── Scene labels below frames ─────────────────────────────────────────────────
text(1.5, -0.07, 'Safe (door open)',        'HorizontalAlignment','center', ...
     'FontSize',9,'Color',[0.12 0.52 0.22]);
text(5.0, -0.07, 'HAZARD — person enters',  'HorizontalAlignment','center', ...
     'FontSize',9,'Color',[0.72 0.10 0.10],'FontWeight','bold');
text(9.0, -0.07, 'Safe (door open)',         'HorizontalAlignment','center', ...
     'FontSize',9,'Color',[0.12 0.52 0.22]);

% ── Scheduled call arrows at F1 and F6  (blue, pointing downward) ─────────────
scCol = [0.29 0.56 0.85];
for f = [1 6]
    plot([f f], [1.05 1.40], '-', 'Color',scCol,'LineWidth',1.8);
    plot(f, 1.05, 'v', 'MarkerFaceColor',scCol,'MarkerEdgeColor','none','MarkerSize',8);
    text(f, 1.52, sprintf('Scheduled\ncall'), 'HorizontalAlignment','center', ...
         'FontSize',8.5,'Color',scCol,'FontWeight','bold');
end

% ── Hybrid trigger arrow at F3  (red, taller) ────────────────────────────────
htCol = [0.85 0.15 0.15];
plot([3 3], [1.05 1.70], '-', 'Color',htCol,'LineWidth',2.2);
plot(3, 1.05, 'v','MarkerFaceColor',htCol,'MarkerEdgeColor','none','MarkerSize',9);
text(3, 1.82, {'Hybrid trigger';'(MediaPipe fires)'}, ...
     'HorizontalAlignment','center','FontSize',8.5,'Color',htCol,'FontWeight','bold');

% ── 3-frame exposure window: double-headed arrow F3 → F6 ─────────────────────
exCol = [0.90 0.48 0.08];
plot([3.10 5.90], [0.93 0.93], '-','Color',exCol,'LineWidth',2.0);
plot(3.10, 0.93, '<','MarkerFaceColor',exCol,'MarkerEdgeColor','none','MarkerSize',7);
plot(5.90, 0.93, '>','MarkerFaceColor',exCol,'MarkerEdgeColor','none','MarkerSize',7);
text(4.5, 0.99, {'3 frames undetected  =  100ms  =  3–5cm travel'}, ...
     'HorizontalAlignment','center','VerticalAlignment','bottom', ...
     'FontSize',8.5,'Color',exCol,'FontWeight','bold');

% ── Legend (inline, top-right) ───────────────────────────────────────────────
hp1 = patch(NaN,NaN,safeColor,  'EdgeColor',[0.70 0.74 0.77],'LineWidth',0.8);
hp2 = patch(NaN,NaN,hazardColor,'EdgeColor',[0.70 0.74 0.77],'LineWidth',0.8);
hl1 = plot(NaN,NaN,'-v','Color',scCol,'MarkerFaceColor',scCol, ...
           'LineWidth',1.8,'MarkerSize',7);
hl2 = plot(NaN,NaN,'-v','Color',htCol,'MarkerFaceColor',htCol, ...
           'LineWidth',2.2,'MarkerSize',8);
legend([hp1,hp2,hl1,hl2], ...
       {'Safe frame','Hazard frame','Scheduled call (F1, F6)','Hybrid trigger (F3)'}, ...
       'Location','northeast','FontSize',9,'Box','on', ...
       'EdgeColor',[0.82 0.82 0.82],'Color',[0.98 0.98 0.98]);

xlim([0.35 10.65]);
ylim([-0.22 2.15]);
axis off;

annotation('textbox',[LFT, BOT1+PHGT+0.005, WID, 0.055], ...
    'String',{'\bf(a)  Detection Timeline: Scheduled vs Hybrid Strategy'; ...
              'Hybrid eliminates the 3-frame (100ms) exposure window — at cost of +1 LLM call per hazard event'}, ...
    'FontSize',11,'EdgeColor','none', ...
    'HorizontalAlignment','center','VerticalAlignment','bottom', ...
    'FitBoxToText','off');

hold off;

%% ══════════════════════════════════════════════════════════════════════════════
%% Panel (b) — Frames Late  (bottom-left)
%% ══════════════════════════════════════════════════════════════════════════════
ax2 = axes('Position',[LFT, BOT2, HWID, PHGT]);

bh2 = bar(1:4, FramesLate, 0.52);
for k = 1:2
    bh2(k).FaceColor = stratColors(k,:);
    bh2(k).FaceAlpha = 0.87;
    bh2(k).EdgeColor = 'none';
end
hold on;

% Value labels above bars
nBars  = 2;  bwFL = 0.52;
offFL  = linspace(-(nBars-1)/2, (nBars-1)/2, nBars) * (bwFL/nBars);
for m = 1:4
    for s = 1:2
        val = FramesLate(m,s);
        text(m + offFL(s), val + 0.10, sprintf('%d',val), ...
             'HorizontalAlignment','center','FontSize',10.5,'FontWeight','bold', ...
             'Color',[0.15 0.15 0.15]);
    end
end

% Footnote inside axes (units = normalized)
text(0.50, -0.13, '3 frames = 100ms at 30fps  =  3–5cm travel at 0.3–0.5 m/s', ...
     'Units','normalized','HorizontalAlignment','center', ...
     'FontSize',8.5,'Color',[0.90 0.48 0.08],'FontAngle','italic');

set(ax2,'XTick',1:4,'XTickLabel',modelLabels,'FontSize',10);
ylim([0 5]);
ax2.YTick = 0:1:5;
ylabel('Mean Frames Late','FontSize',11);
ax2.YGrid='on'; ax2.XGrid='off';
ax2.GridAlpha=0.28; ax2.GridLineStyle='--'; ax2.Box='off';

annotation('textbox',[LFT, BOT2+PHGT+0.005, HWID, 0.055], ...
    'String','\bf(b)  Frames Late to Detect Hazard', ...
    'FontSize',11,'EdgeColor','none', ...
    'HorizontalAlignment','center','VerticalAlignment','bottom', ...
    'FitBoxToText','off');

hold off;

%% ══════════════════════════════════════════════════════════════════════════════
%% Panel (c) — LLM Calls  (bottom-right)
%% ══════════════════════════════════════════════════════════════════════════════
ax3 = axes('Position',[LFT + HWID + LGAP, BOT2, HWID, PHGT]);

bh3 = bar(1:4, LLMCalls, 0.52);
for k = 1:2
    bh3(k).FaceColor = stratColors(k,:);
    bh3(k).FaceAlpha = 0.87;
    bh3(k).EdgeColor = 'none';
end
hold on;

% Value labels above bars
for m = 1:4
    for s = 1:2
        val = LLMCalls(m,s);
        text(m + offFL(s), val + 0.10, sprintf('%d',val), ...
             'HorizontalAlignment','center','FontSize',10.5,'FontWeight','bold', ...
             'Color',[0.15 0.15 0.15]);
    end
end

% Footnote inside axes
text(0.50, -0.13, '+1 LLM call per hazard event  —  Gemini cost: +$0.00012 per trigger', ...
     'Units','normalized','HorizontalAlignment','center', ...
     'FontSize',8.5,'Color',[0.35 0.35 0.35],'FontAngle','italic');

set(ax3,'XTick',1:4,'XTickLabel',modelLabels,'FontSize',10);
ylim([0 5]);
ax3.YTick = 0:1:5;
ylabel('Mean LLM Calls','FontSize',11);
ax3.YGrid='on'; ax3.XGrid='off';
ax3.GridAlpha=0.28; ax3.GridLineStyle='--'; ax3.Box='off';

annotation('textbox',[LFT + HWID + LGAP, BOT2+PHGT+0.005, HWID, 0.055], ...
    'String','\bf(c)  LLM Calls per Sequence', ...
    'FontSize',11,'EdgeColor','none', ...
    'HorizontalAlignment','center','VerticalAlignment','bottom', ...
    'FitBoxToText','off');

hold off;

%% ── Common legend for bar charts (b) & (c) — sits in the gap ────────────────
axLeg = axes('Position',[0 0 1 1],'Visible','off','HandleVisibility','off');
hold(axLeg,'on');
hD = gobjects(1,2);
for k = 1:2
    hD(k) = patch(NaN,NaN,stratColors(k,:), ...
                  'FaceAlpha',0.87,'EdgeColor','none','Parent',axLeg);
end
lgd = legend(axLeg, hD, stratLabels, ...
             'Orientation','horizontal','FontSize',10,'Box','off');
lgd.Units    = 'normalized';
lgd.Position = [0.40, BOT2+PHGT+GAP*0.52, 0.20, 0.04];
hold off;
