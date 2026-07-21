%% plot_ND2_patrol_heatmap.m  →  fig4_3
% ND2 — Agentic patrol outcomes.  TWO heatmaps side by side:
%   Left : Mean quality score (0–5) — did the model reason and produce useful output?
%   Right: Mission complete %       — did the mission terminate with a landing?
%
% Key insight from ND2_observations_20260529.md:
%   "Primary failure modes are not perceptual but behavioural — incorrect termination policies."
%
%   Claude clear_room   : quality=3.8 (reports written, correct reasoning) BUT 0% complete
%                         → termination failure, NOT reasoning failure (never called land())
%   GPT-4o-mini alert   : quality=0.0 (zero words written across all 5 runs)
%                         → LITM total failure — infinite loop, no output at all
%   GPT-4o alert        : quality=4.0 AND 40% complete — hazard-abort is CORRECT doctrine
%                         (do not leave emergency scene → not landing is correct behaviour)
%   Gemini alert        : quality=3.6 AND 40% complete — similar hazard-abort pattern
%
% Data: ND2_summary_20260528_224548.csv (quality + patrol_complete per model × scenario)

clear; clc; close all;
set(groot,'DefaultAxesFontSize',19);

%% ── data: rows = [Claude, GPT-4o, GPT-4o-mini, Gemini]  cols = [clear, alert] ─
%  quality_score mean (0–5)
D_quality = [ 3.8,  3.0; ...   % Claude
              3.8,  4.0; ...   % GPT-4o
              4.0,  0.0; ...   % GPT-4o-mini
              3.8,  3.6 ];     % Gemini

%  mission complete % (patrol + landing)
D_complete = [  0,  100; ...   % Claude
              100,   40; ...   % GPT-4o
              100,    0; ...   % GPT-4o-mini
               60,   40 ];     % Gemini

model_lbl = {'Claude 3.5 Sonnet','GPT-4o','GPT-4o-mini','Gemini 2.5 Flash'};
scene_lbl = {'clear\_room','alert\_room'};

%% ── colormaps ────────────────────────────────────────────────────────────────
nh = 128;
% red → white → green (for both panels, scaled to their range)
cmap_rg = [ linspace(0.90,1.0,nh)', linspace(0.15,1.0,nh)', linspace(0.15,0.15,nh)'; ...
            linspace(1.0,0.12,nh)', linspace(1.0,0.72,nh)', linspace(0.15,0.12,nh)' ];

%% ── Figure ───────────────────────────────────────────────────────────────────
figure('Name','ND2 — Patrol Outcomes','Position',[60 60 1600 820]);

%% ── LEFT: quality score heatmap ─────────────────────────────────────────────
ax1 = subplot(1,2,1);
imagesc(ax1, D_quality);
colormap(ax1, cmap_rg);
clim(ax1, [0 5]);

set(ax1,'XTick',1:2,'XTickLabel',scene_lbl,'YTick',1:4,'YTickLabel',model_lbl, ...
        'XAxisLocation','top','TickLength',[0 0],'FontSize',19,'Box','off');
title(ax1,'Mean Quality Score  (0–5)','FontSize',21,'FontWeight','bold');
cb1 = colorbar(ax1);
cb1.Label.String = 'Quality  (0–5)';
cb1.Label.FontSize = 16;

for r = 1:4
    for c = 1:2
        v  = D_quality(r,c);
        tc = [0 0 0];
        if v <= 1.5 || v >= 3.5; tc = [1 1 1]; end
        text(ax1, c, r, sprintf('%.1f', v), ...
             'HorizontalAlignment','center','VerticalAlignment','middle', ...
             'FontSize',22,'FontWeight','bold','Color',tc);
    end
end

% annotations explaining the key findings
text(ax1, 0.52, 1.0, {'← reports written  (3.8/5)'; 'termination failed'; 'never called land()'}, ...
     'FontSize',13,'Color',[0.15 0.50 0.15],'FontWeight','bold');
text(ax1, 1.52, 3.0, {'← zero words written'; 'LITM total failure'; '(infinite loop)'}, ...
     'FontSize',13,'Color',[0.9 0.2 0.2],'FontWeight','bold');
text(ax1, 1.52, 2.0, {'← hazard-abort'; 'correct safety'; 'doctrine (4.0/5)'}, ...
     'FontSize',13,'Color',[0.15 0.50 0.15],'FontWeight','bold');

%% ── RIGHT: mission complete heatmap ─────────────────────────────────────────
ax2 = subplot(1,2,2);
imagesc(ax2, D_complete);
colormap(ax2, cmap_rg);
clim(ax2, [0 100]);

set(ax2,'XTick',1:2,'XTickLabel',scene_lbl,'YTick',1:4,'YTickLabel',model_lbl, ...
        'XAxisLocation','top','TickLength',[0 0],'FontSize',19,'Box','off');
title(ax2,'Mission Complete  (patrol + landing)  (%)','FontSize',21,'FontWeight','bold');
cb2 = colorbar(ax2);
cb2.Label.String = 'Completion  (%)';
cb2.Label.FontSize = 16;

for r = 1:4
    for c = 1:2
        v  = D_complete(r,c);
        tc = [0 0 0];
        if v <= 30 || v >= 70; tc = [1 1 1]; end
        text(ax2, c, r, sprintf('%d%%', v), ...
             'HorizontalAlignment','center','VerticalAlignment','middle', ...
             'FontSize',22,'FontWeight','bold','Color',tc);
    end
end

% right-panel annotations
text(ax2, 0.52, 1.0, {'← 0% but quality=3.8'; 'reasoning worked'; 'no land() called'}, ...
     'FontSize',13,'Color',[1.0 0.85 0.2],'FontWeight','bold');
text(ax2, 1.52, 3.0, {'← 0% and quality=0'; 'true LITM failure'}, ...
     'FontSize',13,'Color',[0.9 0.2 0.2],'FontWeight','bold');
text(ax2, 1.52, 2.0, {'← 40% complete'; 'rest aborted on'; 'hazard (correct)'}, ...
     'FontSize',13,'Color',[1.0 0.85 0.2],'FontWeight','bold');

sgtitle({'ND2 — Agentic Patrol: Quality of Reasoning  vs  Mission Termination'; ...
         'Left: did model produce useful output?   Right: did mission terminate with landing?'; ...
         'Not landing ≠ mission failed — Claude wrote quality-3.8 reports but never called land()'}, ...
        'FontSize',19,'FontWeight','bold');
