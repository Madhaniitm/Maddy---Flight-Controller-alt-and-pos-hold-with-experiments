%% plot_C2_ambiguity.m
% C2 — Command ambiguity resolution: success-rate heatmap (baseline vs after fix)
% Data hardcoded from CSV files. Rows = command types 1-6, Cols = [Claude, GPT-4o, GPT-4o-mini, Gemini]

clear; clc; close all;
set(groot,'DefaultAxesFontSize',18);

%% ── hardcoded success rates (%)  ────────────────────────────────────────────
%                          Claude  GPT-4o  Mini   Gemini
D_base = [ ...
    100.0,  100.0,  100.0,   93.3; ...   % cmd1 — Explicit
    100.0,  100.0,  100.0,  100.0; ...   % cmd2 — Paraphrase
      0.0,  100.0,   53.3,  100.0; ...   % cmd3 — Relative (no num)
     80.0,   60.0,    6.7,    0.0; ...   % cmd4 — Vague relative
     40.0,   46.7,    6.7,    0.0; ...   % cmd5 — Abstract
     20.0,   20.0,   26.7,    0.0 ];     % cmd6 — Indirect

%                         Claude  GPT-4o  Mini   Gemini
D_fix = [ ...
    100.0,   80.0,  100.0,  100.0; ...   % cmd1 — Explicit
    100.0,  100.0,  100.0,  100.0; ...   % cmd2 — Paraphrase
    100.0,  100.0,   20.0,  100.0; ...   % cmd3 — Relative (no num)
    100.0,   60.0,   40.0,   60.0; ...   % cmd4 — Vague relative
     80.0,   60.0,   40.0,   80.0; ...   % cmd5 — Abstract
     40.0,    0.0,   20.0,   80.0 ];     % cmd6 — Indirect

%% ── labels ───────────────────────────────────────────────────────────────────
cmd_lbl   = {'Explicit','Paraphrase','Relative (no num)','Vague relative','Abstract','Indirect'};
model_lbl = {'Claude','GPT-4o','GPT-4o-mini','Gemini'};

%% colormap: red (0%) → white (50%) → green (100%)
nh   = 128;
cmap = [ linspace(0.85,1,nh)', linspace(0.10,1,nh)', linspace(0.10,1,nh)'; ...
         linspace(1,0.10,nh)', linspace(1,0.60,nh)', linspace(1,0.10,nh)' ];

%% ── Figure ───────────────────────────────────────────────────────────────────
figure('Name','C2 — Ambiguity Heatmap','Position',[60 60 1700 850]);
titles = {'Baseline  (15 runs / model)', 'After Fix  (conservative-default policy, 5 runs / model)'};
D_all  = {D_base, D_fix};

for p = 1:2
    ax = subplot(1,2,p);
    D  = D_all{p};

    imagesc(ax, D);
    colormap(ax, cmap);
    clim(ax, [0 100]);
    set(ax, 'XTick',1:4, 'XTickLabel',model_lbl, ...
            'YTick',1:6, 'YTickLabel',cmd_lbl, ...
            'XAxisLocation','top', 'TickLength',[0 0], ...
            'FontSize',18, 'Box','off');
    title(ax, titles{p}, 'FontSize',20, 'FontWeight','bold');

    cb = colorbar(ax);
    cb.Label.String   = 'Success rate  (%)';
    cb.Label.FontSize = 16;

    for r = 1:6
        for c = 1:4
            v  = D(r,c);
            tc = [0 0 0];
            if v <= 35 || v >= 70; tc = [1 1 1]; end
            text(ax, c, r, sprintf('%.0f%%', v), ...
                 'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
                 'FontSize',17, 'FontWeight','bold', 'Color',tc);
        end
    end

    overall = mean(D(:));
    text(ax, 2.5, 6.7, sprintf('Overall  %.0f%%', overall), ...
         'HorizontalAlignment','center','FontSize',16,'Color',[0.3 0.3 0.3]);
end

sgtitle('C2 — Command Ambiguity: Success Rate by Model and Command Type', ...
        'FontSize',22, 'FontWeight','bold');
