%% plot_C6_square_mission.m
% C6 — Mission planning: XY bounding-box of each model's square-pattern attempt.
% Data hardcoded from CSV files. 5 subplots (2×3, last empty).

clear; clc; close all;
set(groot,'DefaultAxesFontSize',20);

%% ── hardcoded per-run data (passed runs only) ────────────────────────────────
% Claude — 5/5 passed
xr_cl = [0.259, 1.606, 1.167, 2.651, 8.455];
yr_cl = [0.038, 1.039, 0.350, 1.714, 3.588];
pt_cl = [0.509, 4.100, 1.718, 4.853, 10.819];

% GPT-4o — 5/5 passed
xr_g4 = [0.173, 0.177, 0.494, 0.077, 0.330];
yr_g4 = [0.077, 0.224, 0.610, 0.191, 0.202];
pt_g4 = [0.422, 0.534, 1.199, 0.443, 0.729];

% GPT-4o-mini — 5/5 passed
xr_mi = [6.919, 73.270, 7.969, 7.350, 8.008];
yr_mi = [12.111, 20.222, 7.733, 6.284, 8.117];
pt_mi = [23.816, 84.296, 40.775, 39.352, 42.756];

% Gemini (base) — 3/5 passed (runs 1,5 were refusals)
xr_gb = [24.030, 5.434, 7.980];
yr_gb = [9.883,  4.355, 9.518];
pt_gb = [27.708, 15.812, 14.805];
n_tot_gb = 5;

% Gemini + Fix — 5/5 passed
xr_gf = [9.428, 9.030, 7.847, 7.793, 13.416];
yr_gf = [5.700, 6.213, 6.066, 9.711,  9.721];
pt_gf = [12.196, 12.008, 11.122, 14.858, 22.033];

%% ── panel specs: {xr, yr, pt, label, color, n_passed, n_total} ──────────────
specs = { ...
    xr_cl, yr_cl, pt_cl, 'Claude',        [0.20 0.55 0.85], 5, 5; ...
    xr_g4, yr_g4, pt_g4, 'GPT-4o',        [1.00 0.60 0.10], 5, 5; ...
    xr_mi, yr_mi, pt_mi, 'GPT-4o-mini',   [0.60 0.20 0.70], 5, 5; ...
    xr_gb, yr_gb, pt_gb, 'Gemini (base)', [0.85 0.18 0.18], 3, n_tot_gb; ...
    xr_gf, yr_gf, pt_gf, 'Gemini + Fix',  [0.25 0.72 0.38], 5, 5 };

n_panels = size(specs,1);
figure('Name','C6 — Square Mission Planning','Position',[60 60 1600 1000]);

for p = 1:n_panels
    xr    = specs{p,1};  yr    = specs{p,2};  pt    = specs{p,3};
    lbl   = specs{p,4};  clr   = specs{p,5};
    n_ok  = specs{p,6};  n_tot = specs{p,7};

    mx = mean(xr);  my = mean(yr);  mp = mean(pt);
    sq = min(mx,my) / max(mx,my);

    ax = subplot(2,3,p);
    hold on;

    for i = 1:n_ok
        rx = xr(i)/2;  ry = yr(i)/2;
        patch(ax, [-rx rx rx -rx], [-ry -ry ry ry], clr, ...
              'FaceAlpha',0.12,'EdgeColor',clr,'LineWidth',0.8);
    end

    rx = mx/2;  ry = my/2;
    patch(ax, [-rx rx rx -rx], [-ry -ry ry ry], clr, ...
          'FaceAlpha',0.0,'EdgeColor',clr,'LineWidth',3.0);

    plot(ax, 0, 0, '+', 'Color',[0.5 0.5 0.5],'MarkerSize',10,'LineWidth',1.5);

    axis(ax,'equal');
    set(ax,'Box','off','XGrid','on','YGrid','on','GridAlpha',.25,'GridLineStyle','--');
    xlabel(ax,'X (m)','FontSize',20);
    ylabel(ax,'Y (m)','FontSize',20);

    if n_ok < n_tot
        pass_str = sprintf('%d/%d passed', n_ok, n_tot);
    else
        pass_str = 'all passed';
    end
    title(ax, { lbl; sprintf('mean path %.1f m  |  sq = %.2f  |  %s', mp, sq, pass_str) }, ...
          'FontSize',18,'FontWeight','bold','Color',clr);
    hold off;
end

subplot(2,3,6);
axis off;

sgtitle({'C6 — Mission Planning: XY Bounding Box of Square-Pattern Attempt'; ...
         'Bold outline = mean; thin fills = individual runs; scale varies per model'}, ...
        'FontSize',22,'FontWeight','bold');
