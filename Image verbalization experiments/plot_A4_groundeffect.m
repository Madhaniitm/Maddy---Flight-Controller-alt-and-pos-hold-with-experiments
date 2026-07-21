%% plot_A4_groundeffect.m
% A4 — Ground effect thrust multiplier vs normalised altitude (z/R)
%
% Compares simulator model (He 2019) against three published empirical models:
%   Cheeseman-Bennett (1955), Li (2015, rho=3.4), Kan (2019, Crazyflie)
%
% Data: A4_ground_effect.csv (embedded)
% Figure reference: fig3_5_a4_groundeffect.pdf

clear; clc; close all;
set(groot, 'DefaultAxesFontSize', 25);

%% ── Data (from A4_ground_effect.csv) ────────────────────────────────────────
z_over_R = [0.0  0.435 0.87  1.304 2.174 3.478 4.348 5.217 6.522 8.696 10.87  13.043 17.391 21.739 32.609 43.478 65.217 86.957 130.435];
k_sim    = [1.37    1.273   1.20142 1.14862 1.08091 1.0325  1.01769 1.00963 1.00387 1.00085 1.00018 1.00004 1.0     1.0     1.0     1.0     1.0     1.0     1.0    ];
k_cb     = [Inf     1.49393 1.0901  1.03814 1.0134  1.00519 1.00332 1.0023  1.00147 1.00083 1.00053 1.00037 1.00021 1.00013 1.00006 1.00003 1.00001 1.00001 1.0    ];
k_li     = [Inf     Inf     1.39088 1.14273 1.04708 1.01788 1.01137 1.00787 1.00502 1.00282 1.0018  1.00125 1.0007  1.00045 1.0002  1.00011 1.00005 1.00003 1.00001];
k_kan    = [Inf     1.59566 1.21455 1.12499 1.06232 1.03004 1.01971 1.01294 1.00626 0.99967 0.99575 0.99316 0.98993 0.98801 0.98546 0.98419 0.98292 0.98229 0.98165];

%% ── Colours ──────────────────────────────────────────────────────────────────
cSim = [0.00 0.33 0.71];   % blue   — simulator (He 2019)
cCB  = [1.00 0.55 0.00];   % orange — Cheeseman-Bennett 1955
cLi  = [0.10 0.55 0.10];   % green  — Li 2015
cKan = [0.85 0.10 0.10];   % red    — Kan 2019 (Crazyflie)

%% ── Figure ───────────────────────────────────────────────────────────────────
XLIM = [0 6];
YLIM = [0.97 2.05];
axP  = [0.10 0.13 0.86 0.74];   % [left bottom width height] in figure units

figure('Name','A4 — Ground effect','Position',[60 60 1200 680]);
ax = axes('Position', axP);
hold on;

% Per-model masks: each curve filtered independently, clipped to x ≤ 6
vsim = z_over_R >= 0.435 & z_over_R <= XLIM(2);   % sim starts at z=10mm (leg clearance)
vcb  = isfinite(k_cb)  & z_over_R <= XLIM(2);
vli  = isfinite(k_li)  & z_over_R <= XLIM(2);
vkan = isfinite(k_kan) & z_over_R <= XLIM(2);

% Shaded CB-undefined region (h < R/4 = 5.75 mm → z/R < 0.25, R = 23 mm)
patch(ax, [0 0.25 0.25 0], [YLIM(1) YLIM(1) YLIM(2) YLIM(2)], ...
      [1.00 0.85 0.70], 'EdgeColor','none', 'FaceAlpha',0.40, ...
      'DisplayName','CB undefined  (h < R/4 = 5.8 mm)');
text(ax, 0.04, 1.80, {'CB undefined'; '(h < R/4 = 5.8 mm)'}, ...
     'FontSize',16, 'Color',[0.65 0.40 0.10], 'Rotation',90, 'VerticalAlignment','top');

% k_ge = 1 reference
yline(ax, 1.0, ':', 'Color',[0.65 0.65 0.65], 'LineWidth',1.2, 'HandleVisibility','off');

% z = 5R vertical marker (Kan 2019 GE limit)
xline(ax, 5.0, '--', 'Color',[0.60 0.60 0.60], 'LineWidth',1.2, 'HandleVisibility','off');
text(ax, 5.06, 1.09, {'z = 5R'; '(GE limit'; 'Kan 2019)'}, ...
     'FontSize',16, 'Color',[0.50 0.50 0.50]);

% Four model curves
plot(ax, z_over_R(vcb),  k_cb(vcb),  '--', 'Color',cCB,  'LineWidth',2.0, ...
     'DisplayName','Cheeseman-Bennett 1955:  1/(1-(R/4h)^2)  [helicopter, has singularity]');
plot(ax, z_over_R(vli),  k_li(vli),  '-.', 'Color',cLi,  'LineWidth',2.0, ...
     'DisplayName','Li et al. 2015:  1/(1-3.4(R/4h)^2)  [quadrotor, \rho = 3.4]');
plot(ax, z_over_R(vkan), k_kan(vkan), ':',  'Color',cKan, 'LineWidth',2.5, ...
     'DisplayName','Kan et al. 2019:  1/(1.02-0.171\cdotR/z)  [Crazyflie validated \surd]');
plot(ax, z_over_R(vsim), k_sim(vsim), '-',  'Color',cSim, 'LineWidth',2.8, ...
     'DisplayName','Sim — He et al. 2019:  1 + Ca\cdotexp(-Cb\cdotz/R)  [THIS DRONE]');

% Blue circle markers at sim data points
plot(ax, z_over_R(vsim), k_sim(vsim), 'o', 'Color',cSim, ...
     'MarkerFaceColor',cSim, 'MarkerSize',8, 'HandleVisibility','off');

% Arrow annotation: "Sim starts at z/R = 0.43"
% Tail at ~(0.90, 1.39) in data; tip at first sim point (0.435, 1.273)
x_tail = 0.90;  k_tail = 1.39;
x_tip  = 0.435; k_tip  = 1.273;
xn_tail = axP(1) + axP(3)*(x_tail - XLIM(1)) / (XLIM(2) - XLIM(1));
yn_tail = axP(2) + axP(4)*(k_tail - YLIM(1)) / (YLIM(2) - YLIM(1));
xn_tip  = axP(1) + axP(3)*(x_tip  - XLIM(1)) / (XLIM(2) - XLIM(1));
yn_tip  = axP(2) + axP(4)*(k_tip  - YLIM(1)) / (YLIM(2) - YLIM(1));
annotation('arrow', [xn_tail xn_tip], [yn_tail yn_tip], ...
           'Color','k', 'LineWidth',1.2, 'HeadLength',8, 'HeadWidth',6);
text(ax, x_tail + 0.05, k_tail, {'Sim starts at z/R = 0.43'; '(z = 10 mm, leg clearance)'}, ...
     'FontSize',18, 'Color','k');

set(ax, 'XLim',XLIM, 'YLim',YLIM, 'Box','off', ...
        'YGrid','on', 'XGrid','off', 'GridAlpha',0.25, 'GridLineStyle','--');
xlabel(ax, 'Normalised height  z / R_{rotor}', 'FontSize',25, 'Interpreter','tex');
ylabel(ax, 'Thrust multiplier  k_{ge}',         'FontSize',25, 'Interpreter','tex');
legend(ax, 'FontSize',17, 'Location','northeast', 'Box','off', 'Interpreter','tex');
hold off;

%% ── Export ───────────────────────────────────────────────────────────────────
exportgraphics(gcf,'../Thesis/figures/fig3_5_a4_groundeffect.pdf', ...
    'ContentType','vector','Resolution',300);
fprintf('Saved fig3_5_a4_groundeffect.pdf\n');
