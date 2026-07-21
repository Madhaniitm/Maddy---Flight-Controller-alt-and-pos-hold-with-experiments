%% plot_B5_hover_soc.m
% B5 — Hover throttle vs battery SOC (thesis version)
% Two figures: required hover PWM vs SOC, terminal voltage vs SOC

clear; clc; close all;
set(groot, 'DefaultAxesFontSize', 25);

%% ── Data (all 9 SOC points) ─────────────────────────────────────────────────
soc      = [100 90 80 70 60 50 40 30 20];
thr_frac = [0.536582 0.559736 0.584401 0.610735 0.638920 0.669162 0.701704 0.736829 0.774870];
hover_pwm= [1536 1559 1584 1610 1638 1669 1701 1736 1774];
v_term   = [4.1072 3.9844 3.8613 3.7381 3.6146 3.4909 3.3668 3.2425 3.1178];

BAT_V_EMPTY = 3.0;

%% ── A6 model prediction (V^2 scaling) ───────────────────────────────────────
thr_100 = thr_frac(1);
vt_100  = v_term(1);
a6_pwm  = thr_100 .* (vt_100 ./ v_term) * 1000 + 1000;

delta_pwm = hover_pwm(end) - hover_pwm(1);

%% ── Figure 1 — Hover PWM vs SOC ─────────────────────────────────────────────
fig1 = figure('Name','B5 — Hover PWM vs SOC','Position',[60 60 840 1080]);
ax1 = axes(fig1);  hold on;
plot(ax1, soc, hover_pwm, '-o', 'Color','blue', 'LineWidth',2.0, 'MarkerSize',8, ...
     'DisplayName','Analytical hover PWM (this exp)');
plot(ax1, soc, a6_pwm, '--^', 'Color',[1.00 0.55 0.10], 'LineWidth',2.0, 'MarkerSize',7, ...
     'DisplayName','A6 model prediction (V^2 scaling) [Ref 1, 2]');
text(ax1, 50, (hover_pwm(1)+hover_pwm(end))/2, ...
     sprintf('\DeltaPWM=%d over SOC range', delta_pwm), ...
     'FontSize',22, 'Color',[0.00 0.00 0.55]);
set(ax1,'XDir','reverse','Box','off','YGrid','on','XGrid','off','GridAlpha',.25,'GridLineStyle','--');
xlabel(ax1,'Battery SOC  (%)','FontSize',25);
ylabel(ax1,'Hover throttle  (PWM)','FontSize',25);
title(ax1,'Required Hover PWM vs SOC','FontSize',25);
leg1 = legend(ax1,'FontSize',25,'NumColumns',1);  leg1.Location='southoutside';
hold off;

%% ── Figure 2 — Terminal Voltage vs SOC ──────────────────────────────────────
fig2 = figure('Name','B5 — Terminal Voltage vs SOC','Position',[80 60 840 1080]);
ax2 = axes(fig2);  hold on;
plot(ax2, soc, v_term, '-s', 'Color',[1.00 0.55 0.10], 'LineWidth',2.0, 'MarkerSize',8, ...
     'DisplayName','V_{term} at hover current');
yline(ax2, BAT_V_EMPTY, '--', 'Color','red', 'LineWidth',1.5, ...
      'DisplayName', sprintf('V_{empty} = %.1f V', BAT_V_EMPTY));
set(ax2,'XDir','reverse','Box','off','YGrid','on','XGrid','off','GridAlpha',.25,'GridLineStyle','--');
xlabel(ax2,'Battery SOC  (%)','FontSize',25);
ylabel(ax2,'Terminal voltage  V_{term}  (V)','FontSize',25);
title(ax2,'Terminal Voltage vs SOC at Hover Current [Ref 1]','FontSize',25);
leg2 = legend(ax2,'FontSize',25,'NumColumns',1);  leg2.Location='southoutside';
hold off;
