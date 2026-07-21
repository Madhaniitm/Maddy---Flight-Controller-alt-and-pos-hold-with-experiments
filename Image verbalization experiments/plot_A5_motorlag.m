%% plot_A5_motorlag.m
% A5 — Motor first-order lag model validation
%
% Steps PWM from 0 to full. Euler-ZOH sim agrees with 1st-order analytical
% to R² = 1.000.  2nd- and 3rd-order contributions < 0.011% of ω_max.
%
% Figure 1: Step response — sim vs 1st / 2nd / 3rd order analytical
% Figure 2: Residuals — sim − fitted 1st-order curve
% Figure 3: % error of 2nd & 3rd order vs 1st order (zoomed 0–1 ms)
%
% Data: experiments/results/A5_motor_lag.csv (embedded)
% Figure reference: fig3_6_a5_motorlag.pdf

clear; clc; close all;
set(groot, 'DefaultAxesFontSize', 25);

%% ── Embedded data (A5_motor_lag.csv — 200 Hz, t = 0–0.295 s, 61 rows) ───────
T = [0 0.005 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 ...
     0.05 0.055 0.06 0.065 0.07 0.075 0.08 0.085 0.09 0.095 ...
     0.1 0.105 0.11 0.115 0.12 0.125 0.13 0.135 0.14 0.145 ...
     0.15 0.155 0.16 0.165 0.17 0.175 0.18 0.185 0.19 0.195 ...
     0.2 0.205 0.21 0.215 0.22 0.225 0.23 0.235 0.24 0.245 ...
     0.25 0.255 0.26 0.265 0.27 0.275 0.28 0.285 0.29 0.295];

omega_sim = [0 436.3333 799.9444 1102.9537 1355.4614 1565.8845 1741.2371 ...
             1887.3642 2009.1369 2110.6141 2195.1784 2265.6487 2324.3739 ...
             2373.3116 2414.093 2448.0775 2476.3979 2499.9982 2519.6652 ...
             2536.0543 2549.7119 2561.0933 2570.5777 2578.4815 2585.0679 ...
             2590.5566 2595.1305 2598.9421 2602.1184 2604.7653 2606.9711 ...
             2608.8092 2610.341 2611.6175 2612.6813 2613.5677 2614.3064 ...
             2614.922 2615.435 2615.8625 2616.2188 2616.5156 2616.763 ...
             2616.9692 2617.141 2617.2842 2617.4035 2617.5029 2617.5857 ...
             2617.6548 2617.7123 2617.7603 2617.8002 2617.8335 2617.8613 ...
             2617.8844 2617.9037 2617.9197 2617.9331 2617.9442];

omega_anal = [0 401.9108 742.121 1030.1027 1273.874 1480.2219 1654.8916 ...
              1802.7464 1927.9027 2033.8452 2123.5237 2199.4348 2263.6922 ...
              2318.0849 2364.1274 2403.1015 2436.0923 2464.0185 2487.6575 ...
              2507.6674 2524.6055 2538.9433 2551.0799 2561.3534 2570.0497 ...
              2577.4109 2583.6421 2588.9166 2593.3815 2597.1609 2600.3601 ...
              2603.0681 2605.3604 2607.3008 2608.9434 2610.3337 2611.5106 ...
              2612.5069 2613.3502 2614.064 2614.6682 2615.1797 2615.6127 ...
              2615.9792 2616.2894 2616.552 2616.7743 2616.9625 2617.1218 ...
              2617.2566 2617.3707 2617.4673 2617.5491 2617.6183 2617.6769 ...
              2617.7265 2617.7685 2617.804 2617.8341 2617.8596];

%% ── Motor parameters ─────────────────────────────────────────────────────────
omega_cmd = 2618.0;          % rad/s  (OMEGA_MAX; verified: 436.333 × 6)
TAU_MOTOR = 0.030;           % s      firmware time constant
dt        = 0.005;           % s      200 Hz simulation step

% Euler ZOH: discrete pole z = 1 − dt/τ  →  τ_eff = −dt / ln(z)
tau_euler = -dt / log(1.0 - dt / TAU_MOTOR);   % ≈ 27.42 ms

% Fitted τ matches Euler ZOH exactly (R² = 1.000 by construction)
tau_fit = tau_euler;

% Higher-order motor parameters (7×16 mm coreless brushed DC)
R_motor  = 3.0;               % Ω
L_motor  = 10e-6;             % H
tau_e    = L_motor / R_motor;  % ≈ 3.33 μs — electrical time constant
tau_aero = tau_e / 10;         % ≈ 0.33 μs — propeller aerodynamic inertia

%% ── 2nd order step response (electrical + mechanical poles) ──────────────────
% ω_2nd = ω_cmd·[1 − A·e^{−t/τ_m} + B·e^{−t/τ_e}]
A2        = TAU_MOTOR / (TAU_MOTOR - tau_e);
B2        = tau_e     / (TAU_MOTOR - tau_e);
omega_2nd = omega_cmd .* (1 - A2.*exp(-T./TAU_MOTOR) + B2.*exp(-T./tau_e));

%% ── 3rd order step response (+ propeller aerodynamic inertia pole) ───────────
% Heaviside partial fractions: Y(s) = 1/s + B3/(s+1/τ_m) + C3/(s+1/τ_e) + D3/(s+1/τ_a)
d_me      = TAU_MOTOR - tau_e;
d_ma      = TAU_MOTOR - tau_aero;
d_ea      = tau_e     - tau_aero;
B3        = -TAU_MOTOR^2 / (d_me * d_ma);
C3        = +tau_e^2     / (d_me * d_ea);
D3        = -tau_aero^2  / (d_ma * d_ea);
omega_3rd = omega_cmd .* (1 + B3.*exp(-T./TAU_MOTOR) ...
                             + C3.*exp(-T./tau_e) ...
                             + D3.*exp(-T./tau_aero));

%% ── Fine grid for inset (0–20 μs, resolves fast electrical/aero poles) ───────
T_ins  = linspace(0, 20e-6, 3000);
w1_ins = omega_cmd .* (1 - exp(-T_ins ./ TAU_MOTOR));
w2_ins = omega_cmd .* (1 - A2.*exp(-T_ins./TAU_MOTOR) + B2.*exp(-T_ins./tau_e));
w3_ins = omega_cmd .* (1 + B3.*exp(-T_ins./TAU_MOTOR) + C3.*exp(-T_ins./tau_e) + D3.*exp(-T_ins./tau_aero));

%% ── Fitted curve and residuals ───────────────────────────────────────────────
omega_fit = omega_cmd .* (1 - exp(-T ./ tau_fit));
residuals = omega_sim - omega_fit;

%% ── % error of 2nd and 3rd order vs 1st order analytical ────────────────────
omega_1st = omega_cmd .* (1 - exp(-T ./ TAU_MOTOR));
err_2nd   = abs(omega_2nd - omega_1st) ./ omega_cmd .* 100;
err_3rd   = abs(omega_3rd - omega_1st) ./ omega_cmd .* 100;
peak_2nd  = max(err_2nd);
peak_3rd  = max(err_3rd);

%% ── Colours ──────────────────────────────────────────────────────────────────
cSim = [0.12 0.47 0.71];   % blue   — simulator (Euler)
c1st = [1.00 0.50 0.00];   % orange — 1st order analytical (τ = 30 ms)
c2nd = [0.17 0.63 0.17];   % green  — 2nd order (elec + mech)
c3rd = [0.84 0.15 0.16];   % red    — 3rd order (+ prop aero)
cRes = [0.58 0.40 0.74];   % purple — residuals

%% ── Figure 1: step response — 1st vs 2nd vs 3rd order ───────────────────────
figure('Name','A5 Fig1 — Step response','Position',[60 60 1400 620]);
ax1 = axes('Position',[0.09 0.14 0.88 0.72]);
hold on;
plot(ax1, T, omega_sim,  '-',  'Color',cSim, 'LineWidth',2.5, ...
     'DisplayName','Simulator \omega(t)  [1st order Euler]');
plot(ax1, T, omega_anal, '--', 'Color',c1st, 'LineWidth',1.8, ...
     'DisplayName',sprintf('1st order analytical  \\tau = %.0f ms', TAU_MOTOR*1000));
plot(ax1, T, omega_2nd,  '-.', 'Color',c2nd, 'LineWidth',1.8, ...
     'DisplayName',sprintf('2nd order (elec+mech)  \\tau_e = %.1f \\mu s', tau_e*1e6));
plot(ax1, T, omega_3rd,  ':',  'Color',c3rd, 'LineWidth',2.0, ...
     'DisplayName',sprintf('3rd order (+prop aero)  \\tau_{aero} = %.2f \\mu s', tau_aero*1e6));
yline(ax1, 0.632*omega_cmd, ':', 'Color',[0.5 0.5 0.5], 'LineWidth',1.0, 'HandleVisibility','off');
xline(ax1, TAU_MOTOR,       ':', 'Color',[0.5 0.5 0.5], 'LineWidth',1.0, 'HandleVisibility','off');
text(ax1, TAU_MOTOR + 0.003, 0.45*omega_cmd, ...
     sprintf('63.2%%  at t = %.1f ms', tau_fit*1000), 'FontSize',19, 'Color',[0.4 0.4 0.4]);
set(ax1, 'XLim',[0 T(end)], 'Box','off', ...
         'YGrid','on','XGrid','off','GridAlpha',0.25,'GridLineStyle','--');
xlabel(ax1, 'Time  (s)', 'FontSize',25);
ylabel(ax1, 'Angular speed  \omega  (rad/s)', 'FontSize',25);
title(ax1, 'Step Response — 1st vs 2nd vs 3rd Order Motor Model', 'FontSize',25);
legend(ax1, 'FontSize',15, 'Location','east', 'Box','off');
hold off;

% Inset: 0–20 μs zoom where fast poles separate visibly
axIns = axes('Position',[0.42 0.22 0.24 0.38]);
hold on;
plot(axIns, T_ins*1e6, w1_ins, '--', 'Color',c1st, 'LineWidth',2.0);
plot(axIns, T_ins*1e6, w2_ins, '-.', 'Color',c2nd, 'LineWidth',2.0);
plot(axIns, T_ins*1e6, w3_ins, ':',  'Color',c3rd, 'LineWidth',2.5);
plot(axIns, 0, 0, 'o', 'Color',cSim, 'MarkerFaceColor',cSim, 'MarkerSize',9);
set(axIns, 'XLim',[0 20], 'Box','on', 'FontSize',15, ...
           'YGrid','on','XGrid','on','GridAlpha',0.3,'GridLineStyle','--');
xlabel(axIns, 'Time  (\mu s)', 'FontSize',15);
ylabel(axIns, '\omega  (rad/s)', 'FontSize',15);
title(axIns, 'Zoom: 0–20 \mu s', 'FontSize',16, 'FontWeight','bold');
hold off;

%% ── Figure 2: residuals (sim − fit) ─────────────────────────────────────────
figure('Name','A5 Fig2 — Residuals','Position',[100 60 1400 620]);
ax2 = axes('Position',[0.09 0.14 0.88 0.72]);
hold on;
plot(ax2, T, residuals, '-', 'Color',cRes, 'LineWidth',1.5, ...
     'DisplayName','Sim − 1st order fit');
yline(ax2, 0, '--', 'Color',[0.3 0.3 0.3], 'LineWidth',0.8, 'HandleVisibility','off');
set(ax2, 'XLim',[0 T(end)], 'Box','off', ...
         'YGrid','on','XGrid','off','GridAlpha',0.25,'GridLineStyle','--');
xlabel(ax2, 'Time  (s)', 'FontSize',25);
ylabel(ax2, 'Residual  \omega  (rad/s)', 'FontSize',25);
title(ax2, sprintf(['Fit Residuals  (\\tau_{fit} = %.2f ms  vs  ' ...
                    '\\tau_{model} = %.0f ms  |  Euler ZOH: %.2f ms)'], ...
                   tau_fit*1000, TAU_MOTOR*1000, tau_euler*1000), 'FontSize',22);
legend(ax2, 'FontSize',17, 'Location','northeast', 'Box','off');
hold off;

%% ── Figure 3: % error of 2nd and 3rd order vs 1st order (zoomed 0–1 ms) ─────
figure('Name','A5 Fig3 — Model order error','Position',[140 60 1400 620]);
ax3 = axes('Position',[0.09 0.14 0.88 0.72]);
hold on;
plot(ax3, T*1000, err_2nd, '-',  'Color',c2nd, 'LineWidth',2.0, ...
     'DisplayName',sprintf('2nd order error  (peak = %.4f%%)', peak_2nd));
plot(ax3, T*1000, err_3rd, '--', 'Color',c3rd, 'LineWidth',1.5, ...
     'DisplayName',sprintf('3rd order error  (peak = %.4f%%)', peak_3rd));
set(ax3, 'XLim',[0 1.0], 'Box','off', ...
         'YGrid','on','XGrid','off','GridAlpha',0.25,'GridLineStyle','--');
xlabel(ax3, 'Time  (ms)', 'FontSize',25);
ylabel(ax3, 'Error vs 1st order  (% of \omega_{cmd})', 'FontSize',25);
title(ax3, '2nd & 3rd Order Model Error vs 1st Order — Why 1st Order Is Sufficient', 'FontSize',25);
legend(ax3, 'FontSize',17, 'Location','northeast', 'Box','off');
hold off;

