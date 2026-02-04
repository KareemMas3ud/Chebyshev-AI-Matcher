% SECTION 1: Parameters & Dataset Generation
clear; clc; close all;

% --- Design Constants ---
N_design = 5;       % The main design is for 5 sections (for the dataset)
am = 0.05;          % Maximum ripple factor
Z0_const = 85;      % Characteristic Impedance (Ohms)
NumSamples = 250;   % Total size of the dataset

% --- Generate ZL (Load Impedance) Ranges ---
% Real Part: 280 down to 30 (Integers, Descending)
RL_values = round(linspace(280, 30, NumSamples))'; 
RL_values = sort(RL_values, 'descend'); 

% Imaginary Part: 190 down to -60 (Integers, Descending)
XL_values = round(linspace(190, -60, NumSamples))';
XL_values = sort(XL_values, 'descend');

% Combine into Complex ZL
ZL_complex = RL_values + 1j * XL_values;

% String Representation for Excel (e.g., "280+190i")
ZL_String = compose("%d%+di", RL_values, XL_values);

% SECTION 2: Chebyshev Transformer Calculation (N=5 Only for Dataset)
% Pre-allocate arrays
Z1 = zeros(NumSamples, 1);
Z2 = zeros(NumSamples, 1);
Z3 = zeros(NumSamples, 1);
Z4 = zeros(NumSamples, 1);
Z5 = zeros(NumSamples, 1);

for i = 1:NumSamples
    ZL = ZL_complex(i);
    
    % 1. Magnitude of mismatch
    Gamma_L_mag = abs((ZL - Z0_const) / (ZL + Z0_const));
    
    % 2. Chebyshev Constant S (for N=5)
    val_check = (1/N_design) * acosh((1/am) * Gamma_L_mag);
    
    % Check for real/valid S calculation (if mismatch < ripple, S=1)
    if ((1/am) * Gamma_L_mag) < 1
        S = 1; 
    else
        S = cosh(val_check);
    end
    
    % 3. Reflection Coefficients (Gamma_n) for N = 5
    Gamma0 = (am/2) * S^5;
    Gamma1 = (am/2) * (5*S^5 - 5*S^3);
    Gamma2 = (am/2) * (10*S^5 - 15*S^3 + 5*S);
    
    % Symmetry for N=5
    Gamma3 = Gamma2;
    Gamma4 = Gamma1;
    Gamma5 = Gamma0;
    
    % 4. Section Impedances (Zn)
    Z1(i) = Z0_const * exp(2 * Gamma0);
    Z2(i) = Z1(i) * exp(2 * Gamma1);
    Z3(i) = Z2(i) * exp(2 * Gamma2);
    Z4(i) = Z3(i) * exp(2 * Gamma3);
    Z5(i) = Z4(i) * exp(2 * Gamma4);
end

% SECTION 3: Create and Export Dataset
Z0_col = repmat(Z0_const, NumSamples, 1);

% Create Table with ALL columns
Dataset = table(Z0_col, ZL_String, RL_values, XL_values, Z1, Z2, Z3, Z4, Z5, ...
    'VariableNames', {'Z0', 'ZL_Complex', 'ZL_Real', 'ZL_Imag', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5'});

filename = 'Chebyshev_Dataset.xlsx';
writetable(Dataset, filename);
fprintf('Dataset successfully created and saved as %s\n', filename);
disp('First 5 rows of the dataset:');
disp(head(Dataset, 5));

% SECTION 4: Plotting Comparison (N = 1, 2, 3, 4, 5)
% We plot the response for Sample #1 to compare bandwidths.

sample_idx = 1; 
ZL_plot = ZL_complex(sample_idx);
Gamma_L_mag_plot = abs((ZL_plot - Z0_const) / (ZL_plot + Z0_const));

% Frequency Vector
theta = linspace(0, pi, 500); 
f_norm = 2 * theta / pi;

figure('Color', 'w', 'Position', [100, 100, 800, 600]);
hold on;
colors = lines(5); % Generate 5 distinct colors
legends = {};

% Loop through N=1 to N=5
for n_val = 1:5
    
    % 1. Calculate S for this specific N
    % Note: S changes as N changes to maintain the same ripple (am)
    val_check = (1/am) * Gamma_L_mag_plot;
    if val_check < 1
        S_n = 1;
    else
        S_n = cosh( (1/n_val) * acosh(val_check) );
    end
    
    % 2. Calculate Chebyshev Polynomial T_n(x)
    % Argument x = S * cos(theta)
    x = S_n * cos(theta);
    
    switch n_val
        case 1
            Tn = x;
        case 2
            Tn = 2*x.^2 - 1;
        case 3
            Tn = 4*x.^3 - 3*x;
        case 4
            Tn = 8*x.^4 - 8*x.^2 + 1;
        case 5
            Tn = 16*x.^5 - 20*x.^3 + 5*x;
    end
    
    % 3. Calculate Gamma Magnitude
    % |Gamma| = am * |Tn(x)|
    Gamma_mag_n = am * abs(Tn);
    
    % 4. Plot (Check if N=5 to make it BOLD)
    if n_val == 5
        % Make N=5 extra thick (LineWidth = 4)
        plot(f_norm, Gamma_mag_n, 'LineWidth', 4, 'Color', colors(n_val,:));
    else
        % Standard thickness for others
        plot(f_norm, Gamma_mag_n, 'LineWidth', 1.5, 'Color', colors(n_val,:));
    end
    
    legends{end+1} = ['N = ' num2str(n_val)];
end

% Formatting
yline(am, '--r', 'Ripple Limit (0.05)', 'LineWidth', 1.2);
title(['Chebyshev Response vs Number of Sections (N)']);
subtitle(['Z_L = ' num2str(real(ZL_plot)) ' + j' num2str(imag(ZL_plot)) ' \Omega']);
xlabel('Normalized Frequency (f/f0)');
ylabel('Reflection Coefficient |\Gamma|');
legend([legends, 'Ripple Limit'], 'Location', 'northeast');
grid on; grid minor;

% Adjust limits to see the behavior clearly
ylim([0, max(Gamma_mag_n)*1.2]); % Dynamic Y-limit
xlim([0, 2]);
set(gca, 'FontSize', 12);
hold off;