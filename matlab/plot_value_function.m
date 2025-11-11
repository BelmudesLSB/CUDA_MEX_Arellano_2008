function plot_value_function(solution)
% =====================================================================
% Plots the value function V(y,b) for selected income levels.
% Produces a clean white-background, LaTeX-styled figure.
%
% INPUT:
%   solution.V        : flattened value function (Ny*Nb x 1)
%   solution.b_grid   : bond grid (Nb x 1)
%   solution.y_grid   : income grid (Ny x 1)
% =====================================================================

    % Extract fields
    V = solution.V;
    b_grid = solution.b_grid;
    y_grid = solution.y_grid;

    % Dimensions
    Ny = length(y_grid);
    Nb = length(b_grid);

    if numel(V) ~= Ny * Nb
        error('V size mismatch: expected Ny*Nb = %d, got %d.', Ny*Nb, numel(V));
    end

    % --- Reshape V(y,b)
    Vmat = reshape(V, [Nb, Ny])';  % rows = y, columns = b

    % --- Select 3 representative income states
    mid_idx  = round(Ny / 2);
    low_idx  = max(mid_idx - 1, 1);
    high_idx = min(mid_idx + 1, Ny);
    y_sel    = [y_grid(low_idx), y_grid(mid_idx), y_grid(high_idx)];

    % --- Create figure
    figure('Color','w'); hold on; box on;
    set(gca, 'TickLabelInterpreter', 'latex', ...
             'FontSize', 12, ...
             'LineWidth', 1);

    % --- Plot curves
    plt1 = plot(b_grid, Vmat(low_idx,:),  'LineWidth', 1.8);
    plt2 = plot(b_grid, Vmat(mid_idx,:),  'LineWidth', 1.8);
    plt3 = plot(b_grid, Vmat(high_idx,:), 'LineWidth', 1.8);

    % --- Labels and title with LaTeX
    xlabel('$b$ (bond holdings)', 'Interpreter','latex', 'FontSize',14);
    ylabel('$V(y,b)$', 'Interpreter','latex', 'FontSize',14);
    title('Value Function for Selected Income Levels', ...
          'Interpreter','latex', 'FontSize',15);

    % --- Legend
    legend([plt1, plt2, plt3], ...
        {sprintf('$y = %.3f$ (low)', y_sel(1)), ...
         sprintf('$y = %.3f$ (mid)', y_sel(2)), ...
         sprintf('$y = %.3f$ (high)', y_sel(3))}, ...
        'Interpreter','latex', 'Location','best', 'Box','off');

    grid on;

    hold off;

    % --- Save figure automatically in /figures folder
figDir = fullfile(pwd, 'figures');
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

saveas(gcf, fullfile(figDir, 'value_function.png'));

end
