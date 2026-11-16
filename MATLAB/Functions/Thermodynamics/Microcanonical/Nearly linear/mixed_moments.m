function MM = mixed_moments(U,P,epsilon,alpha)
% MIXED_MOMENTS Normalized microcanonical moments of modal occupations.
%   MM = MIXED_MOMENTS(U,P,EPSILON,ALPHA) returns <prod J_k^ALPHA_k>.
%   WEIGHTED_VOLUME supplies the factorial-inclusive numerator and ordinary
%   volume. Their ratio is formed from signs and logarithms so intermediate
%   underflow or overflow does not destroy a representable moment.
%   Exact spectral edges use the finite normalized simplex limit.

    validateattributes(epsilon,{'numeric'}, ...
        {'real','vector','finite','nonempty'},mfilename,'epsilon',3);
    validateattributes(alpha,{'numeric'}, ...
        {'real','vector','finite','nonnegative','integer', ...
         'numel',numel(epsilon)},mfilename,'alpha',4);
    epsilon = double(epsilon(:).');
    alpha = double(alpha(:).');

    if ~isscalar(U) && ~isscalar(P) && ~isequal(size(U),size(P))
        error('mixed_moments:InputSize', ...
            'U and P must be scalars or arrays of equal size.')
    end
    [U_grid,P_grid,output_size] = expand_inputs(U,P);

    % Preserve the zero-order identity, including outside physical support.
    if all(alpha == 0)
        MM = ones(output_size);
        return
    end

    [D,~,d] = degeneracy(epsilon);
    [edge,edge_value] = spectral_edge_moment( ...
        U_grid,P_grid,epsilon,alpha,D,d);
    if all(edge,'all')
        MM = edge_value;
        return
    end

    [~,sign_numerator,log_numerator] = weighted_volume( ...
        U_grid,P_grid,epsilon,alpha,[0 0]);
    [~,sign_V,log_V] = weighted_volume( ...
        U_grid,P_grid,epsilon,[],[0 0]);
    sign_MM = sign_numerator.*sign_V;
    log_MM = log_numerator - log_V;

    % Nonzero-order moments outside support retain their zero convention.
    MM = zeros(output_size);
    nonzero = sign_MM ~= 0 & ~edge;
    MM(nonzero) = sign_MM(nonzero).*exp(log_MM(nonzero));
    MM(edge) = edge_value(edge);
end

function [U_grid,P_grid,output_size] = expand_inputs(U,P)
    if isscalar(U)
        output_size = size(P);
        U_grid = repmat(U,output_size);
    else
        output_size = size(U);
        U_grid = U;
    end
    if isscalar(P)
        P_grid = repmat(P,output_size);
    else
        P_grid = P;
    end
end

function [edge,value] = spectral_edge_moment( ...
        U,P,epsilon,alpha,D,d)
% At a spectral edge all power lies in its original degenerate subspace:
% <prod J_k^alpha_k> = P^A*Gamma(D)/Gamma(D+A)*prod Gamma(alpha_k+1).

    epsilon_min = min(epsilon);
    epsilon_max = max(epsilon);
    lower = epsilon_min.*P;
    upper = epsilon_max.*P;
    scale = max(abs(U),max(abs(lower),abs(upper)));
    tolerance = 32*eps(scale);
    near_lower = U < lower & U >= lower - tolerance;
    near_upper = U > upper & U <= upper + tolerance;
    U(near_lower) = lower(near_lower);
    U(near_upper) = upper(near_upper);

    lower_edge = isfinite(U) & isfinite(P) & P > 0 & U == lower;
    upper_edge = isfinite(U) & isfinite(P) & P > 0 ...
        & U == upper & ~lower_edge;
    edge = lower_edge | upper_edge;
    value = zeros(size(U));
    moment_order = sum(alpha);

    lower_index = find(epsilon == epsilon_min,1);
    lower_modes = d{lower_index};
    if all(alpha(setdiff(1:numel(epsilon),lower_modes)) == 0)
        log_coefficient = gammaln(D(lower_index)) ...
            - gammaln(D(lower_index) + moment_order) ...
            + sum(gammaln(alpha(lower_modes) + 1));
        value(lower_edge) = exp(log_coefficient ...
            + moment_order*log(P(lower_edge)));
    end

    upper_index = find(epsilon == epsilon_max,1);
    upper_modes = d{upper_index};
    if all(alpha(setdiff(1:numel(epsilon),upper_modes)) == 0)
        log_coefficient = gammaln(D(upper_index)) ...
            - gammaln(D(upper_index) + moment_order) ...
            + sum(gammaln(alpha(upper_modes) + 1));
        value(upper_edge) = exp(log_coefficient ...
            + moment_order*log(P(upper_edge)));
    end
end
