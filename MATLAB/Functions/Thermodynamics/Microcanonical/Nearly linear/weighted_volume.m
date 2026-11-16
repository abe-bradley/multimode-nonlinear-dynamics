function [dW,sign_dW,log_dW] = weighted_volume(U,P,epsilon,alpha,dO)
%WEIGHTED_VOLUME Monomial-weighted microcanonical volume and derivatives.
%   [DW,SIGN_DW,LOG_DW] = WEIGHTED_VOLUME(U,P,EPSILON,ALPHA,DO) evaluates
%
%     d_U^DO(1) d_P^DO(2) integral prod(J.^ALPHA)
%         * delta(U-EPSILON*J) * delta(P-sum(J)) dJ.
%
%   ALPHA contains one nonnegative integer per mode. Empty or omitted ALPHA
%   selects the ordinary volume. Empty or omitted DO selects [0 0]. U and P
%   may be scalars or equal-sized arrays, with either scalar expanded.
%   Derivatives hold EPSILON and ALPHA fixed, in the input U/P coordinates.
%   At spectral knots, derivatives retain the piecewise-polynomial convention;
%   distributional derivatives beyond the available smoothness are excluded.
%
%   SIGN_DW and LOG_DW are sign(DW) and log(abs(DW)), including the modal
%   factorial prefactor. They remain usable when the raw double is zero or
%   infinite solely because of its magnitude. Signed-log sums, energy
%   centering/reflection, and adaptive variable precision limit cancellation.
%   Adaptive recovery requires the Symbolic Math Toolbox.
%
%   Internally, augmented pole orders g_n=D_n+sum(ALPHA in level n) select
%   one factorial-free integral. A bounded cache shares that integral across
%   modal labels and derivatives. Scalar distinct-spectrum first moments
%   retain a shared all-mode variable-precision fallback.

    validateattributes(epsilon,{'numeric'},{'real','finite','vector','nonempty'}, ...
        mfilename,'epsilon');
    epsilon = double(epsilon(:).');
    if nargin < 4 || isempty(alpha)
        alpha = zeros(size(epsilon));
    end
    if nargin < 5 || isempty(dO)
        dO = [0 0];
    end
    validateattributes(alpha,{'numeric'}, ...
        {'real','finite','vector','nonnegative','integer','numel',numel(epsilon)}, ...
        mfilename,'alpha');
    validateattributes(dO,{'numeric'}, ...
        {'real','finite','vector','nonnegative','integer','numel',2}, ...
        mfilename,'dO');
    validateattributes(U,{'numeric'},{'real'},mfilename,'U');
    validateattributes(P,{'numeric'},{'real'},mfilename,'P');
    alpha = double(alpha(:).');
    dO = double(dO(:).');
    if ~isscalar(U) && ~isscalar(P) && ~isequal(size(U),size(P))
        error('weighted_volume:InputSize', ...
            'U and P must be scalars or arrays of equal size.')
    end
    [U,P,output_size] = expand_inputs(double(U),double(P));

    [sign_dW,log_dW] = cached_integral(U,P,epsilon,alpha,dO);
    log_dW = log_dW + sum(gammaln(alpha + 1));
    dW = zeros(output_size);
    nonzero = sign_dW ~= 0;
    dW(nonzero) = sign_dW(nonzero).*exp(log_dW(nonzero));
end

function [sign_value,log_value] = cached_integral(U,P,epsilon,alpha,dO)
% Cache Phi_g, without prod(alpha!), in the original input coordinates.
    persistent context keys signs logs first_sign first_log D f groups g_base energy_center
    if isempty(context) || ~isequaln(context.U,U) ...
            || ~isequaln(context.P,P) || ~isequaln(context.epsilon,epsilon)
        context = struct('U',U,'P',P,'epsilon',epsilon);
        [D,f,groups] = degeneracy(epsilon);
        g_base = zeros(size(epsilon));
        g_base(f) = D(f);
        energy_center = mean(epsilon);
        [~,~,centered_groups] = degeneracy(epsilon - energy_center);
        % Subtraction can round a gap across the degeneracy tolerance.
        % Keep the original coordinates if centering changes any group.
        if ~isequal(groups,centered_groups)
            energy_center = 0;
        end
        keys = zeros(0,numel(epsilon) + 2);
        signs = {};
        logs = {};
        first_sign = [];
        first_log = [];
    end
    g = g_base;
    if any(alpha)
        for nn = f
            g(nn) = g(nn) + sum(alpha(groups{nn}));
        end
    end
    key = [g,dO];
    hit = find(all(keys == key,2),1);
    if ~isempty(hit)
        sign_value = signs{hit};
        log_value = logs{hit};
        keys(hit,:) = [];
        signs(hit) = [];
        logs(hit) = [];
    else
        first_moment = isscalar(U) && isscalar(P) && all(D(f) == 1) ...
            && sum(g) == numel(epsilon) + 1 && isequal(dO,[0 0]);
        if first_moment && ~isempty(first_log)
            mode = find(g > D,1);
            sign_value = first_sign(mode);
            log_value = first_log(mode);
        else
            edge_orders = zeros(size(epsilon));
            for nn = f
                edge_orders(groups{nn}) = g(nn);
            end
            [sign_value,log_value,batch_sign,batch_log] = evaluate_integral( ...
                U,P,epsilon,g,dO,D,f,edge_orders,first_moment,energy_center);
            if ~isempty(batch_log)
                first_sign = batch_sign;
                first_log = batch_log;
            end
        end
    end
    keys(end+1,:) = key;
    signs{end+1} = sign_value;
    logs{end+1} = log_value;
    if size(keys,1) > 128
        keys(1,:) = [];
        signs(1) = [];
        logs(1) = [];
    end
end

function [sign_value,log_value,first_sign,first_log] = evaluate_integral( ...
        U,P,epsilon,g,dO,D,f,edge_orders,first_moment,energy_center)
    epsilon_centered = epsilon - energy_center;
    U_centered = U - energy_center.*P;
    reflect = U_centered > 0;
    sign_value = zeros(size(U));
    log_value = -Inf(size(U));
    first_sign = [];
    first_log = [];
    for reflected = [false true]
        selected = reflect == reflected;
        if ~any(selected,'all')
            continue
        end
        direction = 1 - 2*reflected;
        [s,l,first_sign,first_log] = integral_centered( ...
            direction*U_centered(selected),P(selected), ...
            direction*epsilon_centered,direction*epsilon,g,dO,D,f, ...
            edge_orders,first_moment);
        sign_value(selected) = s*((-1)^(reflected*dO(1)));
        log_value(selected) = l;
    end
end

function [sign_value,log_value,first_sign,first_log] = integral_centered( ...
        U,P,epsilon,epsilon_derivative,g,dO,D,f,edge_orders,first_moment)
    [U,support] = physical_support(U,P,epsilon,edge_orders,dO,sum(g));
    sign_value = zeros(size(U));
    log_value = -Inf(size(U));
    first_sign = [];
    first_log = [];
    if ~any(support,'all')
        return
    end
    % Eligible first moments must reach the batch fallback before any
    % separate augmented-weight VPA recovery is attempted for each mode.
    [~,sign_w,log_w,weights_reliable] = ...
        compute_weights(epsilon,g,~first_moment);
    log_positive = -Inf(size(U));
    log_negative = -Inf(size(U));
    g_tot = sum(g);
    for nn = f
        Qn = U - epsilon(nn).*P;
        Q = max(Qn,0);
        for mm = 1:g(nn)
            ii = mm - 1;
            jj = g_tot - ii - 2;
            for pp = 0:dO(2)
                II = dO(2) - pp;
                JJ = dO(1) + pp;
                if II > ii || JJ > jj
                    continue
                end
                power_P = ii - II;
                power_Q = jj - JJ;
                [factor_sign,factor_log] = ...
                    derivative_factor(dO(2),pp,epsilon_derivative(nn));
                coefficient_sign = sign_w(nn,mm)*factor_sign;
                coefficient_log = log_w(nn,mm) + factor_log ...
                    - gammaln(power_P + 1) - gammaln(power_Q + 1);
                if JJ == 0
                    active = support & Qn >= 0;
                else
                    active = support & Qn > 0;
                end
                [log_positive,log_negative] = accumulate_term( ...
                    log_positive,log_negative,coefficient_sign, ...
                    coefficient_log,P,power_P,Q,power_Q,active);
            end
        end
    end
    [sign_value,log_value,unresolved] = ...
        resolve_signed_logs(log_positive,log_negative);
    has_terms = isfinite(log_positive) | isfinite(log_negative);
    fallback = unresolved | (~weights_reliable & has_terms);
    if isequal(dO,[0 0])
        fallback = fallback | (has_terms & sign_value <= 0);
    end
    if any(fallback,'all')
        if first_moment
            [first_sign,first_log] = adaptive_vpa( ...
                @() first_moments_vpa_once(U,P,epsilon), ...
                max(64,ceil(numel(epsilon)/10)));
            mode = find(g > D,1);
            sign_value(fallback) = first_sign(mode);
            log_value(fallback) = first_log(mode);
        else
            [sign_value(fallback),log_value(fallback)] = adaptive_vpa( ...
                @() integral_vpa_once(U(fallback),P(fallback), ...
                    epsilon,epsilon_derivative,dO,g,f), ...
                max(64,ceil(sum(g)/10)));
        end
    end
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

function [U,support] = physical_support(U,P,epsilon,edge_orders,dO,g_tot)
% The constrained simplex is empty unless
%
%   epsilon_min*P <= U <= epsilon_max*P,   P > 0.
%
% Snap roundoff-sized exterior excursions to the exact boundary. At a
% spectral edge the volume vanishes with power sum(g)-g_edge-1, so derivatives
% below that order are known zeros and need no numerical cancellation.

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

    support = isfinite(U) & isfinite(P) & P > 0 ...
        & U >= lower & U <= upper;

    derivative_order = sum(dO);
    lower_index = find(epsilon == epsilon_min,1);
    upper_index = find(epsilon == epsilon_max,1);
    lower_power = g_tot - edge_orders(lower_index) - 1;
    upper_power = g_tot - edge_orders(upper_index) - 1;

    known_zero = U == lower & lower_power > derivative_order;
    known_zero = known_zero ...
        | (U == upper & upper_power > derivative_order);
    support(known_zero) = false;
end

function [factor_sign,factor_log] = derivative_factor(dO_P,pp,epsilon_n)
    factor_log = gammaln(dO_P + 1) ...
        - gammaln(pp + 1) - gammaln(dO_P - pp + 1);

    if pp == 0
        factor_sign = 1;
    elseif epsilon_n == 0
        factor_sign = 0;
        factor_log = -Inf;
    else
        factor_sign = sign(-epsilon_n)^pp;
        factor_log = factor_log + pp*log(abs(epsilon_n));
    end
end

function [log_positive,log_negative] = accumulate_term( ...
        log_positive,log_negative,coefficient_sign,coefficient_log, ...
        P,power_P,Q,power_Q,active)

    if coefficient_sign == 0 || coefficient_log == -Inf
        return
    end

    log_term = coefficient_log + zeros(size(P));
    term_sign = coefficient_sign*ones(size(P));
    valid = active;

    if power_P > 0
        valid = valid & P ~= 0;
        log_term(valid) = log_term(valid) ...
            + power_P*log(abs(P(valid)));
        term_sign(valid) = term_sign(valid).*sign(P(valid)).^power_P;
    end

    if power_Q > 0
        valid = valid & Q > 0;
        log_term(valid) = log_term(valid) ...
            + power_Q*log(Q(valid));
    end

    valid = valid & term_sign ~= 0 & ~isnan(log_term);
    positive = valid & term_sign > 0;
    negative = valid & term_sign < 0;

    log_positive(positive) = logaddexp( ...
        log_positive(positive),log_term(positive));
    log_negative(negative) = logaddexp( ...
        log_negative(negative),log_term(negative));
end

function [sign_value,log_value,unresolved] = ...
        resolve_signed_logs(log_positive,log_negative)

    sign_value = zeros(size(log_positive));
    log_value = -Inf(size(log_positive));

    only_positive = isfinite(log_positive) & ~isfinite(log_negative);
    only_negative = ~isfinite(log_positive) & isfinite(log_negative);
    both = isfinite(log_positive) & isfinite(log_negative);

    sign_value(only_positive) = 1;
    log_value(only_positive) = log_positive(only_positive);

    sign_value(only_negative) = -1;
    log_value(only_negative) = log_negative(only_negative);

    positive_result = both & log_positive > log_negative;
    negative_result = both & log_negative > log_positive;

    sign_value(positive_result) = 1;
    log_value(positive_result) = logdiffexp( ...
        log_positive(positive_result),log_negative(positive_result));

    sign_value(negative_result) = -1;
    log_value(negative_result) = logdiffexp( ...
        log_negative(negative_result),log_positive(negative_result));

    nonzero = sign_value ~= 0;

    separation = abs(log_positive - log_negative);
    unresolved = both & (separation < 1e-8 | ~isfinite(log_value));
    unresolved = unresolved | (nonzero & ~isfinite(log_value));
end

function value = logaddexp(log_x,log_y)
    value = log_x;

    use_y = log_x == -Inf;
    value(use_y) = log_y(use_y);

    y_positive_inf = isinf(log_y) & log_y > 0;
    value(y_positive_inf) = Inf;

    regular = isfinite(log_x) & isfinite(log_y);
    if any(regular,'all')
        largest = max(log_x(regular),log_y(regular));
        value(regular) = largest + log( ...
            exp(log_x(regular) - largest) ...
            + exp(log_y(regular) - largest));
    end
end

function value = logdiffexp(log_x,log_y)
    value = log_x + log(-expm1(log_y - log_x));
end

function [sign_value,log_value] = adaptive_vpa(evaluate,precision)
    if exist('vpa','file') ~= 2
        error('weighted_volume:VariablePrecisionUnavailable', ...
            'Adaptive cancellation recovery requires the Symbolic Math Toolbox.')
    end
    old_digits = digits;
    cleanup = onCleanup(@() digits(old_digits));
    previous = [];
    for attempt = 1:4
        digits(precision);
        current = evaluate();
        if ~isempty(previous) && vpa_converged(current,previous)
            sign_value = double(sign(current));
            log_value = -Inf(size(current));
            nonzero = sign_value ~= 0;
            log_value(nonzero) = double(log(abs(current(nonzero))));
            return
        end
        previous = current;
        precision = precision + 32;
    end
    error('weighted_volume:Precision', ...
        'The weighted volume did not converge at variable precision.')
end

function result = integral_vpa_once( ...
        U,P,epsilon,epsilon_derivative,dO,g,f)
    U_vpa = vpa(sym(U,'f'));
    P_vpa = vpa(sym(P,'f'));
    epsilon_vpa = vpa(sym(epsilon,'f'));
    epsilon_derivative_vpa = vpa(sym(epsilon_derivative,'f'));
    w = compute_weights(epsilon_vpa,g);

    result = sym(zeros(size(U_vpa)));
    g_tot = sum(g);
    dO_U = dO(1);
    dO_P = dO(2);

    for nn = f
        Qn = U - epsilon(nn).*P;
        Q = (U_vpa - epsilon_vpa(nn).*P_vpa).*vpa(Qn > 0);

        for mm = 1:g(nn)
            ii = mm - 1;
            jj = g_tot - ii - 2;

            for pp = 0:dO_P
                II = dO_P - pp;
                JJ = dO_U + pp;

                if II > ii || JJ > jj
                    continue
                end

                power_P = ii - II;
                power_Q = jj - JJ;

                if JJ == 0
                    active = Qn >= 0;
                else
                    active = Qn > 0;
                end

                term = w(nn,mm)*nchoosek(dO_P,pp) ...
                    * (-epsilon_derivative_vpa(nn))^pp ...
                    .* P_vpa.^power_P/gamma(vpa(power_P + 1)) ...
                    .* Q.^power_Q/gamma(vpa(power_Q + 1)) ...
                    .* vpa(active);

                result = result + term;
            end
        end
    end
end

function unnormalized_moments = first_moments_vpa_once(U,P,epsilon)
% For alpha=e_k and distinct energies, all augmented weights follow from
% the simple-pole weights. This is the same partial-fraction equation,
% evaluated for every k in one matrix operation.

    epsilon_vpa = vpa(sym(epsilon(:),'f'));
    M = numel(epsilon);

    base_weights = compute_weights(epsilon_vpa,ones(M,1));
    base_weights = base_weights(:,1);

    Qn = U - epsilon(:)*P;
    U_vpa = vpa(sym(U,'f'));
    P_vpa = vpa(sym(P,'f'));
    Q = (U_vpa - epsilon_vpa*P_vpa).*vpa(Qn > 0);

    A = Q.^(M-1)/gamma(vpa(M));
    B = P_vpa*Q.^(M-2)/gamma(vpa(M-1));

    energy_difference = epsilon_vpa - epsilon_vpa.';
    energy_difference(1:M+1:end) = vpa(1);

    inverse_difference = 1./energy_difference;
    inverse_difference(1:M+1:end) = vpa(0);

    s1 = sum(inverse_difference,2);
    cross = inverse_difference*(base_weights.*A);

    unnormalized_moments = ...
        cross + base_weights.*s1.*A + base_weights.*B;
end

function converged = vpa_converged(current,previous)
    converged = true;

    for kk = 1:numel(current)
        if isAlways(current(kk) == 0)
            converged = converged && isAlways(previous(kk) == 0);
        else
            relative_change = abs((current(kk) - previous(kk))/current(kk));
            converged = converged && double(relative_change) < 1e-12;
        end
    end
end
