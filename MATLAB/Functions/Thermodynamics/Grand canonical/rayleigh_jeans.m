function J = rayleigh_jeans(T,mu,epsilon)
    % Uses the convention that U = epsilon*J0.
    % If the energy is given as U = -beta*J0, then epsilon = -beta must be used.

    [epsilon,I_order] = sort(epsilon,'descend');

    J = T./(epsilon - mu);
    J(I_order) = J;
end
