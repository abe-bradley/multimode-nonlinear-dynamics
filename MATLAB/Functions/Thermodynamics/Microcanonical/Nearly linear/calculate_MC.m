function [T,mu] = calculate_MC(U,P,epsilon)
%CALCULATE_MC Microcanonical temperature and chemical potential.
%   T and MU accept the same scalar or compatible arrays as WEIGHTED_VOLUME.
%   Modal moments are evaluated separately with MIXED_MOMENTS.

    epsilon = epsilon(:).';

    M = length(epsilon);

    %% Volume and its derivative
    V = weighted_volume(U,P,epsilon,[],[0 0]);
    dV = weighted_volume(U,P,epsilon,[],[1 0]);

    %% Temperature
    T = V./dV;

    %% Chemical potential
    mu = (U - (M - 2).*T)./P;

end
