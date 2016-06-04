function [u] = f(theta)
    u = zeros(2,1);
    u(1) = cos(theta(1)) + cos(theta(2));
    u(2) = sin(theta(1)) + sin(theta(2));
end