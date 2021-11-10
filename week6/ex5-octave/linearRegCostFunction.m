function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
    %LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
    %regression with multiple variables
    %   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
    %   cost of using theta as the parameter for linear regression to fit the
    %   data points in X and y. Returns the cost in J and the gradient in grad

    % Initialize some useful values
    m = length(y); % number of training examples

    % You need to return the following variables correctly
    J = 0;
    grad = zeros(size(theta));

    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the cost and gradient of regularized linear
    %               regression for a particular choice of theta.
    %
    %               You should set J to the cost and grad to the gradient.
    %

    %% get data length
    % size(X) # 12, 2
    % size(y) # 12, 1
    % size(theta) # 2, 1

    %% compute J
    h = X * theta; % 12,1
    k = ones(size(theta));
    k(1) = 0;
    J = (sum(power(h - y, 2)) + lambda * sum(power(k .* theta, 2))) / (2 * m);

    %% compute Gradient
    grad = ((sum((h - y) .* X))' + lambda * k.* theta) / m;

    % =========================================================================

    grad = grad(:);

end
