function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%% h=theta0 + theta1 * x1 + ...
%  same as one feature

h_multi = X * theta - y;

% size(theta) 4,1   n = 4
% size(h_multi) 20,1

J = sum(h_multi .^ 2 ) ./ (2 * m);


% =========================================================================

end
