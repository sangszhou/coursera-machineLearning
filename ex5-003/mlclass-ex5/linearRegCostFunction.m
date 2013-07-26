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

row_grad = size( grad, 1);

h_theta = X * theta;
error = ( h_theta - y );
J = sum(error.*error)/(2*m);
theta_trans = [zeros(1,1);theta(2:size(theta,1),:)];
J = J + lambda/(2*m)*sum(theta_trans.*theta_trans);

[~,b] = size(X);
left = ( X'*error )./m;
grad = left + (lambda/m).* theta_trans;


% =========================================================================

grad = grad(:);

end
