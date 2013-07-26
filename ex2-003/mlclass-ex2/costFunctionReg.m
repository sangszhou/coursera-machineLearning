function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
 
gTheta = X * theta;
hTheta = sigmoid ( gTheta );

% don't forget to maintain the value of theta(1)
eliminateFirst = zeros(size(theta));
eliminateFirst(1) = theta(1);

leftPart = -1 * y .* log( hTheta );
rightPart = ( 1 - y ) .* log ( 1 - hTheta );
J = sum( leftPart - rightPart )/m +lambda*sum(theta.*theta - theta.*eliminateFirst)/( 2 * m);

% keep the grad(1) to handle at last
grad = ((hTheta - y)'* X)'/m + lambda*theta/m;
grad(1) = ((hTheta - y)'* X(:, 1))/m;




% =============================================================

end
