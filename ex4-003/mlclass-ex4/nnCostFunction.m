function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%





% first add one row to the top of X
[row, ~] = size(X);
X1 = [ones(row, 1), X];
z1 = (Theta1 * X1');
a1 = sigmoid(z1);

% a1 is also matrix, but a1 has a slight difference with  X
% X's row represent a training example yet a1's column is a "training example"
% so, we do not need to transpose a1
[~, column] = size(a1);
a1 = [ones(1, column); a1];
z2 = Theta2 * a1;
a2 = sigmoid(z2);

% now, calculate the J_theta
% a little bit complex. last time Iew just need to input y directly and
% fmincg can do the rest for us. This time, we have to tranform the y

o = a2;
%//transform_y = zeros(num_labels, m);
%//for i= 1 :m
%//    for j=1 : num_labels
%//        if (y(i) == j)
%//            transform_y(j,i) = 1;
%//        end
%//    end
%end
inter = eye(max(y));
transform_y = inter(:, y);

left_part = (-1 * transform_y).*(log(o));
right_part = (1-transform_y).*(log(1-o));
column_sum = ones(1,num_labels)*(left_part-right_part);
row_sum = column_sum * ones(m, 1);
J = row_sum/m;

% regulization cost function
[~, column] = size( Theta1);
Theta1_trim = Theta1(:, 2:column );
[~, column] = size( Theta2);
Theta2_trim = Theta2(:, 2:column);

regulation_theta1 = sum(sum(Theta1_trim.*Theta1_trim));
regulation_theta2 = sum(sum(Theta2_trim.*Theta2_trim));

J = J + (regulation_theta1 + regulation_theta2)*lambda/(2*m);


% backpropagation algorithm

% output unit


sigma3 = (a2-transform_y);
sigma2 = a1.*(1-a1).*(Theta2'*sigma3);
delta3 = zeros(size(Theta2));
delta2 = zeros(size(Theta1));



for i=1:m
%backpropagation algorithm 
 help1 = sigma3(:,i)*(a1(:,i))';
 delta3 = delta3 + help1;
 help_20 = sigma2(:,i);
 
 help2 = help_20(2:size(sigma2,1))*X1(i,:);
 delta2 = delta2 + help2;
end

% -------------------------------------------------------------

 [~,b]=size(Theta2);
 
Theta2_grad = ( delta3./m ) + ( (lambda/m) .* [ zeros( num_labels, 1 ), Theta2( :, 2:b ) ] ); 
Theta2_grad(:,1) = (delta3(:,1))./m;
 [~,b]=size(Theta1);
 Theta1_grad = ( delta2./m ) + ( (lambda/m) .* [ zeros( hidden_layer_size, 1 ), Theta1( :, 2:b ) ] );
 Theta1_grad(:,1) = (delta2(:,1))./m;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
