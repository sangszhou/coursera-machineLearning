function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


    function [best_C, best_sigma] = EX6PARAMS (X, y, Xval, yval )
        
        choices = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 ];
        min_error = 999;
        min_c =  -1;
        min_sigma = -1;
        for i = 1:size(choices, 2)
            for j = 1:size(choices, 2)
                model= svmTrain(X, y, choices(i), @(x1, x2)gaussianKernel(x1, x2, choices(j))); 
                predictions = svmPredict( model, Xval );
                current_error = mean(double(predictions ~= yval));
                if ( current_error < min_error )
                    min_error = current_error;
                    min_c = choices(i); min_sigma = choices(j);
                end
            end
        end
        best_C = min_c;
        best_sigma = min_sigma;
    end

[C, sigma] = EX6PARAMS(X, y, Xval, yval );

% =========================================================================

end
