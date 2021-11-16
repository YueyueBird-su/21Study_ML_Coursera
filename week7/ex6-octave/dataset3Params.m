function [C, sigma] = dataset3Params(X, y, Xval, yval)
    %DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
    %where you select the optimal (C, sigma) learning parameters to use for SVM
    %with RBF kernel
    %   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
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

    % %% Train a svm model, like the coding in ex6.m:
    % %       model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);

    % model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

    % %% Predict 
    % predictions = svmPredict(model, Xval);

    % %% Compute err && acc
    % err = mean(double(predictions ~= yval));
    % acc = 1 - mean(double(predictions ~= yval));

    %% loop to find C and sigma
    K = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
    num = 0;
    err = [];
    acc = [];

    for i = [1:8]
        for j = [1:8]
            num = num + 1;
            model = svmTrain(X, y, K(i), @(x1, x2) gaussianKernel(x1, x2, K(j)));
            predictions = svmPredict(model, Xval);
            err(num) = mean(double(predictions ~= yval));
            acc(num) = mean(double(predictions == yval));
            fprintf('num: %d , err: %f %%, acc: %f %%;\n', num, err(num) * 100, acc(num) * 100);
        end
    end



    [val, dim] = max(acc);
    C = K(fix(dim / length(K)));
    sigma = K(mod(dim, length(K)));
    fprintf('The best param is C = %f, sigma = %f!', C, sigma);
    %% visualize the acc
    [m, n] = meshgrid(K);
    acc = reshape(acc, [8, 8]);
    surf(m,n,acc);
    set(get(gca, 'XLabel'), 'String', 'C');
    set(get(gca, 'YLabel'), 'String', 'sigma');
    set(get(gca, 'ZLabel'), 'String', 'acc');
    set(get(gca, 'Title'), 'String', 'Parameters Solving');
    pause;
    % =========================================================================

end
