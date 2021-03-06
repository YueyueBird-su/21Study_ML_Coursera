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

    % -------------------------------------------------------------

    %% Part One Compute J(theta) &unregularization
    % size(X) # 5000, 400
    % size(y) # 5000, 1
    % size(Theta1) # 25, 401
    % size(Theta2) # 10, 26
 
    % hidden layer
    a1 = [ones(m,1) X]; % 5000, 401
    z2 = a1 * Theta1'; % 5000, 25
    a2 = sigmoid(z2); 

    % output layer
    a2 = [ones(m, 1), a2]; % 5000, 26
    z3 = a2 * Theta2'; % 5000,10
    a3 = sigmoid(z3);  

    % compute J 
    % we need to trans y to one-hot vectors to compute J
    % or the y's value will expand the Cost value
    Y = zeros(m, num_labels); % 5000, 10
    for i = 1 : m
        Y(i,y(i)) = 1; 
    end

    J = -1 * sum(sum(Y .* log(a3) + (1 - Y) .* log(1 - a3))) / m;

    %% Part3.1 J(thrta) &regularization
    k1 = ones(size(Theta1));
    k1(:,1) = 0;
    k2 = ones(size(Theta2));
    k2(:,1) = 0;
    regular = lambda .* (sum(sum(k1 .* Theta1 .^2)) + sum(sum(k2 .* Theta2 .^2))) / (2 * m);
    J = J + regular; 

    % -------------------------------------------------------------
    
    %% Part Two the backpropagation algorithem &unregularzation

    % compute delta3
    d3 = a3 - Y; % 5000, 10

    % compute delta2
    d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2); % 5000,25

    Theta2_grad = d3' * a2 / m; % 10, 26
    Theta1_grad = d2' * a1 / m; % 25, 401

    %% regularzation backpropagation
    Theta2_grad = Theta2_grad + lambda * k2 .* Theta2 / m;
    Theta1_grad = Theta1_grad + lambda * k1 .* Theta1 / m; 



    % =========================================================================

    % Unroll gradients
    grad = [Theta1_grad(:); Theta2_grad(:)];

end
