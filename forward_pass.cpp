#include <iostream>
#include <Eigen/Dense>

Eigen::VectorXf sigmoid(Eigen::MatrixXf a){
    return 1.0f / (1.0f + (-a.array()).exp().matrix());
}

Eigen::VectorXf forward_pass(const Eigen::VectorXf& a, const Eigen::MatrixXf& W, const Eigen::VectorXf& b){
    return sigmoid(W * a + b);
}

