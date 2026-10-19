#include <iostream>
#include <Eigen/Dense>

float cost_function(const Eigen::VectorXf& prediction, const Eigen::VectorXf& target) {
    return (prediction - target).array().square().mean();
}
