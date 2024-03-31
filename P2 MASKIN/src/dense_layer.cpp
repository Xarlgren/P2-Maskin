#include "dense_layer.hpp"

namespace yrgo {
namespace machine_learning {

namespace {

// --------------------------------------------------------------------------------
double GetActFuncOutput(const double sum, const enum ActFunc act_func) {
    return act_func == ActFunc::kRelu ?
        utils::math::Relu(sum) : utils::math::Tanh(sum);
}

// --------------------------------------------------------------------------------
double GetActFuncDelta(const double output, const enum ActFunc act_func) {
    return act_func == ActFunc::kRelu ?
        utils::math::ReluDelta(output) : utils::math::TanhDelta(output);
}

} /* namespace */

// --------------------------------------------------------------------------------
DenseLayer::DenseLayer(const std::size_t num_nodes,
                       const std::size_t num_weights_per_node,
                       const enum ActFunc act_func) 
    : act_func_{act_func} {
    utils::random::Init();
    output_.resize(num_nodes, 0);
    utils::random::InitVector<double>(bias_, num_nodes, 0, 1);
    error_.resize(num_nodes, 0);
    utils::random::InitVector<double>(weights_, num_nodes, num_weights_per_node, 0, 1);
}

// --------------------------------------------------------------------------------
void DenseLayer::Feedforward(const std::vector<double>& inputs) {
    for (std::size_t i{}; i < NumNodes(); ++i) {
        double sum{ bias_[i] };
        for (std::size_t j{}; j < NumWeightsPerNode() && j < inputs.size(); ++j) {
            sum += inputs[j] * weights_[i][j];
        }
        output_[i] = GetActFuncOutput(sum, act_func_);
    }
}

// --------------------------------------------------------------------------------
void DenseLayer::Backpropagate(const std::vector<double>& reference) {
    for (std::size_t i{}; i < NumNodes() && i < reference.size(); ++i) {
        const double error = reference[i] - output_[i];
        error_[i] = error * GetActFuncDelta(output_[i], act_func_);
    }
}

// --------------------------------------------------------------------------------
void DenseLayer::Backpropagate(const DenseLayer& next_layer) {
    for (std::size_t i{}; i < NumNodes(); ++i) {
        double error{};
        for (std::size_t j{}; j < next_layer.NumNodes(); ++j) {
            error += next_layer.error_[j] * next_layer.weights_[j][i];
        }
        error_[i] = error * GetActFuncDelta(output_[i], act_func_);
    }
}

// --------------------------------------------------------------------------------
void DenseLayer::Optimize(const std::vector<double>& inputs, const double learning_rate) {
    for (std::size_t i{}; i < NumNodes(); ++i) {
        bias_[i] += error_[i] * learning_rate;
        for (std::size_t j{}; j < NumWeightsPerNode() && j < inputs.size(); ++j) {
            weights_[i][j] += error_[i] * learning_rate * inputs[j];
        }
    }
}

} /* namespace machine_learning */
} /* namespace yrgo */
