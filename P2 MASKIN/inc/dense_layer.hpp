/********************************************************************************
 * @brief Contains class for implementation of dense layers.
 ********************************************************************************/
#pragma once

#include <vector>
#include <utils.hpp>

namespace yrgo {
namespace machine_learning {

/********************************************************************************
 * @brief Enumeration for selecting activation function between ReLU and tanH.
 *
 * @param kRelu Enumerator for selecting ReLU (Rectified Linear Unit)-
 * @param kTanh Enumerator for selecting Tanh (the hyperbolic tangent function).
 ********************************************************************************/
enum class ActFunc { kRelu, kTanh };

class DenseLayer {
  public:
  
    /********************************************************************************
     * @brief Default constructor deleted.
     ********************************************************************************/
    DenseLayer(void) = delete;

    /********************************************************************************
     * @brief Creates new dense layer with specified number of nodes and weights.
     * 
     * @param num_nodes            The number of nodes in the layer.
     * @param num_weights_per_node The number of weights per node in the layer.
     * @param act_func             Activation function (default = ReLU).
     ********************************************************************************/
    DenseLayer(const std::size_t num_nodes, 
               const std::size_t num_weights_per_node,
               const enum ActFunc act_func = ActFunc::kRelu);

    /********************************************************************************
     * @brief Provides the output values of the dense layer.
     * 
     * @return A reference to vector holding the output values.
     ********************************************************************************/
    const std::vector<double>& Output(void) const { return this->output_; }

    /********************************************************************************
     * @brief Provides the number of nodes in the dense layer.
     * 
     * @return The number of nodes in the layer.
     ********************************************************************************/
    std::size_t NumNodes(void) const { return output_.size(); }

    /********************************************************************************
     * @brief Provides the number of weights per node in the dense layer.
     * 
     * @return The number of weights per node in the layer.
     ********************************************************************************/
    std::size_t NumWeightsPerNode(void) const { 
        return weights_.size() > 0 ? weights_[0].size() : 0;
    }
    
    /********************************************************************************
     * @brief Updates the output of all nodes in the layer.
     * 
     * @param inputs Reference to vector holding the new input values.
     ********************************************************************************/
    void Feedforward(const std::vector<double>& inputs);

    /********************************************************************************
     * @brief Calculates current errors in output layer by comparing the output
     *        values with corresponding reference values.
     * 
     * @note This function is for output layers only.
     * 
     * @param reference Reference to vector holding the reference values.
     ********************************************************************************/
    void Backpropagate(const std::vector<double>& reference);

    /********************************************************************************
     * @brief Calculates current error in hidden layer by using the errors and
     *        weights in the next layer.
     * 
     * @note This function is for hidden layers only.
     * 
     * @param next_layer Reference to next layer (holds errors and weights we need).
     ********************************************************************************/
    void Backpropagate(const DenseLayer& next_layer);

    /********************************************************************************
     * @brief Adjusts bias och weights in the dense layer to increase the precision.
     * 
     * @param inputs Reference to vector holding input values (for adjusting weights).
     * @param learning_rate The amount of adjustment (default = 1 %).
     ********************************************************************************/
    void Optimize(const std::vector<double>& inputs, const double learning_rate = 0.01);

  private:
    std::vector<double> output_{};               /* Holds output values. */
    std::vector<double> bias_{};                 /* Holds bias values. */
    std::vector<double> error_{};                /* Holds calculated errors. */
    std::vector<std::vector<double>> weights_{}; /* Holds weights for all nodes. */
    enum ActFunc act_func_{ActFunc::kRelu};      /* Selected activation function. */
};

} /* namespace machine_learning */
} /* namespace yrgo */