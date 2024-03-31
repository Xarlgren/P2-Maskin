/********************************************************************************
 * @brief Contains implementation of neural network.
 ********************************************************************************/
#pragma once 

#include <iomanip>  
#include <iostream> 
#include <type_traits>
#include <vector>   

#include <dense_layer.hpp>
#include <utils.hpp>

namespace yrgo {
namespace machine_learning {

class NeuralNetwork {
public:

    /********************************************************************************
     * @brief Default constructor deleted.
     ********************************************************************************/
    NeuralNetwork(void) = delete;

    /********************************************************************************
     * @brief Creates new neural network.
     * 
     * @param num_inputs       The number of inputs (nodes in the input layer).
     * @param num_hidden_nodes The number of nodes in the hidden layer.
     * @param num_output       The number of outputs (nodes in the output layer).
     * @param act_func_hidden  Activation function of the hidden layer (default = ReLU).
     * @param act_func_output  Activation function of the output layer (default = ReLU).
     ********************************************************************************/
    NeuralNetwork(const std::size_t num_inputs, 
                  const std::size_t num_hidden_nodes, 
                  const std::size_t num_outputs,
                  const ActFunc act_func_hidden = ActFunc::kRelu, 
                  const ActFunc act_func_output = ActFunc::kRelu);

    /********************************************************************************
     * @brief Provides the number of inputs in the network.
     * 
     * @return The number of inputs, i.e. the number of nodes in the input layer.
     ********************************************************************************/
    std::size_t NumInputs(void) const { return hidden_layer_.NumWeightsPerNode(); }

    /********************************************************************************
     * @brief Provides the number of nodes in the hidden layer of the network.
     * 
     * @return The number of nodes in the hidden layer.
     ********************************************************************************/
    std::size_t NumHiddenNodes(void) const { return hidden_layer_.NumNodes(); }

    /********************************************************************************
     * @brief Provides the number of outputs in the network.
     * 
     * @return The number of outputs, i.e. the number of nodes in the output layer.
     ********************************************************************************/
    std::size_t NumOutputs(void) const { return output_layer_.NumNodes(); }

    /********************************************************************************
     * @brief Provides the number of stored training sets.
     * 
     * @return The number of stored training sets, if any.
     ********************************************************************************/
    std::size_t NumTrainingSets(void) const { return train_order_.size(); }

    /********************************************************************************
     * @brief Adds training sets to the network.
     * 
     * @param train_input Reference to vector storing input sets.
     * @param train_output Reference to vector storing output sets.
     * 
     * @return True if at least one training set has been added.
     ********************************************************************************/
    bool AddTrainingData(const std::vector<std::vector<double>>& train_input,
                         const std::vector<std::vector<double>>& train_output);
    
    /********************************************************************************
     * @brief Trains the neural network.
     * 
     * @param num_epochs    The number of epochs to train.
     * @param learning_rate The learning rate, sets the adjustment rate of the
     *                      network parameters upon error (default = 0.01, i.e. 1 %).
     * 
     * @return True if training was performed, else false.
     ********************************************************************************/
    bool Train(const std::size_t num_epochs, const double learning_rate = 0.01);

    /********************************************************************************
     * @brief Performs prediction with specified input values.
     * 
     * @param input Reference to vector holding input values.
     * 
     * @return Reference to vector holding the predicted output values.
     ********************************************************************************/
    const std::vector<double>& Predict(const std::vector<double>& input);

    /********************************************************************************
     * @brief Performs predictions with all input sets and prints the output.
     * 
     * @param input_sets   Reference to vector holding all input sets to predict with.
     * @param num_decimals The number of decimals to print (default = 0).
     * @param ostream      Reference to output stream (default = terminal print).
     ********************************************************************************/
    void PrintPredictions(const std::vector<std::vector<double>>& input_sets,
                          const std::size_t num_decimals = 0,
                          std::ostream& ostream = std::cout);

private:

    void CheckNumTrainingSets();
    void InitTrainOrderVector();
    void RandomizeTrainingOrder();
    void Feedforward(const std::vector<double>& input);
    void Backpropagate(const std::vector<double>& reference);
    void Optimize(const std::vector<double>& input, const double learning_rate);

    DenseLayer hidden_layer_ = DenseLayer(3, 2, ActFunc::kRelu);
    DenseLayer output_layer_ = DenseLayer(1, 3, ActFunc::kTanh);
    std::vector<std::vector<double>> train_input_{};
    std::vector<std::vector<double>> train_output_{};
    std::vector<std::size_t> train_order_{};
};

} /* namespace machine_learning */
} /* namespace yrgo */