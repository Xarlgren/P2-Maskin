#include <neural_network.hpp>

namespace {
    
// --------------------------------------------------------------------------------
template <typename T = int>
void Print(const std::vector<T>& data, std::ostream& ostream = std::cout) {
    static_assert(std::is_arithmetic<T>::value, 
        "Non-arithmetic type selected for method ::Print!");
    ostream << "[";
    for (const auto& i : data) {
        ostream << i;
        if (&i < &data[data.size() - 1]) { ostream << ", "; }
    }
    ostream << "]\n";
}
} // namespace

namespace yrgo {
namespace machine_learning {

// --------------------------------------------------------------------------------
NeuralNetwork::NeuralNetwork(const std::size_t num_inputs, 
                             const std::size_t num_hidden_nodes, 
                             const std::size_t num_outputs,
                             const ActFunc act_func_hidden, 
                             const ActFunc act_func_output) 
    : hidden_layer_(DenseLayer(num_hidden_nodes, num_inputs, act_func_hidden))
    , output_layer_(DenseLayer(num_outputs, num_hidden_nodes, act_func_output)) {}

// --------------------------------------------------------------------------------
bool NeuralNetwork::AddTrainingData(const std::vector<std::vector<double>>& train_input,
                                    const std::vector<std::vector<double>>& train_output) {
    train_input_ = train_input; 
    train_output_ = train_output;
    CheckNumTrainingSets(); 
    InitTrainOrderVector(); 
    return NumTrainingSets() > 0;
}    

// --------------------------------------------------------------------------------
bool NeuralNetwork::Train(const std::size_t num_epochs, const double learning_rate) {
    if (NumTrainingSets() == 0 || num_epochs == 0 || learning_rate <= 0) { return false; }
    
    for (std::size_t i{}; i < num_epochs; ++i) {
        RandomizeTrainingOrder(); 
        for (const auto& j : train_order_) {
            Feedforward(train_input_[j]);
            Backpropagate(train_output_[j]);
            Optimize(train_input_[j], learning_rate);
        }
    }
    return true;
}

// --------------------------------------------------------------------------------
const std::vector<double>& NeuralNetwork::Predict(const std::vector<double>& input) {
     Feedforward(input);
     return output_layer_.Output(); 
}

// --------------------------------------------------------------------------------
void NeuralNetwork::PrintPredictions(const std::vector<std::vector<double>>& input_sets,
                                     const std::size_t num_decimals,
                                     std::ostream& ostream) {
    if (input_sets.size() == 0) { return; }
    ostream << std::fixed << std::setprecision(num_decimals);
    ostream << "--------------------------------------------------------------------------------";
    for (const auto& input: input_sets) {
        ostream << "\nInput:\t";
        Print<double>(input, ostream);
        ostream << "Output:\t";
        Print<double>(Predict(input), ostream);
    }
    ostream << "--------------------------------------------------------------------------------\n\n";
}

// --------------------------------------------------------------------------------
void NeuralNetwork::CheckNumTrainingSets() {
    if (train_input_.size() != train_output_.size()) {
        const auto num_sets{train_input_.size() < train_output_.size() ?
            train_input_.size() : train_output_.size()};
        train_input_.resize(num_sets);
        train_output_.resize(num_sets);
    }
}

// --------------------------------------------------------------------------------
void NeuralNetwork::InitTrainOrderVector() {
    train_order_.resize(train_input_.size());
    for (std::size_t i{}; i < train_order_.size(); ++i) {
        train_order_[i] = i; 
    }
}

// --------------------------------------------------------------------------------
void NeuralNetwork::RandomizeTrainingOrder() {
    utils::random::ShuffleVector<std::size_t>(train_order_);
}

// --------------------------------------------------------------------------------
void NeuralNetwork::Feedforward(const std::vector<double>& input) {
    hidden_layer_.Feedforward(input); 
    output_layer_.Feedforward(hidden_layer_.Output());
}

// --------------------------------------------------------------------------------
void NeuralNetwork::Backpropagate(const std::vector<double>& reference) {
    output_layer_.Backpropagate(reference);
    hidden_layer_.Backpropagate(output_layer_);
}

// --------------------------------------------------------------------------------
void NeuralNetwork::Optimize(const std::vector<double>& input, const double learning_rate) {
    hidden_layer_.Optimize(input, learning_rate);
    output_layer_.Optimize(hidden_layer_.Output(), learning_rate);
}

} /* namespace machine_learning */
} /* namespace yrgo */