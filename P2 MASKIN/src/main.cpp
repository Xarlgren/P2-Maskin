/********************************************************************************
 * @brief Simple neural network in C++
 ********************************************************************************/

#include <vector>
#include <neural_network.hpp>
#include <led.hpp>
#include <button.hpp>

using namespace yrgo::machine_learning;
using namespace yrgo::rpi;


namespace 
{

double sum(const std::vector<double>& data) 
{
    double num{};
    for (const auto& i : data)
    {
        num += i;
    }
    return num;
}
} 

/********************************************************************************
 * @brief Creates a neural network trained to predict a 4-bit XOR pattern.
 *        The network consists of four inputs, ten hidden nodes and one output.
 *        TanH is used as activation function in the hidden layer in order
 *        to make the network better at predicting complex patterns, while
 *        ReLU is used in the output layer.
 * 
 *        The model is trained during 10 000 epochs with a 1 % learning rate.
 *        If the training is successful, the training inputs are used for
 *        prediction, which is printed in the terminal.
 ********************************************************************************/

int main(void) 
{

    const std::vector<std::vector<double>> train_input{{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 0, 1, 1},
                                                       {0, 1, 0, 0}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 1},
                                                       {1, 0, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 0}, {1, 0, 1, 1},
                                                       {1, 1, 0, 0}, {1, 1, 0, 1}, {1, 1, 1, 0}, {1, 1, 1, 1}};

    const std::vector<std::vector<double>> train_output{{0}, {1}, {1}, {0},
                                                        {1}, {0}, {0}, {1},
                                                        {1}, {0}, {0}, {1},
                                                        {0}, {1}, {1}, {0}};


    Led led1{17};
    Button b1(25), b2(22), b3(23), b4(24);
    std::vector<double> inputs {0, 0, 0, 0};
    std::vector<double> previous_inputs {0, 0, 0, 0};
    std::vector<Button*> buttons {&b1, &b2, &b3, &b4};


    NeuralNetwork network{4, 10, 1, ActFunc::kTanh};
    network.AddTrainingData(train_input, train_output);
    network.Train(10000, 0.01);

    while(1)
    {        
        for(std::size_t i{}; i<buttons.size(); ++i)
        {
            inputs[i] = buttons[i]->isPressed() ? 1 : 0;
        }

        if (sum(inputs) != sum(previous_inputs))
        {
            const auto output{network.Predict(inputs)};     
            const auto led_val{static_cast<int>(output[0] + 0.5)};

            if(led_val > 0)
            {
                led1.on();
                
            }
            else
            {
                led1.off();
            }

            previous_inputs = inputs;   
        }
    }
    return 0;
}