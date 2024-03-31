/********************************************************************************
 * @brief Contains miscellaneous utility functions for generation of random
 *        numbers, initialization of vectors and mathematical operations.
 ********************************************************************************/
#pragma once

#include <vector>
#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cmath>

namespace yrgo {
namespace utils {
namespace {
namespace random {

/********************************************************************************
 * @brief Initializes the random number generator. This function should be called
 *        before using generating any random numbers to ensure that each run of
 *        the program yields different numbers.
 ********************************************************************************/
inline void Init(void) {
    static bool rand_initialized{false};
    if (!rand_initialized) {
        std::srand(std::time(nullptr));
        rand_initialized = true;
    }
}

/********************************************************************************
 * @brief Generates a random number in the range of specified min and max values.
 * 
 * @tparam T The type of the random number to generate.
 * 
 * @param min The minimum permitted random number (default = 0).
 * @param max The maximum permitted random number (default = 100).
 * 
 * @return The generated random number.
 ********************************************************************************/
template <typename T>
T GetNumber(const T min = 0, const T max = 100) {
    static_assert(std::is_arithmetic<T>::value, 
        "Non-arithmetic type selected for new random number!");
    if constexpr (std::is_integral<T>::value) {
        return static_cast<T>((std::rand() % (max + 1 - min)) + min);
    } else {
        return (std::rand() / static_cast<T>(RAND_MAX)) * (max - min) + min;
    }
}

/********************************************************************************
 * @brief Initializes one-dimensional vector with random numbers.
 * 
 * @tparam T The vector type.
 * 
 * @param vector Reference to the vector to initialize.
 * @param size The new size of the vector.
 * @param min The minimum permitted random number (default = 0).
 * @param max The maximum permitted random number (default = 100).
 ********************************************************************************/
template <typename T> 
void InitVector(std::vector<T>& vector,
                const std::size_t size, 
                const T min = 0, 
                const T max = 100) {
    static_assert(std::is_arithmetic<T>::value, 
        "Cannot assign random numbers for non-arithmetic types!");
    vector.resize(size);
    for (auto& i : vector) {
        i = GetNumber<T>(min, max);
    }
}

/********************************************************************************
 * @brief Initializes two-dimensional vector with random numbers.
 * 
 * @tparam T The vector type.
 * 
 * @param vector Reference to the vector to initialize.
 * @param num_columns The new number of columns of the vector.
 * @param num_rows The new number of rows of the vector.
 * @param min The minimum permitted random number (default = 0).
 * @param max The maximum permitted random number (default = 100).
 ********************************************************************************/
template <typename T> 
void InitVector(std::vector<std::vector<T>>& vector,
                const std::size_t num_columns,
                const std::size_t num_rows,
                const T min = 0,
                const T max = 100) {
    static_assert(std::is_arithmetic<T>::value, 
        "Cannot assign random numbers for non-arithmetic types!");
    vector.resize(num_columns, std::vector<T>(num_rows));
    for (auto& i : vector) {
        for (auto& j : i) {
            j = GetNumber<T>(min, max);
        }
    }
}

/********************************************************************************
 * @brief Shuffle the content of one-dimensional vector.
 * 
 * @tparam T The vector type.
 * 
 * @param vector Reference to the vector whose content will be shuffled.
 ********************************************************************************/
template <typename T>
void ShuffleVector(std::vector<T>& vector) {
    for (size_t i{}; i < vector.size(); ++i) {
        const auto r{static_cast<std::size_t>(std::rand() % vector.size())};
        const auto temp{vector[i]};
        vector[i] = vector[r];
        vector[r] = temp;
    }
}

/********************************************************************************
 * @brief Shuffle the content of two-dimensional vector.
 * 
 * @tparam T The vector type.
 * 
 * @param vector Reference to the vector whose content will be shuffled.
 ********************************************************************************/
template <typename T>
void ShuffleVector(std::vector<std::vector<T>>& vector) {
    for (size_t i{}; i < vector.size(); ++i) {
        const auto r{static_cast<std::size_t>(std::rand() % vector.size())};
        const auto temp{vector[i]};
        vector[i] = vector[r];
        vector[r] = temp;
    }
}
} /* namespace random */

namespace math {

/********************************************************************************
 * @brief Provides the sum of an arbitrary amount of numbers.
 * 
 * @tparam T The type of the numbers.
 * @tparam Numbers Value type for parameter pack.
 * 
 * @param numbers Parameter pack holding numbers.
 * 
 * @return The sum of the numbers.
 ********************************************************************************/
template <typename T, typename... Numbers>
constexpr T Add(const Numbers&... numbers) {
    static_assert(std::is_arithmetic<T>::value, 
        "Cannot perform mathematical operations with non-arithmetic types!");
    T sum{};
    for (const auto& i : {numbers...}) {
        sum += i;
    }
    return sum;
}

/********************************************************************************
 * @brief Provides the difference of an arbitrary amount of numbers.
 * 
 * @tparam T The type of the numbers.
 * @tparam Numbers Value type for parameter pack.
 * 
 * @param numbers Parameter pack holding numbers.
 * 
 * @return The difference of the numbers.
 ********************************************************************************/
template <typename T, typename... Numbers>
constexpr T Subtract(const Numbers&... numbers) {
    static_assert(std::is_arithmetic<T>::value, 
        "Cannot perform mathematical operations with non-arithmetic types!");
    T sum{};
    for (const auto& i : {numbers...}) {
        sum -= i;
    }
    return sum;
}

/********************************************************************************
 * @brief Provides the product of an arbitrary amount of numbers.
 * 
 * @tparam T The type of the numbers.
 * @tparam Numbers Value type for parameter pack.
 * 
 * @param numbers Parameter pack holding numbers.
 * 
 * @return The product of the numbers.
 ********************************************************************************/
template <typename T, typename... Numbers>
constexpr T Multiply(const Numbers&... numbers) {
    static_assert(std::is_arithmetic<T>::value, 
        "Cannot perform mathematical operations with non-arithmetic types!");
    T sum{1};
    for (const auto& i : {numbers...}) {
        sum *= i;
    }
    return sum;
}

/********************************************************************************
 * @brief Provides the quotient of specified numbers.
 * 
 * @tparam T1 The type of the dividend.
 * @tparam T2 The type of the divisor.

 * @param dividend The dividend/numerator.
 * @param divisor The divisor/denominator.
 * 
 * @return The quotient of the numbers or 0 if the divisor is 0.
 ********************************************************************************/
template <typename T1, typename T2>
constexpr double Divide(const T1 dividend, const T2 divisor) {
    static_assert(std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value, 
        "Cannot perform mathematical operations with non-arithmetic types!");
    return divisor != 0 ? dividend / (static_cast<double>(divisor)) : 0;
}

/********************************************************************************
 * @brief Rounds floating-point number to the nearest integer.
 * 
 * @tparam T The integral type to round to (default = std::int32_t).
 * 
 * @param number The floating-point number to round.
 * 
 * @return The nearest integer.
 ********************************************************************************/
template <typename T = std::int32_t>
constexpr T Round(const double number) {
    static_assert(std::is_arithmetic<T>::value, "Cannot round to non-arithmetic type!");
    return static_cast<T>(number + 0.5);
}

/********************************************************************************
 * @brief Provides the hyperbolic tangent of specified angle.
 * 
 * @param v The angle to calculate the hyperbolic tangent with.
 * 
 * @return The hyperbolic tangent of specified angle.
 ********************************************************************************/
constexpr double Tanh(const double v) { return std::tanh(v); }

/********************************************************************************
 * @brief Provides the derivate of the hyperbolic tangent of specified angle.
 * 
 * @param v The angle to calculate the derivate of the hyperbolic tangent with.
 * 
 * @return The derivate of the hyperbolic tangent of specified angle.
 ********************************************************************************/
constexpr double TanhDelta(const double x) { return 1 - std::pow(std::tanh(x), 2); }

/********************************************************************************
 * @brief Provides the ReLU (Rectified Linear Unit) output of specified value.
 * 
 * @param x The value to calculate the ReLU output with.
 * 
 * @return The ReLU output, i.e. x if x > 0, else 0.
 ********************************************************************************/
constexpr double Relu(const double x) { return x > 0 ? x : 0; }

/********************************************************************************
 * @brief Provides the derivate of the ReLU (Rectified Linear Unit) output of 
 *        specified value.
 * 
 * @param x The value to calculate the derivate of the ReLU output with.
 * 
 * @return The derivate of the ReLU output, i.e. 1 if x > 0, else 0.
 ********************************************************************************/
constexpr double ReluDelta(const double x) { return x > 0 ? 1 : 0; }

} /* namespace math */
} /* namespace */
} /* namespace utils */
} /* namespace yrgo */