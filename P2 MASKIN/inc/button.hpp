#pragma once

#include <cstdint>

#include "gpiod_utils.h"

namespace yrgo 
{
namespace rpi
{

class Button
{
public:

    enum class Edge
    {
        Rising = GPIOD_LINE_EDGE_RISING,
        Falling = GPIOD_LINE_EDGE_FALLING,
        Both = GPIOD_LINE_EDGE_BOTH
    };

    Button();
    Button(const std::uint8_t pin, const std::uint8_t activeHigh = 1);
    ~Button();

    std::uint8_t pin() const;

    bool init(const std::uint8_t pin, const std::uint8_t activeHigh = 1);

    bool isPressed();

    bool isEventDetected(const Edge edge = Edge::Rising);

private:
    gpiod_line* myLine{nullptr};
    std::uint8_t myActiveHigh{1};
    std::uint8_t myLastInput{!myActiveHigh};
};

} // namespace rpi
} // namespace yrgo