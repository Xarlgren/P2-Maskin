#pragma once

#include <cstdint>

#include "gpiod_utils.h"

namespace yrgo 
{
namespace rpi 
{

class Led 
{
public:
    Led();
    Led(const std::uint8_t pin, const bool enabled = false);
    ~Led();

    std::uint8_t pin() const;
    bool isEnabled() const;

    bool init(const std::uint8_t pin, const bool enabled = false);

    void on();
    void off();
    void toggle();

    void blink(const std::uint16_t blinkSpeedMs);

private:
    gpiod_line* myLine{nullptr};
};

} // namespace rpi
} // namespace yrgo