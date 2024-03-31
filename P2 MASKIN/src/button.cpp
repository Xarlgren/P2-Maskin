#include "button.hpp"

namespace yrgo 
{
namespace rpi
{

// -----------------------------------------------------------------------------
Button::Button() = default;

// -----------------------------------------------------------------------------
Button::Button(const std::uint8_t pin, const std::uint8_t activeHigh)
{
    init(pin, activeHigh);
}

// -----------------------------------------------------------------------------
Button::~Button() { gpiod_line_release(myLine); }

// -----------------------------------------------------------------------------
std::uint8_t Button::pin() const { return gpiod_line_offset(myLine); }

// -----------------------------------------------------------------------------
bool Button::init(const std::uint8_t pin, const std::uint8_t activeHigh)
{
    if (myLine != nullptr) { return false; }
    myLine = gpiod_line_new(pin, GPIOD_LINE_DIRECTION_IN);
    myActiveHigh = activeHigh;
    return myLine != nullptr;
}

// -----------------------------------------------------------------------------
bool Button::isPressed() 
{ 
    myLastInput = gpiod_line_get_value(myLine);
    return myActiveHigh != 0 ? myLastInput : !myLastInput;
}

// -----------------------------------------------------------------------------
bool Button::isEventDetected(const Edge edge)
{
    return gpiod_line_event_detected(myLine, static_cast<enum gpiod_line_edge>(edge), &myLastInput);
}

} // namespace rpi
} // namespace yrgo