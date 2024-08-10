#pragma once

template<typename T>
class With
{
public:
    With(T& variable)
        : var(variable), original_value(variable)
    {
    }
    With(T& variable, T temporary_value)
        : var(variable), original_value(variable)
    {
        variable = temporary_value;
    }
    ~With()
    {
        var = original_value;
    }

private:
    T& var;
    T original_value;
};
