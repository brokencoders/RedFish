#pragma once

template<int size, typename T>
inline void swap(T& n)
{
    char* b = (char*)&n;
    int i = 0, e = size-1;
    while (i < e)
        std::swap(b[i++], b[e--]);
    n = *((T*)b);
}

template<typename Tp>
inline void swap_endian(Tp& n)
{
    swap<sizeof(Tp), Tp>(n);
}

template<typename Tp, typename... T>
void swap_endian(Tp& n, T&... in)
{
    swap<sizeof(Tp), Tp>(n);
    if (sizeof...(in) > 0) swap_endian(in...);
}
