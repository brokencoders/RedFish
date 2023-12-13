#include "Tensor.h"

namespace RedFish
{

    void conv_1d_impl(float64 *dst, const float64 *t, const float64 *kernel, size_t t_size, size_t kernel_size, size_t stride, size_t dilation)
    {
        constexpr size_t block_size = 32;
        size_t end = (t_size + stride - (kernel_size - 1) * dilation - 1) / stride;
        size_t end_b = end - end % block_size;

        for (size_t cb = 0; cb < end_b; cb += block_size)
            for (size_t ck = 0; ck < kernel_size; ck++)
                for (size_t c = cb; c < cb + block_size; c++)
                    dst[c] += t[c * stride + ck * dilation] * kernel[kernel_size - ck - 1];

        for (size_t ck = 0; ck < kernel_size; ck++)
            for (size_t c = end_b; c < end; c++)
                dst[c] += t[c * stride + ck * dilation] * kernel[kernel_size - ck - 1];
    }

    void conv_2d_impl(float64 *dst, const float64 *t, const float64 *kernel, Tuple2d t_size, Tuple2d kernel_size, Tuple2d stride, Tuple2d dilation)
    {
        constexpr size_t block_size_r = 4;
        constexpr size_t block_size_c = 32;
        Tuple2d end = {(t_size.y + stride.y - (kernel_size.y - 1) * dilation.y - 1) / stride.y,
                       (t_size.x + stride.x - (kernel_size.x - 1) * dilation.x - 1) / stride.x};
        Tuple2d end_b = {end.y - end.y % block_size_r, end.x - end.x % block_size_c};

        const auto conv1d = [=](float64 *dst, const float64 *t, const float64 *kernel)
        {
            for (size_t cb = 0; cb < end_b.x; cb += block_size_c)
                for (size_t ck = 0; ck < kernel_size.x; ck++)
                    for (size_t c = cb; c < cb + block_size_c; c++)
                        dst[c] += t[c * stride.w + ck * dilation.w] * kernel[kernel_size.w - ck - 1];

            for (size_t ck = 0; ck < kernel_size.x; ck++)
                for (size_t c = end_b.x; c < end.x; c++)
                    dst[c] += t[c * stride.w + ck * dilation.w] * kernel[kernel_size.w - ck - 1];
        };

        //#pragma omp parallel for
        for (size_t rb = 0; rb < end_b.y; rb += block_size_r)
            for (size_t rk = 0; rk < kernel_size.y; rk++)
                for (size_t r = rb; r < rb + block_size_r; r++)
                    conv1d(dst + r * end.w, t + (r * stride.h + rk * dilation.h) * t_size.w, kernel + (kernel_size.h - rk - 1) * kernel_size.w);

        for (size_t rk = 0; rk < kernel_size.y; rk++)
            for (size_t r = end_b.y; r < end.y; r++)
                conv1d(dst + r * end.w, t + (r * stride.h + rk * dilation.h) * t_size.w, kernel + (kernel_size.h - rk - 1) * kernel_size.w);
    }

    void cross_correlation_1d_impl(float64 *dst, const float64 *t, const float64 *kernel, size_t t_size, size_t kernel_size, size_t stride, size_t dilation)
    {
        constexpr size_t block_size = 32;
        size_t end = (t_size + stride - (kernel_size - 1) * dilation - 1) / stride;
        size_t end_b = end - end % block_size;

        for (size_t cb = 0; cb < end_b; cb += block_size)
            for (size_t ck = 0; ck < kernel_size; ck++)
                for (size_t c = cb; c < cb + block_size; c++)
                    dst[c] += t[c * stride + ck * dilation] * kernel[ck];

        for (size_t ck = 0; ck < kernel_size; ck++)
            for (size_t c = end_b; c < end; c++)
                dst[c] += t[c * stride + ck * dilation] * kernel[ck];
    }

    void cross_correlation_2d_impl(float64 *dst, const float64 *t, const float64 *kernel, Tuple2d t_size, Tuple2d kernel_size, Tuple2d stride, Tuple2d dilation)
    {
        constexpr size_t block_size_r = 4;
        constexpr size_t block_size_c = 32;
        Tuple2d end = {(t_size.y + stride.y - (kernel_size.y - 1) * dilation.y - 1) / stride.y,
                       (t_size.x + stride.x - (kernel_size.x - 1) * dilation.x - 1) / stride.x};
        Tuple2d end_b = {end.y - end.y % block_size_r, end.x - end.x % block_size_c};

        const auto conv1d = [=](float64 *dst, const float64 *t, const float64 *kernel)
        {
            for (size_t cb = 0; cb < end_b.x; cb += block_size_c)
                for (size_t ck = 0; ck < kernel_size.x; ck++)
                    for (size_t c = cb; c < cb + block_size_c; c++)
                        dst[c] += t[c * stride.w + ck * dilation.w] * kernel[ck];

            for (size_t ck = 0; ck < kernel_size.x; ck++)
                for (size_t c = end_b.x; c < end.x; c++)
                    dst[c] += t[c * stride.w + ck * dilation.w] * kernel[ck];
        };

        //#pragma omp parallel for
        for (size_t rb = 0; rb < end_b.y; rb += block_size_r)
            for (size_t rk = 0; rk < kernel_size.y; rk++)
                for (size_t r = rb; r < rb + block_size_r; r++)
                    conv1d(dst + r * end.w, t + (r * stride.h + rk * dilation.h) * t_size.w, kernel + rk * kernel_size.w);

        for (size_t rk = 0; rk < kernel_size.y; rk++)
            for (size_t r = end_b.y; r < end.y; r++)
                conv1d(dst + r * end.w, t + (r * stride.h + rk * dilation.h) * t_size.w, kernel + rk * kernel_size.w);
    }

    void cross_correlation_3d_impl(float64 *dst, const float64 *t, const float64 *kernel, Tuple3d t_size, Tuple3d kernel_size, Tuple3d stride, Tuple3d dilation)
    {
        constexpr size_t block_size_r = 4;
        constexpr size_t block_size_c = 32;
        Tuple3d end = {0, (t_size.y + stride.y - (kernel_size.y - 1) * dilation.y - 1) / stride.y,
                       (t_size.x + stride.x - (kernel_size.x - 1) * dilation.x - 1) / stride.x};
        Tuple3d end_b = {0, end.y - end.y % block_size_r, end.x - end.x % block_size_c};

        const auto conv1d = [=](float64 *dst, const float64 *t, const float64 *kernel)
        {
            for (size_t cb = 0; cb < end_b.x; cb += block_size_c)
                for (size_t ck = 0; ck < kernel_size.x; ck++)
                    for (size_t c = cb; c < cb + block_size_c; c++)
                        dst[c] += t[c * stride.w + ck * dilation.w] * kernel[ck];

            for (size_t ck = 0; ck < kernel_size.x; ck++)
                for (size_t c = end_b.x; c < end.x; c++)
                    dst[c] += t[c * stride.w + ck * dilation.w] * kernel[ck];
        };

        const auto conv2d = [=](float64 *dst, const float64 *t, const float64 *kernel)
        {
            for (size_t rb = 0; rb < end_b.y; rb += block_size_r)
                for (size_t rk = 0; rk < kernel_size.y; rk++)
                    for (size_t r = rb; r < rb + block_size_r; r++)
                        conv1d(dst + r * end.w, t + (r * stride.h + rk * dilation.h) * t_size.w, kernel + rk * kernel_size.w);

            for (size_t rk = 0; rk < kernel_size.y; rk++)
                for (size_t r = end_b.y; r < end.y; r++)
                    conv1d(dst + r * end.w, t + (r * stride.h + rk * dilation.h) * t_size.w, kernel + rk * kernel_size.w);
        };

        for (size_t dk = 0; dk < kernel_size.z; dk++)
            for (size_t d = 0; d < end.z; d++)
                conv2d(dst + d * end.h * end.w, t + (d * stride.d + dk * dilation.d) * t_size.h * t_size.w, kernel + dk * kernel_size.h * kernel_size.w);
    }

}