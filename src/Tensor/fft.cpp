#include "Tensor.h"
#include <complex>

namespace RedFish
{
    uint8_t reverse(uint8_t b)
    {
        b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
        b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
        b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
        return b;
    }


    size_t reverse_bits(size_t i, size_t shift)
    {
        size_t ret;
        uint8_t *bt = (uint8_t *)&i;
        uint8_t *btr = (uint8_t *)&ret;
        if constexpr (sizeof(size_t) > 0)
            btr[sizeof(size_t) - 1] = reverse(bt[0]);
        if constexpr (sizeof(size_t) > 1)
            btr[sizeof(size_t) - 2] = reverse(bt[1]);
        if constexpr (sizeof(size_t) > 2)
            btr[sizeof(size_t) - 3] = reverse(bt[2]);
        if constexpr (sizeof(size_t) > 3)
            btr[sizeof(size_t) - 4] = reverse(bt[3]);
        if constexpr (sizeof(size_t) > 4)
            btr[sizeof(size_t) - 5] = reverse(bt[4]);
        if constexpr (sizeof(size_t) > 5)
            btr[sizeof(size_t) - 6] = reverse(bt[5]);
        if constexpr (sizeof(size_t) > 6)
            btr[sizeof(size_t) - 7] = reverse(bt[6]);
        if constexpr (sizeof(size_t) > 7)
            btr[sizeof(size_t) - 8] = reverse(bt[7]);
        if constexpr (sizeof(size_t) > 8)
            btr[sizeof(size_t) - 9] = reverse(bt[8]);
        if constexpr (sizeof(size_t) > 9)
            btr[sizeof(size_t) - 10] = reverse(bt[9]);
        return ret >> shift;
    }

    void fft_impl(std::complex<float64> *dst, const float64 *src, size_t n)
    {
        constexpr float64 pi = 3.1415926535897931;
        constexpr float64 pi2 = 2 * pi;
        size_t shift = sizeof(size_t) * 8 - std::log2(n);
        for (size_t k = 0; k < n; k++)
            dst[reverse_bits(k, shift)] = src[k];

        for (size_t m = 2; m <= n; m *= 2)
        {
            using namespace std;
            complex<float64> wm = exp(-pi2 / m * 1i);
            for (size_t k = 0; k < n; k += m)
            {
                complex<float64> w = 1.;
                for (size_t j = 0; j < m / 2; j++)
                {
                    auto t = dst[k + j + m / 2] * w;
                    auto u = dst[k + j];
                    dst[k + j] = u + t;
                    dst[k + j + m / 2] = u - t;
                    w *= wm;
                }
            }
        }
    }

    float64 *fft_impl_reversed(const float64 *src, size_t n)
    {
        constexpr float64 pi = 3.1415926535897931;
        constexpr float64 pi2 = 2 * pi;
        struct complex
        {
            float64 re, im;
            constexpr complex(float64 n) : re(n), im(0) {}
            constexpr complex(float64 re, float64 im) : re(re), im(im) {}
            constexpr complex operator+(const complex &c) const { return {re + c.re, im + c.im}; }
            constexpr complex operator-(const complex &c) const { return {re - c.re, im - c.im}; }
            constexpr complex operator*(const complex &c) const { return {re * c.re - im * c.im, im * c.re + re * c.im}; }
        };

        constexpr auto fft_step = [](complex *X, complex *x, size_t n, size_t s)
        {
            if (n == 2)
            {
                for (int q = 0; q < s; q++)
                {
                    const complex a = x[q + 0];
                    const complex b = x[q + s];
                    x[q + 0] = a + b;
                    x[q + s] = a - b;
                }
            }
            else
            {
                constexpr complex j = {0., 1.};
                const int n1 = n / 4;
                const int n2 = n / 2;
                const int n3 = n1 + n2;
                const float64 theta0 = pi2 / n;
                const complex wn = {std::cos(theta0), -std::sin(theta0)};
                complex w1k = 1.;
                for (size_t k = 0; k < n1; k++)
                {
                    const complex w2k = w1k * w1k;
                    const complex w3k = w1k * w2k;
                    for (size_t q = 0; q < s; q++)
                    {
                        const complex a = x[q + s * (k + 0)];
                        const complex b = x[q + s * (k + n1)];
                        const complex c = x[q + s * (k + n2)];
                        const complex d = x[q + s * (k + n3)];
                        const complex apc = a + c;
                        const complex amc = a - c;
                        const complex bpd = b + d;
                        const complex jbmd = j * (b - d);
                        X[q + s * (4 * k + 0)] = apc + bpd;
                        X[q + s * (4 * k + 1)] = w1k * (amc - jbmd);
                        X[q + s * (4 * k + 2)] = w2k * (apc - bpd);
                        X[q + s * (4 * k + 3)] = w3k * (amc + jbmd);
                    }
                    w1k = w1k * wn;
                }
            }
        };

        complex *X = (complex *)std::aligned_alloc(32, 2 * n * sizeof(float64)), *x = (complex *)std::aligned_alloc(32, 2 * n * sizeof(float64));

        for (size_t i = 0; i < n; i++)
        {
            x[i] = src[i];
        }

        for (size_t m = n, s = 1; m > 1; m /= 4, s *= 4)
        {
            fft_step(X, x, m, s);
            std::swap(x, X);
        }

        std::free(X);
        return (float64 *)x;
    }

    void ifft_impl_reversed(float64 *dst, std::complex<float64> *src, size_t n)
    {
        constexpr float64 pi = 3.1415926535897931;
        constexpr float64 pi2 = 2 * pi;

        for (size_t m = 2; m <= n; m *= 2)
        {
            using namespace std;
            complex<float64> wm = exp(pi2 / m * 1i);
            for (size_t k = 0; k < n; k += m)
            {
                complex<float64> w = 1.;
                for (size_t j = 0; j < m / 2; j++)
                {
                    auto u = src[k + j];
                    auto t = src[k + j + m / 2] * w;
                    src[k + j] = (u + t) / 2.;
                    src[k + j + m / 2] = (u - t) / 2.;
                    w *= wm;
                }
            }
        }
        for (size_t k = 0; k < n; k++)
            dst[k] = src[k].real();
    }

    void fft2d_impl_reversed(std::complex<float64> *dst, const float64 *src, size_t n)
    {
        constexpr float64 pi = 3.1415926535897931;
        constexpr float64 pi2 = 2 * pi;
        constexpr size_t block_size = 32;
        for (size_t k = 0; k < n * n; k++)
            dst[k] = src[k];
        ;
        for (size_t m = n; m > 1; m /= 2)
        {
            using namespace std;
            complex<float64> wm = exp(-pi2 / m * 1i);
            //#pragma omp parallel for
            for (size_t l = 0; l < n; l += m)
                for (size_t k = 0; k < n; k += m)
                {
                    complex<float64> wr = 1.;
                    for (size_t i = 0; i < m / 2; i++)
                    {
                        complex<float64> wc = 1.;
                        for (size_t j = 0; j < m / 2; j++)
                        {
                            auto s00 = dst[(l + i) * n + k + j];
                            auto s01 = dst[(l + i) * n + k + j + m / 2];
                            auto s10 = dst[(l + i + m / 2) * n + k + j];
                            auto s11 = dst[(l + i + m / 2) * n + k + j + m / 2];
                            dst[(l + i) * n + k + j] = s00 + s01 + s10 + s11;
                            dst[(l + i) * n + k + j + m / 2] = (s00 - s01 + s10 - s11) * wc;
                            dst[(l + i + m / 2) * n + k + j] = (s00 + s01 - s10 - s11) * wr;
                            dst[(l + i + m / 2) * n + k + j + m / 2] = (s00 - s01 - s10 + s11) * wc * wr;
                            wc *= wm;
                        }
                        wr *= wm;
                    }
                }
        }
    }

    void ifft2d_impl_reversed(float64 *dst, std::complex<float64> *src, size_t n)
    {
        constexpr float64 pi = 3.1415926535897931;
        constexpr float64 pi2 = 2 * pi;

        for (size_t m = 2; m <= n; m *= 2)
        {
            using namespace std;
            complex<float64> wm = exp(pi2 / m * 1i);
            //#pragma omp parallel for
            for (size_t l = 0; l < n; l += m)
                for (size_t k = 0; k < n; k += m)
                {
                    complex<float64> wr = 1.;
                    for (size_t i = 0; i < m / 2; i++)
                    {
                        complex<float64> wc = 1.;
                        for (size_t j = 0; j < m / 2; j++)
                        {
                            auto s00 = src[(l + i) * n + k + j];
                            auto s01 = src[(l + i) * n + k + j + m / 2] * wc;
                            auto s10 = src[(l + i + m / 2) * n + k + j] * wr;
                            auto s11 = src[(l + i + m / 2) * n + k + j + m / 2] * wc * wr;
                            src[(l + i) * n + k + j] = (s00 + s01 + s10 + s11) / 4.;
                            src[(l + i) * n + k + j + m / 2] = (s00 - s01 + s10 - s11) / 4.;
                            src[(l + i + m / 2) * n + k + j] = (s00 + s01 - s10 - s11) / 4.;
                            src[(l + i + m / 2) * n + k + j + m / 2] = (s00 - s01 - s10 + s11) / 4.;
                            wc *= wm;
                        }
                        wr *= wm;
                    }
                }
        }
        for (size_t k = 0; k < n * n; k++)
            dst[k] = src[k].real();
    }
}