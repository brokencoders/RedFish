#include "Tensor.h"

namespace RedFish
{
    void matmul_gotoblas(float64 *dst, const float64 *m1, const float64 *m2, size_t rows, size_t mid, size_t cols, size_t ld0, size_t ld1, size_t ld2)
    {
        const size_t block_size = 8;
        size_t j_end = rows - rows % block_size;
        size_t i_end = cols - cols % block_size;
        size_t k_end = mid - mid % block_size;

        //#pragma omp parallel for
        for (size_t jc = 0; jc < j_end; jc += block_size)
        {
            for (size_t kc = 0; kc < k_end; kc += block_size)
            {
                for (size_t ic = 0; ic < i_end; ic += block_size)
                    for (size_t jr = jc; jr < jc + block_size; jr++)
                        for (size_t ir = ic; ir < ic + block_size; ir++)
                            for (size_t k = kc; k < kc + block_size; k++)
                                dst[jr * ld0 + ir] += m1[jr * ld1 + k] * m2[ir + k * ld2];

                    for (size_t jr = jc; jr < jc + block_size; jr++)
                        for (size_t ir = i_end; ir < cols; ir++)
                            for (size_t k = kc; k < kc + block_size; k++)
                                dst[jr * ld0 + ir] += m1[jr * ld1 + k] * m2[ir + k * ld2];
            }

                for (size_t ic = 0; ic < i_end; ic += block_size)
                    for (size_t jr = jc; jr < jc + block_size; jr++)
                        for (size_t ir = ic; ir < ic + block_size; ir++)
                            for (size_t k = k_end; k < mid; k++)
                                dst[jr * ld0 + ir] += m1[jr * ld1 + k] * m2[ir + k * ld2];

                    for (size_t jr = jc; jr < jc + block_size; jr++)
                        for (size_t ir = i_end; ir < cols; ir++)
                            for (size_t k = k_end; k < mid; k++)
                                dst[jr * ld0 + ir] += m1[jr * ld1 + k] * m2[ir + k * ld2];
        }
        for (size_t kc = 0; kc < k_end; kc += block_size)
        {
            for (size_t ic = 0; ic < i_end; ic += block_size)
                for (size_t jr = j_end; jr < rows; jr++)
                    for (size_t ir = ic; ir < ic + block_size; ir++)
                        for (size_t k = kc; k < kc + block_size; k++)
                            dst[jr * ld0 + ir] += m1[jr * ld1 + k] * m2[ir + k * ld2];

            for (size_t jr = j_end; jr < rows; jr++)
                for (size_t ir = i_end; ir < cols; ir++)
                    for (size_t k = kc; k < kc + block_size; k++)
                        dst[jr * ld0 + ir] += m1[jr * ld1 + k] * m2[ir + k * ld2];
        }

        for (size_t ic = 0; ic < i_end; ic += block_size)
            for (size_t jr = j_end; jr < rows; jr++)
                for (size_t ir = ic; ir < ic + block_size; ir++)
                    for (size_t k = k_end; k < mid; k++)
                        dst[jr * ld0 + ir] += m1[jr * ld1 + k] * m2[ir + k * ld2];

        for (size_t jr = j_end; jr < rows; jr++)
            for (size_t ir = i_end; ir < cols; ir++)
                for (size_t k = k_end; k < mid; k++)
                    dst[jr * ld0 + ir] += m1[jr * ld1 + k] * m2[ir + k * ld2];
    }

    void matmul_left_T(float64 *dst, const float64 *m1, const float64 *m2, size_t rows, size_t mid, size_t cols, size_t ld0, size_t ld1, size_t ld2)
    {
        const size_t block_size = 256;
        size_t j_end = rows - rows % block_size;
        size_t i_end = cols - cols % block_size;
        size_t k_end = mid - mid % block_size;

        for (size_t jc = 0; jc < j_end; jc += block_size)
        {
            for (size_t kc = 0; kc < k_end; kc += block_size)
            {
                for (size_t ic = 0; ic < i_end; ic += block_size)
                    for (size_t jr = jc; jr < jc + block_size; jr++)
                        for (size_t k = kc; k < kc + block_size; k++)
                            for (size_t ir = ic; ir < ic + block_size; ir++)
                                dst[jr * ld0 + ir] += m1[jr + k * ld1] * m2[ir + k * ld2];

                for (size_t jr = jc; jr < jc + block_size; jr++)
                    for (size_t ir = i_end; ir < cols; ir++)
                        for (size_t k = kc; k < kc + block_size; k++)
                            dst[jr * ld0 + ir] += m1[jr + k * ld1] * m2[ir + k * ld2];
            }

            for (size_t ic = 0; ic < i_end; ic += block_size)
                for (size_t jr = jc; jr < jc + block_size; jr++)
                    for (size_t k = k_end; k < mid; k++)
                        for (size_t ir = ic; ir < ic + block_size; ir++)
                            dst[jr * ld0 + ir] += m1[jr + k * ld1] * m2[ir + k * ld2];

            for (size_t jr = jc; jr < jc + block_size; jr++)
                for (size_t k = k_end; k < mid; k++)
                    for (size_t ir = i_end; ir < cols; ir++)
                        dst[jr * ld0 + ir] += m1[jr + k * ld1] * m2[ir + k * ld2];
        }
        for (size_t kc = 0; kc < k_end; kc += block_size)
        {
            for (size_t ic = 0; ic < i_end; ic += block_size)
                for (size_t jr = j_end; jr < rows; jr++)
                    for (size_t k = kc; k < kc + block_size; k++)
                        for (size_t ir = ic; ir < ic + block_size; ir++)
                            dst[jr * ld0 + ir] += m1[jr + k * ld1] * m2[ir + k * ld2];

            for (size_t jr = j_end; jr < rows; jr++)
                for (size_t k = kc; k < kc + block_size; k++)
                    for (size_t ir = i_end; ir < cols; ir++)
                        dst[jr * ld0 + ir] += m1[jr + k * ld1] * m2[ir + k * ld2];
        }

        for (size_t ic = 0; ic < i_end; ic += block_size)
            for (size_t jr = j_end; jr < rows; jr++)
                for (size_t k = k_end; k < mid; k++)
                    for (size_t ir = ic; ir < ic + block_size; ir++)
                        dst[jr * ld0 + ir] += m1[jr + k * ld1] * m2[ir + k * ld2];

        for (size_t jr = j_end; jr < rows; jr++)
            for (size_t k = k_end; k < mid; k++)
                for (size_t ir = i_end; ir < cols; ir++)
                    dst[jr * ld0 + ir] += m1[jr + k * ld1] * m2[ir + k * ld2];
    }

}