#ifndef float64
#define float64 double
#endif

kernel void tensor_tensor_convolution_1d(const int s_size,
                                         const int d_size,
                                         const int k_size,
                                         const int stride,
                                         const int dilation,
                                         global   float64* dst,
                                         constant float64* src,
                                         constant float64* kern) 
{
    const int lcol = get_local_id(0);

    const int col = TS1 * get_group_id(0) + lcol;
    const int col_stride = col * stride;

    float64 acc = 0.;

    for (size_t c = 0; c < k_size; c++)
        acc += src[col_stride + c * dilation] * kern[k_size - c - 1];

    dst[col] = acc;
}


kernel void tensor_tensor_convolution_2d(const int s_size_x,   const int s_size_y,
                                         const int d_size_x,   const int d_size_y,
                                         const int k_size_x,   const int k_size_y,
                                         const int stride_x,   const int stride_y,
                                         const int dilation_x, const int dilation_y,
                                         global   float64* dst,
                                         constant float64* src,
                                         constant float64* kern) 
{
    const int lrow = get_local_id(0);
    const int lcol = get_local_id(1);

    const int row = TS2 * get_group_id(0) + lrow;
    const int col = TS2 * get_group_id(1) + lcol;
    const int row_stride_y = row * stride_y;
    const int col_stride_x = col * stride_x;
    const int off = row_stride_y*s_size_x + col_stride_x;

    float64 acc = 0.;

    for (size_t r = 0; r < k_size_y; r++)
    for (size_t c = 0; c < k_size_x; c++)
        acc += src[off + r * dilation_y * s_size_x + c * dilation_x] * kern[(k_size_y - r)*k_size_x - c - 1];

    dst[row*d_size_x + col] = acc;
}
