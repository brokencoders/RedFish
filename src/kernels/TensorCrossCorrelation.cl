#define float64 double

kernel void tensor_tensor_cross_correlation_1d(const int s_size,
                                               const int d_size,
                                               const int k_size,
                                               const int stride,
                                               const int dilation,
                                               global   float64* dst,
                                               constant float64* src,
                                               constant float64* kern) 
{
    #define TS 16
    const int lcol = get_local_id(0);

    const int col = TS * get_group_id(0) + lcol;
    const int col_stride = col * stride;

    float64 acc = 0.;

    for (size_t c = 0; c < k_size; c++)
        acc += src[col_stride + c * dilation] * kern[c];

    dst[col] = acc;
}


kernel void tensor_tensor_cross_correlation_2d(const int s_size_x,   const int s_size_y,
                                               const int d_size_x,   const int d_size_y,
                                               const int k_size_x,   const int k_size_y,
                                               const int stride_x,   const int stride_y,
                                               const int dilation_x, const int dilation_y,
                                               global   float64* dst,
                                               constant float64* src,
                                               constant float64* kern) 
{
    #define TS 16
    const int lrow = get_local_id(0); // Local row ID (max: TS)
    const int lcol = get_local_id(1); // Local col ID (max: TS)

    const int row = TS * get_group_id(0) + lrow; // Row ID of C (0..M)
    const int col = TS * get_group_id(1) + lcol; // Col ID of C (0..N)
    const int row_stride_y = row * stride_y;
    const int col_stride_x = col * stride_x;
    const int off = row_stride_y*s_size_x + col_stride_x;

    float64 acc = 0.;

    for (size_t r = 0; r < k_size_y; r++)
    for (size_t c = 0; c < k_size_x; c++)
        acc += src[off + r * dilation_y * s_size_x + c * dilation_x] * kern[r*k_size_x + c];

    dst[row*d_size_x + col] = acc;
}
