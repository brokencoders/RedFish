#ifndef float64
#define float64 double
#endif

kernel void tensor_tensor_cross_correlation_1d(const int s_size,
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
    const int lrow = get_local_id(0); // Local row ID (max: TS)
    const int lcol = get_local_id(1); // Local col ID (max: TS)

    const int row = TS2 * get_group_id(0) + lrow; // Row ID of C (0..M)
    const int col = TS2 * get_group_id(1) + lcol; // Col ID of C (0..N)
    const int row_stride_y = row * stride_y;
    const int col_stride_x = col * stride_x;
    const int off = row_stride_y*s_size_x + col_stride_x;

    float64 acc = 0.;

    for (size_t r = 0; r < k_size_y; r++)
    for (size_t c = 0; c < k_size_x; c++)
        acc += src[off + r * dilation_y * s_size_x + c * dilation_x] * kern[r*k_size_x + c];

    dst[row*d_size_x + col] = acc;
}


kernel void tensor_tensor_cross_correlation_1d(const int s_size,
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

    local float64 k_tile[TS1];
    float64 acc = 0.;

    const int numTiles = k_size / TS1;
    for (size_t t = 0; t < numTiles; t++)
    {
        k_tile[lcol] = kern[t*TS1 + lcol];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t c = 0; c < TS1; c++)
            acc += src[col_stride + (t*TS1 + c) * dilation] * k_tile[c];
    }

    if (k_size > numTiles*TS1)
    {
        if (lcol > k_size)
            k_tile[lcol] = 0;
        else
            k_tile[lcol] = kern[numTiles*TS1 + lcol];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t c = 0; c < TS1; c++)
            acc += src[col_stride + (numTiles*TS1 + c) * dilation] * k_tile[c];
    }

    dst[col] = acc;
}