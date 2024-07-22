#ifndef float64
#define float64 double
#endif

__kernel void tensor_tensor_math_mul(const int M, const int N, const int K, const __global float64* A, const __global float64* B, __global float64* C) 
{
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)

    const int globalRow = TS2 * get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS2 * get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local float64 Asub[TS2][TS2];
    __local float64 Bsub[TS2][TS2];   

    // Initialise the accumulation register
    float64 acc = 0.0f;

    const int numTiles = K/TS2;
    for (int t = 0; t < numTiles; t++) 
    {
        // Load one tile of A and B into local memory
        const int tiledRow = TS2 * t + row;
        const int tiledCol = TS2 * t + col;

        Asub[row][col] = A[tiledCol + globalRow * K];
        Bsub[row][col] = B[globalCol + tiledRow * N];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k = 0; k < TS2; k++)
            acc += Asub[row][k] * Bsub[k][col];
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    C[globalCol + globalRow * N] = acc;
}