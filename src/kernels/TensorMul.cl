#define float64 double

__kernel void tensor_tensor_math_mul(const int M, const int N, const int K, const __global float64* A, const __global float64* B, __global float64* C) 
{
    #define TS 16
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)

    const int globalRow = TS * get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS * get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];   

    // Initialise the accumulation register
    float acc = 0.0f;

    const int numTiles = K/TS;
    for (int t = 0; t < numTiles; t++) 
    {
        // Load one tile of A and B into local memory
        const int tiledRow = TS * t + row;
        const int tiledCol = TS * t + col;

        Asub[row][col] = A[tiledCol + globalRow * K];
        Bsub[row][col] = B[globalCol + tiledRow * N];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k = 0; k < TS; k++)
            acc += Asub[row][k] * Bsub[k][col];
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    C[globalCol + globalRow * N] = acc;
}