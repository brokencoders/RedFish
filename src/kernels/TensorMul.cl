#define float64 double

// C = A * B
// A size (M, K)
// B size (K, N)
// C size (M, N)
__kernel void tensor_tensor_math_mul(const int M, const int N, const int K, const __global float64* A, const __global float64* B, __global float64* C) 
{
    #define TS 10

    const int row = get_local_id(0);                  // Local row ID (max: TS)
    const int col = get_local_id(1);                  // Local col ID (max: TS)

    const int globalRow = TS * get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS * get_group_id(1) + col; // Col ID of C (0..N)

    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    float64 acc = 0.0f;
    
    const int numTiles = K / TS;
    for (int t = 0; t < numTiles; t++) 
    {
        const int tiledRow = TS * t + row;
        const int tiledCol = TS * t + col;

        Asub[col][row] = A[tiledCol * M + globalRow];
        Bsub[col][row] = B[globalCol * K + tiledRow];
 
        barrier(CLK_LOCAL_MEM_FENCE);
 
        for (int k=0; k < TS; k++) {
            acc += Asub[k][row] * Bsub[col][k];
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    C[globalCol * M + globalRow] = acc;
}