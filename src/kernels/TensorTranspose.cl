#define float64 double

kernel void tensor_transpose_blocked(const int rows, const int cols, constant float64* A, global float64* B) 
{
    #define TS 16
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)

    const int globalRow = TS * get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS * get_group_id(1) + col; // Col ID of C (0..N)

    B[globalCol * rows + globalRow] = A[globalCol + globalRow * cols];
}

kernel void tensor_transpose(const int rows, const int cols, constant float64* A, global float64* B) 
{
    const int globalRow = get_group_id(0); // Row ID of C (0..M)
    const int globalCol = get_group_id(1); // Col ID of C (0..N)
    
    B[globalCol * rows + globalRow] = A[globalCol + globalRow * cols];
}
