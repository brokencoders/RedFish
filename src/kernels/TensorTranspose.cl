#ifndef float64
#define float64 double
#endif

kernel void tensor_transpose(const int rows, const int cols, constant float64* A, global float64* B) 
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);

    const int globalRow = TS2 * get_group_id(0) + row;
    const int globalCol = TS2 * get_group_id(1) + col;

    local float64 Asub[TS2][TS2];
    local float64 Bsub[TS2][TS2];

    if (globalCol < cols && globalRow < rows)
    {
        Asub[row][col] = A[globalRow*cols + globalCol];
        barrier(CLK_LOCAL_MEM_FENCE);

        Bsub[col][row] = Asub[row][col];
        barrier(CLK_LOCAL_MEM_FENCE);

        B[globalCol*rows + globalRow] = Bsub[col][row];
    }
}
