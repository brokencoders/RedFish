#pragma once

#include <iostream>
#include <functional>
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cfloat>

#ifdef _WIN32
namespace std
{
    inline void* aligned_alloc(size_t alignment, size_t size) { return _aligned_malloc(size, alignment); }
}
#endif

namespace Algebra
{
    typedef double float64;

    class MatrixView;

    class Matrix
    {
    protected:
        Matrix(float64* m, size_t row, size_t col, size_t roff) : m(m), row(row), col(col), roff(roff), size(row*col), own(false) {}

    public:
        Matrix(size_t length = 0);
        Matrix(size_t row, size_t col);
        Matrix(const std::initializer_list<float64> &lst);
        Matrix(size_t row, size_t col, const std::initializer_list<float64> &lst);
        Matrix(const Matrix &mat);
        Matrix(Matrix &&mat);
        Matrix(const MatrixView& mv);
        ~Matrix();

        Matrix &operator=(const Matrix &m) &;
        Matrix &operator=(const Matrix &m) && = delete;
        Matrix &operator=(const MatrixView &m) &;
        Matrix &operator=(const MatrixView &m) && = delete;

        Matrix  operator+(const Matrix &m) const &;
        Matrix& operator+(const Matrix &m) && { return operator+=(m); }
        Matrix  operator+(float64 s) const &;
        Matrix& operator+(float64 s) && { return operator+=(s); }
        Matrix& operator+=(const Matrix &m);
        Matrix& operator+=(float64 s);

        Matrix  operator-(const Matrix &m) const &;
        Matrix& operator-(const Matrix &m) && { return operator-=(m); }
        Matrix  operator-() const &;
        Matrix& operator-() &&;
        Matrix  operator-(float64 s) const &;
        Matrix& operator-(float64 s) && { return operator-=(s); }
        Matrix& operator-=(const Matrix &m);
        Matrix& operator-=(float64 s);

        Matrix  operator*(const Matrix &m) const;
        Matrix  operator*(float64 s) const &;
        Matrix& operator*(float64 s) && { return operator*=(s); }
        Matrix& operator*=(const Matrix &m);
        Matrix& operator*=(float64 s);
        Matrix  transposeTimes(const Matrix &m) const;

        Matrix  operator/(const Matrix& m) const &;
        Matrix& operator/(const Matrix& m) && { return operator/=(m); }
        Matrix  operator/(float64 s) const &;
        Matrix& operator/(float64 s) && { return operator/=(s); }
        Matrix& operator/=(const Matrix& m);
        Matrix& operator/=(float64 s);

        explicit operator float64() const;

        float64 norm() const;
        float64 pnorm(size_t p) const;
        float64 normSquare() const;
        float64 sum() const;
        Matrix trace() const;
        Matrix rowPermutation(const std::vector<size_t>& perm_v) const;
        Matrix upTriInverse();
        Matrix forEach(std::function<float64(float64)> fn) const;
        Matrix forEachRow(std::function<Matrix(Matrix)> fn) const;

        Matrix conv(const Matrix& input, const Matrix& kernel, bool padding) const;

        float64 max() const;
        float64 absMax() const;
        bool hasNaN() const;

        Matrix  hom() const;
        Matrix  hom_i() const &;
        Matrix& hom_i() &&;

        Matrix solve(Matrix b) const;
        Matrix cholesky() const;
        Matrix partialQR(int64_t t = 0, int64_t l = 0, int64_t b = -1, int64_t r = -1) const;
        std::tuple<Matrix, Matrix> QR() const;
        std::tuple<Matrix, Matrix> tridiagQR() const;
        std::tuple<Matrix, Matrix> implicitelyShiftedQR(const float64 threshold = 1e-12) const;
        Matrix solveMin() const;
        std::tuple<Matrix, Matrix, Matrix> svd() const;

        Matrix& reshape(size_t row, size_t col);
        Matrix& resize(size_t row, size_t col);

        inline float64& operator()(size_t i);
        inline float64& operator()(size_t i, size_t j);
        inline const float64& operator()(size_t i) const;
        inline const float64& operator()(size_t i, size_t j) const;
        inline float64* operator[](size_t i);
        inline const float64* operator[](size_t i) const;
        
        Matrix& vstack(const Matrix& mat);
        Matrix& vstack(const std::vector<Matrix>& mats);
        Matrix& hstack(const Matrix& mat);

        void setCol(int64_t c, const std::initializer_list<float64> &lst);
        void setCol(int64_t c, const Matrix& v);
        void setRow(int64_t r, const std::initializer_list<float64> &lst);
        void setRow(int64_t r, const Matrix& v);
        void setSubMatrix(const Matrix& mat, int64_t r = 0, int64_t c = 0, bool transpose = false);
        Matrix getCol(int64_t c) const;
        //MatrixView getCol(int64_t c);
        Matrix getRow(int64_t r) const;
        //MatrixView getRow(int64_t r);
        Matrix subMatrix(int64_t top, int64_t left, int64_t bottom = -1, int64_t right = -1) const;
        //MatrixView subMatrix(int64_t top, int64_t left, int64_t bottom = -1, int64_t right = -1);

        void print() const;
        Matrix& transpose();
        Matrix T() const &;
        Matrix T() &&;

    private:

        inline static float64* alloc(size_t size) { return (float64*)std::aligned_alloc(byte_alignment, size * sizeof(float64)); }
        inline static void   dealloc(void* ptr) {
#ifdef _WIN32
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
        }
        inline static constexpr size_t aligned(size_t len)  { return len == 1 ? 1 : ((float_alignment-1) & len ? (len & ~(float_alignment-1)) + float_alignment : len); }

        inline bool isVector() { return col == 1 || row == 1; }
        void checkRangeRow(int64_t& r) const;
        void checkRangeCol(int64_t& c) const;
        void checkRangeSubMatrix(int64_t& t, int64_t& l, int64_t& b, int64_t& r) const;

        static void realTranspose(const float64* A, float64* B, const size_t r, const size_t c, const size_t lda, const size_t ldb);
        static void realTransposeInPlace(float64*& A, const size_t r, const size_t c, const size_t lda, bool local_buff);
        static Matrix householderReflect(const Matrix& u);
        void householderReflectSubMatLeft(const Matrix& u, size_t r, size_t c, size_t cols = 0);
        void householderReflectSubMatRight(const Matrix& u, size_t r, size_t c, size_t rows = 0);
        void householderReflectSubMatForwardLeft(const Matrix& u, size_t r, size_t c, bool store = true);
        void householderReflectSubMatForwardRight(const Matrix& u, size_t r, size_t c, bool store = true);
        void givensRotateLeft(float64 c, float64 s, int64_t r1, int64_t r2, int64_t l = 0, int64_t r = -1);
        void givensRotateRight(float64 c, float64 s, int64_t c1, int64_t c2, int64_t t = 0, int64_t b = -1);

    private:
        float64 *m;
        size_t row, col, roff, size;
        bool own;

        static const size_t buff_size = 16;
        static const size_t byte_alignment = 32;
        static const size_t float_alignment = byte_alignment / sizeof(float64);
    public:
        union alignas(byte_alignment) {
            struct { float64 x,y,z,w; };
            struct { float64 r,g,b,a; };
            float64 buff[buff_size];
        };

    public:
        size_t rows() const { return row; }
        size_t cols() const { return col; }
        size_t getSize() const { return size; }

        friend void zero(Matrix&);
        friend void random_matrix(Matrix&, float64, float64);
        friend Matrix ones(size_t rows, size_t cols);
        friend Matrix ones_like(const Matrix&);
        friend Matrix  identity(size_t);
        friend Matrix  identity(size_t, size_t);
        friend Matrix  std_base_vector(size_t, size_t);
        friend Matrix  rodriguesToMatrix(Matrix);
        friend Matrix  matrixToRodrigues(Matrix);
        friend Matrix  operator+(float64, const Matrix&);
        friend Matrix& operator+(float64, Matrix&&);
        friend Matrix  operator-(float64, const Matrix&);
        friend Matrix& operator-(float64, Matrix&&);
        friend Matrix  operator*(float64, const Matrix&);
        friend Matrix& operator*(float64, Matrix&&);
        friend float64 operator/(float64, const Matrix&);
        friend float64 operator/=(float64, const Matrix&);
        friend Matrix  vstack(const std::vector<Matrix>& mats);
        friend Matrix  hstack(const std::vector<Matrix>& mats);
        friend Matrix  cross(const Matrix& v1, const Matrix& v2);
        friend int     upperTriangInvert(Matrix& m);
    };

    class MatrixView : public Matrix
    {
        MatrixView(float64* m, size_t row, size_t col, size_t roff) : Matrix(m, row, col, roff) {}

    public:

        Matrix& operator*=(const Matrix &m) = delete;
        Matrix& operator*=(float64 s) { return Matrix::operator*=(s); }

        Matrix realize() const { return Matrix(*this); }

        friend class Matrix;
    };

    using Vector = Matrix;
    void zero(Matrix& mat);
    Matrix ones(size_t rows, size_t cols = 1);
    Matrix ones_like(const Matrix& m);
    Matrix identity(size_t size);
    Matrix identity(size_t r, size_t c);
    Vector std_base_vector(size_t dim, size_t n = 0);
    Matrix rodriguesToMatrix(Matrix rodrigues_vector);
    Matrix matrixToRodrigues(Matrix rotation_matrix);
    Matrix cross(const Matrix& v1, const Matrix& v2);
    inline float64 square(float64 n);
    int upperTriangInvert(Matrix& m);

/* #ifdef ALGEBRA_IMPL */

    inline Matrix::Matrix(size_t length)
        : row(length), col(1), roff(1), size(length), own(true)
    {
        if (size <= buff_size)
            m = buff;
        else
            m = alloc(aligned(size));
    }

    inline Matrix::Matrix(size_t row, size_t col)
        : row(row), col(col), roff(aligned(col)), size(row * col), own(true)
    {
        if (row*roff <= buff_size)
            m = buff;
        else
            m = alloc(row*roff);

        if (roff != col)
        for (float64* r = m, *end = m + row*roff; r < end; r += roff)
            std::fill(r + col, r + roff, 0.0);
    }

    inline Matrix::Matrix(const std::initializer_list<float64> &lst)
        : row(lst.size()), col(1), roff(1), size(lst.size()), own(true)
    {
        if (row <= buff_size)
            m = buff;
        else
            m = alloc(aligned(row));
        std::copy(lst.begin(), lst.end(), m);
    }

    inline Matrix::Matrix(size_t row, size_t col, const std::initializer_list<float64> &lst)
        : row(row), col(col), roff(aligned(col)), size(row * col), own(true)
    {
        if (row*roff <= buff_size)
            m = buff;
        else
            m = alloc(row*roff);
        auto it = lst.begin();
        size_t r = 0;
        for (; r < row && it < lst.begin() + size; r++, it += col)
            std::copy(it, std::min(it + col, lst.begin() + size), m + r*roff),
            std::fill(m + r*roff + col, m + (r+1)*roff, 0.0);

        std::fill(m + (r-1)*roff + (lst.begin() + size - it + col), m + row*roff, 0.0);
    }

    inline Matrix::Matrix(const Matrix &mat)
        : row(mat.row), col(mat.col), roff(aligned(mat.col)), size(mat.size), own(true)
    {
        if (row*roff <= buff_size)
            m = buff;
        else
            m = alloc(row*roff);

        for (float64* r = m, *mt = mat.m, *end = m + row*roff; r < end; mt += mat.roff, r += roff)
            std::copy(mt, mt + col, r),
            std::fill(r + col, r + roff, 0.0);
        // std::cout << "Matrix copy constructor\n";
    }

    inline Matrix::Matrix(Matrix &&mat)
        : row(mat.row), col(mat.col), roff(mat.roff), size(mat.size), own(true)
    {
        if (mat.m != mat.buff)
        {
            m = mat.m;
            mat.m = nullptr;
            mat.row = mat.col = mat.size = 0;
            mat.own = false;
        }
        else
        {
            m = buff;
            std::copy(mat.m, mat.m + row*roff, m);
        }
        // std::cout << "Matrix move constructor\n";
    }

    inline Matrix::Matrix(const MatrixView &mat)
        : row(mat.row), col(mat.col), roff(aligned(mat.col)), size(mat.size), own(true)
    {
        if (row*roff <= buff_size)
            m = buff;
        else
            m = alloc(row*roff);

        for (float64* r = m, *mt = mat.m, *end = m + row*roff; r < end; mt += mat.roff, r += roff)
            std::copy(mt, mt + col, r),
            std::fill(r + col, r + roff, 0.0);
        // std::cout << "Matrix copy constructor\n";
    }

    inline Matrix::~Matrix()
    {
        if (m != buff && m && own)
            dealloc(m);
        own = false;
        // std::cout << "Matrix destructor\n";
    }

    inline Matrix& Matrix::operator=(const Matrix &mat) & /* TODO: check ownership */
    {
        size_t capacity = row*roff, mat_capacity = mat.row*mat.roff;
        if (mat_capacity > capacity || mat_capacity < capacity * .75)
        {
            if (m != buff)
                dealloc(m);
            if (mat_capacity > buff_size)
                m = alloc(mat_capacity);
            else
                m = buff;
        }
        if (&mat != this)
        {
            row = mat.row, col = mat.col, roff = mat.roff, size = mat.size;
            std::copy(mat.m, mat.m + mat_capacity, m);
        }
        return *this;
    }

    inline Matrix& Matrix::operator=(const MatrixView &mat) & /* TODO: check ownership */
    {
        size_t capacity = row*roff, mat_capacity = mat.row*(mat.col + (mat.col & 0b11 ? 4 : 0));
        if (mat_capacity > capacity || mat_capacity < capacity * .75)
        {
            if (m != buff)
                dealloc(m);
            if (mat_capacity > buff_size)
                m = alloc(mat_capacity);
            else
                m = buff;
        }
        if (&mat != this)
        {
            row = mat.row, col = mat.col, roff = mat.roff, size = mat.size;
            std::copy(mat.m, mat.m + mat.row*mat.roff, m);
        }
        return *this;
    }

    /* --------- SUM --------- */

    inline Matrix Matrix::operator+(const Matrix &mat) const &
    {
        if (row != mat.row || col != mat.col)
            throw std::length_error("Matrix sizes not matching in sum operation");
        Matrix sum(row, col);
        
        __m256d thisv, matv;
        const float64* matp = mat.m, *thisp = m;
        for (float64* sump = sum.m, *end = sum.m + row*roff; sump < end; sump+=4, matp+=4, thisp+=4)
        {
            thisv = _mm256_load_pd(thisp);
            matv  = _mm256_load_pd(matp);
            thisv = _mm256_add_pd(thisv, matv);
            _mm256_store_pd(sump, thisv);
        }
        
        return sum;
    }

    inline Matrix Matrix::operator+(float64 s) const &
    {
        Matrix sum(row, col);
        alignas(byte_alignment) float64 ss[] = { s,s,s,s };
        
        __m256d thisv, addv = _mm256_load_pd(ss);
        const float64* thisp = m;
        for (float64* sump = sum.m, *end = sum.m + row*roff; sump < end; sump+=4, thisp+=4)
        {
            thisv = _mm256_load_pd(thisp);
            thisv = _mm256_add_pd(thisv, addv);
            _mm256_store_pd(sump, thisv);
        }
        return sum;
    }

    inline Matrix operator+(float64 s, const Matrix& mat)
    {
        return mat + s;
    }

    inline Matrix& operator+(float64 s, Matrix&& mat)
    {
        return mat.operator+=(s);
    }

    inline Matrix& Matrix::operator+=(const Matrix &mat)
    {
        if (row != mat.row || col != mat.col)
            throw std::length_error("Matrix sizes not matching in sum operation");
        
        __m256d thisv, matv;
        const float64* matp = mat.m, *thisp = m;
        for (float64* thisp = m, *end = m + row*roff; thisp < end; matp+=4, thisp+=4)
        {
            thisv = _mm256_load_pd(thisp);
            matv  = _mm256_load_pd(matp);
            thisv = _mm256_add_pd(thisv, matv);
            _mm256_store_pd(thisp, thisv);
        }
        return *this;
    }

    inline Matrix& Matrix::operator+=(float64 s)
    {
        alignas(byte_alignment) float64 ss[] = { s,s,s,s };
        
        __m256d thisv, addv = _mm256_load_pd(ss);
        for (float64* thisp = m, *end = m + row*roff; thisp < end; thisp+=4)
        {
            thisv = _mm256_load_pd(thisp);
            thisv = _mm256_add_pd(thisv, addv);
            _mm256_store_pd(thisp, thisv);
        }
        return *this;
    }

    /* --------- SUB --------- */

    inline Matrix Matrix::operator-(const Matrix &mat) const &
    {
        if (row != mat.row || col != mat.col)
            throw std::length_error("Matrix sizes not matching in sub operation");
        Matrix sub(row, col);
        
        /* __m256d thisv, matv;
        const float64* matp = mat.m, *thisp = m;
        for (float64* subp = sub.m, *end = sub.m + row*roff; subp < end; subp+=4, matp+=4, thisp+=4)
        {
            thisv = _mm256_load_pd(thisp);
            matv  = _mm256_load_pd(matp);
            thisv = _mm256_sub_pd(thisv, matv);
            _mm256_store_pd(subp, thisv);
        } */
        for (size_t i = 0; i < row*roff; i++)
        {
            sub.m[i] = m[i] - mat.m[i];
        }
        
        return sub;
    }

    inline Matrix Matrix::operator-() const &
    {
        Matrix sub(row, col);
        
        for (size_t r = 0; r < row; r++) {
            float64* subp = &sub(r,0);
            const float64* thisp = &(*this)(r,0);
            for (size_t c = 0; c < col; c++)
                subp[c] = -thisp[c];
        }
        return sub;
    }

    inline Matrix& Matrix::operator-() &&
    {
        
        for (size_t r = 0; r < row; r++) {
            float64* thisp = &(*this)(r,0);
            for (size_t c = 0; c < col; c++)
                thisp[c] = -thisp[c];
        }
        return *this;
    }

    inline Matrix operator-(float64 s, const Matrix& mat)
    {
        Matrix sub(mat.row, mat.col);
        alignas(mat.byte_alignment) float64 ss[] = { s,s,s,s };
        
        __m256d thisv, addv = _mm256_load_pd(ss);
        const float64* thisp = mat.m;
        for (float64* subp = sub.m, *end = sub.m + mat.row*mat.roff; subp < end; subp+=4, thisp+=4)
        {
            thisv = _mm256_load_pd(thisp);
            thisv = _mm256_sub_pd(addv, thisv);
            _mm256_store_pd(subp, thisv);
        }
        return sub;
    }

    inline Matrix& operator-(float64 s, Matrix&& mat)
    {
        alignas(mat.byte_alignment) float64 ss[] = { s,s,s,s };
        
        __m256d thisv, addv = _mm256_load_pd(ss);
        for (float64* thisp = mat.m, *end = mat.m + mat.row*mat.roff; thisp < end; thisp+=4)
        {
            thisv = _mm256_load_pd(thisp);
            thisv = _mm256_sub_pd(addv, thisv);
            _mm256_store_pd(thisp, thisv);
        }
        return mat;
    }

    inline Matrix Matrix::operator-(float64 s) const &
    {
        Matrix sub(row, col);
        alignas(byte_alignment) float64 ss[] = { s,s,s,s };
        
        __m256d thisv, addv = _mm256_load_pd(ss);
        const float64* thisp = m;
        for (float64* subp = sub.m, *end = sub.m + row*roff; subp < end; subp+=4, thisp+=4)
        {
            thisv = _mm256_load_pd(thisp);
            thisv = _mm256_sub_pd(thisv, addv);
            _mm256_store_pd(subp, thisv);
        }
        return sub;
    }

    inline Matrix& Matrix::operator-=(const Matrix &mat)
    {
        if (row != mat.row || col != mat.col)
            throw std::length_error("Matrix sizes not matching in subtraction operation");
        
        __m256d thisv, matv;
        const float64* matp = mat.m, *thisp = m;
        for (float64* thisp = m, *end = m + row*roff; thisp < end; matp+=4, thisp+=4)
        {
            thisv = _mm256_load_pd(thisp);
            matv  = _mm256_load_pd(matp);
            thisv = _mm256_sub_pd(thisv, matv);
            _mm256_store_pd(thisp, thisv);
        }
        return *this;
    }

    inline Matrix& Matrix::operator-=(float64 s)
    {
        alignas(byte_alignment) float64 ss[] = { s,s,s,s };
        
        __m256d thisv, addv = _mm256_load_pd(ss);
        for (float64* thisp = m, *end = m + row*roff; thisp < end; thisp+=4)
        {
            thisv = _mm256_load_pd(thisp);
            thisv = _mm256_sub_pd(thisv, addv);
            _mm256_store_pd(thisp, thisv);
        }
        return *this;
    }

    /* --------- MULT --------- */

    inline Matrix Matrix::operator*(const Matrix &mat) const
    {
        if (col != mat.row)
            throw std::length_error("Matrix sizes not matching in multiplication operation");
        Matrix mul(row, mat.col);
        float64 tmp_buff[buff_size];
        float64* t = tmp_buff;
        auto t_roff = aligned(row);
        if (col*t_roff > buff_size) t = alloc(col*t_roff);
        realTranspose(m, t, row, col, roff, t_roff);
        
        for (size_t i = 0; i < row; i++)
        {
            //__m256d mulv = _mm256_load_pd(mul.m + i*mul.roff + j);
            for (size_t j = 0; j < mat.col; j++)
                //mul(i,j) = 0;
                mul(i,j) = t[i] * mat.m[j];
        }

        for (size_t k = 1; k < col; k++)
            for (size_t i = 0; i < row; i++)
            {
                float64 t_ik = t[i + k * t_roff];
                for (size_t j = 0; j < mat.col; j++)
                    mul(i,j) += t_ik * mat(k,j);
            }

        /* const size_t BS = 64;
        for (size_t kb = 0; kb < col; kb+=BS)
            for (size_t jb = 0; jb < mul.col; jb+=BS)
                for (size_t i = 0; i < mul.row; i++)
                    for (size_t k = kb; k < kb + BS; k++)
                    {
                        float64 t_ik = t[i + k * t_roff];
                        for (size_t j = jb; j < jb + BS; j++)
                        {
                            mul(i,j) += t_ik * mat(k,j);
                        }
                    } */

        
        if (t != tmp_buff)
            dealloc(t);
        return mul;
    }

    inline Matrix Matrix::operator*(float64 s) const &
    {
        Matrix mul(row, col);
        alignas(byte_alignment) float64 ss[] = { s,s,s,s };
        
        __m256d thisv, addv = _mm256_load_pd(ss);
        const float64* thisp = m;
        for (float64* mulp = mul.m, *end = mul.m + row*roff; mulp < end; mulp+=4, thisp+=4)
        {
            thisv = _mm256_load_pd(thisp);
            thisv = _mm256_mul_pd(thisv, addv);
            _mm256_store_pd(mulp, thisv);
        }
        return mul;
    }

    inline Matrix operator*(float64 s, const Matrix& mat)
    {
        return mat * s;
    }

    inline Matrix& operator*(float64 s, Matrix&& mat)
    {
        return mat.operator*=(s);
    }

    inline Matrix& Matrix::operator*=(const Matrix& mat)
    {
        if (col != mat.row)
            throw std::length_error("Matrix sizes not matching in multiplication operation");

        float64 tmp_buff[buff_size];
        float64* t = tmp_buff;
        auto t_roff = aligned(row);
        if (col*t_roff > buff_size) t = alloc(col*t_roff);
        realTranspose(m, t, row, col, roff, t_roff);
        
        float64 new_tmp_buff[buff_size];
        float64* new_m = new_tmp_buff;
        auto n_roff = aligned(mat.col);
        if (row*n_roff > buff_size) new_m = alloc(row*n_roff);

        for (size_t i = 0; i < row; i++)
            for (size_t j = 0; j < mat.col; j++)
                new_m[i * n_roff + j] = t[i] * mat.m[j];

        for (size_t k = 1; k < col; k++)
            for (size_t i = 0; i < row; i++)
            {
                float64 t_ik = t[i + k * t_roff];
                for (size_t j = 0; j < mat.col; j++)
                    new_m[i * n_roff + j] += t_ik * mat(k,j);
            }

        col = mat.col;
        size = row*col;
        roff = n_roff;
        if (t != tmp_buff) dealloc(t);
        if (m != buff)     dealloc(m);
        if (new_m == new_tmp_buff)
            std::copy(new_m, new_m + row*roff, buff),
            new_m = buff;
            
        m = new_m;
        return *this;
    }

    inline Matrix& Matrix::operator*=(float64 s)
    {
        alignas(byte_alignment) float64 ss[] = { s,s,s,s };
        
        __m256d thisv, addv = _mm256_load_pd(ss);
        for (float64* thisp = m, *end = m + row*roff; thisp < end; thisp+=4)
        {
            thisv = _mm256_load_pd(thisp);
            thisv = _mm256_mul_pd(thisv, addv);
            _mm256_store_pd(thisp, thisv);
        }
        return *this;
    }

    inline Matrix Matrix::transposeTimes(const Matrix& mat) const
    {
        if (row != mat.row)
            throw std::length_error("Matrix sizes not matching in multiplication operation");
        Matrix mul(col, mat.col);
        
        for (size_t i = 0; i < mul.row; i++)
            for (size_t j = 0; j < mul.col; j++)
                mul(i,j) = m[i] * mat.m[j];

        for (size_t k = 1; k < mat.row; k++)
            for (size_t i = 0; i < mul.row; i++)
            {
                float64 t_ik = m[i + k * roff];
                for (size_t j = 0; j < mul.col; j++)
                    mul(i,j) += t_ik * mat(k,j);
            }
        
        return mul;
    }

    /* --------- DIV --------- */

    inline Matrix Matrix::operator/(const Matrix &mat) const &
    {
        if (row != mat.row || col != mat.col)
            throw std::length_error("Matrix sizes not matching in division operation");
        
        Matrix div(row, col);
        
        __m256d thisv, matv;
        const float64* matp = mat.m, *thisp = m;
        for (float64* divp = div.m, *end = div.m + row*roff; divp < end; divp+=4, matp+=4, thisp+=4)
        {
            thisv = _mm256_load_pd(thisp);
            matv  = _mm256_load_pd(matp);
            thisv = _mm256_div_pd(thisv, matv);
            _mm256_store_pd(divp, thisv);
        }
        return div;
    }

    inline Matrix Matrix::operator/(float64 s) const &
    {
        Matrix div(row, col);
        alignas(byte_alignment) float64 ss[] = { s,s,s,s };
        
        __m256d thisv, addv = _mm256_load_pd(ss);
        const float64* thisp = m;
        for (float64* divp = div.m, *end = div.m + row*roff; divp < end; divp+=4, thisp+=4)
        {
            thisv = _mm256_load_pd(thisp);
            thisv = _mm256_div_pd(thisv, addv);
            _mm256_store_pd(divp, thisv);
        }
        return div;
    }

    inline Matrix& Matrix::operator/=(const Matrix &mat)
    {
        if ((row != mat.row || col != mat.col) && (mat.col != 1 || mat.row != 1))
            throw std::length_error("Matrix sizes not matching in division operation");
        
        __m256d thisv, matv;
        const float64* matp = mat.m, *thisp = m;
        for (float64* thisp = m, *end = m + row*roff; thisp < end; matp+=4, thisp+=4)
        {
            thisv = _mm256_load_pd(thisp);
            matv  = _mm256_load_pd(matp);
            thisv = _mm256_div_pd(thisv, matv);
            _mm256_store_pd(thisp, thisv);
        }
        return *this;
    }

    inline Matrix& Matrix::operator/=(float64 s)
    {
        alignas(byte_alignment) float64 ss[] = { s,s,s,s };
        
        __m256d thisv, addv = _mm256_load_pd(ss);
        for (float64* thisp = m, *end = m + row*roff; thisp < end; thisp+=4)
        {
            thisv = _mm256_load_pd(thisp);
            thisv = _mm256_div_pd(thisv, addv);
            _mm256_store_pd(thisp, thisv);
        }
        return *this;
    }

    inline float64 operator/(float64 s, const Matrix& mat)
    {
        if (mat.col != 1 || mat.row != 1)
            throw std::length_error("Matrix is not 1x1 in scalar matrix division operation");
        return s / mat.m[0];
    }

    inline float64 operator/=(float64 s, const Matrix& mat)
    {
        if (mat.col != 1 || mat.row != 1)
            throw std::length_error("Matrix is not 1x1 in scalar matrix division operation");
        return s /= mat.m[0];
    }

    inline Matrix::operator float64() const
    {
        if (row != 1 || col != 1)
            throw std::length_error("Matrix is not 1x1 in double cast");
        return m[0];
    }

    inline float64 Matrix::norm() const
    {
        return std::sqrt(normSquare());
    }

    inline float64 Matrix::pnorm(size_t p) const
    {
        float64 sum = 0.;
        for (size_t r = 0; r < row; r++) {
            const float64* mat_row = &(*this)(r,0);
            for (size_t c = 0; c < col; c++)
                sum += pow(std::abs(mat_row[c]), p);
        }
        return pow(sum, 1./p);
    }

    inline float64 Matrix::normSquare() const
    {
        float64 sum = 0.;
        for (size_t r = 0; r < row; r++) {
            const float64* mat_row = &(*this)(r,0);
            for (size_t c = 0; c < col; c++)
                sum += mat_row[c] * mat_row[c];
        }
        return sum;
    }

    inline float64 Matrix::sum() const
    {
        float64 sum = 0.;
        for (size_t r = 0; r < row; r++) {
            const float64* mat_row = &(*this)(r,0);
            for (size_t c = 0; c < col; c++)
                sum += mat_row[c];
        }
        return sum;
    }

    inline Matrix Matrix::trace() const
    {
        size_t min = std::min(row, col);
        Matrix tr(min);
        std::swap(tr.row, tr.col);
        for (size_t i = 0; i < min; i++)
            tr(i) = (*this)(i,i);
        return tr;
    }

    inline Matrix Matrix::rowPermutation(const std::vector<size_t>& perm_v) const
    {
        Matrix perm(row, col);
        
        for (size_t i = 0; i < row; i++)
            std::copy(m + perm_v[i]*col, m + perm_v[i]*col + roff, perm[i]);
        return perm;
    }

    inline Matrix Matrix::upTriInverse()
    {
        if (col != row)
            throw std::invalid_argument("Invalid non-triangular Matrix given in upperTriangInvert(Matrix)");
        Matrix mat(*this);
        int i, j, k;
        float64 *p_i, *p_j, *p_k;
        float64 sum;

        // diagonal
        for (k = 0, p_k = mat.m; k < col; p_k += (roff + 1), k++) {
            if (*p_k == 0.0) return -1;
            else *p_k = 1.0 / *p_k;
        }

        // upper part
        for (i = col - 2, p_i = mat.m + roff * (col - 2); i >=0; p_i -= roff, i-- ) {
            for (j = col - 1; j > i; j--) {
                sum = 0.0;
                for (k = i + 1, p_k = p_i + roff; k <= j; p_k += roff, k++ ) {
                    sum += *(p_i + k) * *(p_k + j);
                }
                *(p_i + j) = - *(p_i + i) * sum;
            }
        }
        
        return mat;
    }

    inline Matrix Matrix::forEach(std::function<float64(float64)> fn) const
    {
        Matrix ret(row, col);

        
        for (size_t r = 0; r < row; r++) {
            float64* mat_row = &ret(r,0);
            const float64* thisp = &(*this)(r,0);
            for (size_t c = 0; c < col; c++)
                mat_row[c] = fn(thisp[c]);
        }
        
        return ret;
    }

    inline Matrix Matrix::forEachRow(std::function<Matrix(Matrix)> fn) const
    {
        Matrix ret(row, col);

        
        for (size_t i = 0; i < row; i++)
            ret.setRow(i, fn(this->getRow(i)));
        
        return ret;
    }

    inline Matrix Matrix::conv(const Matrix &input, const Matrix &kernel, bool padding) const
    {



        return Matrix();
    }

    inline float64 Matrix::max() const
    {
        float64 max = -std::numeric_limits<float64>::infinity();
        for (size_t r = 0; r < row; r++) {
            const float64* mat_row = &(*this)(r,0);
            for (size_t c = 0; c < col; c++)
                if(mat_row[c] > max) max = mat_row[c];
        }
        return max;
    }


    inline float64 Matrix::absMax() const
    {
        float64 max = 0;
        for (size_t r = 0; r < row; r++) {
            const float64* mat_row = &(*this)(r,0);
            for (size_t c = 0; c < col; c++)
                if(std::abs(mat_row[c]) > max) max = std::abs(mat_row[c]);
        }
        return max;
    }

    inline bool Matrix::hasNaN() const
    {
        
        for (size_t r = 0; r < row; r++) {
            const float64* mat_row = &(*this)(r,0);
            for (size_t c = 0; c < col; c++)
                if (std::isnan(mat_row[c]))
                    return true;
        }
        return false;
    }

    inline Matrix Matrix::hom() const
    {
        Matrix hv(size + 1);
        std::copy(m, m + size, hv.m);
        hv.m[size] = 1;
        if (col != 1) std::swap(hv.row, hv.col);
        return hv;
    }

    inline Matrix Matrix::hom_i() const &
    {
        if (row != 1 && col != 1)
            throw std::length_error("Only Vectors can be converted from homogeneus to cartesian coordinates in Matrix::hom_i() &");
        Matrix hv(size - 1);
        for (size_t i = 0; i < hv.size; i++)
            hv.m[i] = m[i] / m[hv.size];
        if (col != 1) std::swap(hv.row, hv.col);
        return hv;
    }

    inline Matrix& Matrix::hom_i() &&
    {
        if (row != 1 && col != 1)
            throw std::length_error("Only Vectors can be converted from homogeneus to cartesian coordinates in Matrix::hom_i() &&");
        float64 w = m[--size];
        for (size_t i = 0; i < size; i++)
            m[i] /= m[size];
        if (row != 1) row--;
        else col--;
        return *this;
    }

    inline Matrix& Matrix::reshape(size_t row, size_t col)
    {
        if (size != row * col)
            throw std::length_error("Matrix wrong reshape size");
        this->row = row, this->col = col;
        return *this;
    }

    inline Matrix& Matrix::resize(size_t row, size_t col)
    {
        if (row*col > size || row*col < size * .75)
        {
            if (m != buff)
                dealloc(m);
            if (row*col > buff_size)
                m = alloc(row*col);
            else
                m = buff;
        }
        this->row = row, this->col = col, this->size = row*col;
        return *this;
    }

    inline float64& Matrix::operator()(size_t i)
    {
        if (i >= size)
            throw std::out_of_range("Matrix[] : index is out of range");
        return m[i];
    }

    inline float64 &Matrix::operator()(size_t i, size_t j)
    {
        if (i >= this->row || j >= this->col)
            throw std::out_of_range("Matrix[] : index is out of range");
        return m[i*roff + j];
    }

    inline const float64& Matrix::operator()(size_t i) const
    {
        if (i >= size)
            throw std::out_of_range("Matrix[] : index is out of range");
        return m[i];
    }

    inline const float64& Matrix::operator()(size_t i, size_t j) const
    {
        if (i >= this->row || j >= this->col)
            throw std::out_of_range("Matrix[] : index is out of range");
        return m[i*roff + j];
    }

    inline float64* Matrix::operator[](size_t i)
    {
        return m + i*roff;
    }

    inline const float64* Matrix::operator[](size_t i) const
    {
        return m + i*roff;
    }

    inline Matrix& Matrix::vstack(const Matrix &mat)
    {
        if (mat.col != col)
            throw std::length_error("Matrix columns not matching for vertical stacking");

        float64* new_buf;

        if (col * (row + mat.row) < buff_size)
            new_buf = buff;
        else
            new_buf = alloc(roff * (row + mat.row));
        
        if (new_buf != m)
            std::copy(m, m + row*roff, new_buf);

        
        for (size_t r = 0; r < mat.row; r++)
            std::copy(mat.m + r*mat.roff, mat.m + r*mat.roff + mat.col, new_buf + (row + r)*roff),
            std::fill(new_buf + (row + r)*roff + col, new_buf + (row + r + 1)*roff, 0.);

        if (m != buff && m != new_buf) dealloc(m);
        m = new_buf;
        row += mat.row;
        size += mat.size;

        return *this;
    }

    inline Matrix &Matrix::vstack(const std::vector<Matrix> &mats)
    {
        size_t new_row = 0;
        for (auto& mat : mats)
        {
            if (mat.col != col)
                throw std::length_error("Matrix columns not matching for vertical stacking");
            new_row += mat.row;
        }

        float64* new_buf;

        if (roff * (row + new_row) < buff_size)
            new_buf = buff;
        else
            new_buf = alloc(roff * (row + new_row));
        
        if (new_buf != m)
            std::copy(m, m + row*roff, new_buf);

        float64* tmp_buff = new_buf + row*roff;
        for (auto& mat : mats)
        {
            
            for (size_t r = 0; r < mat.row; r++)
                std::copy(mat.m + r*mat.roff, mat.m + r*mat.roff + mat.col, tmp_buff + r*roff),
                std::fill(tmp_buff + r*roff + col, tmp_buff + (r + 1)*roff, 0.);
                
            tmp_buff += mat.row * roff;
        }

        if (m != buff && m != new_buf) dealloc(m);
        m = new_buf;
        row += new_row;
        size = row*col;

        return *this;
    }

    inline Matrix vstack(const std::vector<Matrix>& mats)
    {
        size_t col = mats.empty() ? 0 : mats[0].col;
        size_t row = 0;
        for (auto& mat : mats)
        {
            if (mat.col != col)
                throw std::length_error("Matrix columns not matching for vertical stacking");
            row += mat.row;
        }

        Matrix stack(row, col);
        float64* buff = stack.m;
        size_t roff = stack.roff;
        for (auto& mat : mats)
        {
            
            for (size_t r = 0; r < mat.row; r++)
                std::copy(mat.m + r*mat.roff, mat.m + r*mat.roff + mat.col, buff + r*roff),
                std::fill(buff + r*roff + col, buff + (r + 1)*roff, 0.);
                
            buff += mat.row * roff;
        }
        
        return stack;
    }

    inline Matrix& Matrix::hstack(const Matrix &mat)    /* TODO roff */
    {
        if (mat.row != row)
            throw std::length_error("Matrix rows not matching for horizontal stacking");

        float64* new_buf;

        if (row * (col + mat.col) < buff_size)
            new_buf = buff;
        else
            new_buf = alloc(row * (col + mat.col));
        
        if (new_buf == m)
            /* NOT parallelizable */
            for (size_t i = row-1; i > 0; i--)
                std::copy(m + i*col, m + (i+1)*col, new_buf + i*(col+mat.col));
        else
            for (size_t i = 0; i < row; i++)
                std::copy(m + i*col, m + (i+1)*col, new_buf + i*(col+mat.col));

        for (size_t i = 0; i < row; i++)
            std::copy(mat.m + i*mat.col, mat.m + (i+1)*mat.col, new_buf + i*(col+mat.col) + col);

        if (m != buff && m != new_buf) dealloc(m);
        m = new_buf;
        col += mat.col;
        size += mat.size;
        
        return *this;
    }

    inline Matrix hstack(const std::vector<Matrix>& mats)    /* TODO roff */
    {
        size_t row = mats.empty() ? 0 : mats[0].row;
        size_t col = 0;
        for (auto& mat : mats)
        {
            if (mat.row != row)
                throw std::length_error("Matrix columns not matching for vertical stacking");
            col += mat.col;
        }

        Matrix stack(row, col);
        float64* buff = stack.m;
        for (auto& mat : mats)
        {
            for (size_t i = 0; i < row; i++)
                std::copy(mat.m + i*mat.col, mat.m + (i+1)*mat.col, buff + i*col);

            buff += mat.col;
        }
        
        return stack;
    }

    inline void Matrix::setCol(int64_t c, const std::initializer_list<float64> &lst)
    {
        if (c < 0) c += col;
        if (c >= col || c < 0)
            throw std::out_of_range("Matrix column out of range in Matrix::setCol");

        size_t end = std::min(lst.size(), row);
        auto column = lst.begin();
        auto buff = m + c;
        for (size_t i = 0; i < end; i++, buff += roff)
            *buff = column[i];
    }

    inline void Matrix::setCol(int64_t c, const Matrix& v)
    {
        if (c < 0) c += col;
        if (c >= col || c < 0)
            throw std::out_of_range("Matrix column out of range in Matrix::setCol");

        size_t end = std::min(v.row, row);
        auto buff = m + c, column = v.m;
        for (size_t i = 0; i < end; i++, buff += roff, column += v.roff)
            *buff = *column;
    }

    inline void Matrix::setRow(int64_t r, const std::initializer_list<float64> &lst)
    {
        if (r < 0) r += row;
        if (r >= row || r < 0)
            throw std::out_of_range("Matrix row out of range in Matrix::setRow");

        std::copy(lst.begin(), std::min(lst.end(), lst.begin() + col), m + r * roff);
    }

    inline void Matrix::setRow(int64_t r, const Matrix& v)
    {
        if (r < 0) r += row;
        if (r >= row || r < 0)
            throw std::out_of_range("Matrix row out of range in Matrix::setRow");

        std::copy(v.m, v.m + std::min(v.col, col), m + r * roff);
    }

    inline void Matrix::setSubMatrix(const Matrix &mat, int64_t r, int64_t c, bool trns)
    {
        if (c < 0) c += col;
        if (r < 0) r += row;
        if (r >= row || r < 0 || c >= col || c < 0)
            throw std::out_of_range("Matrix row out of range in Matrix::setSubMatrix");

        if (!trns)
        {
            size_t endr = std::min(row - r, mat.row);
            size_t endc = std::min(col - c, mat.col);
            auto dst = m + c + r*roff, src = mat.m;
            for (size_t i = 0; i < endr; i++, src += mat.roff, dst += roff)
                std::copy(src, src + endc, dst);
        }
        else
        {
            size_t endr = std::min(row - r, mat.row);
            size_t endc = std::min(col - c, mat.col);
            realTranspose(mat.m, m, endr, endc, mat.roff, roff);
            /* for (size_t i = 0; i < endr; i++)
                for (size_t j = 0; j < endc; j++)
                    (*this)(r+i, c+j) = mat(j,i); */
        }
    }

    void Matrix::checkRangeCol(int64_t& c) const
    {
        if (c < 0) c += col;
        if (c >= col || c < 0)
            throw std::out_of_range("Matrix column out of range in Matrix::getCol");
    }

    inline Matrix Matrix::getCol(int64_t c) const
    {
        checkRangeCol(c);
        Matrix v(row);
        for (size_t i = 0; i < row; i++)
            v.m[i] = m[c + i * roff];
        return v;
    }

    /* inline MatrixView Matrix::getCol(int64_t c)
    {
        checkRangeCol(c);
        return {m + c, row, 1, roff};
    } */

    void Matrix::checkRangeRow(int64_t& r) const
    {
        if (r < 0) r += row;
        if (r >= row || r < 0)
            throw std::out_of_range("Matrix row out of range in Matrix::getRow");
    }

    inline Matrix Matrix::getRow(int64_t r) const
    {
        checkRangeRow(r);
        Matrix v(1, col);
        std::copy(m + r*roff, m + (r+1)*roff, v.m);
        return v;
    }

    /* inline MatrixView Matrix::getRow(int64_t r)
    {
        checkRangeRow(r);
        return {m + r*roff, 1, col, roff};
    } */

    inline void Matrix::checkRangeSubMatrix(int64_t& t, int64_t& l, int64_t& b, int64_t& r) const
    {
        if (t < 0) t += row + 1;
        if (l < 0) l += col + 1;
        if (b < 0) b += row + 1;
        if (r < 0) r += col + 1;
        if (l > col || l < 0 || r > col || r < 0 || t > row || t < 0 || b > row || b < 0)
            throw std::out_of_range("Matrix coordinates out of range in Matrix::subMatrix");
        if (t > b || l > r)
            throw std::out_of_range("Matrix coordinates negatively overlapping in Matrix::subMatrix");
    }

    inline Matrix Matrix::subMatrix(int64_t t, int64_t l, int64_t b, int64_t r) const
    {
        checkRangeSubMatrix(t,l,b,r);
        Matrix sm(b-t, r-l);
        for (size_t i = t; i <= b; i++)
            std::copy(m + i*roff + l, m + i*roff + r, sm.m + (i-t)*sm.roff);
        return sm;
    }

    /* inline MatrixView Matrix::subMatrix(int64_t t, int64_t l, int64_t b, int64_t r)
    {
        checkRangeSubMatrix(t,l,b,r);
        return {m + t*roff + l, b-t, r-l, roff};
    } */

    inline void Matrix::print() const
    {
        //std::cout.precision(17);
        std::cout << "[";
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
                std::cout << (*this)(i,j) << " ";
            if (i < this->row - 1) std::cout << "\n ";
        }
        std::cout << "]\n";
    }

    inline Matrix& Matrix::transpose()
    {
        auto new_roff = aligned(row);
        auto new_size = new_roff*col;
        if (row > 1 && col > 1)
        {
            alignas(byte_alignment) float64 tmp_buff[buff_size];
            auto t = new_size > buff_size ? alloc(new_size) : (m == buff ? tmp_buff : buff);
            realTranspose(m, t, row, col, roff, new_roff);
            if (m == buff && new_size <= buff_size) std::copy(t, t + new_size, buff), t = buff;
            if (m != buff)
                dealloc(m);
            m = t;
        }
        roff = new_roff;
        std::swap(row, col);
        return *this;
    }

    inline Matrix Matrix::T() const &
    {
        Matrix t(col, row);
        if (row > 1 && col > 1) realTranspose(this->m, t.m, row, col, roff, t.roff);
        else std::copy(this->m, this->m + size, t.m);
        return t;
    }

    inline Matrix Matrix::T() &&
    {
        return transpose();
    }

    inline void transpose4x4_SSE(const float64 *A, float64 *B, const int lda, const int ldb) {
        __m256d r0 = _mm256_load_pd(A        );
        __m256d r1 = _mm256_load_pd(A + 1*lda);
        __m256d r2 = _mm256_load_pd(A + 2*lda);
        __m256d r3 = _mm256_load_pd(A + 3*lda);
        //_MM_TRANSPOSE4_PS(row1, row2, row3, row4);
        auto t0 = _mm256_unpacklo_pd(r0, r1);
        auto t1 = _mm256_unpacklo_pd(r2, r3);
        auto t2 = _mm256_unpackhi_pd(r0, r1);
        auto t3 = _mm256_unpackhi_pd(r2, r3);
        
        _mm256_store_pd(B +   ldb, r0);
        _mm256_store_pd(B + 1*ldb, r1);
        _mm256_store_pd(B + 2*ldb, r2);
        _mm256_store_pd(B + 3*ldb, r3);
    }

    inline void Matrix::realTranspose(const float64 *A, float64 *B, const size_t r, const size_t c, const size_t lda, const size_t ldb)
    {
        const size_t block_size = 4UL;
        const size_t max_r = r & ~(block_size-1);
        const size_t max_c = c & ~(block_size-1);

        
        for (size_t i = 0; i < max_r; i += block_size)
        {
            for (size_t j = 0; j < max_c; j += block_size)
                for (size_t k = 0; k < block_size; k++)
                    for (size_t l = 0; l < block_size; l++)
                        B[(j + l)*ldb + (i + k)] = A[(i + k)*lda + (j + l)];

            for (size_t k = 0; k < block_size; k++)
                for (size_t j = max_c; j < c; j++)
                    B[j*ldb + (i + k)] = A[(i + k)*lda + j];
        }

        for (size_t i = max_r; i < r; i++)
            for (size_t j = 0; j < c; j++)
                B[j*ldb + i] = A[i*lda + j];


/* 
        const size_t max_r = r & ~((size_t)0b11);
        const size_t max_c = c & ~((size_t)0b11);

        #pragma omp parallel for
        for(size_t i=0; i<max_r; i+=block_size) {
            for(size_t j=0; j<max_c; j+=block_size) {
                size_t max_i2 = i+block_size < max_r ? i + block_size : max_r;
                size_t max_j2 = j+block_size < max_c ? j + block_size : max_c;
                for(size_t i2=i; i2<max_i2; i2+=4) {
                    for(size_t j2=j; j2<max_j2; j2+=4) {
                        transpose4x4_SSE(&A[i2*lda +j2], &B[j2*ldb + i2], lda, ldb);
                    }
                }
            }
        }

        for (size_t i = 0; i < max_r; i++)
            for (size_t j = max_c; j < c; j++)
                B[j*ldb + i] = A[i*lda + j];

        for (size_t i = max_r; i < r; i++)
            for (size_t j = 0; j < c; j++)
                B[j*ldb + i] = A[i*lda + j]; */
    }

    inline void Matrix::realTransposeInPlace(float64*& A, const size_t r, const size_t c, const size_t lda, bool lb)
    {
        if (lb)
        {
            alignas(16) float64 tmp_buff[buff_size];
            std::copy(A, A + r*c, tmp_buff);
            realTranspose(tmp_buff, A, r, c, lda, r);
        }
        else
        {
            float64* tmp_buff = alloc(r*c);
            realTranspose(A, tmp_buff, r, c, lda, r);
            dealloc(A);
            A = tmp_buff;
        }
    }

    inline Matrix Matrix::householderReflect(const Matrix &u)
    {
        float64 tau_i = 2 / u.normSquare();
        Matrix ref = -tau_i*u*u.T();
        for (size_t i = 0; i < u.getSize(); i++)
            ref(i,i) += 1.;
        return ref;
    }

    inline void Matrix::householderReflectSubMatLeft(const Matrix &u, size_t r, size_t c, size_t len)
    {
        if (len == 0 || len > col - c) len = col - c;
        float64 tau_i = 2 / u.normSquare();
        Vector wt(1, len);

        for (size_t i = 0; i < wt.size; i++)
        {
            wt.m[i] = m[r*roff + i + c];
            for (size_t k = 1; k < u.size; k++)
                wt.m[i] += u.m[k] * m[(r+k)*roff + i + c];
            wt.m[i] *= tau_i;
        }
        for (size_t i = 0; i < wt.size; i++)
            m[r*roff + c + i] -= wt.m[i];
        for (size_t i = 1; i < u.size; i++)
            for (size_t j = 0; j < wt.size; j++)
                m[(r+i)*roff + c + j] -= u.m[i] * wt.m[j];
    }

    inline void Matrix::householderReflectSubMatRight(const Matrix &u, size_t r, size_t c, size_t len)
    {
        if (len == 0 || len > row - r) len = row - r;
        float64 tau_i = 2 / u.normSquare();
        //Vector wt(len);

        for (size_t i = 0; i < len; i++)
        {
            float64 wtmi = m[(r+i)*roff + c];
            for (size_t k = 1; k < u.size; k++)
                wtmi += u.m[k] * m[(r+i)*roff + c+k];
            wtmi *= tau_i;
            for (size_t j = 0; j < u.size; j++)
                m[(r+i)*roff + c + j] -= u.m[j] * wtmi;
        }
        //for (size_t i = 0; i < wt.size; i++)
        //    m[(r+i)*roff + c] -= wt.m[i];
        //for (size_t i = 0; i < wt.size; i++)
        //    for (size_t j = 0; j < u.size; j++)
        //        m[(r+i)*roff + c + j] -= u.m[j] * wt.m[i];
    }

    inline void Matrix::householderReflectSubMatForwardLeft(const Matrix &u, size_t r, size_t c, bool store)
    {
        float64 tau_i = 2 / u.normSquare(), theta = m[r*roff + c];
        Vector wt(1, col - c - 1);

        for (size_t k = 1; k < u.size; k++)
            theta += u.m[k] * m[(r+k)*roff + c];
        m[r*roff + c] -= tau_i * theta;

        if (store)
            for (size_t i = 1; i < u.size; i++)
                m[(r+i)*roff + c] = u.m[i];
        else
            for (size_t i = 1; i < u.size; i++)
                m[(r+i)*roff + c] = 0;
        
        for (size_t i = 0; i < wt.size; i++)
        {
            wt.m[i] = m[r*roff + i + c+1];
            for (size_t k = 1; k < u.size; k++)
                wt.m[i] += u.m[k] * m[(r+k)*roff + i + c+1];
            wt.m[i] *= tau_i;
        }
        for (size_t i = 0; i < wt.size; i++)
            m[r*roff + c+1 + i] -= wt.m[i];
        for (size_t i = 1; i < u.size; i++)
            for (size_t j = 0; j < wt.size; j++)
                m[(r+i)*roff + c+1 + j] -= u.m[i] * wt.m[j];
    }

    inline void Matrix::householderReflectSubMatForwardRight(const Matrix &u, size_t r, size_t c, bool store)
    {
        float64 tau_i = 2 / u.normSquare(), theta = m[r*roff + c];
        Vector wt(row - r - 1);

        for (size_t k = 1; k < u.size; k++)
            theta += u.m[k] * m[r*roff + c+k];
        m[r*roff + c] -= tau_i * theta;

        if (store)
            for (size_t i = 1; i < u.size; i++)
                m[r*roff + c+i] = u.m[i];
        else
            for (size_t i = 1; i < u.size; i++)
                m[r*roff + c+i] = 0;
        
        for (size_t i = 0; i < wt.size; i++)
        {
            wt.m[i] = m[(r+i+1)*roff + c];
            for (size_t k = 1; k < u.size; k++)
                wt.m[i] += u.m[k] * m[(r+i+1)*roff + c+k];
            wt.m[i] *= tau_i;
        }
        for (size_t i = 0; i < wt.size; i++)
            m[(r+i+1)*roff + c] -= wt.m[i];
        for (size_t i = 0; i < wt.size; i++)
            for (size_t j = 1; j < u.size; j++)
                m[(r+i+1)*roff + c + j] -= u.m[j] * wt.m[i];
    }

    inline void Matrix::givensRotateLeft(float64 c, float64 s, int64_t r1, int64_t r2, int64_t l, int64_t r)
    {
        if (r1 < 0) r1 += row;
        if (r2 < 0) r2 += row;
        if (l < 0) l += col;
        if (r < 0) r += col;
        l = std::clamp<int64_t>(l, 0L, col-1);
        r = std::clamp<int64_t>(r, 0L, col-1);
        if (r1 >= row || r1 < 0 || r2 >= row || r2 < 0)
            throw std::out_of_range("Matrix row indices out of range in Matrix::givensRotateLeft()");

        for (size_t i = l; i <= r; i++)
        {
            float64 x1 = m[r1*roff + i], x2 = m[r2*roff + i];
            m[r1*roff + i] = c*x1 - s*x2;
            m[r2*roff + i] = s*x1 + c*x2;
        }
    }

    inline void Matrix::givensRotateRight(float64 c, float64 s, int64_t c1, int64_t c2, int64_t t, int64_t b)
    {
        if (c1 < 0) c1 += col;
        if (c2 < 0) c2 += col;
        if (t < 0) t += row;
        if (b < 0) b += row;
        t = std::clamp<int64_t>(t, 0L, row-1);
        b = std::clamp<int64_t>(b, 0L, row-1);
        if (c1 >= col || c1 < 0 || c2 >= col || c2 < 0)
            throw std::out_of_range("Matrix column indices out of range in Matrix::givensRotateRight()");

        for (size_t i = t; i <= b; i++)
        {
            float64 x1 = m[i*roff + c1], x2 = m[i*roff + c2];
            m[i*roff + c1] = c*x1 - s*x2;
            m[i*roff + c2] = s*x1 + c*x2;
        }
    }

    inline Matrix Matrix::solve(Matrix b) const
    {
        Matrix x(col, b.col), R(*this);
        auto min = std::min(row-1, col);
        zero(x);

        for (size_t i = 0; i < min; i++)
        {
            Vector v = R.subMatrix(i,i, -1,i);
            v.m[0] += v.m[0] < 0 ? -v.norm() : v.norm();
            if (v.m[0] > DBL_EPSILON || v.m[0] < -DBL_EPSILON)
            {
                v /= v.m[0];
                R.householderReflectSubMatForwardLeft(v, i,i, false);
                b.householderReflectSubMatLeft(v, i,0);
            }
        }
        for (int64_t i = x.row - 1; i >= 0; i--)
        {
            auto divisor = R.m[i*col + i];
            if (divisor > DBL_EPSILON || divisor < -DBL_EPSILON)
                x.setRow(i, (b.getRow(i) - R.getRow(i)*x)/divisor);
        }

        return x;
    }

    inline Matrix Matrix::cholesky() const
    {
        if (row != col)
            throw std::length_error("Matrix not square in Matrix::cholesky()");
        
        Matrix L(row, col);
        for (size_t i = 0; i < row; i++)
        {
            size_t j;
            for (j = 0; j < i; j++)
                L[i][j] = 0;
            for (; j < col; j++)
                L[i][j] = (*this)[i][j];
        }

        for (size_t i = 0; i < L.col; i++)
        {
            float64& a11 = L[i][i];
            a11 = std::sqrt(a11);
            if (std::isnan(a11))
                throw std::invalid_argument("Matrix not positive defined in Matrix::cholesky()");

            float64 a11_i = 1./a11;
            for (size_t j = i+1; j < row; j++)
                L[i][j] *= a11_i;

            for (size_t j = i+1; j < row; j++)
                for (size_t k = j; k < col; k++)
                    L[j][k] -= L[i][j] * L[i][k];
        }

        return L.T();
    }

    inline Matrix Matrix::partialQR(int64_t t, int64_t l, int64_t b, int64_t r) const
    {
        if (t < 0) t += row;
        if (l < 0) l += col;
        if (b < 0) b += row;
        if (r < 0) r += col;
        if (l >= col || l < 0 || r >= col || r < 0 || t >= row || t < 0 || b >= row || b < 0)
            throw std::out_of_range("Matrix coordinates out of range in Matrix::partialQR");
        if (t > b || l > r)
            throw std::out_of_range("Matrix coordinates negatively overlapping in Matrix::partialQR &&");

        Matrix R = (*this).subMatrix(t,l,b,r);
        auto min = std::min(R.row-1, R.col);

        for (size_t i = 0; i < min; i++)
        {
            Vector v = R.subMatrix(i,i, -1,i);
            v.m[0] += v.m[0] < 0 ? -v.norm() : v.norm();
            if (v.m[0] > DBL_EPSILON || v.m[0] < -DBL_EPSILON)
            {
                v /= v.m[0];
                R.householderReflectSubMatForwardLeft(v, i,i);
            }
        }
        return R;
    }

    inline std::tuple<Matrix, Matrix> Matrix::QR() const
    {
        std::tuple<Matrix, Matrix> tpl = {identity(row), *this};
        auto& [Q, R] = tpl;
        auto min = std::min(row-1, col);

        for (size_t i = 0; i < min; i++)
        {
            Vector v = R.subMatrix(i,i, -1,i);
            v.m[0] += v.m[0] < 0 ? -v.norm() : v.norm();
            if (v.m[0] > DBL_EPSILON || v.m[0] < -DBL_EPSILON)
            {
                v /= v.m[0];
                R.householderReflectSubMatForwardLeft(v, i,i);
            }
        }
        for (int64_t i = min - 1; i >= 0; i--)
        {
            Vector u = R.subMatrix(i,i, -1,i);
            if (u.m[0] > DBL_EPSILON || u.m[0] < -DBL_EPSILON)
            {
                u.m[0] = 1.;
                Q.householderReflectSubMatLeft(u, i,i);
            }
        }

        for (size_t i = 1; i < row; i++)
            std::fill(R.m + i*col, R.m + i*col + i, 0.);
        return tpl;
    }

    inline std::tuple<Matrix, Matrix> Matrix::tridiagQR() const
    {
        std::tuple<Matrix, Matrix> tpl = {identity(row), *this};
        auto& [Q, R] = tpl;

        for (size_t i = 0; i < row-1; i++)
        {
            float64 x1 = R.m[i*col + i], x2 = R.m[(i+1)*col + i];
            R.m[i*col + i] = std::sqrt(x1*x1+x2*x2);
            R.m[(i+1)*col + i] = 0;
            float64 norm_i = 1./R.m[i*col + i];
            float64 c = x1*norm_i, s = -x2*norm_i;
            float64 a = R.m[i*col + i+1],
                   d = R.m[(i+1)*col + i+1],
                   e = R.m[(i+1)*col + i+2];
            R.m[i*col + i+1] = c*a-s*d;
            R.m[(i+1)*col + i+1] = s*a+c*d;
            if (i+2 < row)
            {
                R.m[i*col + i+2] = -s*e;
                R.m[(i+1)*col + i+2] = c*e;
            }
            Q.givensRotateRight(c, -s, i, i+1);
        }

        return tpl;
    }

    inline std::tuple<Matrix, Matrix> Matrix::implicitelyShiftedQR(const float64 threshold) const
    {
        if (row != col)
            throw std::length_error("Matrix not square in Matrix::implicitelyShiftedQR()");

        Matrix V = identity(row);
        Matrix A = *this;
        size_t bshift = col-1, tshift = 0;
        int it=0;
        
        for (size_t i = 0; i < col-2; i++)
        {
            Vector v = A.subMatrix(i+1,i, -1,i);
            v.m[0] += v.m[0] < 0 ? -v.norm() : v.norm();
            if (v.m[0] > DBL_EPSILON || v.m[0] < -DBL_EPSILON)
            {
                v /= v.m[0];
                A.householderReflectSubMatForwardLeft(v, i+1,i);
                A.householderReflectSubMatForwardRight(v, i,i+1);
            }
        }
        for (int64_t i = col-3; i >= 0; i--)
        {
            Vector u = A.subMatrix(i+1,i, -1,i);
            if (u.m[0] > DBL_EPSILON || u.m[0] < -DBL_EPSILON)
            {
                u.m[0] = 1.;
                V.householderReflectSubMatLeft(u, i+1,i);
            }
        }
        for (size_t i = 0; i < row-2; i++)
            std::fill(A.m + i*col + i+2, A.m + (i+1)*col, 0.);
        for (size_t i = 2; i < row; i++)
            std::fill(A.m + i*col, A.m + i*col + i-1, 0.);
            
        while (bshift > tshift)
        {
            float64 Amm = A[bshift][bshift];
            
            float64 x1 = A[tshift][tshift] - Amm, x2 = A[tshift][tshift+1];
            float64 norm = std::hypot(x1,x2);
            float64 c = x1/norm, s = -x2/norm;
            if (s == 0.) tshift++;
            else
            {
                A.givensRotateLeft (c, s, tshift,tshift+1, tshift, std::min(tshift+3UL, bshift));
                A.givensRotateRight(c, s, tshift,tshift+1, tshift, std::min(tshift+3UL, bshift));
                V.givensRotateRight(c, s, tshift,tshift+1);

                for (size_t i = tshift; i < bshift-1; i++)
                {
                    float64 x1 = A[i+1][i], x2 = A[i+2][i];
                    float64 norm = std::hypot(x1,x2);
                    if (norm == 0.) 
                        break; 
                    float64 c = x1/norm, s = -x2/norm;
                    A.givensRotateLeft(c, s, i+1,i+2, i,std::min(i+3, bshift));
                    A[i+2][i] = 0.;
                    A.givensRotateRight(c, s, i+1,i+2, i,std::min(i+3, bshift));
                    A[i][i+2] = 0.;
                    V.givensRotateRight(c, s, i+1,i+2);
                }
                float64 error = square(A[bshift][bshift-1]);
                if (error < threshold) bshift--;
            }
            it++;
        }
        //A.print();
        //for (size_t i = 0; i < col; i++)
        //    A[i+1][i] = A[i][i+1] = 0.;
        
        std::cout << it << " iterations\n";

        return {V,A};
    }

    inline Matrix Matrix::solveMin() const
    {
        float64 scale = this->absMax();
        Matrix B = (*this) / scale;
        Matrix Vt = identity(col);
        const float64 eps = 1e-6;

        for (size_t i = 0; i < col; i++)
        {
            Vector v = B.subMatrix(i,i, -1,i);
            v.m[0] += v.m[0] < 0 ? -v.norm() : v.norm();
            if (v.m[0] > DBL_EPSILON || v.m[0] < -DBL_EPSILON)
            {
                v /= v.m[0];
                B.householderReflectSubMatForwardLeft(v, i,i, false);
            }
            if (i < col - 1)
            {
                Vector v = B.subMatrix(i,i+1,i);
                v.m[0] += v.m[0] < 0 ? -v.norm() : v.norm();
                if (v.m[0] > DBL_EPSILON || v.m[0] < -DBL_EPSILON)
                {
                    v /= v.m[0];
                    B.householderReflectSubMatForwardRight(v, i,i+1);
                }
            }
        }
        for (int64_t i = col - 2; i >= 0; i--)
        {
            Vector u = B.subMatrix(i,i+1,i);
            if (u.m[0] > DBL_EPSILON || u.m[0] < -DBL_EPSILON)
            {
                u.m[0] = 1.;
                Vt.householderReflectSubMatRight(u, i,i+1);
            }
        }
        
        for (size_t i = 0; i < col-1; i++)
            std::fill(B[i] + i + 2, B[i+1], 0.);

        size_t bshift = col-1, tshift = 0;

        while (bshift > tshift)
        {
            float64 mu = square(B[bshift][bshift]) + square(B[bshift-1][bshift]);
            
            float64 x1 = square(B[tshift][tshift]) - mu, x2 = B[tshift][tshift] * B[tshift][tshift+1];
            float64 norm = std::hypot(x1,x2);
            float64 c = x1/norm, s = -x2/norm;
            if (s == 0.) 
                tshift++;
            else
            {
                B.givensRotateRight(c, s, tshift,tshift+1, tshift,std::min(tshift + 3UL, bshift));
                Vt.givensRotateLeft(c, s, tshift,tshift+1);

                for (size_t i = tshift; i < bshift; i++)
                {
                    float64 x1 = B[i][i], x2 = B[i+1][i];
                    float64 norm = std::hypot(x1,x2);
                    if (norm == 0.) 
                        break; 
                    float64 c = x1/norm, s = -x2/norm;
                    B.givensRotateLeft(c, s, i,i+1, i+1,std::min(i+3, bshift));
                    B[i][i] = norm;
                    B[i+1][i] = 0.;

                    if (i < bshift-1)
                    {
                        x1 = B[i][i+1], x2 = B[i][i+2];
                        norm = std::hypot(x1,x2);
                        if (norm == 0.) 
                            break; 
                        c = x1/norm, s = -x2/norm;
                        B.givensRotateRight(c, s, i+1,i+2, i+1,std::min(i+3, bshift));
                        B[i][i+1] = norm;
                        B[i][i+2] = 0.;
                        Vt.givensRotateLeft(c, s, i+1,i+2);
                    }
                }
                float64 error = std::abs(B[bshift-1][bshift]);
                if (error < 1e-30) bshift--;
            }
        }
        if (B[B.col-1][B.col-1] < 0.)
        {
            B[B.col-1][B.col-1] *= -1.;
            auto row = Vt[Vt.row-1];
            for (size_t i = 0; i < Vt.col; i++)
                row[i] = -row[i];
        }

        for (size_t i = 0; i < B.col-1; i++)
        {
            B[i][i] *= scale;
            B[i][i+1] = 0.;
        }
        B.m[(B.col-1)*B.col - 1] *= scale;
        
        size_t minsvi = 0;
        float64 minsv = std::abs(B[0][0]);
        for (size_t i = 1; i < B.col; i++)
            if (std::abs(B[i][i]) < minsv)
                minsv = std::abs(B[i][i]), minsvi = i;

        if (B[minsvi][minsvi] < 0.)
            for (size_t j = 0; j < Vt.col; j++)
                Vt[minsvi][j] = -Vt[minsvi][j];

        return Vt.getRow(minsvi);
    }

    inline std::tuple<Matrix, Matrix, Matrix> Matrix::svd() const
    {
        float64 scale = this->absMax();
        Matrix B = (*this) / scale;
        Matrix U = identity(col, row);
        Matrix Vt = identity(col);
        const float64 eps = 1e-6;

        for (size_t i = 0; i < col; i++)
        {
            Vector v = B.subMatrix(i,i, -1,i);
            v.m[0] += v.m[0] < 0 ? -v.norm() : v.norm();
            if (v.m[0] > DBL_EPSILON || v.m[0] < -DBL_EPSILON)
            {
                v /= v.m[0];
                B.householderReflectSubMatForwardLeft(v, i,i);
            }
            if (i < col - 1)
            {
                Vector v = B.subMatrix(i,i+1,i);
                v.m[0] += v.m[0] < 0 ? -v.norm() : v.norm();
                if (v.m[0] > DBL_EPSILON || v.m[0] < -DBL_EPSILON)
                {
                    v /= v.m[0];
                    B.householderReflectSubMatForwardRight(v, i,i+1);
                }
            }
        }
        for (int64_t i = col - 1; i >= 0; i--)
        {
            Vector u = B.subMatrix(i,i, -1,i);
            if (u.m[0] > DBL_EPSILON || u.m[0] < -DBL_EPSILON)
            {
                u.m[0] = 1.;
                U.householderReflectSubMatRight(u, i,i);
            }
            if (i < col - 1)
            {
                Vector u = B.subMatrix(i,i+1,i);
                if (u.m[0] > DBL_EPSILON || u.m[0] < -DBL_EPSILON)
                {
                    u.m[0] = 1.;
                    Vt.householderReflectSubMatRight(u, i,i+1);
                }
            }
        }
        
        for (size_t i = 1; i < col; i++)
            std::fill(B[i], B[i] + i, 0.);
        for (size_t i = 0; i < col-1; i++)
            std::fill(B[i] + i + 2, B[i+1], 0.);

        size_t bshift = col-1, tshift = 0;

        while (bshift > tshift)
        {
            float64 mu = square(B[bshift][bshift]) + square(B[bshift-1][bshift]);
            
            float64 x1 = square(B[tshift][tshift]) - mu, x2 = B[tshift][tshift] * B[tshift][tshift+1];
            float64 norm = std::hypot(x1,x2);
            float64 c = x1/norm, s = -x2/norm;
            if (s == 0.) 
                tshift++;
            else
            {
                B.givensRotateRight(c, s, tshift,tshift+1, tshift,std::min(tshift + 3UL, bshift));
                Vt.givensRotateLeft(c, s, tshift,tshift+1);

                for (size_t i = tshift; i < bshift; i++)
                {
                    float64 x1 = B[i][i], x2 = B[i+1][i];
                    float64 norm = std::hypot(x1,x2);
                    if (norm == 0.) 
                        break; 
                    float64 c = x1/norm, s = -x2/norm;
                    B.givensRotateLeft(c, s, i,i+1, i+1,std::min(i+3, bshift));
                    B[i][i] = norm;
                    B[i+1][i] = 0.;
                    U.givensRotateLeft(c, s, i,i+1);

                    if (i < bshift-1)
                    {
                        x1 = B[i][i+1], x2 = B[i][i+2];
                        norm = std::hypot(x1,x2);
                        if (norm == 0.) 
                            break; 
                        c = x1/norm, s = -x2/norm;
                        B.givensRotateRight(c, s, i+1,i+2, i+1,std::min(i+3, bshift));
                        B[i][i+1] = norm;
                        B[i][i+2] = 0.;
                        Vt.givensRotateLeft(c, s, i+1,i+2);
                    }
                }
                float64 error = std::abs(B[bshift-1][bshift]);
                if (error < 1e-30) bshift--;
            }
        }
        if (B[B.col-1][B.col-1] < 0.)
        {
            B[B.col-1][B.col-1] *= -1.;
            auto row = Vt[Vt.row-1];
            for (size_t i = 0; i < Vt.col; i++)
                row[i] = -row[i];
        }

        for (size_t i = 0; i < B.col-1; i++)
        {
            B[i][i] *= scale;
            B[i][i+1] = 0.;
        }
        B.m[(B.col-1)*B.col - 1] *= scale;
        
        std::vector<std::pair<float64, int64_t>> svals;
        std::vector<size_t> p(B.col);
        svals.reserve(B.col);
        for (int64_t i = 0; i < B.col; i++)
        {
            svals.emplace_back(std::abs(B[i][i]), i);
            if (B[i][i] < 0.)
                for (size_t j = 0; j < Vt.col; j++)
                    Vt[i][j] = -Vt[i][j];
        }

        std::sort(svals.begin(), svals.end(), std::greater<>());

        Matrix S(B.col);
        for (int64_t i = 0; i < B.col; i++)
            S.m[i] = svals[i].first,
            p[i]   = svals[i].second;

        return {(U.rowPermutation(p)).T(), S, Vt.rowPermutation(p)};
    }


    /* -------- outsize class functions */

    inline void zero(Matrix& mat)
    {
        std::fill(mat.m, mat.m + mat.row*mat.roff, 0);
    }

    inline void random_matrix(Matrix& mat, float64 min, float64 max)
    {
        float64 range = (max - min) / RAND_MAX;
        for (float64* i = mat.m, * end = mat.m + mat.row * mat.roff; i < end; i++)
        {
            *i = rand() * range - min;
        }
    }
    

    inline Matrix ones(size_t rows, size_t cols)
    {
        Matrix mat(rows, cols);
        std::fill(mat.m, mat.m + mat.row*mat.roff, 1.0);
        return mat;
    }

    inline Matrix ones_like(const Matrix& m)
    {
        Matrix mat(m.row, m.col);
        std::fill(mat.m, mat.m + mat.row*mat.roff, 1.0);
        return mat;
    }

    inline Matrix identity(size_t size)
    {
        Matrix I(size, size);
        zero(I);
        for (size_t i = 0; i < size; i++)
            I(i,i) = 1.0;
        return I;
    }

    inline Matrix identity(size_t r, size_t c)
    {
        Matrix I(r, c);
        zero(I);
        auto min = std::min(r, c);
        for (size_t i = 0; i < min; i++)
            I(i,i) = 1.0;
        return I;
    }

    inline Vector std_base_vector(size_t dim, size_t n)
    {
        if (n >= dim) n = dim-1;
        Vector base_v(dim);
        zero(base_v);
        base_v(n) = 1.;
        return base_v;
    }

    inline Matrix rodriguesToMatrix(Matrix rod_v)
    {
        if (rod_v.col * rod_v.row != 3)
            throw std::invalid_argument("Invalid Vector dimensions in Algebra::rodriguesToMatrix(Matrix)");
        Matrix rot = identity(3);
        float64 theta = rod_v.norm();
        if (theta > DBL_EPSILON)
        {
            float64 costh = std::cos(theta), sinth = std::sin(theta);
            rot(0,0) = rot(1,0) = rot(2,2) = costh;
            rod_v /= theta;
            for (size_t i = 0; i < 3; i++)
                for (size_t j = 0; j < 3; j++)
                    rot.m[i*3 + j] += (1 - costh) * rod_v.m[j]*rod_v.m[i];
            rod_v *= sinth;
            rot(0,1) -= rod_v(0,2), rot(0,2) += rod_v(0,1);
            rot(1,0) += rod_v(0,2), rot(1,2) -= rod_v(0,0);
            rot(2,0) -= rod_v(0,1), rot(2,1) += rod_v(0,0);
        }
        return rot;
    }

    inline Matrix matrixToRodrigues(Matrix R)
    {
        Matrix r({R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1)}); // r = [a32, a13, a21]
        float64 s = r.norm()*.5;
        float64 c = (R(0,0) + R(1,1) + R(2,2) - 1.) * .5;
        c = c > 1. ? 1. : c < -1. ? -1. : c;
        float64 theta = std::acos(c);

        if(s < 1e-5)
        {
            float64 t;

            if( c > 0 )
                zero(r);
            else
            {
                t = (R(0,0) + 1) * .5;
                r(0,0) = std::sqrt(std::max(t, 0.));
                t = (R(1,1) + 1) * 0.5;
                r(0,1) = std::sqrt(std::max(t, 0.)) * (R(0,1) < 0 ? -1. : 1.);
                t = (R(2,2) + 1) * 0.5;
                r(0,2) = std::sqrt(std::max(t, 0.)) * (R(0,2) < 0 ? -1. : 1.);
                if(std::fabs(r(0,0)) < std::fabs(r(0,1)) && std::fabs(r(0,0)) < std::fabs(r(0,2)) && (R(1,2) > 0) != (r(0,1)*r(0,2) > 0) )
                    r(0,2) = -r(0,2);
                theta /= r.norm();
                r *= theta;
            }
        }
        else
        {
            float64 vth = .5/s;
            vth *= theta;
            r *= vth;
        }
        return r;
    }

    inline Matrix cross(const Matrix &v1, const Matrix &v2)
    {
        Matrix c(3);
        c(0) = v1(1) * v2(2) - v1(2) * v2(1);
        c(1) = v1(2) * v2(0) - v1(0) * v2(2);
        c(2) = v1(0) * v2(1) - v1(1) * v2(0);
        return c;
    }

    inline float64 square(float64 n)
    {
        return n*n;
    }

#ifdef ALGEBRA_SHORT_NAMES
    using Vec = Matrix;
    using Mat = Matrix;
#endif

}