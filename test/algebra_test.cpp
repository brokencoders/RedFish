#include <gtest/gtest.h>

#define ALGEBRA_IMPL
#include "../include/Algebra.h"
using namespace Algebra;

bool testSum(const size_t row, const size_t col)
{
    Matrix m1(row,col);
    for (size_t i = 0; i < row; i++)
        for (size_t j = 0; j < col; j++)
            m1(i,j) = i + j + 1;
    const Matrix m2 = m1;

    auto m3 = m1 + m2;
    m1 += m2;

    for (size_t i = 0; i < row; i++)
        for (size_t j = 0; j < col; j++)
            if (m3(i,j) != (i + j + 1)*2 || m1(i,j) != (i + j + 1)*2) return testing::AssertionFailure();
    
    auto m4 = m3 + 10;
    m3 += 10;
    for (size_t i = 0; i < row; i++)
        for (size_t j = 0; j < col; j++)
            if (m4(i,j) != (i + j + 1)*2 + 10 || m3(i,j) != (i + j + 1)*2 + 10) return testing::AssertionFailure();


    return testing::AssertionSuccess();
}

bool testSub(const size_t row, const size_t col)
{
    Matrix m1(row,col);
    for (size_t i = 0; i < row; i++)
        for (size_t j = 0; j < col; j++)
            m1(i,j) = i + j + 1;
    const Matrix m2 = m1;

    auto m3 = m1 - m2;
    m1 -= m2;

    for (size_t i = 0; i < row; i++)
        for (size_t j = 0; j < col; j++)
            if (m3(i,j) != 0 || m1(i,j) != 0) return false;
    
    auto m4 = m3 - 10;
    m3 -= 10;
    for (size_t i = 0; i < row; i++)
        for (size_t j = 0; j < col; j++)
            if (m4(i,j) != -10 || m3(i,j) != -10) return false;


    return true;
}

bool testMinus(const size_t row, const size_t col)
{
    Matrix m1(row,col);
    for (size_t i = 0; i < row; i++)
        for (size_t j = 0; j < col; j++)
            m1(i,j) = i + j + 1;
    const Matrix m2 = -m1;

    for (size_t i = 0; i < row; i++)
        for (size_t j = 0; j < col; j++)
            if (m2(i,j) != -((double)i+j+1)) return false;

    return true;
}

bool testDiv(const size_t row, const size_t col)
{
    Matrix m1(row,col);
    for (size_t i = 0; i < row; i++)
        for (size_t j = 0; j < col; j++)
            m1(i,j) = i + j + 1;
    const Matrix m2 = m1;

    auto m3 = m1 / m2;
    m1 /= m2;

    for (size_t i = 0; i < row; i++)
        for (size_t j = 0; j < col; j++)
            if (m3(i,j) != 1 || m1(i,j) != 1) return false;
    
    auto m4 = m3 / 10;
    m3 /= 10;
    for (size_t i = 0; i < row; i++)
        for (size_t j = 0; j < col; j++)
            if (m4(i,j) != .1 || m3(i,j) != .1) return false;


    return true;
}

bool testTranspose(const size_t row, const size_t col)
{
    Matrix m1(row,col);
    for (size_t i = 0; i < row; i++)
        for (size_t j = 0; j < col; j++)
            m1(i,j) = j + i*col;
    const Matrix m2 = m1.T();
    m1.transpose();

    for (size_t i = 0; i < col; i++)
        for (size_t j = 0; j < row; j++)
            if (m2(i,j) != i + j*col || m1(i,j) != i + j*col) return false;
    
    return true;
}

bool testMul(const size_t row, const size_t col)
{
    Matrix m1(row,col);
    for (size_t i = 0; i < row; i++)
        for (size_t j = 0; j < col; j++)
            m1(i,j) = i + j + 1;
    const Matrix m2 = m1.T();

    auto m3 = m1 * m2;
    auto mE = m2.transposeTimes(m2);
    m1 *= m2;

    for (size_t i = 0; i < row; i++)
        for (size_t j = 0; j < row; j++)
        {
            size_t n = 0;
            for (size_t k = 0; k < col; k++) n += (i+k+1)*(k+j+1);
            if (m3(i,j) != n || m1(i,j) != n || mE(i,j) != n) return false;
        }
    
    auto m4 = m3 * 10;
    m3 *= 10;
    for (size_t i = 0; i < row; i++)
        for (size_t j = 0; j < row; j++)
        {
            size_t n = 0;
            for (size_t k = 0; k < col; k++) n += (i+k+1)*(k+j+1);
            if (m3(i,j) != n*10 || m4(i,j) != n*10) return false;
        }


    return true;
}

TEST(TestAlgebra, testSum)
{
    EXPECT_TRUE(!testSum(1, 10) || !testSum(10, 1) || !testSum(10,10) || !testSum(3,3));
}

TEST(TestAlgebra, testSub)
{
    EXPECT_TRUE(!testSub(1, 10) || !testSub(10, 1) || !testSub(10,10) || !testSub(3,3));
}

TEST(TestAlgebra, testDiv)
{
    EXPECT_TRUE(!testDiv(1, 10) || !testDiv(10, 1) || !testDiv(10,10) || !testDiv(3,3));
}

TEST(TestAlgebra, testTranspose)
{
    EXPECT_TRUE(!testTranspose(1, 10) || !testTranspose(10, 1) || !testTranspose(10,10) || !testTranspose(3,3));
}

TEST(TestAlgebra, testMul)
{
    EXPECT_TRUE(!testMul(1, 10) || !testMul(10, 1) || !testMul(10,10) || !testMul(3,3));
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}