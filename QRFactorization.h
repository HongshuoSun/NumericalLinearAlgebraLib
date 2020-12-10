//
// Created by bugma on 2020/11/4.
//

#ifndef MAIN_QRFACTORIZATION_H
#define MAIN_QRFACTORIZATION_H


#include <Eigen/Core>
#include <Eigen/Dense>

using Eigen::MatrixXd;
class QRFactorization{
public:
    static bool NormalGramSchmidt(const Eigen::MatrixXd& mat,Eigen::MatrixXd& Q,Eigen::MatrixXd& R);
    static bool ModifiedGramSchmidt(const Eigen::MatrixXd& mat,Eigen::MatrixXd& Q,Eigen::MatrixXd& R);
    static bool HouseHolderMethod(const Eigen::MatrixXd& mat,Eigen::MatrixXd& Q,Eigen::MatrixXd& R );
    static bool GivensMethod(const Eigen::MatrixXd& mat,Eigen::MatrixXd& Q,Eigen::MatrixXd& R );
    static bool HessenbergGivenMethod(const Eigen::MatrixXd& mat,Eigen::MatrixXd& Q,Eigen::MatrixXd& R );
    static bool HessenbergDecomposition(const Eigen::MatrixXd& mat,Eigen::MatrixXd& Q,Eigen::MatrixXd& hessen);
    static bool LUDecomposition(const Eigen::MatrixXd& mat,Eigen::MatrixXd& L,Eigen::MatrixXd& U);
    static bool LUDecompositionColPivot(const Eigen::MatrixXd& sqaredMat,Eigen::MatrixXd& P,Eigen::MatrixXd& L,Eigen::MatrixXd& U);
    static bool LUDecompositionFullPivot(const Eigen::MatrixXd& mat,Eigen::MatrixXd& P,Eigen::MatrixXd& L,Eigen::MatrixXd& U,Eigen::MatrixXd& Q);
    static bool CholeskyFactorization(const Eigen::MatrixXd&  HermitianMatrix,Eigen::MatrixXd& R);
    static bool PowerIteratorMethod(const Eigen::MatrixXd& squaredMatrix,Eigen::MatrixXd::value_type eigenValue,Eigen::VectorXd& eigenVector);
    static bool InverseIterator(const Eigen::MatrixXd& squaredMatrix,Eigen::MatrixXd::value_type eigenValue,Eigen::VectorXd& eigenVector);
    static bool SolveEigenByQR(const Eigen::MatrixXd& mat,Eigen::MatrixXd& eigenVectors,Eigen::MatrixXd& eigenValues);
    static bool GetMatrixKernel(const Eigen::MatrixXd& mat,Eigen::MatrixXd& kernel);
    static bool GetMatrixImage(const Eigen::MatrixXd& mat,Eigen::MatrixXd& image);
    static bool SolveEigenByHouseHolderQR(const Eigen::MatrixXd& mat,Eigen::MatrixXd& eigenValue,Eigen::MatrixXd& eigenVec);
    static bool SolveEigenByHouseHolderQR2(const Eigen::MatrixXd& mat,Eigen::MatrixXd& eigenValue);
};
#endif //MAIN_QRFACTORIZATION_H
