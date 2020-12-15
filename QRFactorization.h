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
    constexpr static const double doubleEps = 0.00000001;
    static bool NormalGramSchmidt(const Eigen::MatrixXd& mat,Eigen::MatrixXd& Q,Eigen::MatrixXd& R,double eps=doubleEps);
    static bool ModifiedGramSchmidt(const Eigen::MatrixXd& mat,Eigen::MatrixXd& Q,Eigen::MatrixXd& R,double eps=doubleEps);
    static bool HouseHolderMethod(const Eigen::MatrixXd& mat,Eigen::MatrixXd& Q,Eigen::MatrixXd& R ,double eps=doubleEps);
    static bool GivensMethod(const Eigen::MatrixXd& mat,Eigen::MatrixXd& Q,Eigen::MatrixXd& R ,double eps=doubleEps);
    static bool HessenbergGivenMethod(const Eigen::MatrixXd& mat,Eigen::MatrixXd& Q,Eigen::MatrixXd& R ,double eps=doubleEps);
    static bool HessenbergDecomposition(const Eigen::MatrixXd& mat,Eigen::MatrixXd& Q,Eigen::MatrixXd& hessen,double eps=doubleEps);
    static bool LUDecomposition(const Eigen::MatrixXd& mat,Eigen::MatrixXd& L,Eigen::MatrixXd& U,double eps=doubleEps);
    static bool LUDecompositionColPivot(const Eigen::MatrixXd& sqaredMat,Eigen::MatrixXd& P,Eigen::MatrixXd& L,Eigen::MatrixXd& U,double eps=doubleEps);
    static bool LUDecompositionFullPivot(const Eigen::MatrixXd& mat,Eigen::MatrixXd& P,Eigen::MatrixXd& L,Eigen::MatrixXd& U,Eigen::MatrixXd& Q,double eps=doubleEps);
    static bool CholeskyFactorization(const Eigen::MatrixXd&  HermitianMatrix,Eigen::MatrixXd& R,double eps=doubleEps);
    static bool PowerIteratorMethod(const Eigen::MatrixXd& squaredMatrix,Eigen::MatrixXd::value_type eigenValue,Eigen::VectorXd& eigenVector,double eps=doubleEps);
    static bool InverseIterator(const Eigen::MatrixXd& squaredMatrix,Eigen::MatrixXd::value_type eigenValue,Eigen::VectorXd& eigenVector,double eps=doubleEps);
    static bool SolveEigenByQR(const Eigen::MatrixXd& mat,Eigen::MatrixXd& eigenVectors,Eigen::MatrixXd& eigenValues,double eps=doubleEps);
    static bool GetMatrixKernel(const Eigen::MatrixXd& mat,Eigen::MatrixXd& kernel,double eps=doubleEps);
    static bool GetMatrixImage(const Eigen::MatrixXd& mat,Eigen::MatrixXd& image,double eps=doubleEps);
    static bool SolveEigenByHouseHolderQR(const Eigen::MatrixXd& mat,Eigen::MatrixXd& eigenValue,Eigen::MatrixXd& eigenVec,double eps=doubleEps);
    static bool SolveEigenByHouseHolderQR2(const Eigen::MatrixXd& mat,Eigen::VectorXd& eigenValue,double eps=doubleEps);

};
#endif //MAIN_QRFACTORIZATION_H
