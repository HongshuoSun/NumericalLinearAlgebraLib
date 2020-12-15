#include <iostream>
#include <chrono>
#include <zconf.h>
#include <omp.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <QRFactorization.h>
#include <cmath>
#define EIGEN_VECTORIZE_SSE4_
using namespace std;

void TestQR1() {
    size_t rows,cols;
    Eigen::MatrixXd q,r,p,l,u;
    Eigen::MatrixXd mat;
    Eigen::MatrixXd X,Diag;
    Eigen::VectorXd diag;
    bool flag = false;
    for (int _i = 0; _i < 10000; _i++) {
        rows = cols =rand()%30+1;
        mat = MatrixXd::Random(rows, cols)*0.1+MatrixXd::Random(rows, cols)*0.5+MatrixXd::Random(rows, cols)*1;
        mat = mat*mat.transpose();

        flag = QRFactorization::SolveEigenByHouseHolderQR2(mat,diag);
        std::cout<<"size:"<<rows<<" flag:"<<flag<<endl;

    }
}
int main(){
    Eigen::initParallel();
    try{
        TestQR1();
    } catch (std::exception& exp) {
        return 0;
    }
    return 0;
}


