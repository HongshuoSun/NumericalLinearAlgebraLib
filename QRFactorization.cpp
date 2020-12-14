//
// Created by bugma on 2020/11/4.
//

#include "QRFactorization.h"

#include <iostream>
#include <cmath>
#include <limits>
#include <vector>
#include <Eigen/LU>
#include <chrono>
#include <zconf.h>
using std::vector;
using std::pair;
using Eigen::VectorXd;
using Eigen::Matrix2Xd;
using Eigen::VectorXi;
using std::endl;
using std::cout;
inline float sign(float val) {
    return val>0.0?1.0:-1.0;
}
const double dZero = 0.000000001;

bool QRFactorization::NormalGramSchmidt(const Eigen::MatrixXd& mat, Eigen::MatrixXd& Q, Eigen::MatrixXd& R){
    int rols = mat.rows();
    int cols = mat.cols();
    Q = mat;
    R=Eigen::MatrixXd(cols,cols);
    R.setZero();
    for(int i=0;i<cols;i++){
        for(int j=0;j<i-1;j++){
            R(j,i) =  Q.col(i).dot( Q.col(j));
            Q.col(i) = Q.col(i)-(R(j,i)*Q.col(j));
        }
        R(i,i) = Q.col(i).norm();
        Q.col(i).stableNormalize();
    }
    return true;
}
bool QRFactorization::ModifiedGramSchmidt(const Eigen::MatrixXd& mat,Eigen::MatrixXd& Q,Eigen::MatrixXd& R) {
    int cols = mat.cols();
    Q = mat;
    R = Eigen::MatrixXd(cols, cols);
    R.setZero();
    for (int i = 0; i < cols; i++) {
        R(i, i) = Q.col(i).norm();
        Q.col(i).stableNormalize();

        for (int j = i + 1; j < cols; j++) {
            R(i, j) = Q.col(i).dot(Q.col(j));
        }
        for (int j = i + 1; j < cols; j++) {
            Q.col(j) = Q.col(j) - R(i, j) * Q.col(i);
        }
    }
    return true;
}

bool QRFactorization::HouseHolderMethod(const Eigen::MatrixXd& mat,Eigen::MatrixXd& Q,Eigen::MatrixXd& R ) {
    int cols = mat.cols();
    int rows = mat.rows();
    Q = Eigen::MatrixXd(rows, rows);

    R = mat.eval();
    Q.setIdentity();

    VectorXd x;
    VectorXd e = VectorXd(rows, 1);;
    size_t rRows, rCols, qRows, qCols;
    rRows = R.rows(), rCols = R.cols(), qRows = Q.rows(), qCols = Q.cols();

    for (int k = 0; k < rCols; k++) {
        x = R.col(k).tail(rows - k).eval();
        e.resize(rows-k,1);
        e.setZero();
        e(0, 0) = 1.0;
        x = sign(x(0, 0)) * x.norm() * e + x;
        x.normalize();

        R.bottomRightCorner(rRows - k, rCols - k) = R.bottomRightCorner(rRows - k, rCols - k) - (2 * x *
                                                                                                 (x.transpose() *
                                                                                                  R.bottomRightCorner(
                                                                                                          rRows - k,
                                                                                                          rCols -
                                                                                                          k)));
        Q.bottomRows(qRows - k) = Q.bottomRows(qRows - k) - (2 * x * (x.transpose() * Q.bottomRows(qRows - k)));
    }
    Q.transposeInPlace();
    return true;
}
template <typename  T>
inline void GetGivensMatrix(T x1,T x2,Matrix2Xd& givensMatrix){
    T cos=0.0,sin=0.0;
    T norm = sqrt(x1*x1+x2*x2);
    if(std::abs(x2)< std::numeric_limits<T>::epsilon()){
        cos=1,sin = 0;
    }else{
        cos = x1/norm;
        sin = x2/norm;
    }
    givensMatrix(0,0)=cos;
    givensMatrix(0,1)=sin;
    givensMatrix(1,0)=-sin;
    givensMatrix(1,1)=cos;
}
bool QRFactorization::GivensMethod(const Eigen::MatrixXd& mat,Eigen::MatrixXd& Q,Eigen::MatrixXd& R ){
    int cols = mat.cols();
    int rows = mat.rows();
    Q = Eigen::MatrixXd(rows, rows);
    R = mat.eval();
    Q.setIdentity();
    size_t rRows,rCols,qRows,qCols;

    rRows = R.rows(),rCols = R.cols(),qRows = Q.rows(),qCols = Q.cols();
    Eigen::Matrix2Xd givens(2,2);
    for (int c = 0; c < rCols-1; c++) {
        for(int r=rRows-1;r>c;r--){
            GetGivensMatrix(R(r-1,c), R(r,c),givens);
            R.block(r-1,c,2,rCols-c) =givens*     R.block(r-1,c,2,rCols-c);
            Q.block(r-1,0,2,qCols) =   givens* Q.block(r-1,0,2,qCols);
        }
    }
    Q.transposeInPlace();
    return true;
}

bool QRFactorization::HessenbergGivenMethod(const Eigen::MatrixXd& mat,Eigen::MatrixXd& Q,Eigen::MatrixXd& R ){
    int cols = mat.cols();
    int rows = mat.rows();
    Q = Eigen::MatrixXd(rows, rows);
    R = mat.eval();
    Q.setIdentity();
    size_t rRows,rCols,qRows,qCols;

    rRows = R.rows(),rCols = R.cols(),qRows = Q.rows(),qCols = Q.cols();
    Eigen::Matrix2Xd givens(2,2);
    for (int c = 0; c < rCols-1; c++) {
        int r = c + 1;
        GetGivensMatrix(R(r - 1, c), R(r, c), givens);
        R.block(r - 1, c, 2, rCols - c) = givens * R.block(r - 1, c, 2, rCols - c);
        Q.block(r - 1, 0, 2, qCols) = givens * Q.block(r - 1, 0, 2, qCols);

    }
    Q.transposeInPlace();
    return true;
}

bool QRFactorization::HessenbergDecomposition(const Eigen::MatrixXd& mat,Eigen::MatrixXd& Q,Eigen::MatrixXd& hessen){
    hessen = mat;
    int cols = hessen.cols();
    int rows = hessen.rows();
    if(cols!=rows || cols <1){
        return false;
    }
    VectorXd e;
    VectorXd w;
    Q = MatrixXd::Identity(rows,cols);
    for(int i=0;i<cols-2;i++){
        w = hessen.col(i).tail(rows-i-1).eval();
        e = VectorXd (rows-i-1,1);
        e.setZero();
        e(0,0)=1.0f;
        w =  sign(w(0,0))*w.norm()*e+w;
        w.stableNormalize();
        hessen.bottomRightCorner(rows-i-1,cols-i) = hessen.bottomRightCorner(rows-i-1,cols-i) - (2*w*(w.transpose()* hessen.bottomRightCorner(rows-i-1,cols-i)));
        hessen.rightCols(cols-i-1) =  hessen.rightCols(cols-i-1) - (2* (hessen.rightCols(cols-i-1)*w)*w.transpose());
        Q.bottomRows(cols-i-1) =   Q.bottomRows(cols-i-1)-(2*w*(w.transpose()*  Q.bottomRows(cols-i-1)));
    }
   Q.transposeInPlace();
    return true;
}

bool QRFactorization::LUDecomposition(const Eigen::MatrixXd& mat,Eigen::MatrixXd& L,Eigen::MatrixXd& U){
    size_t rows = mat.rows();
    size_t cols = mat.cols();
    if(rows!=cols || cols<1){
        return false;
    }
    L = MatrixXd::Identity(rows,cols);
    U = mat;
    for(int c=0;c<cols;c++){
        MatrixXd::value_type current = U(c,c);
        if(std::abs(current)<dZero){
            return false;
        }
        for(int r=c+1;r<rows;r++){
            MatrixXd::value_type scale = -U(r,c)/current;
            L(r,c)=-scale;
            for(int c1=c;c1<cols;c1++){
                U(r,c1) += U(c,c1)*scale;
            }
        }
    };
    return true;
}
bool QRFactorization::LUDecompositionColPivot(const Eigen::MatrixXd& mat,Eigen::MatrixXd& P,Eigen::MatrixXd& L,Eigen::MatrixXd& U){
    size_t rows = mat.rows();
    size_t cols = mat.cols();

    P=MatrixXd(rows,rows);
    P.setIdentity();
    L = MatrixXd::Identity(rows,rows);
    U = mat;
    int r1= 0;
    for(int c=0;c<cols&&r1<rows;c++,r1++){
        size_t maxIndex=c;
        U.col(c).tail(rows-r1).cwiseAbs().maxCoeff(&maxIndex);
        maxIndex +=c;
        MatrixXd::value_type current = U(maxIndex,c);
        if(maxIndex!=c){
            U.row(maxIndex).tail(cols-c).swap(U.row(c).tail(cols-c));
            L.row(maxIndex).head(c).swap(L.row(c).head(c));
            P.row(maxIndex).swap(P.row(c));
        }
        if(std::abs(U(r1,c))<dZero){
            continue;
        }
        for(int r = r1+1 ;r<rows;r++){
            MatrixXd::value_type scale = -U(r,c)/current;
            L(r,c)=-scale;
            for(int c1=c;c1<cols;c1++){
                U(r,c1) += U(c,c1)*scale;
            }
        }
    }
    return true;
}
bool QRFactorization::LUDecompositionFullPivot(const Eigen::MatrixXd& mat,Eigen::MatrixXd& P,Eigen::MatrixXd& L,Eigen::MatrixXd& U,Eigen::MatrixXd& Q) {
    size_t rows = mat.rows();
    size_t cols = mat.cols();
    P = MatrixXd(rows, rows);
    P.setIdentity();
    Q = MatrixXd(cols, cols);
    Q.setIdentity();
    MatrixXd lu = mat;
    int size = std::min(rows, cols);
    for (int k = 0; k < size; k++) {
        size_t maxCoeffRow = k;
        size_t maxCoeffCol = k;
        lu.bottomRightCorner(rows - k, cols - k).cwiseAbs().maxCoeff(&maxCoeffRow, &maxCoeffCol);
        maxCoeffRow += k;
        maxCoeffCol += k;

        double current = lu(maxCoeffRow, maxCoeffCol);
        if (std::abs(current) < dZero) {
            break;
        }
        if (maxCoeffRow != k || maxCoeffCol != k) {
            lu.row(maxCoeffRow).swap(lu.row(k));
            lu.col(maxCoeffCol).swap(lu.col(k));
            P.row(maxCoeffRow).swap(P.row(k));
            Q.col(maxCoeffCol).swap(Q.col(k));
        }

        for (int r = k + 1; r < rows; r++) {
            MatrixXd::value_type scale = -lu(r, k) / current;
            lu(r, k) = -scale;
            for (int c1 = k + 1; c1 < cols; c1++) {
                lu(r, c1) += lu(k, c1) * scale;
            }
        }
    }
    L = MatrixXd::Identity(rows,rows);
    U = MatrixXd::Zero(rows,cols);
    L.topLeftCorner(rows,std::min(cols,rows)).triangularView<Eigen::StrictlyLower>() =lu.topLeftCorner(rows,std::min(cols,rows)).triangularView<Eigen::StrictlyLower>();
    U = lu.triangularView<Eigen::Upper>();
    return true;
}

bool QRFactorization::CholeskyFactorization(const Eigen::MatrixXd&  HermitianMatrix,Eigen::MatrixXd& R) {
    int rows = HermitianMatrix.rows();
    int cols = HermitianMatrix.cols();
    if (rows != cols || rows < 1) {
        return false;
    }
    size_t size = rows;
    R =HermitianMatrix;
    double alpha = 0;
    for(size_t i = 0;i<size-1;i++){
        alpha = 1/R(i,i);
        for(size_t j = i+1;j<size;j++){
            R.col(j).tail(size-i-1) = R.col(j).tail(size-i-1) - R(i,j)*alpha*R.col(i).tail(size-i-1);
        }
        R.row(i).tail(size-i-1) = sqrt(alpha)*R.row(i).tail(size-i-1);
        R.col(i).tail(size-i-1).setZero();
        R(i,i) = sqrt(R(i,i));
    }
    R(size-1,size-1) = sqrt(R(size-1,size-1));
    return true;
}
bool QRFactorization::PowerIteratorMethod(const Eigen::MatrixXd& squaredMatrix,Eigen::MatrixXd::value_type eigenValue,Eigen::VectorXd& eigenVector){
    size_t rows = squaredMatrix.rows();
    size_t cols = squaredMatrix.cols();
    if(rows!=cols || cols<1){
        return false;
    }
    VectorXd vec(rows,1);
    for(int i=0;i<cols;i++){
        vec+=squaredMatrix.col(i);
    }
    eigenValue = vec.norm();
    vec.normalize();
    MatrixXd::value_type  laseValue = 0;
    int i=0;
    int iteratorCount = rows*10;
    while(i++<iteratorCount&& abs(eigenValue-laseValue)>(dZero)){
        vec = squaredMatrix*vec;
        vec.normalize();
        laseValue = eigenValue;
        eigenValue = (vec.transpose()*squaredMatrix*vec)(0,0);
    }
    eigenVector = vec;
    return i<iteratorCount;
}
inline bool IsMatLowerTriangleZero(const Eigen::MatrixXd& mat,const int matSize=-1,const double zero=(dZero*10)){
    int size = matSize;
    if(size<1){
        size = std::min(mat.cols(),mat.rows());
    }
    for(int c=0;c<size-1;c++){
        for(int r=c+1;r<size;r++){
            if( std::abs(mat(r,c))>zero){
                return false;
            }
        }
    }
    return true;
}

inline void FillLowerZero(Eigen::MatrixXd& mat,size_t size){
    for(int i=0;i<size-1;i++){
        mat.col(i).tail(size-i-1).setZero();
    }
}
bool QRFactorization::SolveEigenByQR(const Eigen::MatrixXd& mat,Eigen::MatrixXd& eigenVectors,Eigen::MatrixXd& eigenValues) {
    size_t rows = mat.rows();
    size_t cols = mat.cols();
    if (rows != cols || cols < 1) {
        return false;
    }
    MatrixXd Q, R;
    MatrixXd Ak = mat;
    Q = MatrixXd::Identity(rows, cols);
    eigenVectors = MatrixXd::Zero(rows, cols);
    R = Ak;
    int iteratorCount = rows * 100;
    int index = 0;
    while (index++ < iteratorCount && !IsMatLowerTriangleZero(Ak, rows)) {
        QRFactorization::HouseHolderMethod(Ak, Q, R);
        Ak = R * Q;
    }
    if (index >= iteratorCount) {
        return false;
    }
    FillLowerZero(Ak, rows);
    VectorXd eigenValue = Ak.diagonal();
    // std::sort(eigenValue.data(),eigenValue.data()+eigenValue.size(),std::greater<decltype(eigenValue)::value_type>());
    std::sort(eigenValue.data(), eigenValue.data() + eigenValue.size(),
              [](const double a, const double b) -> bool { return std::abs(a) >= std::abs(b); });
    eigenValues = Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic>(eigenValue);

    int eigenValueIndex = 0;
    while (eigenValueIndex < eigenValue.size()) {
        int count = 1;
        int uniqueEnd = eigenValueIndex + 1;
        while (uniqueEnd < eigenValue.size() && std::abs(eigenValue[uniqueEnd] - eigenValue[eigenValueIndex]) < dZero) {
            uniqueEnd++;
            count++;
        }
        Eigen::MatrixXd kernel;
        GetMatrixKernel(mat - eigenValue(eigenValueIndex) * MatrixXd::Identity(rows, cols), kernel);
        if (kernel.cols() < count) {
            return false;
        }
        eigenVectors.middleCols(eigenValueIndex, count) = kernel;
        eigenValueIndex = uniqueEnd;
    }
    std::cout << "mat:\n" << mat << endl;

    Eigen::EigenSolver<MatrixXd> solver1(mat);
    VectorXd vec1 = solver1.eigenvalues().real();
    std::sort(vec1.data(), vec1.data() + vec1.size(),
              [](const double a, const double b) -> bool { return std::abs(a) >= std::abs(b); });
    return index < iteratorCount;
}

bool QRFactorization::SolveEigenByHouseHolderQR(const Eigen::MatrixXd& mat,Eigen::MatrixXd& eigenValue,Eigen::MatrixXd& eigenVec){
    size_t rows = mat.rows();
    size_t cols = mat.cols();
    if(rows !=cols || rows<1){
        return false;
    }
    size_t size = rows;
    Eigen::MatrixXd hessenbergMat,hessenQ;
    QRFactorization::HessenbergDecomposition(mat,hessenQ,hessenbergMat);
    Eigen::MatrixXd  q,r;
    int itCount;
    int itUpper =size*size*100;
    for(itCount=0;itCount<itUpper;itCount++){
        QRFactorization::HessenbergGivenMethod(hessenbergMat,q,r);
        hessenbergMat = r*q;
        if(IsMatLowerTriangleZero(hessenbergMat)){
            break;
        }
    }
    if(itCount>=itUpper){
        assert(false);
        return false;
    }

    VectorXd eigenValVec = hessenbergMat.diagonal();
    std::sort(eigenValVec.data(), eigenValVec.data() + eigenValVec.size(),
              [](const double a, const double b) -> bool { return std::abs(a) >= std::abs(b); });
    bool flag;
    eigenValue = MatrixXd::Zero(size,size);
    eigenVec = MatrixXd::Zero(size,size);
    for(int i=0;i<eigenValVec.rows();i++){
        VectorXd vec;
       flag =  QRFactorization::InverseIterator(mat,eigenValVec(i,0),vec);
       if(!flag){
           return false;
       }else{
           eigenValue(i,i)=eigenValVec(i,0);
           eigenVec.col(i) = vec;
       };
    }

    return true;
}

bool QRFactorization::SolveEigenByHouseHolderQR2(const Eigen::MatrixXd& mat,Eigen::VectorXd& eigenValue){
    size_t rows = mat.rows();
    size_t cols = mat.cols();
    if(rows !=cols || rows<1){
        return false;
    }
    size_t size = rows;
    Eigen::MatrixXd hessenbergMat,hessenQ;
    QRFactorization::HessenbergDecomposition(mat,hessenQ,hessenbergMat);
    Eigen::MatrixXd  q,r;
    int itCount;
    int itUpper =size*size;
    double mu = 0;
    size_t currentSize = size;
    eigenValue=Eigen::VectorXd(size,1);
    if(size==1){
        eigenValue(0,0)=mat(0,0);
        return true;
    }
    for(itCount=0;itCount<itUpper;itCount++){
        mu = hessenbergMat(currentSize-1,currentSize-1);
        hessenbergMat = hessenbergMat - mu*MatrixXd::Identity(currentSize,currentSize);
        QRFactorization::HessenbergGivenMethod(hessenbergMat,q,r);
        hessenbergMat = r*q;
        hessenbergMat = hessenbergMat + mu*MatrixXd::Identity(currentSize,currentSize);
        if(std::abs(hessenbergMat(currentSize-1,currentSize-2))<dZero){
            if(currentSize>2){
                eigenValue(currentSize-1,0) = hessenbergMat(currentSize-1,currentSize-1);
                currentSize--;
                hessenbergMat = hessenbergMat.topLeftCorner(currentSize,currentSize).eval();
            }
            else{
                eigenValue(0,0) = hessenbergMat(0,0);
                eigenValue(1,0) = hessenbergMat(1,1);
                break;
            }
        }
    }
    if(itCount>=itUpper){
        return false;
    }

    std::sort(eigenValue.data(), eigenValue.data() + eigenValue.size(),
              [](const double a, const double b) -> bool { return std::abs(a) >= std::abs(b); });
    return true;

}
bool QRFactorization::GetMatrixKernel(const Eigen::MatrixXd& mat,Eigen::MatrixXd& kernel) {
    size_t rows ,cols ;
    size_t size;
    rows = mat.rows();
    cols = mat.cols();
    size = std::min(rows,cols);
    if(std::min(rows,cols)<1){
        return false;
    }
    MatrixXd  l,u,p,q;
    QRFactorization::LUDecompositionFullPivot(mat,p,l,u,q);
    size_t pivotCount = 0;
    for(pivotCount= 0;pivotCount<size;pivotCount++){
        if( std::abs(u(pivotCount,pivotCount))<dZero){
            break;
        }
    }
    if(pivotCount>=cols){
        return true;
    }
    kernel = MatrixXd::Zero(cols,cols-pivotCount);
    for(size_t i =pivotCount;i<cols;i++){
        VectorXd current=VectorXd::Zero(cols,1);
        current(i,0)=1;
        for(int j = i-1;j>=0;j--){
            double dot = u.row(j)*current;
            if(std::abs(dot)<dZero || std::abs(u(j,j))<dZero ){
                continue;
            }else{
                current(j,0) = -dot/u(j,j);
            }
        }
        current.normalize();
        kernel.col(i-pivotCount)=current;
    }
    kernel = q*kernel;
    std::cout<<"res:\n"<< mat*kernel<<endl;
    return true;
}
bool QRFactorization::GetMatrixImage(const Eigen::MatrixXd& mat,Eigen::MatrixXd& image){
    size_t rows ,cols;
    size_t size;
    rows = mat.rows();
    cols = mat.cols();
    size = std::min(rows,cols);
    if(std::min(rows,cols)<1){
        return false;
    }
    MatrixXd  l,u,p,q;
    QRFactorization::LUDecompositionFullPivot(mat,p,l,u,q);
    size_t pivotCount = 0;
    for(pivotCount= 0;pivotCount<size;pivotCount++){
        if( std::abs(u(pivotCount,pivotCount))<dZero){
            break;
        }
    }
    if(pivotCount<1){
        return true;
    }
    image =  p.transpose()*l * u.leftCols(pivotCount).eval();
    return true;
}
bool QRFactorization::InverseIterator(const Eigen::MatrixXd& squaredMatrix,Eigen::MatrixXd::value_type eigenValue,Eigen::VectorXd& eigenVector){
    size_t rows,cols,size;
    rows = squaredMatrix.rows();
    cols = squaredMatrix.cols();
    if(rows!=cols || rows<1){
        return false;
    }
    size = rows;
    eigenVector = VectorXd::Random(size,1);
    VectorXd last = VectorXd::Random(size,1);
    Eigen::FullPivLU<MatrixXd> lu(squaredMatrix-eigenValue*MatrixXd::Identity(size,size));
    size_t current=0,total=size*size;
    while( (current++<total)&&  (((last-eigenVector).cwiseAbs().maxCoeff()>dZero) && ((last+eigenVector).cwiseAbs().maxCoeff()>dZero)) ){
        last = eigenVector;
        eigenVector = lu.solve(eigenVector);
        eigenVector.normalize();
    }
    if(current>=total){
        std::cout<<"last:\n"<<last.transpose()<<endl;
        std::cout<<"evec:\n"<<eigenVector.transpose()<<endl;
        return false;
    }
    return true;
}