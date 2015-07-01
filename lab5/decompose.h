// Header file for homography decomposition, which includes the decomposition and 
// all associated helper functions. 

#include <Eigen/Dense>
#include <Eigen/SVD>

// All constructs must use doubles, not floats, in the interest of numerical accuracy.
struct decompose_return{
	Eigen::Matrix3d Ra;
	Eigen::Matrix3d Rb;
	Eigen::MatrixXd ta;
	Eigen::MatrixXd tb;
	Eigen::MatrixXd na;
	Eigen::MatrixXd nb;
};

Eigen::MatrixXd submatrix(Eigen::MatrixXd matrix, int indexRow, int indexColumn);
int sign(double number);
Eigen::MatrixXd normalize(Eigen::MatrixXd mat);
decompose_return homogDecompose(Eigen::Matrix3d H, Eigen::Matrix3d K);

Eigen::MatrixXd submatrix(Eigen::MatrixXd matrix, int indexRow, int indexColumn){
	// Returns the original matrix without the row of indexRow and column
	// of indexColumn. submatrix([1,2,3;4,5,6;7,8,9], 1, 1) will return
	// [1, 3; 7, 9].
	// Make it zero-indexed for consistency with C++ and Eigen convention.
	
	int rows = matrix.rows() - 1;
	int cols = matrix.cols() - 1;
	
	if (indexRow > rows || indexColumn > cols){ // Then it's a 
	 		// problem since we're trying to take out an index tat doesn't 
	 		// exist in the input matrix.
		exit(-1);
	}

	Eigen::MatrixXd ret(rows, cols);

	// If we're past the row or column that we want to exclude from the submatrix, then
	// we still want to pick the appropriate value from matrix (input) but subtract one from
	// the row/column value that we're going to be assigning that value to in ret. 
	for (int i = 0; i < matrix.rows(); i++){
		for (int j = 0; j < matrix.rows(); j++){
			if (i != indexRow && j != indexColumn){
				int jIndex = (j < indexColumn) ? j : j - 1;
				int iIndex = (i < indexRow) ? i : i - 1;
				//cout << "i " << iIndex << " j " << jIndex << endl;
				ret(iIndex, jIndex) = matrix(i, j);
			}
		}
	}
	
	//std::cout << "submatrix is " << ret << std::endl;
	
	//std::cout << "Determinant is " << ret.determinant() << std::endl;
	
	return ret;
}

int sign(double number){
// sgn=@(a)2*(a>=0)-1 -- MATLAB
// Gets the sign of the number, 0 being positive (+1). 
	int num = 2 * (number >= 0) - 1;
	return num;
}

Eigen::MatrixXd normalize(Eigen::MatrixXd mat){
// Vector divided by the norm of the vector. 
	return mat / mat.norm();
}

// Decompose a 3x3 homography with a 3x3 intrinsic camera calibration matrix, returning
// two valid solutions, R/n/t a and b. Method is from paper "A deeper understanding of the 
// homography decomposition for vision-based control." 
decompose_return homogDecompose(Eigen::Matrix3d H, Eigen::Matrix3d K){
	H = K.inverse() * H * K;
	
	
	// Take the SVD
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(H);
	
	Eigen::MatrixXd svs(3,1); // Array of singular values
	svs = svd.singularValues();
	
	// Find the median singular value, just by deciding which value is in between the other
	// two values. It's probably the middle value from the .singularValues function, but 
	// it doesn't hurt to check. 
	double medianSVD;
	if ( (svs(1) > svs(0) && svs(1) < svs(2)) || (svs(1) < svs(0) && svs(1) > svs(2) ) )
		medianSVD = svs(1);
	if ( (svs(0) > svs(2) && svs(0) < svs(1)) || (svs(0) < svs(2) && svs(0) > svs(1) ) )
		medianSVD = svs(0);
	if ( (svs(2) > svs(0) && svs(2) < svs(1)) || (svs(2) < svs(0) && svs(2) > svs(1) ) )
		medianSVD = svs(2);
		
	//std::cout << "SVDs: " << svs(0) << "  " << svs(1) << "  " << svs(2) << "   " << medianSVD << std::endl; // Good to here, 
	// except minor numerical discrepencies. 
		
	// Divide H by the median singular value so that it has... uh... wait... H isn't in SL(3) group any more.
	H = H / medianSVD;
	Eigen::MatrixXd eye3;
	eye3 = Eigen::MatrixXd::Identity(3, 3);
	Eigen::MatrixXd Sr(3,3);
	Sr = H * H.transpose() - eye3;
	//std::cout << "H_mark is " << std::endl << Sr << std::endl << std::endl;
	
	double sr11 = Sr(0, 0);
	double sr12 = Sr(0, 1);
	double sr13 = Sr(0, 2);
	double sr22 = Sr(1, 1);
	double sr23 = Sr(1, 2);
	
	double M_sr11 = abs(-1 * submatrix(Sr,0,0).determinant()); // This needs to be fixed.
	double M_sr22 = abs(-1 * submatrix(Sr,1,1).determinant());
	double M_sr33 = abs(-1 * submatrix(Sr,2,2).determinant());
	double M_sr13 = (-1 * submatrix(Sr,0,2).determinant());
	double M_sr23 = (-1 * submatrix(Sr,1,2).determinant());
	
	//std::cout << "srs are " << sr11 << "  " << sr12 << "  " << sr13 << "  " << sr22 << "  " << sr23 << std::endl;

	int er23 = sign(M_sr23);
	int er13 = sign(M_sr13);
	
	Eigen::MatrixXd ta_sr11(3,1), tb_sr11(3,1);
	ta_sr11 << sr11, sr12 + sqrt(M_sr33), sr13 + er23 * sqrt(M_sr22);
	tb_sr11 << sr11, sr12 - sqrt(M_sr33), sr13 - er23 * sqrt(M_sr22);
	
	//std::cout << ta_sr11 << " ta  tb" << tb_sr11 << " sr22" << M_sr22<< std::endl;
	
	double nu_r = 2 * sqrt(abs(1+Sr.trace() - M_sr11 -  M_sr22 - M_sr33) );
	double norm_te_r_squared = 2 + Sr.trace() - nu_r;
	double norm_te_r = sqrt(norm_te_r_squared);
	double rho_r = sqrt(2 + Sr.trace() + nu_r);// These four are good
	
	//std::cout << "Check: " << nu_r << "  " << norm_te_r_squared << "  " << norm_te_r << "  " << rho_r << std::endl;
	
	Eigen::MatrixXd tar_11(3,1), tbr_11(3,1);
	tar_11 = normalize(ta_sr11) * norm_te_r;
	tbr_11 = normalize(tb_sr11) * norm_te_r;
   // std::cout << tar_11 << std::endl;
	
	Eigen::MatrixXd nar_11(3,1), nbr_11(3,1); 
	nar_11 = normalize(0.5 * (sign(sr11) * rho_r / norm_te_r * tbr_11 - tar_11 ) );
    nbr_11 = normalize(0.5 * (sign(sr11) * rho_r / norm_te_r * tar_11 - tbr_11) );
	//std::cout << nar_11 << " ta  tb" << nbr_11 << " sr22  " << M_sr22<< std::endl;
    
    // Answers to the decomposition
    decompose_return rVal;
    
    rVal.Ra = (eye3 - (2/ nu_r) * tar_11 * nar_11.transpose() ) * H;
    rVal.Rb = (eye3 - (2/ nu_r) * tbr_11 * nbr_11.transpose() ) * H;
    
    
    rVal.ta = normalize(tar_11.transpose());
    rVal.tb = normalize(tbr_11.transpose());
    rVal.na = rVal.Ra.transpose() * nar_11;
    rVal.nb = rVal.Rb.transpose() * nbr_11;
        
    return rVal;
}
