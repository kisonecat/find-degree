#include <iostream>
#include <string>
#include <list>
#include <stdexcept>
#include <Eigen/Dense>
#include <nlopt.hpp>
#include <random>

const double epsilon = 1e-4;

using namespace std;

typedef std::vector<double> SU;
const double pi = 3.14159265358979323846;

////////////////////////////////////////////////////////////////
// COMPUTE THE INDEX AT THE FIXED POINT!!!

// Seed with a real random value, if available
std::random_device rr;
// Choose a random mean between 1 and 6
std::default_random_engine randgen(rr());

typedef std::complex<double> scalar;
typedef Eigen::Matrix3cd matrix;

double random_in_range(double fMin, double fMax)
{
  std::uniform_real_distribution<> dis(fMin, fMax);
  return dis(randgen);
}

std::complex<double> cis(double angle) {
  return std::complex<double>( cos(angle), sin(angle) );
}


unsigned int binomial(unsigned int n, unsigned int k) {
  unsigned int c = 1, i;
  
  if (k > n-k) // take advantage of symmetry
    k = n-k;
  
  for (i = 1; i <= k; i++, n--) {
    if (c/i > UINT_MAX/n) // return 0 on overflow
      return 0;
      
    c = c / i * n + c % i * n / i;  // split c * n / i into (c / i * i + c % i) * n / i
  }
  
  return c;
}

int minus_one_to(int k) {
  if (k%2 == 0) return 1;

  return -1;
}

std::complex<double> ck( int k, std::complex<double>r ) {
  std::complex<double> result = 0;

  k = abs(k);
  
  for( int i = 0; i <= k/2; i++ ) {
    result += (double)minus_one_to(i) * binomial(k, 2*i) * std::pow(r,i) * std::pow(1.0-r, (k/2)-i);
  }

  return result;
}

std::complex<double> sk( int k, std::complex<double>r ) {
  std::complex<double> result = 0;

  int sign_k = (k > 0) ? 1 : -1;
  k = abs(k);
  
  for( int i = 0; i <= (k-1)/2; i++ ) {
    result += (double)minus_one_to(i) * binomial(k, 2*i+1) * std::pow(r,i) * std::pow(1.0-r, ((k-1)/2)-i);
  }

  return (double)sign_k * result;
}


// https://physics.stackexchange.com/questions/237988/good-reference-on-the-parametrization-of-su3-and-sun
double vector_to_su(const SU &x, matrix &m ) {
  double theta1 = x[0];
  double theta2 = x[1];
  double theta3 = x[2];
  double phi1 = x[3];
  double phi2 = x[4];
  double phi3 = x[5];
  double phi4 = x[6];
  double phi5 = x[7];
  
  m.row(0)[0] = cos(theta1) * cos(theta2) * cis(phi1);
  m.row(0)[1] = sin(theta1) * cis(phi3);
  m.row(0)[2] = cos(theta1) * sin(theta2) * cis(phi4);

  m.row(1)[0] = sin(theta2) * sin(theta3) * cis(- phi4 - phi5)
                - sin(theta1) * cos(theta2) *cos(theta3) * cis(phi1 + phi2 - phi3);
  m.row(1)[1] = cos(theta1) * cos(theta3) * cis(phi2);
  m.row(1)[2] = - cos(theta2) * sin(theta3) * cis(-phi1 - phi5)
                - sin(theta1) * sin(theta2) * cos(theta3) * cis(phi2 - phi3 + phi4);

  /*
  m.row(2)[0] = - sin(theta1) * cos(theta2) * sin(theta3) * cis(phi1-phi3+phi5)
                - sin(theta2) * cos(theta3) * cis(-phi2 - phi4);
  m.row(2)[1] = cos(theta1) * sin(theta3) * cis(phi5);
  m.row(2)[2] = cos(theta2) * cos(theta3) * cis(-phi1 - phi2)
                - sin(theta1) * sin(theta2) * sin(theta3) * cis(-phi3 + phi4 + phi5);
  */

  m.row(2) = m.row(0).cross( m.row(1) );
}

double random_su( matrix &m ) {
  SU x(8);
  for( int i = 0; i < 3; i++ ) x[i] = random_in_range(0,pi/2);
  for( int i = 3; i < 8; i++ ) x[i] = random_in_range(0,2*pi);
  vector_to_su( x, m );
}

bool is_special_unitary( const matrix &m ) {
  scalar one(1,0);

  if (abs( (m * m.adjoint() - matrix::Identity()).norm() ) > epsilon)
    return false;
  
  if (abs(m.determinant() - one) > epsilon)
    return false;

  return true;
}

void gramschmidt( matrix & A ) {
  // First vector just gets normalized
  A.row(0).normalize();
	
  for(unsigned int j = 1; j < A.rows(); ++j) {
    // Replace inner loop over each previous vector in A with fast matrix-vector multiplication
    A.row(j) -= A.topRows(j).transpose() * (A.topRows(j).conjugate() * A.row(j).transpose());
    // Normalize vector if possible (othw. means colums of A almsost lin. dep.
    if( A.row(j).norm() <= 10e-14 * A.row(j).norm() ) {
      std::cerr << "Gram-Schmidt failed because A has lin. dep columns. Bye." << std::endl;
      break;
    } else {
      A.row(j).normalize();
    }
  }

  A.row(A.rows()-1) /= A.determinant();
  return;
}

void selfmap_squaring( matrix &output, matrix &m ) {
  output = m*m;
}

void selfmap( matrix &output, const matrix &m ) {
  int k = 3;

  matrix bar = m.conjugate();
  
  matrix product = m * bar;
  scalar trace = product.row(0)[0] + product.row(1)[1] + product.row(2)[2];
  scalar cos2t = (trace - 1.0) / 2.0;
  scalar cost2 = 0.5 + cos2t / 2.0;
  scalar sint2 = 1.0 - cost2;
  scalar r = sint2;
                  
  Eigen::Matrix<scalar,3,1> column;
  column.col(0)[0] = bar.row(1)[2] - bar.row(2)[1];
  column.col(0)[1] = bar.row(2)[0] - bar.row(0)[2];
  column.col(0)[2] = bar.row(0)[1] - bar.row(1)[0];

  Eigen::Matrix<scalar,1,3> row = column.transpose();

  matrix remainder = column * row / 4 / sint2;
  
  output = sk(k,r) * (m - m.transpose())/2.0 + ck(k,r) * ((m + m.transpose())/2 - remainder) + remainder;

  //cout << "is so bad?" << output * output.adjoint() << endl;
  //cout << "is good?" << is_special_unitary(output) << endl;

  output = output * m;
}

void selfmap_bad( matrix &output, const matrix &m ) {
  
  output = m;
  
  scalar a = output.row(0)[0];
  scalar b = output.row(0)[1];
  scalar c = output.row(0)[2];
  //output.row(0)[2] *= c*c/abs(c)/abs(c);
  output.row(0)[2] *= c/conj(c);

  scalar D = output.row(1)[0];
  scalar E = output.row(1)[1];

  //output.row(1)[0] *= (-conj(c)*conj(c)*conj(c)/abs(c)/abs(c));
  //output.row(1)[1] *= (-conj(c)*conj(c)*conj(c)/abs(c)/abs(c));
  //output.row(1)[2]  = (D*conj(a) + E*conj(b));

  //   output.row(1)[2] = (D*conj(a) + E*conj(b)) / (-conj(c)*conj(c)*conj(c)/abs(c)/abs(c));

  output.row(1)[0] *= -conj(c) * conj(c)/c;
  output.row(1)[1] *= -conj(c) * conj(c)/c;
  output.row(1)[2]  = (D*conj(a) + E*conj(b));

  /*
  output.row(1)[0] *= -abs(c*c*c*c) / c/c/c;
  output.row(1)[1] *= -abs(c*c*c*c) / c/c/c;
  output.row(1)[2]  = (D*conj(a) + E*conj(b));
  */
  
  output.row(1).normalize();  

  /*
  scalar p = output.row(0).dot( output.row(1) );
  if (abs(conj(output.row(0)[2])) > epsilon)
    output.row(1)[2] = -p / conj(output.row(0)[2]);
  else
    output.row(1)[2] = 1;
  */

  //cout << "nrm = " << output.row(1).dot( output.row(1) ) << endl;
  //cout << "dot = " << output.row(0).dot( output.row(1) ) << endl;
  
  output.row(2) = output.row(0).cross( output.row(1) );
  
  //  output = output * m;
  
  //cout << is_special_unitary(output) << endl;
}

int find_index( const matrix &m ) {
  Eigen::Matrix<double,8,8> derivative;

  matrix f;
  selfmap( f, m );

  int i;
  for( i = 0; i < 8; i++ ) {
    matrix e = matrix::Zero();
    if (i == 0) { e.row(0)[0].imag(0.5); e.row(2)[2].imag(-0.5); }
    if (i == 1) e.row(0)[1].real(1);
    if (i == 2) e.row(0)[1].imag(1);
    if (i == 3) e.row(0)[2].real(1);
    if (i == 4) e.row(0)[2].imag(1);
    if (i == 5) { e.row(1)[1].imag(0.5); e.row(2)[2].imag(-0.5); }
    if (i == 6) e.row(1)[2].real(1);
    if (i == 7) e.row(1)[2].imag(1); 

    matrix x = e - e.adjoint();

    //cout << "x = " << endl;
    //cout << x << endl;

    //matrix p = x * 1e-1;
    //matrix ep = matrix::Identity() + p + p*p/2 + p*p*p/6 + p*p*p*p/24 + p*p*p*p*p/120;
    //cout << ep << endl;
    //cout << "good?" << is_special_unitary( ep ) << endl;

    matrix f_plus_hx;
    double h = 1e-5;
    // could do this better but oh well
    matrix exp_hx = matrix::Identity() + h*x + h*x*h*x / 2 + h*x*h*x*h*x / 6 + h*x*h*x*h*x*h*x / 24;
    //cout << "exp(h x) in su? " << is_special_unitary( exp_hx ) << endl;
    selfmap( f_plus_hx, m * exp_hx );

    //matrix shifted = m + h*x;
    //cout << "is su? " << is_special_unitary( shifted ) << endl;

    matrix df = (f_plus_hx - f) / h;
    //shifted = f + 1e-10 * df;
    //cout << "is also su? " << is_special_unitary( shifted ) << endl;

    //cout << "f(m+hx) = " << f_plus_hx << endl;
    //cout << "f(m) = " << f << endl;
    //cout << "f(m) + h df = " << f + h * df<< endl;

    //cout << "f(m) in su?  " << is_special_unitary( f ) << endl;
    //cout << "f(m) + h df in su?  " << is_special_unitary( f + h*df ) << endl;
    //cout << "f(m+hx) in su?  " << is_special_unitary( f_plus_hx ) << endl;
    
    //cout << "is skew-hermitian?" << endl;
    df = f.inverse() * df;
    //cout << df << endl; // in the tangent space at f
    
    derivative.row(i)[0] = (df.row(0)[0].imag() - df.row(2)[2].imag())/2;
    derivative.row(i)[1] = df.row(0)[1].real();
    derivative.row(i)[2] = df.row(0)[1].imag();
    derivative.row(i)[3] = df.row(0)[2].real();
    derivative.row(i)[4] = df.row(0)[2].imag();
    derivative.row(i)[5] = (df.row(1)[1].imag() - df.row(2)[2].imag())/2;
    derivative.row(i)[6] = df.row(1)[2].real();
    derivative.row(i)[7] = df.row(1)[2].imag();
  }

  //cout << "df" << endl << derivative << endl;
  
  double det = derivative.determinant();

  cout << "det = " << det << endl;
  
  if (det < 0)
    return -1;

  if (det > 0)
    return 1;  

  return 0;
}

void derivative_selfmap( matrix &output, matrix &m ) {
  output = 4*m*m*m;
}


std::list<matrix> preimages;

double objective_function(const SU &x, SU &grad, void *my_func_data)
{
  if (!grad.empty()) {
    std::cout << "WARNING: I cannot compute the gradient.\n" << std::endl;
  }

  matrix m;
  vector_to_su( x, m );

  matrix output;
  selfmap( output, m );

  matrix image = *(matrix*)(my_func_data);
  //double error = (output - image).squaredNorm();
  double error = (output - image).lpNorm<2>();

  // trace((m^4 - image) * (m^4 - image).adjoint)
  
  //matrix derivative_output;
  //derivative_selfmap( derivative_output, m );
  //su_to_vector( derivative_output, grad );
  /*
  if (error < 1e-4) return error;
  */

  return error;
  
  double d = 0;
  for (auto const& p : preimages) {
    d += 1.0 / ((p - m).lpNorm<2>());
  }

  if (error < 1e-2) return error;
  
  return d + error;
}


float find_preimage (matrix &initial, matrix &image) {
  nlopt::opt opt(nlopt::LN_BOBYQA, 8 );

  ////////////////////////////////////////////////////////////////
  //// LOWER BOUNDS
  SU lb(8);

  for(int i=0;i < 8; i++ ) lb[i] = 0;
  
  opt.set_lower_bounds(lb);

  ////////////////////////////////////////////////////////////////
  //// UPPER BOUNDS
  SU ub(8);
  for(int i=0;i < 3; i++ ) ub[i] = pi/2;
  for(int i=3;i < 8; i++ ) ub[i] = 2*pi;
  
  opt.set_upper_bounds(ub);  

  opt.set_min_objective(objective_function, &image);
  
  opt.set_ftol_rel(1e-10);
  opt.set_stopval(epsilon);
  opt.set_maxeval(10000);
  
  SU x(8);
  double minf;
  for( int i = 0; i < 3; i++ ) x[i] = random_in_range(0,pi/2);
  for( int i = 3; i < 8; i++ ) x[i] = random_in_range(0,2*pi);

  try {
    nlopt::result result = opt.optimize(x, minf);
  } catch (nlopt::roundoff_limited r) {
    // ignore roundoff problems
    cout << "roundoff error" << endl;
  }
  vector_to_su(x, initial);
  
  return minf;
}

//using Eigen::MatrixXd;

int main(int argc, char** argv)
{
  /*
  int k = 11;
  double theta = random_in_range(0,2*pi);

  cout << "cos (kt) = " << cos(k*theta) << endl;
  cout << "k even, then cos (kt) ?= " << ck(k,sin(theta)*sin(theta)) << endl;
  cout << "k odd, then cos (kt) ?= " << ck(k,sin(theta)*sin(theta))*cos(theta) << endl;

  cout << "sin (kt) = " << sin(k*theta) << endl;
  cout << "k even, then sin (kt) ?= " << sk(k,sin(theta)*sin(theta))*cos(theta)*sin(theta) << endl;
  cout << "k odd, then sin (kt) ?= " << sk(k,sin(theta)*sin(theta))*sin(theta) << endl;

  */  
  int sum_of_indices = 0;
  
  matrix m;
  random_su(m);

  /*
  double farthest = 0;

  for( int i = 0 ; i < 1000 ; i ++ ) {
    matrix n;
    matrix output;
    random_su(n);
    selfmap( output, n );
    double d = (output - n).norm();
    if (d > farthest) {
      farthest = d;
      m = n;
      cout << "farthest = " << farthest << endl;      
    }
  }
  */

  
  /*
  matrix  random_input;
  random_su(random_input);
  selfmap( m, random_input );
  */
  
  cout << "searching for preimages..." << endl;
  
  //for(int i = 0; i<10000;i ++) {
  for(;;) {
    matrix x;
    float dist = find_preimage (x, m);

    matrix output;
    selfmap( output, x );
    double error = (output - m).lpNorm<2>();
    //cout << "error=" << error << endl;
    //cout << "error = " << error << endl;
    if (error > epsilon) {
      //cout << "Bad preimage: " << error << endl;
      continue;
    }
    
    std::list<matrix>::iterator it;
    bool match = false;
    double closest = 10000;
    for (auto const& p : preimages) {
      double d = (p - x).squaredNorm();
      if (d < closest) closest = d;
      
      if (d < 1e-4) {
        match = true;
      }
    }

    if (!match) {
      preimages.push_front( x );
      int index_of_preimage = find_index(x);
      sum_of_indices += index_of_preimage;
      cout << "found preimage with index " << index_of_preimage << " and error " << error << endl;
      cout << "degree ?= " << sum_of_indices  << endl;
      cout << "preimages >= " << preimages.size() << " with closest " << closest << endl;
    } else {
      //ncout << "Already found!"  << endl;
    }


  }

  /*
  for (auto const& p : preimages) {
    cout << p << endl << endl;
    }*/

  
  return 0;
}

