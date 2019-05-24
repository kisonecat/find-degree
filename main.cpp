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
typedef std::complex<double> scalar;
typedef Eigen::Matrix3cd matrix;

const double pi = 3.14159265358979323846;

std::random_device rr; // Seed with a real random value, if available
std::default_random_engine randgen(rr());

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

// From J. B. Bronzan "Parametrization of SU(3)." Physical Review D, 38(6), 1994.
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

  m.row(2) = m.row(0).cross( m.row(1) );
}

double random_su( matrix &m ) {
  SU x(8);
  for( int i = 0; i < 3; i++ ) x[i] = random_in_range(0,pi/2);
  for( int i = 3; i < 8; i++ ) x[i] = random_in_range(0,2*pi);
  vector_to_su( x, m );
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

  output = output * m;
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

    matrix f_plus_hx;
    double h = 1e-5;

    // could do this better but oh well
    matrix exp_hx = matrix::Identity() + h*x + h*x*h*x / 2 + h*x*h*x*h*x / 6 + h*x*h*x*h*x*h*x / 24;
    selfmap( f_plus_hx, m * exp_hx );

    matrix df = (f_plus_hx - f) / h;

    df = f.inverse() * df;
    
    derivative.row(i)[0] = (df.row(0)[0].imag() - df.row(2)[2].imag())/2;
    derivative.row(i)[1] = df.row(0)[1].real();
    derivative.row(i)[2] = df.row(0)[1].imag();
    derivative.row(i)[3] = df.row(0)[2].real();
    derivative.row(i)[4] = df.row(0)[2].imag();
    derivative.row(i)[5] = (df.row(1)[1].imag() - df.row(2)[2].imag())/2;
    derivative.row(i)[6] = df.row(1)[2].real();
    derivative.row(i)[7] = df.row(1)[2].imag();
  }

  double det = derivative.determinant();

  cout << "det = " << det << endl;
  
  if (det < 0)
    return -1;

  if (det > 0)
    return 1;  

  return 0;
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
  double error = (output - image).lpNorm<2>();

  return error;
}

float find_preimage (matrix &initial, matrix &image) {
  nlopt::opt opt(nlopt::LN_BOBYQA, 8 );

  SU lb(8); // lower bounds
  for(int i=0;i < 8; i++ ) lb[i] = 0;
  opt.set_lower_bounds(lb);

  SU ub(8); // upper bounds
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
    cout << "roundoff error" << endl;
  }
  
  vector_to_su(x, initial);
  
  return minf;
}


int main(int argc, char** argv)
{
  int sum_of_indices = 0;
  
  matrix m;
  random_su(m);

  cout << "searching for preimages..." << endl;
  
  for(;;) {
    matrix x;
    float dist = find_preimage (x, m);

    matrix output;
    selfmap( output, x );
    double error = (output - m).lpNorm<2>();
    if (error > epsilon) {
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
    }
  }

  return 0;
}

