
#include <boost/lambda/lambda.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions/normal.hpp>
#include <iostream>
#include <stdio.h> // For printf
#include <stdbool.h> // For boolian
#include <iterator>
#include <algorithm>
#include <iomanip>
#include <map>
#include <random>
#include <cmath>
#include <fstream>
#include <sstream>

struct weights{
	double d1;
	double d2;
};


double Get_StdNormDistr_RandomNumber();
double Evaluate_BonusDerivat_loop(double S_0, const double Sigma, const double r_f, const double Barrier, const double Bonus_Level, const int maturity, const int N_timeSteps );
double Evaluate_BonusDerivat_Rekursiv(double S_0, const double Sigma, const double r_f, const double Barrier, const double Bonus_Level, const int maturity, const int N_timeSteps );

weights Get_weight_d1_and_d2(const double S_0, const double K, const double maturity, const double Sigma, const double r_f, const double CoC_q);
double Get_Put(const double S_0, const double K, const double maturity, const double Sigma, const double r_f, const double CoC_q);
double Get_Call(const double S_0, const double K, const double maturity, const double Sigma, const double r_f, const double CoC_q);
double Get_Down_and_In_Put(const double S_0, const double K, const double Barrier, const double maturity, const double Sigma, const double r_f, const double CoC_q);
double Get_Down_and_Out_Put(const double S_0, const double K, const double Barrier, const double maturity, const double Sigma, const double r_f, const double CoC_q);
double Get_Theory_Value(double Put_down_out_value, double Call_value);
double Cumul_StNormFunc(const double x);

// Global variables

bool TouchBarrier_L = false; // Boolian for checking whether barrier was touched or undershoot
bool TouchBarrier_R = false; // Boolian for checking whether barrier was touched or undershoot
int tmp_step_noX_R = 0;	     // Current time step that is calculated 
std::random_device rd{};     // Random seed from the computer; Syntax is equal to: std::random_device rd;
//std::mt19937 gen{rd()};    // Method with which the random numbers are generated, Therefore, one needs to provide the seed at which to start (here rd or arbitrary number), Syntax equal to std::mt19937 gen = 2;
std::mt19937 gen{2};         // Method with which the random numbers are generated, Therefore, one needs to provide the seed at which to start (here 2), Syntax equal to std::mt19937 gen = 2;
boost::math::normal dist(0.0,1.0); 

int main(){

	//printf("Hello World\n");

	int N_paths 	 ;      	   		// Number of paths that are calculated
	int N_timeSteps;        	 		// Number of time steps that are calculated
	int maturity       = 1;    		// Maturity of the derivative in [a]
	double S_0         = 72.93;   // Initial value of the underlying // https://www.onvista.de/aktien/BASF-Aktie-DE000BASF111

	double Barrier     = 50;      // Barrier of the Bonus certificate // https://www.onvista.de/derivate/bonus-zertifikate/BONUS-ZERTIFIKAT-AUF-BASF-DE000DC3YKS7
	double Bonus_Level = 100;     // Bonuslevel of the Bonus certificate // https://www.onvista.de/derivate/bonus-zertifikate/BONUS-ZERTIFIKAT-AUF-BASF-DE000DC3YKS7
	double K 					 = 0.0;     // Strike price for the Call which is used to rebild the Bonus certificate
	double CoC_q 			 = 0.0;     // Discountfactor for dividends "Cost of carry"
	double r_f         = -0.0055;   // Estimated risk free interest rate
														    // Einjährige deutsche staatsanleihe mit Laufzeit 1a und Ausgabedatum 16/04/19 r_f = -0.550% //https://de.investing.com/rates-bonds/germany-1-year-bond-yield
														    // LIBOR r_f = -0,2113 % 	// https://www.finanzen.net/zinsen/historisch/libor/libor-eur-12-monate
														    // Mittelwert aus Staatsanleihe und LIBOR: r_f = -0.38065%
	double Sigma       = 0.2429;  // Estimated volatility of the underlying BASF11 aktie //https://www.onvista.de/aktien/BASF-Aktie-DE000BASF111
														 		// Standartabweichung geschätzt aus Daten ab 03.02.2010. Aus Daten stetige Renditen ln(S_t1/S_t) berechnet. 
														 		// Varianz über stetige Tagesrenditen berechnet und dann mit 250 tagen multipliziert


	double time 				 ; // Point of time at which 
	double tmp_price     ; // Temporary price loop method 
	double tmp_sum       ; // Sum of the temporary prices recursive methodof all paths calculated
	double mean_price_nd ; // Final price loop method is eqaul to mean value of N_paths results. This one gets not discounted for calculating sigma value.
	double price       	 ; // Final price loop method is eqaul to mean value of N_paths results. This one gets discounted
	double tmp_sigma   	 ; // Sum used for calculating sigma: Sum((S_tmp-S_0)²)
	double sigma       	 ; // Sigma of the final result calculated as Sqrt(Sum((S_tmp-S_0)²)/N_paths)
 
	std::vector<int> Diff_Number_Of_Paths     = {10, 100, 1000};
	std::vector<int> Diff_Number_Of_timeSteps = {10, 100, 1000};

	std::ofstream file_output;
	file_output.open("Prices_and_Sigmas.csv");
	file_output << "N_paths" << "," << "N_timeSteps" << "," << "price" << "," << "sigma" <<"\n";

  for(int pathStyle =0; pathStyle <= Diff_Number_Of_Paths.size()-1; pathStyle++){
  	for(int timeStep_Style =0; timeStep_Style <= Diff_Number_Of_timeSteps.size()-1; timeStep_Style++){

  		N_paths 		= Diff_Number_Of_Paths[pathStyle];
			N_timeSteps = Diff_Number_Of_timeSteps[timeStep_Style];


			tmp_price     				= 0.0;
			tmp_sum       				= 0.0;
			mean_price_nd			    = 0.0;
			price       					= 0.0; 
			tmp_sigma   					= 0.0; 
			sigma       					= 0.0; 

			std::ofstream myfile;
			myfile.open("tmp_Prices_Zertifikate" + std::to_string(N_paths) + "_" + std::to_string(N_timeSteps) + ".csv");
			myfile << 0 << "," << S_0 << "\n";

			for(int path = 1; path <= N_paths; path++){

				time = (double) path/N_paths;

				//____ IMPORTANT______//
				// You can not evaluate the bonus certificate
				// with the loop method and the recursive method
				// at the same time. Results are then inonsistent
				// because random numbers are different. 
				// Always comment one function call !!

				//Use for the Loop method
				tmp_price = Evaluate_BonusDerivat_loop(S_0, Sigma, r_f, Barrier, Bonus_Level, maturity, N_timeSteps );
				tmp_sum += tmp_price;

				//Use for the recursive method
				//tmp_price = Evaluate_BonusDerivat_Rekursiv(S_0, Sigma, r_f, Barrier, Bonus_Level, maturity, N_timeSteps );
				//tmp_sum += tmp_price;
				//tmp_step_noX_R =0;

				
				myfile << time << "," << tmp_price << "\n";

			}//end loop over number of paths
			myfile.close();

			mean_price_nd = tmp_sum/N_paths;
			price 				= tmp_sum/(N_paths )* exp(-r_f * maturity);
			//printf("Price bonus certificate: Loop = %3.10f\n", price);

			 // read file line by line
			std::ifstream file("tmp_Prices_Zertifikate" + std::to_string(N_paths) + "_" + std::to_string(N_timeSteps) + ".csv");
			std::string line;
			double tmp_value_file = 0.0;

			std::size_t a, b;
			
			while(getline(file, line)){
				a = line.find(',', 0);
				b = line.find('\"', a + 1);
				if (b != std::string::npos){
				std::string tmp_string = line.substr(a, 10);
				tmp_value_file = std::stof(tmp_string);
				}
				//a = line.find('', b + 1);
				//tmp_value_file = std::stof(line);
				tmp_sigma += pow((tmp_value_file - mean_price_nd),2);
			}
			file.close();

			sigma = sqrt(tmp_sigma/(N_paths*(N_paths-1)));
			//printf("Sigma deviation of price estimated via MC = %3.12f\n", sigma);

			file_output << N_paths << "," << N_timeSteps << "," << price << "," << sigma <<"\n";
		} // end loop through vector of different number of time steps -> e.g. this loop is just for taking various sizes of time intervalls
	} // end loop through vector of different number of paths -> e.g. this loop is just for taking different numbers of paths
	file_output.close();
	
	double Theory_call_value 						  = Get_Call(S_0, K, maturity, Sigma, r_f, CoC_q);
  double Theory_put_down_and_out_value  = Get_Down_and_Out_Put(S_0, Bonus_Level, Barrier, maturity, Sigma, r_f, CoC_q);
  double Theory_bonus_certificate_value = Get_Theory_Value(Theory_put_down_and_out_value, Theory_call_value);

  printf("Theory value for Bonus certificate = %3.10f\n", Theory_bonus_certificate_value);

  


}//end main function


//_____________________________________________________________________________________________________________________________________________________________________________________________________
//

double Evaluate_BonusDerivat_loop(const double S_0, const double Sigma, const double r_f, const double Barrier, const double Bonus_Level, const int maturity, const int N_timeSteps ){
	
	double delta_t = (double)maturity/N_timeSteps; // time intervall for each evaluation
	double S_tmp = 0;
	double S_tmp_initial = S_0;
	double random_number;
	bool TouchBarrier = false;

	for(int tmp_step_noX = 1; tmp_step_noX <= N_timeSteps; tmp_step_noX++){
		random_number = Get_StdNormDistr_RandomNumber();                                                                               // Get the standard normal distributed random number
		S_tmp = S_tmp_initial * exp( (r_f - (Sigma*Sigma)/2.) * delta_t + Sigma * sqrt(delta_t) * random_number );   // Calculate the price of the underlying at the n-st time step: delta_t* tmp_step_noX
		S_tmp_initial = S_tmp;

		//Check if barrier was touched
		if(S_tmp <= Barrier) TouchBarrier = true;                                        // If barrier was touched set the boolian to kTRUE 
    //printf("Wert St_Loop = %3.5f and number tmpStep = %d, timeStep*delta_t = %3.3f\n",S_tmp,  tmp_step_noX, tmp_step_noX* delta_t);
	}

  //Determine the correct price of the bonus certificate
	if(TouchBarrier || (S_tmp > Bonus_Level) )  return S_tmp;                       // If barrier was (touched or undershoot) or if (the value of the underlying S_t is greater than the Bonuslevel Bonus_Level) than return value of the underlying = Price of Bonus certificate
	else return Bonus_Level;																												// If the value of the underlying has never touched or undershoot the Barrier and if the value of the underlying is smaller than the Bonuslevel than return the Bonuslevel = Price of Bonus certificate
}

//_____________________________________________________________________________________________________________________________________________________________________________________________________
//

//In the recursion the bool is still wrong and is never set to false again after being set to true...
//...not changed as recursion will not be used due to stack overflow problem

//Beware of the stack overflow for too deep recursions!!!
double Evaluate_BonusDerivat_Rekursiv(double S_0, const double Sigma, const double r_f, const double Barrier, const double Bonus_Level, const int maturity, const int N_timeSteps ){
	
	double delta_t = ((double) maturity)/N_timeSteps; // time intervall for each evaluation
  tmp_step_noX_R += 1;

	double random_number = Get_StdNormDistr_RandomNumber();                                                                               // Get the standard normal distributed random number                                                                               // Get the standard normal distributed random number
	double S_tmp = S_0 * exp( (r_f - (Sigma*Sigma)/2.) * delta_t + Sigma * sqrt(delta_t) * random_number );   // Calculate the price of the underlying at the n-st time step: delta_t* tmp_step_noX

	//Check if barrier was touched
	if(S_tmp <= Barrier) TouchBarrier_R = true;                                        // If barrier was touched set the boolian to kTRUE 

	//Check if the end of maturity is reached and determine the correct price of the bonus certificate
	if(tmp_step_noX_R < N_timeSteps){																						// Check if the end of maturity is reached; If not than repeat the procedure; if yes than return the appropriate value
	  //printf("Wert St_Rekursiv = and number tmpStep = %d, timeStep*delta_t = %3.3f\n",  tmp_step_noX_R, tmp_step_noX_R* delta_t);
		return Evaluate_BonusDerivat_Rekursiv(S_tmp, Sigma, r_f, Barrier, Bonus_Level, maturity, N_timeSteps);
	
	}
	else{
		if(TouchBarrier_R || (S_tmp > Bonus_Level) )  return S_tmp;                       // If barrier was (touched or undershoot) or if (the value of the underlying S_t is greater than the Bonuslevel Bonus_Level) than return value of the underlying = Price of Bonus certificate
		else return Bonus_Level;																												// If the value o the underlying has never touched or undershoot the Barrier and if the value of the underlying is smaller than the Bonuslevel than return the Bonuslevel = Price of Bonus certificate
	}

}

//_____________________________________________________________________________________________________________________________________________________________________________________________________
//

double Get_StdNormDistr_RandomNumber(){

  std::normal_distribution<> nd{0,1}; // Define normal distribution with name nd and mean = 0 and sigma =1 which is equal to standard normal distribution; Return type is double
	double random_number = nd(gen);           // Generate random number which is standard normal distributed

	return random_number;
}

//_____________________________________________________________________________________________________________________________________________________________________________________________________

double Get_Theory_Value(double Put_down_out_value, double Call_value){

	double theory_value = Put_down_out_value + Call_value;
	return theory_value;
}

double Get_Down_and_Out_Put(const double S_0, const double K, const double Barrier, const double maturity, const double Sigma, const double r_f, const double CoC_q){

	double P_DI = Get_Down_and_In_Put(S_0, K, Barrier, maturity, Sigma, r_f, CoC_q);
	double put  = Get_Put(S_0, K, maturity, Sigma, r_f, CoC_q);
	double P_DO = put - P_DI;
	return P_DO;

}

//_____________________________________________________________________________________________________________________________________________________________________________________________________
double Get_Down_and_In_Put(const double S_0, const double K, const double Barrier, const double maturity, const double Sigma, const double r_f, const double CoC_q){

	double lambda = (r_f - CoC_q + (Sigma*Sigma)/2.)/(Sigma*Sigma);
	double x_1    = log(S_0/Barrier) / (Sigma * sqrt(maturity)) + lambda * Sigma * sqrt(maturity);
	double y      = log(Barrier*Barrier / (S_0 * K)) / (Sigma*sqrt(maturity)) + lambda * Sigma * sqrt(maturity);
	double y_1    = log(Barrier/S_0) / (Sigma*sqrt(maturity)) + lambda * Sigma * sqrt(maturity);

	struct weights tmp_weight = Get_weight_d1_and_d2(S_0, K, maturity, Sigma, r_f, CoC_q);
	double d1 = tmp_weight.d1;
	double d2 = tmp_weight.d2;

	double value_down_and_in_put = -S_0 * Cumul_StNormFunc(-x_1) * exp(-CoC_q * maturity) 
																 + K * exp(-r_f * maturity) * Cumul_StNormFunc(-x_1 + Sigma * sqrt(maturity))
																 + S_0 * exp(-CoC_q * maturity) * pow(Barrier/S_0, 2*lambda) * (Cumul_StNormFunc(y) - Cumul_StNormFunc(y_1))
																 - K * exp(-r_f * maturity) * pow(Barrier/S_0, 2*lambda -2) * (Cumul_StNormFunc(y - Sigma*sqrt(maturity)) - Cumul_StNormFunc(y_1 - Sigma*sqrt(maturity)));
	return value_down_and_in_put;
}
//_____________________________________________________________________________________________________________________________________________________________________________________________________

double Get_Call(const double S_0, const double K, const double maturity, const double Sigma, const double r_f, const double CoC_q){

	struct weights tmp_weight = Get_weight_d1_and_d2(S_0, K, maturity, Sigma, r_f, CoC_q);
	double d1 = tmp_weight.d1;
	double d2 = tmp_weight.d2;
	double call_value = -100.0;

	if(K == 0.){
		call_value = S_0 * exp(-CoC_q * maturity);
	}
	else{
		call_value = S_0 * exp(-CoC_q * maturity) * Cumul_StNormFunc(d1) - K * exp(-r_f * maturity) * Cumul_StNormFunc(d2); 
  }
	return call_value;
}
//_____________________________________________________________________________________________________________________________________________________________________________________________________

double Get_Put(const double S_0, const double K, const double maturity, const double Sigma, const double r_f, const double CoC_q){

	struct weights tmp_weight = Get_weight_d1_and_d2(S_0, K, maturity, Sigma, r_f, CoC_q);
	double d1 = tmp_weight.d1;
	double d2 = tmp_weight.d2;
	double put_value = -1000.0;

	if(K == 0.){
		put_value = 0.0;
	}
	else{
		put_value = K * exp(-r_f * maturity) * Cumul_StNormFunc(-d2) - S_0 * exp(-CoC_q * maturity) * Cumul_StNormFunc(-d1);
	}

	return put_value;
}
//_____________________________________________________________________________________________________________________________________________________________________________________________________

weights Get_weight_d1_and_d2(const double S_0, const double K, const double maturity, const double Sigma, const double r_f, const double CoC_q){
	// K = strike price

	struct weights tmp_struct;
	tmp_struct.d1 = (log(S_0/K) + (r_f - CoC_q + Sigma*Sigma/2.)/maturity)/(Sigma/sqrt(maturity));
	tmp_struct.d2 = tmp_struct.d1 - Sigma * sqrt(maturity);

	return tmp_struct;
}
//_____________________________________________________________________________________________________________________________________________________________________________________________________

double Cumul_StNormFunc(const double d_x){        // Phi(-∞, x) aka N(x)
    return std::erfc(-d_x/std::sqrt(2))/2;				// https://en.cppreference.com/w/cpp/numeric/math/erfc --> https://en.wikipedia.org/wiki/Error_function#Complementary_error_function
}