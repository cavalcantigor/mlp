package main;

import mlp.MultilayerPerceptron;

public class Main {
	
	static final double TAXA_APRENDIZADO = 0.1;
	static final int CAMADAS_ESCONDIDAS = 2;
	static final int NEURONIOS_CAMADA_ESCONDIDA = 6;
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		// training set
		/*double xI[][] = {
					 {1,1},
					 {0,1},
					 {1,0},
					 {1,1}};
	    */
		// desired output for my training set
		
		//double yD[][] = {{0}, {1}, {1}, {0}};
		
		// training set
		double xI[][] = {
						 {7.1,3.0,5.9,2.1},
						 {4.9,3.0,1.4,0.2},
						 {4.7,3.2,1.3,0.2},
						 {7.0,3.2,4.7,1.4},
						 {5.1,3.5,1.4,0.2},
						 {6.4,3.2,4.5,1.5},
						 {6.9,3.1,4.9,1.5},
						 {6.3,3.3,6.0,2.5},
						 {5.8,2.7,5.1,1.9}
					 	};
	
		// desired output for my training set
		double yD[][] = {
						 {1, 0},
						 {0, 0},
						 {0, 0},
						 {0, 1},
						 {0, 0},
						 {0, 1},
						 {0, 1},
						 {1, 0},
						 {1, 0}
						};
		
		double matrixInputTest[][] = {
									  {4.8,3.0,1.4,0.1},
									  {4.3,3.0,1.1,0.1},
									  {4.9,2.4,3.3,1.0},
									  {6.6,2.9,4.6,1.3},
									  {7.7,2.8,6.7,2.0},
									  {6.3,2.7,4.9,1.8}
									 };
		
		double matrixOutputTest[][] = {
										{0, 0},
										{0, 0},
										{0, 1},
										{0, 1},
										{1, 0},
										{1, 0}
									  };
		
		MultilayerPerceptron mlp = new MultilayerPerceptron(TAXA_APRENDIZADO, xI, yD, CAMADAS_ESCONDIDAS, xI[0].length, NEURONIOS_CAMADA_ESCONDIDA, yD[0].length);
		
		mlp.trainMLP();
		
		mlp.testMLP(matrixInputTest, matrixOutputTest);
	}

}
