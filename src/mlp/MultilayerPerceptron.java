package mlp;

import java.text.DecimalFormat;
import java.util.Random;
import java.util.ArrayList;

import neural.Layer;
import neural.LayerType;

public class MultilayerPerceptron {
	
	private Layer layerInput = null;
	private ArrayList<Layer> layerHidden = null;
	private Layer layerOutput = null;
	
	// learning rate parameter
	private double alpha = 0;
	
	private static int MAX_EPOCH = 1000000;
	private static double ERROR = 0.1;
	
	// input and outputs from my training set
	private double matrixInput[][] = null;
	private double matrixOutput[][] = null;
	
	
	public MultilayerPerceptron(double alpha, double matrixInput[][], double matrixOutput[][], int nHiddenLayers, int nNeuronsInputLayer, int nNeuronsHiddenLayer, int nNeuronsOutputLayer){
		
		this.layerInput = new Layer(nNeuronsInputLayer, LayerType.INPUT);

		this.layerHidden = new ArrayList<>();
		for(int i = 0; i < nHiddenLayers; i++){
			Layer l = new Layer(nNeuronsHiddenLayer, LayerType.HIDDEN);
			this.layerHidden.add(l);
		}

		this.layerOutput = new Layer(nNeuronsOutputLayer, LayerType.OUTPUT);
		
		this.matrixInput = matrixInput;
		this.matrixOutput = matrixOutput;
		this.alpha = alpha;
	}
	
	// new input
	private void generateInput(int index){
		for(int i = 0; i < this.layerInput.getSize(); i++){
			double xInput[] = {(double)this.matrixInput[index][i]};
			// assigns a new input for my layerInput
			this.layerInput.getNodes().get(i).setX(xInput);
		}
	}
	
	private void generateWeights(){
		// generate random weights and thresholds
		
		Random r = new Random();
		
		// weights		
		for(int j = 0; j < this.layerHidden.get(0).getNodes().size(); j++){
			double w1[] = new double [this.matrixInput[0].length];
			for(int k = 0; k < this.matrixInput[0].length; k++){
				w1[k] = r.nextGaussian();
			}
			this.layerHidden.get(0).getNodes().get(j).setW(w1);
			this.layerHidden.get(0).getNodes().get(j).setTeta(r.nextGaussian());
		}
		
		for(int i = 1; i < this.layerHidden.size(); i++){
			for(int j = 0; j < this.layerHidden.get(i).getNodes().size(); j++){
				double w1[] = new double [this.layerHidden.get(0).getNodes().size()];
				for(int k = 0; k < this.layerHidden.get(0).getNodes().size(); k++){
					w1[k] = r.nextGaussian();
				}
				this.layerHidden.get(i).getNodes().get(j).setW(w1);
				this.layerHidden.get(i).getNodes().get(j).setTeta(r.nextGaussian());;
			}
		}
		
		int lastHidden = this.layerHidden.size() - 1;
		
		for(int j = 0; j < this.layerOutput.getNodes().size(); j++){
			double w1[] = new double [this.layerHidden.get(lastHidden).getNodes().size()];
			for(int k = 0; k < this.layerHidden.get(lastHidden).getNodes().size(); k++){
				w1[k] = r.nextGaussian();
			}
			this.layerOutput.getNodes().get(j).setW(w1);
			this.layerOutput.getNodes().get(j).setTeta(r.nextGaussian());;
		}
		
		/*double w1[] = {0.5, 0.4};
		this.layerHidden.get(0).getNodes().get(0).setW(w1);
		double w2[] = {0.9, 1.0};
		this.layerHidden.get(0).getNodes().get(1).setW(w2);
		double w3[] = {-1.2, 1.1};
		this.layerOutput.getNodes().get(0).setW(w3);
		
		this.layerHidden.get(0).getNodes().get(0).setTeta(0.8);
		this.layerHidden.get(0).getNodes().get(1).setTeta(-0.1);
		this.layerOutput.getNodes().get(0).setTeta(0.3);*/
	}
	
	public void trainMLP(){
		
		generateWeights();
		
		int epoch = 0;
		double error[][] = new double[this.matrixInput.length][this.matrixOutput[0].length];
		double sumSquadError = 1000;		
		
		//&& epoch < MultilayerPerceptron.MAX_EPOCH
		while(sumSquadError > MultilayerPerceptron.ERROR && epoch < MultilayerPerceptron.MAX_EPOCH){
			
			for(int i = 0; i < this.matrixInput[0].length; i++){
				System.out.print("x" + (i + 1) + "\t");
			}
			
			for(int i = 0; i < this.matrixOutput[0].length; i++){
				System.out.print("y" + (i + 1) + "\t");
			}
			
			for(int i = 0; i < this.matrixOutput[0].length; i++){
				System.out.print("output" + (i + 1) + "\t\t");
			}
			
			for(int i = 0; i < this.matrixOutput[0].length; i++){
				System.out.print("e" + (i + 1) + "\t\t");
			}
			
			System.out.println();
			
			for(int i = 0; i < this.matrixInput.length; i++){
				
				generateInput(i);
				
				// calculate the actual outputs of the neurons in the hidden layer
				sigmoidHidden(this.matrixInput[i]);
				
				// calculate the actual output of the neuron in the output layer
				sigmoidOutput();
				
				// error obtained
				for(int j = 0; j < this.layerOutput.getNodes().size(); j++){
					error[i][j] = this.matrixOutput[i][j] - this.layerOutput.getNodes().get(j).getY();
				}
				// calculate the error gradient for the neurons in the output layer
				for(int j = 0; j < this.layerOutput.getNodes().size(); j++){
					gradientSigmoidOutput(error[i][j], j);
					weightCorrectionsOutput(j);
				}
				
				gradientSigmoidHidden();
				weightCorrectionsHidden(i);
					
				printIteraction(i);
				
			}
			
			// calculate the sum of squared errors
			sumSquadError = 0;
			for(int j = 0; j < error.length; j++){
				for(int k = 0; k < error[j].length; k++){
					sumSquadError += Math.pow(error[j][k], 2);
				}
			}
			sumSquadError = (sumSquadError / (double)error.length);
			
			epoch++;
			
			printErrorEpoch(sumSquadError);
		}
		
		System.out.println("Total epoch = " + epoch);
		
	}
	
	public void testMLP(double matrixInputTest[][], double matrixOutputTest[][]){
		
		for(int i = 0; i < matrixOutputTest[0].length; i++){
			System.out.print("y" + (i + 1) + "\t");
		}
		
		for(int i = 0; i < matrixOutputTest[0].length; i++){
			System.out.print("output" + (i + 1) + "\t\t");
		}
		
		System.out.println();
		
		for(int i = 0; i < matrixInputTest.length; i++){
			// calculate the actual outputs of the neurons in the hidden layer
			sigmoidHidden(matrixInputTest[i]);
			
			// calculate the actual output of the neuron in the output layer
			sigmoidOutput();
			
			printTestResult(matrixOutputTest[i]);
		}
		
	}
	
	/* SIGMOID */
	
	private void sigmoidHidden(double input[]){
		double X, Y;
		
		X = 0;
		Y = 0;
		
		for(int i = 0; i < this.layerHidden.size(); i++){
			
			// se eh a primeira camada escondida
			if(i == 0){
				for(int j = 0; j < this.layerHidden.get(i).getNodes().size(); j++){
					for(int k = 0; k < input.length; k++){
						X += ((input[k] * this.layerHidden.get(i).getNodes().get(j).getW()[k]));
					}
					
					X -= (this.layerHidden.get(i).getNodes().get(j).getTeta());
					
					Y = (1 / (1 + Math.exp(-X)));
					
					// saida do no j da camada i
					this.layerHidden.get(i).getNodes().get(j).setY(Y);
				}
			}else{
				for(int j = 0; j < this.layerHidden.get(i).getNodes().size(); j++){
					for(int k = 0; k < this.layerHidden.get(i-1).getNodes().size(); k++){
						X += ((this.layerHidden.get(i-1).getNodes().get(k).getY() * this.layerHidden.get(i).getNodes().get(j).getW()[k]));
					}
					
					X -= (this.layerHidden.get(i).getNodes().get(j).getTeta());
					
					Y = (1 / (1 + Math.exp(-X)));
					
					// saida do no j da camada i
					this.layerHidden.get(i).getNodes().get(j).setY(Y);
				}
			}
		}
		
	}
	
	private void sigmoidOutput(){
		double X, Y;
				
		X = 0;
		Y = 0;
		
		for(int i = 0; i < this.layerOutput.getNodes().size(); i++){
			int lastHidden = this.layerHidden.size() - 1;
			for(int j = 0; j < this.layerHidden.get(lastHidden).getNodes().size(); j++){
				X += ((this.layerHidden.get(lastHidden).getNodes().get(j).getY() * this.layerOutput.getNodes().get(i).getW()[j]));
			}
			
			X -= (this.layerOutput.getNodes().get(i).getTeta());
			
			Y = (1 / (1 + Math.exp(-X)));
			
			this.layerOutput.getNodes().get(i).setY(Y);
		}
	}
	
	private double gradientSigmoidOutput(double error, int indexNode){
		double gradient = 0;
		double y = this.layerOutput.getNodes().get(indexNode).getY();
		
		// derivative of the activation function multiplied by the error at the neuron output
		gradient = y * (1 - y) * error;
		
		this.layerOutput.getNodes().get(indexNode).setGradient(gradient);
		
		return gradient;
	}
	
	private void gradientSigmoidHidden(){
		double gradient = 0;
		double gradientWeight = 0;
		double y = 0;
		
		for(int i = this.layerHidden.size() - 1; i >= 0 ; i--){
			for(int j = 0; j < this.layerHidden.get(i).getNodes().size(); j++){
				
				// se eh ultima camada
				if(i == this.layerHidden.size() - 1){
					
					y = 0;
					y = this.layerHidden.get(i).getNodes().get(j).getY();
					
					gradient = y * (1 - y);
					
					gradientWeight = 0;
					for(int k = 0; k < this.layerOutput.getNodes().size(); k++){
						gradientWeight *= this.layerOutput.getNodes().get(k).getW()[j] * this.layerOutput.getNodes().get(k).getGradient();
					}
					
					gradient *= gradientWeight;
					
					this.layerHidden.get(i).getNodes().get(j).setGradient(gradient);
				
				}else{
					
					Layer nextLayer = this.layerHidden.get(i + 1);
					
					y = 0;
					y = this.layerHidden.get(i).getNodes().get(j).getY();
					
					gradient = y * (1 - y);
					
					gradientWeight = 0;
					for(int k = 0; k < nextLayer.getNodes().size(); k++){
						gradientWeight *= nextLayer.getNodes().get(k).getW()[j] * nextLayer.getNodes().get(k).getGradient();
					}
					
					gradient *= gradientWeight;
					
					this.layerHidden.get(i).getNodes().get(j).setGradient(gradient);
				}
			}
		}
	}
		
	/* WEIGHT CORRECTIONS */
	
	private void weightCorrectionsOutput(int indexNode){
		double deltaW[] = new double[this.layerOutput.getNodes().get(indexNode).getW().length];
		double deltaTeta = 0;
		
		for(int j = 0; j < this.layerHidden.get(this.layerHidden.size() - 1).getNodes().size(); j++){
			deltaW[j] = this.alpha * this.layerHidden.get(this.layerHidden.size() - 1).getNodes().get(j).getY() * this.layerOutput.getNodes().get(indexNode).getGradient();
		}
		
		deltaTeta = alpha * (-1) * this.layerOutput.getNodes().get(indexNode).getGradient();
		
		for(int j = 0; j < deltaW.length; j++){
			this.layerOutput.getNodes().get(indexNode).getW()[j] = this.layerOutput.getNodes().get(indexNode).getW()[j] + deltaW[j];
		}
		
		this.layerOutput.getNodes().get(indexNode).setTeta(this.layerOutput.getNodes().get(indexNode).getTeta() + deltaTeta); 
	}

	private void weightCorrectionsHidden(int indexInput){
		double deltaW[];
		double deltaTeta = 0;
		
		for(int i = 0; i < this.layerHidden.size(); i++){
			for(int j = 0; j < this.layerHidden.get(i).getNodes().size(); j++){
				
				if(i == 0){
					
					deltaW = new double[this.layerInput.getNodes().size()];
					for(int k = 0; k < this.layerInput.getNodes().size(); k++){
						deltaW[k] = this.alpha * this.matrixInput[indexInput][k] * this.layerHidden.get(i).getNodes().get(j).getGradient();
					}
					
					for(int k = 0; k < this.layerInput.getNodes().size(); k++){
						deltaW[k] += this.layerHidden.get(i).getNodes().get(j).getW()[k];
					}
					
					this.layerHidden.get(i).getNodes().get(j).setW(deltaW);
					
					deltaTeta = this.alpha * (-1) * this.layerHidden.get(i).getNodes().get(j).getGradient();
					
					deltaTeta += this.layerHidden.get(i).getNodes().get(j).getTeta();
					
					this.layerHidden.get(i).getNodes().get(j).setTeta(deltaTeta);
					
				}else{
					
					deltaW = new double[this.layerHidden.get(i - 1).getNodes().size()];
					for(int k = 0; k < this.layerHidden.get(i - 1).getNodes().size(); k++){
						deltaW[k] = this.alpha * this.layerHidden.get(i - 1).getNodes().get(k).getY() * this.layerHidden.get(i).getNodes().get(j).getGradient();
					}
					
					for(int k = 0; k < this.layerHidden.get(i - 1).getNodes().size(); k++){
						deltaW[k] += this.layerHidden.get(i).getNodes().get(j).getW()[k];
					}
					
					this.layerHidden.get(i).getNodes().get(j).setW(deltaW);
					
					deltaTeta = this.alpha * (-1) * this.layerHidden.get(i).getNodes().get(j).getGradient();
					
					deltaTeta += this.layerHidden.get(i).getNodes().get(j).getTeta();
					
					this.layerHidden.get(i).getNodes().get(j).setTeta(deltaTeta);
				}
			}
		}
	}
	
	private void printIteraction(int index){
		
		DecimalFormat x = new DecimalFormat("0.####");
		String strX;
		for(int i = 0; i < this.matrixInput[index].length; i++){
			strX = x.format(matrixInput[index][i]);
			System.out.print(strX + "\t");
		}
		
		DecimalFormat y = new DecimalFormat("0.####");
		String strY;
		for(int i = 0; i < this.matrixOutput[index].length; i++){
			strY = y.format(this.matrixOutput[index][i]);
			System.out.print(strY + "\t");
		}
		
		DecimalFormat output = new DecimalFormat("0.####");
		String strOut;
		for(int i = 0; i < this.layerOutput.getNodes().size(); i++){
			strOut = output.format(this.layerOutput.getNodes().get(i).getY());
			System.out.print(strOut + "\t\t");
		}
		
		DecimalFormat e = new DecimalFormat("0.####");
		String strE;
		for(int i = 0; i < this.matrixOutput[index].length; i++){
			strE = e.format((this.matrixOutput[index][i] - this.layerOutput.getNodes().get(i).getY()));
			System.out.print(strE + "\t\t");
		}
		
		System.out.println();
		
	}
	
	private void printErrorEpoch(double sumSquadError){
		DecimalFormat sumSquad = new DecimalFormat("0.####");
		String strSumSquad;
		
		strSumSquad = sumSquad.format(sumSquadError);
		
		System.out.println("Sum squared of errors = " + strSumSquad + "\n\n");
	}
	
	private void printTestResult(double vectorOutput[]){
		DecimalFormat y = new DecimalFormat("0.####");
		String strY;
		for(int i = 0; i < vectorOutput.length; i++){
			strY = y.format(vectorOutput[i]);
			System.out.print(strY + "\t");
		}

		DecimalFormat output = new DecimalFormat("0.####");
		String strOut;
		for(int i = 0; i < this.layerOutput.getNodes().size(); i++){
			strOut = output.format(this.layerOutput.getNodes().get(i).getY());
			System.out.print(strOut + "\t\t");
		}
		
		System.out.println();
	}
	
}
