package neural;

public class Perceptron {
	
	private double x[]; // input set
	private double y; // output
	private double w[]; // weight set
	private double teta; // threshold
	private double gradient; // gradient
	
	public Perceptron(){
		this.x = null;
		this.y = 0;
		this.w = null;
		this.teta = 0;
		this.gradient = 0;
	}

	public double[] getX() {
		return x;
	}

	public void setX(double[] x) {
		this.x = x;
	}

	public double getY() {
		return y;
	}

	public void setY(double y) {
		this.y = y;
	}

	public double[] getW() {
		return w;
	}

	public void setW(double[] w) {
		this.w = w;
	}

	public double getTeta() {
		return teta;
	}

	public void setTeta(double teta) {
		this.teta = teta;
	}

	public double getGradient() {
		return gradient;
	}

	public void setGradient(double gradient) {
		this.gradient = gradient;
	}
	
}
