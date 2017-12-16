package neural;

import java.util.ArrayList;

public class Layer {
	
	private ArrayList<Perceptron> nodes; // each layer has a perceptron set
	private int size; // number of nodes
	private LayerType type;
	
	public Layer(int n, LayerType type){
		this.size = n;
		nodes = new ArrayList<>();
		
		for(int i = 0; i < n; i++){
			Perceptron p = new Perceptron();
			nodes.add(p);
		}
		
		this.type = type;

	}
	
	public ArrayList<Perceptron> getNodes() {
		return nodes;
	}

	public int getSize() {
		return size;
	}

	public LayerType getType() {
		return type;
	}
	
}

