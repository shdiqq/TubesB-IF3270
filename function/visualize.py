import os
import sys
import networkx as nx
import matplotlib.pyplot as plt

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', 'class')
sys.path.append(mymodule_dir)

from mbgd import MiniBatchGradientDescent

def visualize(mbgd : MiniBatchGradientDescent):
  mbgd.information()
  print()
  ### INITIALIZATION
  G = nx.Graph()
  pos = {}
  labels = {}
  counter = 1
  posCounter = 1
  inputNode = []
  hiddenNode = []
  outputNode = []
  biasNode = []
    
  #### 1.NODES ADJUSTMENT    
  # 1. Nodes for input layer
  # + the input layer bias
  G.add_node(counter, label="bias")    
  pos[counter] = (0, posCounter)
  labels[counter] = "1"
  biasNode.append(counter)
  posCounter += 1
  counter += 1
  # + input nodes
  for i in range(mbgd.input_size):
    G.add_node(counter)
    pos[counter] = (0, posCounter)
    labels[counter] = f"x{i+1}"
    inputNode.append(counter)
    posCounter += 1
    counter += 1

  posCounter = 1
  # 2. Nodes for hidden layer, if exist
  if (mbgd.n_hidden_layer > 0):
    for i in range(mbgd.n_hidden_layer):            
      # + the hidden layer bias
      G.add_node(counter, label="bias")    
      pos[counter] = (1+i, posCounter)
      labels[counter] = "1"
      biasNode.append(counter)
      posCounter += 1
      counter += 1
      # + hidden nodes
      for j in range(mbgd.hidden_layer[i].n_neuron):
        G.add_node(counter)
        pos[counter] = (1+i, posCounter)
        labels[counter] = f"h{i+1}{j+1}"
        hiddenNode.append(counter)
        posCounter += 1
        counter += 1
      plt.annotate(f"{mbgd.hidden_layer[i].activation_function.__name__}", xy=(1+i, posCounter-1), xytext=(1+i, posCounter-1+0.15), ha='center', fontsize=9, fontweight='bold')
            
  posCounter = 1
  # 3. Nodes for output layer
  # + output nodes
  for i in range(len(mbgd.target[0])):
    G.add_node(counter)
    pos[counter] = (1+mbgd.n_hidden_layer, posCounter)
    labels[counter] = f"y{i+1}"
    # labels[counter] = f"y{i+1}\n{mbgd.output_layer.:.2f}"
    outputNode.append(counter)
    posCounter += 1
    counter += 1
  plt.annotate(f"{mbgd.output_layer.activation_function.__name__}", xy=(1+mbgd.n_hidden_layer, posCounter-1), xytext=(1+mbgd.n_hidden_layer, posCounter-1+0.15), ha='center', fontsize=9, fontweight='bold')
    
  # nodes style
  options = {"edgecolors": "tab:gray", "node_size": 900, "alpha": 1}
  nx.draw_networkx_nodes(G, pos, nodelist=biasNode, node_color="tab:grey", **options)
  nx.draw_networkx_nodes(G, pos, nodelist=inputNode, node_color="tab:red", **options)
  nx.draw_networkx_nodes(G, pos, nodelist=hiddenNode, node_color="tab:blue", **options)
  nx.draw_networkx_nodes(G, pos, nodelist=outputNode, node_color="tab:green", **options)
    
  #### 2.EDGES ADJUSTMENT
  edgeMap = []
  edge_labels = {}
  # 1. Edges for input layer to output layer (hidden is not exist)
  if (mbgd.n_hidden_layer == 0):
    # connect input to output layer
    for i in range(mbgd.input_size+1):
      for j in range(len(mbgd.target[0])):
        G.add_edge(i+1, mbgd.input_size+2+j)
        edgeMap.append((i+1, mbgd.input_size+2+j))
        if (i == 0):
          edge_labels[(i+1, mbgd.input_size+2+j)] = f"{mbgd.output_layer.bias[j]:.2f}"
        else:
          edge_labels[(i+1, mbgd.input_size+2+j)] = f"{mbgd.output_layer.weights[i-1][j]:.2f}"
  # 2. Edges for input layer to hidden layer if exist
  if (mbgd.n_hidden_layer > 0):
    # Connect input to hidden layer
    for i in range(mbgd.input_size+1):
      for j in range(mbgd.hidden_layer[0].n_neuron):
        G.add_edge(i+1, mbgd.input_size+2+j+1)
        edgeMap.append((i+1, mbgd.input_size+2+j+1))
        if (i == 0):
          edge_labels[(i+1, mbgd.input_size+2+j+1)] = f"{mbgd.hidden_layer[0].bias[j]:.2f}"
        else:
          edge_labels[(i+1, mbgd.input_size+2+j+1)] = f"{mbgd.hidden_layer[0].weights[i-1][j]:.2f}"   
    # Connect hidden layer to other hidden layer
    for i in range(mbgd.n_hidden_layer-1):
      for j in range(mbgd.hidden_layer[i].n_neuron+1):
        G.add_edge(mbgd.input_size+2+mbgd.hidden_layer[i].n_neuron+j, mbgd.input_size+2+mbgd.hidden_layer[i].n_neuron+mbgd.hidden_layer[i+1].n_neuron+j+1)
        edgeMap.append((mbgd.input_size+2+mbgd.hidden_layer[i].n_neuron+j, mbgd.input_size+2+mbgd.hidden_layer[i].n_neuron+mbgd.hidden_layer[i+1].n_neuron+j+1))
        if (i == 0):
          edge_labels[(mbgd.input_size+2+mbgd.hidden_layer[i].n_neuron+j, mbgd.input_size+2+mbgd.hidden_layer[i].n_neuron+mbgd.hidden_layer[i+1].n_neuron+j+1)] = f"{mbgd.hidden_layer[i+1].bias[j]:.2f}"
        else:
          edge_labels[(mbgd.input_size+2+mbgd.hidden_layer[i].n_neuron+j, mbgd.input_size+2+mbgd.hidden_layer[i].n_neuron+mbgd.hidden_layer[i+1].n_neuron+j+1)] = f"{mbgd.hidden_layer[i+1].weights[i-1][j]:.2f}"
    # Connect last hidden layer to output layer
    for i in range(mbgd.hidden_layer[mbgd.n_hidden_layer-1].n_neuron+1):
      for j in range(len(mbgd.target[0])):
        totalOthersNeuron = 0
        for k in range(mbgd.n_hidden_layer-1):
          totalOthersNeuron += mbgd.hidden_layer[k].n_neuron
        G.add_edge(mbgd.input_size+2+totalOthersNeuron+i, mbgd.input_size+2+totalOthersNeuron+mbgd.hidden_layer[mbgd.n_hidden_layer-1].n_neuron+1+j)
        edgeMap.append((mbgd.input_size+2+totalOthersNeuron+i, mbgd.input_size+2+totalOthersNeuron+mbgd.hidden_layer[mbgd.n_hidden_layer-1].n_neuron+1+j))
        if (i == 0):
          edge_labels[(mbgd.input_size+2+totalOthersNeuron+i, mbgd.input_size+2+totalOthersNeuron+mbgd.hidden_layer[mbgd.n_hidden_layer-1].n_neuron+1+j)] = f"{mbgd.output_layer.bias[j]:.2f}"
        else:
          edge_labels[(mbgd.input_size+2+totalOthersNeuron+i, mbgd.input_size+2+totalOthersNeuron+mbgd.hidden_layer[mbgd.n_hidden_layer-1].n_neuron+1+j)] = f"{mbgd.output_layer.weights[i-1][j]:.2f}"
                
  # edges style
  edgeOptions = {"width": 2, "alpha": 0.7}
  nx.draw_networkx_edges(G, pos, edgelist=edgeMap, **edgeOptions)
    
  ### 3.LABELS ADJUSTMENT
  # labels
  nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight="bold", font_color="whitesmoke")
  nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10, label_pos=0.25, font_weight="bold", font_color="tab:gray")

  plt.tight_layout()
  plt.axis("off")
  plt.show()