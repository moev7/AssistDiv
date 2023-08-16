from speech_utils import speak
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt

def describe_relationship(selected_obj, detected_objects):
    relationships = []
    for obj in detected_objects:
        if obj is not selected_obj:
            distance_diff = abs(obj['distance'] - selected_obj['distance'])
            if distance_diff <= 1:
                rel = ""
                print("DISTANCE DIFF " + str(distance_diff))
                if obj['box'][1] + obj['box'][3] < selected_obj['box'][1]:
                    rel += "above and "
                elif obj['box'][1] > selected_obj['box'][1] + selected_obj['box'][3]:
                    rel += "below and "
                if obj['centroid'][0] < selected_obj['centroid'][0] - selected_obj['box'][2] * 0.5:
                    rel += "on the left side of "
                elif obj['centroid'][0] > selected_obj['centroid'][0] + selected_obj['box'][2] * 0.5:
                    rel += "on the right side of "

                # Check for object overlap before describing front and behind relationships
                overlap_x = obj['box'][0] + obj['box'][2] > selected_obj['box'][0] and obj['box'][0] < selected_obj['box'][0] + selected_obj['box'][2]
                overlap_y = obj['box'][1] + obj['box'][3] > selected_obj['box'][1] and obj['box'][1] < selected_obj['box'][1] + selected_obj['box'][3]
                if overlap_x and overlap_y:
                    if obj['distance'] < selected_obj['distance'] - 0.3:
                        rel = "in front of " + rel
                    elif obj['distance'] > selected_obj['distance'] + 0.3:
                        rel = "behind " + rel

                if not rel:
                    rel = "around "
                relationships.append((obj, rel))

    if not relationships:
        print(f"No other objects detected around the {selected_obj['name']}.")
    else:
        for obj, rel in relationships:
            print("SELECTED OBJEC RELATIONSHIPS: ")
            print(f"{obj['name']} is {rel}the {selected_obj['name']}.")
            speak(f"{obj['name']} is {rel}the {selected_obj['name']}.")
    return relationships

def describe_all_relationships(detected_objects):
    relationships = []
    for i, obj1 in enumerate(detected_objects):
        for j, obj2 in enumerate(detected_objects):
            if i != j:
                rel = ""

                if obj1['box'][1] + obj1['box'][3] < obj2['box'][1]:
                    rel += "above and "
                elif obj1['box'][1] > obj2['box'][1] + obj2['box'][3]:
                    rel += "below and "

                if obj1['centroid'][0] < obj2['centroid'][0] - obj2['box'][2] * 0.5:
                    rel += "on the left side of "
                elif obj1['centroid'][0] > obj2['centroid'][0] + obj2['box'][2] * 0.5:
                    rel += "on the right side of "


                # Check for object overlap before describing front and behind relationships
                if obj1['box'][0] + obj1['box'][2] > obj2['box'][0] and obj1['box'][0] < obj2['box'][0] + obj2['box'][2] and \
                   obj1['box'][1] + obj1['box'][3] > obj2['box'][1] and obj1['box'][1] < obj2['box'][1] + obj2['box'][3]:
                    if obj1['distance'] < obj2['distance'] - 0.3:
                        rel = "in front of " + rel
                    elif obj1['distance'] > obj2['distance'] + 0.3:
                        rel = "behind " + rel

                if not rel:
                    rel = "around "
                relationships.append((obj1, obj2, rel))

    if not relationships:
        print("No relationships detected between objects.")
    else:
        for obj1, obj2, rel in relationships:
            print("ALL RELATIONSHIPS: ")
            print(f"{obj1['name']} is {rel}the {obj2['name']}.")
            # speak(f"{obj1['name']} is {rel}the {obj2['name']}.")

    return relationships




'''
def generate_scene_graph(selected_obj, relationships):
    G = nx.DiGraph()

    G.add_node(selected_obj['name'])

    for obj, rel in relationships:
        G.add_node(obj['name'])
        G.add_edge(obj['name'], selected_obj['name'], label=rel)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, node_color='lightblue', with_labels=True, node_size=3000, font_size=12)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)
    plt.show()
'''

def generate_scene_graph(relationships):
    G = nx.DiGraph()
    
    # Add nodes to the graph
    for obj1, obj2, rel in relationships:
        G.add_node(obj1['name'])
        G.add_node(obj2['name'])

    # Add edges and their relationship attributes to the graph
    for obj1, obj2, rel in relationships:
        G.add_edge(obj1['name'], obj2['name'], relationship=rel)

    return G

  

def plot_scene_graph(G):
    # Use Pydot and Graphviz to compute the layout
    pos = graphviz_layout(G, prog='dot')

    # Draw the graph using NetworkX and Matplotlib
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrowsize=20)

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'relationship')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    # Show the plot
    plt.axis('off')
    plt.show()

