import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def build_network(Link_path, Node_path):
    links_df = pd.read_csv(Link_path)
    G = nx.Graph()
    nodes_df = pd.read_csv(Node_path)
    for _, row in nodes_df.iterrows():
        G.add_node(int(row['id']), pos=(row['pos_x'], row['pos_y']))
    for _, row in links_df.iterrows():
        G.add_edge(row['Source'], row['Target'], TravelTime=row['TravelTime'], Capacity=row['Capacity'])
        G.add_edge(row['Target'], row['Source'], TravelTime=row['TravelTime'], Capacity=row['Capacity'])
    nx.set_edge_attributes(G, 0, 'flow_temp')
    nx.set_edge_attributes(G, 0, 'flow_real')
    nx.set_edge_attributes(G, nx.get_edge_attributes(G, "TravelTime"), 'weight')
    return G

def show_network(G):
    pos = nx.get_node_attributes(G, "pos")
    weights = [d['TravelTime'] for u, v, d in G.edges(data=True)]
    colors = [plt.cm.Blues(weight / max(weights)) for weight in weights] 
    widths = [weight / max(weights) for weight in weights]  
    nx.draw_networkx_nodes(G, pos, node_size=200)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges, width=widths, edge_color=colors, arrows=False)
    print("绘制网络图")
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    plt.show()

def draw_network(G):
    pos = nx.get_node_attributes(G, "pos")
    weights = [d['flow_real'] for u, v, d in G.edges(data=True)]
    colors = [plt.cm.Blues(weight / max(weights)) for weight in weights] 
    widths = [weight*10 / max(weights) for weight in weights]  
    nx.draw_networkx_nodes(G, pos, node_size=200)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges, width=widths, edge_color=colors, arrows=False)
    print("绘制网络图")
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    '''edge_labels = {(u, v): f'{d["flow_real"]}' for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)'''
    plt.show()

def BPR(FFT, flow, capacity, alpha=0.15, beta=4.0):
    return FFT * (1 + alpha * (flow / capacity) ** beta)

def all_none_initialize(G, od_df):
    for _, od_data in od_df.iterrows():
        source = od_data["o"]
        target = od_data["d"]
        demand = od_data["demand"]
        shortest_path = nx.shortest_path(G, source=source, target=target, weight="weight")
        for i in range(len(shortest_path) - 1):
            u = shortest_path[i]
            v = shortest_path[i + 1]
            G[u][v]['flow_real'] += demand
    for _, _, data in G.edges(data=True):
        data['weight'] = BPR(data['TravelTime'], data['flow_real'], data['Capacity'])

def all_none_temp(G, od_df):
    nx.set_edge_attributes(G, 0, 'flow_temp')
    for _, od_data in od_df.iterrows():
        source = od_data["o"]
        target = od_data["d"]
        demand = od_data["demand"]
    
        shortest_path = nx.shortest_path(G, source=source, target=target, weight="weight")
    
        for i in range(len(shortest_path) - 1):
            u = shortest_path[i]
            v = shortest_path[i + 1]
        
            G[u][v]['flow_temp'] += demand * 0.1
            '''print("u, v, flow_temp", u, v, G[u][v]['flow_temp'])'''


def update_flow_real(G):
    for _, _, data in G.edges(data=True):
        print(data['flow_real'],data['flow_temp']) 
        data['flow_real'] = data['flow_real'] * 0.9 + data['flow_temp']
        print(data['flow_real'],data['flow_temp'])

        data['weight'] = BPR(data['TravelTime'], data['flow_real'], data['Capacity'])

def main():
    G = build_network("Link.csv", "Node.csv") 
    od_df = pd.read_csv("ODPairs.csv") 
    show_network(G)
    all_none_initialize(G, od_df)  

    print("初始化流量", list(nx.get_edge_attributes(G, 'flow_real').values()))
    epoch = 0 
    while epoch < 100:
        epoch += 1
        all_none_temp(G, od_df)  
        update_flow_real(G)  
        print(epoch, "迭代流量", list(nx.get_edge_attributes(G, 'flow_real').values()))

    draw_network(G)
    print("均衡流量", list(nx.get_edge_attributes(G, 'flow_real').values()))
    print("成功")
    df = nx.to_pandas_edgelist(G)
    df = df[["source", "target", "flow_real"]].sort_values(by=["source", "target"])
    df.to_csv("result.csv", index=False)
 
 
if __name__ == '__main__':
    main()