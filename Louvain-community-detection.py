import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import community as community_louvain
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

# 1. 加载并归一化数据
df = pd.read_csv("rings.csv", index_col=0)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df.values)
names = df.index.tolist()

# 2. 计算距离矩阵
dist_matrix = euclidean_distances(X_scaled)
ring_features = ['3-MRs', '4-MRs', '5-MRs', '6-MRs']
threshold = 0.4  # 只关注阈值为0.4
modularities = []

# 3. 创建图并进行Louvain聚类
G = nx.Graph()
G.add_nodes_from(names)

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        dist = dist_matrix[i][j]
        if dist < threshold:
            similarity = 1 - dist
            G.add_edge(names[i], names[j], weight=similarity)

# Louvain 聚类
partition = community_louvain.best_partition(G)
modularity_Q = community_louvain.modularity(partition, G, weight='weight')
modularities.append((threshold, modularity_Q))

# 添加聚类信息
df_clustered = df.copy()
df_clustered['cluster'] = df_clustered.index.map(partition)

# 准备 dot plot 数据
plot_data = []
for cluster_id in sorted(set(partition.values())):
    sub_df = df_clustered[df_clustered['cluster'] == cluster_id]
    n = len(sub_df)
    for feature in ring_features:
        values = sub_df[feature]
        avg = values.mean()
        ratio = (values > 0).sum() / n
        plot_data.append({
            'cluster': f'Group {cluster_id}',
            'feature': feature,
            'avg': avg,
            'ratio': ratio
        })

dot_df = pd.DataFrame(plot_data)
dot_df['cluster'] = pd.Categorical(
    dot_df['cluster'],
    categories=sorted(dot_df['cluster'].unique()),
    ordered=True
)

# 画 dot plot
plt.figure(figsize=(10, 6))
norm = plt.Normalize(vmin=dot_df['avg'].min(), vmax=dot_df['avg'].max())
sns.scatterplot(
    data=dot_df,
    x='feature', y='cluster',
    hue='avg',
    hue_norm=norm,
    size='ratio',
    sizes=(20, 300),
    palette='RdPu',  # 使用渐变
    edgecolor='gray',
    legend=False
)

# 设置颜色条
sm = plt.cm.ScalarMappable(cmap='RdPu', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, orientation='vertical', fraction=0.05, pad=0.01)
cbar.ax.tick_params(left=False, labelleft=False)
cbar.ax.tick_params(labelsize=14)
cbar.set_label("avg", rotation=0, labelpad=25,fontsize=16)

# 设置字体为Arial
plt.rcParams['font.family'] = 'Arial'

# 显示和保存点图，移除标题
plt.xlabel("")
plt.ylabel("Group",fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# 加粗横纵坐标轴线条
ax = plt.gca()
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
# 加粗刻度条线条
plt.tick_params(axis='both', width=2)
plt.tight_layout()
plt.savefig(f"dotplot{threshold:.2f}.png", dpi=300)
plt.close()

# 4. 输出GEXF文件
def export_gexf(G, partition, output_path):
    pos = nx.spring_layout(G, weight='weight', seed=42)
    for node in G.nodes():
        G.nodes[node]['cluster'] = partition.get(node, -1)
        G.nodes[node]['x'] = float(pos[node][0])
        G.nodes[node]['y'] = float(pos[node][1])
        G.nodes[node]['z'] = 0.0

    nx.write_gexf(G, output_path)
    print(f"GEXF 输出成功：{output_path}")

# 使用阈值0.4生成GEXF文件
export_gexf(G, partition, "rings_graph.gexf")