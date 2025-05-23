from graphviz import Digraph

# Create a directed graph (left-to-right layout)
dot = Digraph(format='png')
dot.attr(rankdir='LR', size='10')

# 1. Guandan Environment
dot.node('env', 'Guandan Game\nEnvironment\n(WebSocket)',
         shape='box', style='filled', fillcolor='#e0f7fa')

# 2. Combined Actor Node (with main loop included)
dot.node('actor', '''Actor (4 Processes)''',
         shape='box', style='filled', fillcolor='#fff3e0')

# 3. Memory Pool
dot.node('mem_pool', 'Shared\nMemory Pool',
         shape='box', style='filled', fillcolor='#ede7f6')

# 4. Learner
dot.node('learner', 'Learner\n(PPO Agent)',
         shape='box', style='filled', fillcolor='#e8f5e9')

# 5. Checkpoint Storage
dot.node('ckpt', 'Model Checkpoint\n(Pickle File)',
         shape='box', style='filled', fillcolor='#dcedc8')

# Connections
dot.edge('env', 'actor', label='state')
dot.edge('actor', 'env', label='action or none')
dot.edge('actor', 'mem_pool', label='send data\nafter each game')
dot.edge('mem_pool', 'learner', label='sample batch')
dot.edge('learner', 'ckpt', label='save weights\n(every N updates)')
dot.edge('ckpt', 'actor', xlabel='load latest\nmodel')

# Output the diagram
dot.render('danzero_diagram', format='png', cleanup=True)
