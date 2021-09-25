import tabulate
from IPython.display import HTML, display

data = {}
data['num'] = [1,2,3]
data['alp'] = ['a', 'b', 'c']

headers = ['<img src="/tmp/a.png" width="50" height="50">', 'alp']
table = [data['num'], data['alp']]

html = tabulate.tabulate(table, headers, tablefmt='html')
print(html)
# display(HTML(html))
