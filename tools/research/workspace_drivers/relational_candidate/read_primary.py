import json, urllib.request
from html.parser import HTMLParser
from pathlib import Path
class Parser(HTMLParser):
    def __init__(self):super().__init__();self.parts=[];self.links=[]
    def handle_data(self,data):
        if data.strip():self.parts.append(data.strip())
    def handle_starttag(self,tag,attrs):
        if tag=='a':
            for key,value in attrs:
                if key=='href' and value and 'github.com' in value:self.links.append(value)
raw=urllib.request.urlopen('https://arxiv.org/html/2401.07745v2',timeout=30).read()
p=Parser();p.feed(raw.decode());text=' '.join(p.parts)
print('GITHUB',json.dumps(sorted(set(p.links))))
i=text.find('3.2.3',text.find('3.2.2 Efficient')+20)
print('SECTION',text[i:i+3200])
request=urllib.request.Request('https://api.github.com/search/repositories?q=MaskClustering',headers={'User-Agent':'EviScene-research-audit'})
repos=json.load(urllib.request.urlopen(request,timeout=30))
print('REPOS',json.dumps([(r['full_name'],r.get('description')) for r in repos['items'][:5]]))
