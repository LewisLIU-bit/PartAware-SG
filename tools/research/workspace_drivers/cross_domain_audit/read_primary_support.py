"""Read primary paper text for the audit's unimplemented next hypothesis."""
import hashlib, json, urllib.request
from html.parser import HTMLParser
from pathlib import Path

class Text(HTMLParser):
    def __init__(self): super().__init__(); self.parts=[]
    def handle_data(self,data):
        if data.strip(): self.parts.append(data.strip())

url='https://arxiv.org/html/2401.07745v2'
data=urllib.request.urlopen(url,timeout=30).read()
parser=Text();parser.feed(data.decode('utf-8'))
text=' '.join(parser.parts)
record=dict(primary_url=url,document_sha256=hashlib.sha256(data).hexdigest(),
    title_present='MaskClustering' in text,sections_checked=[])
for term in ['Under-Segmented Mask Filtering','under-segmented','observer','supporter']:
    i=text.lower().find(term.lower())
    record['sections_checked'].append(dict(term=term,found=i>=0))
    if i>=0: print(term, text[max(0,i-200):i+1800])
Path(__file__).with_name('primary_support_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
