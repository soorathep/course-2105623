from pathlib import Path
import re,sys
site=Path('_site'); repo=sys.argv[1]; code=sys.argv[2]
prefix='/'+repo
# Root-relative references produced by hand-written/Quarto files need the project-site prefix.
def convert(url):
 if not url.startswith('/') or url.startswith('//') or url==prefix or url.startswith(prefix+'/'): return url
 if url.startswith('/teaching/'+code+'/') or url.startswith('/teaching/shared/') or url.startswith('/files/'+code+'/') or url.startswith('/assets/') or url in ['/favicon.png','/og.png','/feed.xml','/sitemap.xml']: return prefix+url
 return 'https://www.skhgroup.net'+url
for f in site.rglob('*'):
 if f.suffix.lower() not in ['.html','.css','.js','.json','.svg']: continue
 text=f.read_text(errors='strict')
 if f.suffix=='.html':
  text=re.sub(r'((?:href|src|poster|action)=["\'])(/[^"\']*)',lambda m:m[1]+convert(m[2]),text)
  text=re.sub(r'((?:href|src|poster|action)=["\'])https://(?:www\.)?skhgroup.net(/(?:teaching/2105623/|files/2105623/)[^"\']*)',lambda m:m[1]+prefix+m[2],text)
 text=re.sub(r'(url\(["\']?)(/[^)"\']*)',lambda m:m[1]+convert(m[2]),text)
 f.write_text(text)
