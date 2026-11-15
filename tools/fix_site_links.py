from pathlib import Path
import re,sys
site=Path('_site'); repo=sys.argv[1]; code=sys.argv[2]
prefix='/'+repo
# Root-relative references produced by hand-written/Quarto files need the project-site prefix.
def convert(url):
 if not url.startswith('/') or url.startswith('//') or url in [prefix,prefix+'/']: return url
 if url.startswith(prefix+'/'): url=url[len(prefix):]
 if url.split('?')[0].split('#')[0] in ['/teaching/2105315/slides/exit.html','/teaching/2105603/slides-2027/exit.html']: return 'https://www.skhgroup.net'+url
 if url.startswith('/teaching/'+code+'/') or url.startswith('/teaching/shared/') or url.startswith('/files/'+code+'/') or url.startswith('/assets/') or url in ['/favicon.png','/og.png','/feed.xml','/sitemap.xml']: return prefix+url
 return 'https://www.skhgroup.net'+url
for f in site.rglob('*'):
 if f.suffix.lower() not in ['.html','.css','.js','.json','.svg']: continue
 text=f.read_text(errors='strict')
 if f.suffix=='.html':
  text=re.sub(r'((?:href|src|poster|action)=["\'])(/[^"\']*)',lambda m:m[1]+convert(m[2]),text)
  text=re.sub(r'((?:href|src|poster|action)=["\'])https://(?:www\.)?skhgroup.net(/(?:teaching/2105623/|files/2105623/)[^"\']*)',lambda m:m[1]+prefix+m[2],text)
 text=re.sub(r'''(["'])(/teaching/shared/[^"']*)''',lambda m:m[1]+prefix+m[2],text)
 text=re.sub(r'(url\(["\']?)(/[^)"\']*)',lambda m:m[1]+convert(m[2]),text)
 f.write_text(text)

# Keep submission forms on their existing authorized backend origin.
for route in ['teaching/2105315/slides/exit.html','teaching/2105603/slides-2027/exit.html']:
 f=site/route
 if f.exists():
  dest='https://www.skhgroup.net/'+route
  f.write_text('<!doctype html><meta charset="utf-8"><title>Course feedback</title><script>location.replace('+repr(dest)+'+location.search+location.hash)</script><a href="'+dest+'">Open feedback form</a>')
