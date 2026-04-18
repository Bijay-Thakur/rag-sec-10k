from bs4 import BeautifulSoup
import re
from pathlib import Path
import sys

ITEM_PATTERN=re.compile(r"^ITEM\s+\d+[A-Z]?\.",re.IGNORECASE)
def load_html(path):
    with open(path,"r",encoding="utf-8",errors="ignore")as f:
        html=f.read()
    return BeautifulSoup(html,"lxml")

def clean_soup(soup):
    for tag in soup(["script","style"]):
        tag.decompose()
    
    for ix_tag in soup.find_all(lambda tag:tag.name and tag.name.startswith("ix:")):
        ix_tag.decompose()
    return soup

# extract text and split into sections
def extract_sections(soup,source_file):
    sections=[]
    content_title="PREAMBLE"
    content_text=[]
    
    for element in soup.stripped_strings:
        text=element.strip()
        if not text:
            continue
        #Detect ITEM headings
        if ITEM_PATTERN.match(text):
            #Save previous section
            
            if content_text and content_title!="PREAMBLE":
                sections.append({
                    "section_title":content_title,
                    "text":"\n".join(content_text),
                    "source_file":source_file
                })
                content_text=[]
            content_title=text.upper()
        else:
            content_text.append(text)
    #Save last section
    if content_text and content_title!="PREAMBLE":
        sections.append({
            "section_title":content_title,
            "text":"\n".join(content_text),
            "source_file":source_file
        })
    return sections
if __name__=="__main__":
    path=sys.argv[1]
    soup=clean_soup(load_html(path))
    sections=extract_sections(soup,Path(path).name)
    print(f"found{len(sections)}sections in {path}\n")
    for s in sections[:5]:
        print(f"---{s['section_title']}---")
        print(s['text'][:300])
        print(f"[...{len(s['text'])}chart total]\n")
        
        

