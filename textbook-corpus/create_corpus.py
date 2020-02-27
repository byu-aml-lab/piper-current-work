import uuid
import os
import json
import re

folder = "html_fragments_labeled"

quiz_doc = open("ankura_textbook/scienceh.txt","a")
answer_doc = open("ankura_textbook/scienceh.stars","a")

answers = ['1','5']
i=0

for directory in os.listdir(folder):
  print(directory)
  if directory.startswith("fizyka"):
    continue
  if directory.startswith("anatomy") or directory.startswith("astronomy") or directory.startswith("biology") or directory.startswith("chemistry") or directory.startswith("college-physics") or directory.startswith("concepts") or directory.startswith("microbiology") or directory.startswith("university-physics"):
    i+=1
    for filename in os.listdir(folder + "/" + directory):

      if directory.startswith("anatomy") or directory.startswith("biology") or directory.startswith("chemistry") or directory.startswith("concepts"):
        answer = "1"
      else:
        answer = "5"   

      f = open(folder + "/" + directory + "/" + filename,"r")
      document = f.read()
      f.close()

      document = re.sub(r"\<.*?>",' ',document)
      document = re.sub(r"#?\w+?;"," ",document)
      document = re.sub(r"&\w+?;"," ",document)
      document = re.sub(r"&"," ",document)
      document = re.sub(r'\s+'," ",document).strip()   
      
      if document == "":
        continue   

      uid = directory + "_" + filename #uid = uuid.uuid4().hex
      answer_doc.write(uid + "\t" + answer + "\n")
      quiz_doc.write(uid + "\t" + document + "\n")

quiz_doc.close()
answer_doc.close()
