import os
import re

folder = "no_footer"
new_folder = "html_fragments_labeled"

if not os.path.exists(new_folder):
  os.mkdir(new_folder)

for subdir, dirs, files in os.walk(folder):
  print(subdir)
  for f in files:
    new_filepath = new_folder + "/" + subdir.split('/')[-1]
    if not os.path.exists(new_filepath):
      os.mkdir(new_filepath)
    temp = open(subdir+"/"+f,'r')
    text = temp.read()
    temp.close()
    
    text = re.sub(r'\n',' ',text)
    text = re.split(r"<h\d",text)
    filenum = 0
    new_filepath += "/" + f.replace('.txt','-')
    for i in range(len(text)):
      matches = re.findall(r"</(h\d)",text[i],re.MULTILINE)
      if len(matches) > 0:
        section = "<" + matches[0] + text[i]
      else:
        section = text[i]
      exercises = text[i].split('data-type="exercise"')
      if len(exercises) > 1:
        for exercise in exercises[1:]:
          tag = "exercise"
          if exercise.find('data-type="solution"') > -1:
            tag = "excercisewsoln"
          new_file = open(new_filepath + str(filenum) + "-"+tag+".txt","w")
          new_file.write('<div data-type="excercise"'+exercise)
          new_file.close()  
          filenum += 1        
      else:
        paragraphs = exercises[0].split("<p ")
        if len(paragraphs) > 1:
          for p in paragraphs[1:]:
            new_file = open(new_filepath + str(filenum) + "-text.txt","w")
            new_file.write("<p "+p)
            new_file.close()
            filenum += 1
        else:
          glossary = paragraphs[0].split("<dl ")
          if len(glossary) > 1:
            for term in glossary:
              new_file = open(new_filepath + str(filenum) + "-term.txt","w")
              new_file.write("<dl "+term)
              new_file.close()
              filenum += 1
