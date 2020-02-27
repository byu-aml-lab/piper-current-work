import os

folder = "no_footer"
new_folder = "no_footer"

for subdir, dirs, files in os.walk(folder):
  print(subdir)
  for f in files:
    new_filepath = new_folder+"/"+subdir.split('/')[-1]
    if not os.path.exists(new_filepath):
      os.mkdir(new_filepath)
    temp = open(subdir+"/"+f,'r')
    text = temp.read()
    temp.close()
    text = text.split('<ul class="menu">')
    text = text[0]
    new_file = open(new_filepath + "/" + f,"w")
    new_file.write(text)
    new_file.close()
