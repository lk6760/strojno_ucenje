import os
import json
import shutil

cwd = os.getcwd()
root = os.path.join(cwd, 'data/nsynth/nsynth-train/')
old_path = os.path.join(root, 'audio')
new_path = os.path.join(root, 'audio_selected')

if os.path.isdir(new_path):
        print("Beep-Boop")
else:
        print("Can not find the directory", new_path)

f = open("./json/nsynth_selected_sounds_per_class.json","r") 
json_data = json.loads(f.read())
f.close()

for intrument, list_of_filenames in json_data.items():
	print("Moving '{}' audio files".format(intrument))
	for filename in list_of_filenames:
		#print(filename)
		full_filename = filename + ".wav"
		old_file_location = os.path.join(old_path, full_filename)
		new_file_location = os.path.join(new_path, full_filename)

		#print(old_file_location)
		#print(new_file_location)
		#print("****************************")
		if os.path.isfile(old_file_location):
                        shutil.move(old_file_location, new_file_location)
                        #print("a")
