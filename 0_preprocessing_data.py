
# coding: utf-8

# ### Preprocessing data (remove noisered in audio file and save data)

# Import the libraries

# In[ ]:

import os
import xml.etree.ElementTree as ET
from sklearn.externals import joblib


# Directories

# In[ ]:

# Current directory
current_directory = os.getcwd()
# Directory the metadata are stored
xml_path = os.path.join(current_directory, 'data/xml')
# Directory the audio files are stored
wav_path = os.path.join(current_directory, 'data/wav')


# Functions

# In[ ]:

# Fuction for remove noise from audio files
def noiseReduction(original_audio):
    
    response = False
    new_audio = None
    
    # new audio file
    new_audio = original_audio.replace('.wav', '_sil.wav')
    
    if not os.path.isfile(new_audio):
        
        # remove noise
        resp = os.system('sox ' + original_audio + ' ' + new_audio + '.wav noisered speech.noise-profile .5')
        
        # create new file without silence
        resp = os.system('sox ' + original_audio + ' ' + new_audio + ' silence 1 0.1 1% -1 0.1 1%')
        
        if resp is 0 or os.path.isfile(new_audio):
            
            response = True
    
    return response, new_audio


# In[ ]:

# Function for save in pickle file all metadata
def exportMetadata(xml_path, wav_path):
    
    data = {}
    
    # for each xml file in path
    for xml_file in os.listdir(xml_path):
        
        if xml_file.endswith(".xml"):
            
            dir_xml = os.path.join(xml_path, xml_file)
            dir_wav = os.path.join(wav_path, xml_file.replace('.xml','.wav'))
            
            # verify if exist wav file of xml file
            if os.path.isfile(dir_wav):                
                    
                tmp_dict_xml = {}
                tree = ET.parse(dir_xml)
                root = tree.getroot()       
                
                for child in root:
                    tmp_dict_xml[child.tag] = root.find(child.tag).text
                
                tmp_dict_xml['xmlDir'] = dir_xml
                tmp_dict_xml['wavDir'] = dir_wav

                response, new_audio = noiseReduction(dir_wav)
                
                if response is True:
                    tmp_dict_xml['silDir'] = new_audio                    
                    data[xml_file] = tmp_dict_xml
                    
    # Dump data with compression
    print('Dump data ...')
    joblib.dump(data, 'data.gz', compress=('gzip', 3)) 
    print("Files for process:", len(data))
    print("Done.")
    
    return data


# Process

# In[ ]:

# Main process
data = exportMetadata(xml_path, wav_path)


# In[ ]:

# Visualization of one example of metadata
for key in data:
    
    metadata = data[key]
    
    for key_ in metadata:
        
        print (key_, ':', str(metadata[key_]))
        
    break

