import tensorflow as tf
import random
import numpy as np
import csv
import re
import os
import create_data
from xml.dom import minidom
import xml.etree.ElementTree as ET
from vocab import Vocabulary
from env2 import ConceptData
from create_data import addFile

vocabulary = Vocabulary()

def addFile(name):
    file_name = name + "_structured_final.xml"
    file = minidom.parse(os.path.join(os.path.join('visa_dataset', 'UK'), file_name))
    concepts = file.getElementsByTagName('concept')

    for concept in concepts:
        vocabulary.addConcept(concept)

addFile("ANIMALS")
addFile("APPLIANCES")
addFile("ARTEFACTS")
addFile("CLOTHING")
addFile("CONTAINER")
addFile("DEVICE")
addFile("FOOD")
addFile("HOME")
addFile("INSTRUMENTS")
addFile("MATERIAL")
addFile("PLANTS")
addFile("STRUCTURES")
addFile("TOOLS")
addFile("TOYS")
addFile("VEHICLES")
addFile("WEAPONS")

concept_data = ConceptData(vocabulary, 9)
data = concept_data.getInput()

print("fetched input:")
print(data)
