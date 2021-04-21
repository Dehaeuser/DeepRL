import os

from xml.dom import minidom
import xml.etree.ElementTree as ET
from vocab import Vocabulary

vocabulary = Vocabulary()

def addFile(name):
    file_name = name + "_structured_final.xml"
    file = minidom.parse(os.path.join(os.path.join('visa_dataset', 'UK'), file_name))
    concepts = file.getElementsByTagName('concept')

    for concept in concepts:
        vocabulary.addConcept(concept)

def main():
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

    print("num concepts:")
    print(vocabulary.num_concept)
    print("num attributes:")
    print(vocabulary.num_attributes)
    #print("keys:")
    #print(vocabulary.consWithAttributes.keys())
    #print("attributes zucchini:")
    #print(vocabulary.consWithAttributes['zucchini'])
    concept_data = ConceptData(vocabulary, 9)
    data = concept_data.getInput()
    print("fetched input:")
    print(data)

if __name__ == '__main__':
    main()
