from xml.dom import minidom
import xml.etree.ElementTree as ET
from vocab import Vocabulary

vocabulary = Vocabulary()

def addFile(name):
    file_name = name + "_structured_final.us.xml"
    file = minidom.parse(file_name)
    concepts = file.getElementsByTagName('concept')

    for concept in concepts:
        vocabulary.addConcept(concept)

def main():
    addFile("ANIMALS")
    addFile("APPLIANCES")
    addFile("ARTIFACTS")
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
    print(vocabulary.num_concepts)
    print("num attributes:")
    print(vocabulary.num_attributes)

if __name__ == '__main__':
    main()
