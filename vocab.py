from xml.dom import minidom
import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np

class Vocabulary:
    def __init__(self):
        # dict to convert concept into 0 - 1 vector
        self.attribute2index = {}
        # dict to convert back to attributes
        self.index2attribute = {}
        # dict of concepts with attributes
        self.consWithAttributes = {}
        # dict of number of attribute appearances
        self.attribute2count = {}
        # number of concepts
        self.num_concept = 0
        # number of attributes
        self.num_attributes = 0
        # list of concepts
        self.concept_list = []
        # list of attributes
        self.attribute_list = []
        # dict of concepts and attributes as binary vector
        self.concept2vector = {}

    def iterateConcept(self, animal):
        i = 0
        attributes = []
        for node in animal.childNodes:
            i += 1
            if i%2 == 0:
                for attribute in node.childNodes[0].data.split('\n'):
                    if attribute != '':
                        attributes.append(attribute.strip())
        return attributes

    def addConcept(self, animal):
        #print(animal.attributes['name'].value)
        self.num_concept += 1
        self.concept_list.append(animal)
        attributes = self.iterateConcept(animal)
        self.consWithAttributes[animal.attributes['name'].value] = attributes
        for attribute in attributes:
            self.addAttr(attribute)

    def addAttr(self, attribute):
        if attribute not in self.attribute2index:
            self.attribute2index[attribute] = self.num_attributes
            self.index2attribute[self.num_attributes] = attribute
            self.num_attributes += 1
            self.attribute2count[attribute] = 1
        else:
            self.attribute2count[attribute] += 1

    def parseConcept(self, animal):
        # returns vector of lenght num_attributes
        # ones for attributes of concept, else zero
        if animal.attributes['name'].value not in self.concept2vector:
            attributes = self.consWithAttributes[animal.attributes['name'].value]
            vector = np.zeros(self.num_attributes)
            for attribute in attributes:
                vector[self.attribute2index[attribute]] = 1
            self.concept2vector[animal.attributes['name'].value] = vector
        else:
            vector = self.concept2vector[animal.attributes['name'].value]

        return vector
