import os
from typing import Tuple, List
import math
import xml.etree.ElementTree as ET
from xml.dom import minidom


# this class is responsible for Converting a set of Text-Regions into a String or XML file
class XMLGenerator:
    '''
    Creates and saves textregions to xml file
    '''

    def __init__(self, imageWidth: int, imageHeight: int, imageFilename: str, baselines):
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.imageFilename = imageFilename
        self.baselines = baselines

    def baselines_to_xml_string(self) -> str:
        '''
        creates the xml to the given baselines
        :return: xml-string of baselines
        '''
        xmlns_uris = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15'}
        attr_qname = ET.QName("http://www.w3.org/2001/XMLSchema-instance", "schemaLocation")
        root_node = ET.Element("PcGts", {
            attr_qname: "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"})
        for prefix, uri in xmlns_uris.items():
            root_node.attrib['xmlns'] = uri
        page_node = ET.SubElement(root_node, "Page", imageFilename=str(self.imageFilename),
                                  imageWidth=str(self.imageWidth), imageHeight=str(self.imageHeight))
        reading_order_node = ET.SubElement(page_node, "ReadingOrder")
        ordered_group_node = ET.SubElement(reading_order_node, "OrderedGroup", id="ro1", caption="Region reading order")
        for i in range(0, len(self.baselines)):
            tr_node = ET.SubElement(page_node, "TextRegion", id="TextRegion" + str(i))
            ET.SubElement(ordered_group_node, "RegionRefIndexed", index=str(i), regionRef="TextRegion" + str(i))
            # ET.SubElement(tr_node, "Coords", points=self.coords_to_string(region.coords))
            tl_node = ET.SubElement(tr_node, "TextLine", id="TextLine")
            ET.SubElement(tl_node, "Baseline", points=self.coords_to_string(self.baselines[i]))
        # annotate_with_XMLNS_prefixes(root_node, "pc", False)
        return minidom.parseString(ET.tostring(root_node)).toprettyxml(indent='    ')

    def coords_to_string(self, coords: List[Tuple[int, int]]) -> str:
        '''
        transforms int tuples to string for xml
        :param coords: list of int tuples
        :return: string of coords separated by whitespaces
        '''
        coordstring = ""
        for coord in coords:
            coordstring = coordstring + str(coord[0]) + "," + str(coord[1]) + " "
        return coordstring[:-1]

    def save_textregions_as_xml(self, output_path: str):
        '''
        Transform textregions to xml and save it to output_path
        :param output_path:
        '''
        completeName = os.path.join(output_path, self.imageFilename + ".xml")
        output_String = self.baselines_to_xml_string()
        file = open(completeName, "w")
        file.write(output_String)
        file.close()


def annotate_with_XMLNS_prefixes(tree, xmlns_prefix, skip_root_node=True):
    '''
    annotates the xml with prefixes (like in the example of christoph)
    :param tree:
    :param xmlns_prefix:
    :param skip_root_node:
    :return:
    '''
    if not ET.iselement(tree):
        tree = tree.getroot()
    iterator = tree.iter()
    if skip_root_node:
        iterator.next()
    for e in iterator:
        if not ':' in e.tag:
            e.tag = xmlns_prefix + ":" + e.tag
