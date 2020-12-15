import itertools
import os
from typing import Tuple, List
import math
import xml.etree.ElementTree as ET
from xml.dom import minidom


# class representing a BaseLine
class BaseLine:
    """
    Represents a Baseline. It contains its coords
    """

    def __init__(self, coords: List[Tuple[int, int]]):
        self.coords = coords


# class representing a TextLine
class TextLine:
    """
    Represents a Textline. It contains its coords
    """

    def __init__(self, coords: List[Tuple[int, int]] = None, baseline: BaseLine = None):
        self.coords = coords
        self.baseline = baseline


# class representing a TextRegion
class TextRegion:
    """
    Represents a Textregion. It contains a list of Text- and Baselines and a list of coords.
    """

    def __init__(self, text_lines: List[TextLine] = None, coords: List[Tuple[int, int]] = None):
        self.coords = coords
        self.textLines = text_lines


# this class is responsible for Converting a set of Text-Regions into a String or XML file
def coords_to_string(coords: List[Tuple[int, int]]) -> str:
    """
    transforms int tuples to string for xml
    :param coords: list of int tuples
    :return: string of coords separated by whitespaces
    """
    coord_string = ""
    if coords:
        for coord in coords:
            coord_string = coord_string + str(int(round(coord[0]))) + "," + str(int(round(coord[1]))) + " "
    return coord_string[:-1]


class XMLGenerator:
    """
    Creates and saves textregions to xml file
    """

    def __init__(self, image_width: int, image_height: int, image_filename: str, regions: List[TextRegion]):
        self.imageWidth = image_width
        self.imageHeight = image_height
        self.imageFilename = image_filename
        self.regions = regions

    def baselines_to_xml_string(self) -> str:
        """
        creates the xml to the given baselines
        :return: xml-string of baselines
        """
        def make_id_gen(identifier):
            return map(lambda x: f"{identifier}{x}", itertools.count(start=1))
        textline_id_gen = make_id_gen("TextLine")
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
        for t_ind, t in enumerate(self.regions):
            tr_node = ET.SubElement(page_node, "TextRegion", id="TextRegion" + str(t_ind))
            ET.SubElement(ordered_group_node, "RegionRefIndexed", index=str(t_ind), regionRef="TextRegion" + str(t_ind))
            if t.coords is not None:
                ET.SubElement(tr_node, "Coords", points=coords_to_string(t.coords))

            for i_ind, i in enumerate(t.textLines):
                tl_node = ET.SubElement(tr_node, "TextLine", id=next(textline_id_gen))
                ET.SubElement(tl_node, "Coords", points=coords_to_string(i.coords))
                ET.SubElement(tl_node, "Baseline", points=coords_to_string(i.baseline.coords))
        return minidom.parseString(ET.tostring(root_node)).toprettyxml(indent='    ')

    def save_textregions_as_xml(self, output_path: str):
        """
        Transform textregions to xml and save it to output_path
        :param output_path:
        """
        complete_name = os.path.join(output_path, self.imageFilename.split(".")[0] + ".xml")  # remove .image extensions from filename
        output_string = self.baselines_to_xml_string()
        file = open(complete_name, "w")
        file.write(output_string)
        file.close()


def annotate_with_xmlns_prefixes(tree, xmlns_prefix, skip_root_node=True):
    """
    annotates the xml with prefixes (like in the example of christoph)
    :param tree:
    :param xmlns_prefix:
    :param skip_root_node:
    :return:
    """
    if not ET.iselement(tree):
        tree = tree.getroot()
    iterator = tree.iter()
    if skip_root_node:
        iterator.next()
    for e in iterator:
        if ':' not in e.tag:
            e.tag = xmlns_prefix + ":" + e.tag
