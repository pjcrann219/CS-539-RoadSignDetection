import os
import xml.etree.ElementTree as ET
import shutil

def create_xml(img_path, output_dir, class_name="stop"):
    annotation = ET.Element("annotation")
    folder = ET.SubElement(annotation, "folder")
    folder.text = os.path.basename(output_dir)

    filename = ET.SubElement(annotation, "filename")
    filename.text = os.path.basename(img_path)

    obj = ET.SubElement(annotation, "object")
    name= ET.SubElement(obj, "name")
    name.text = class_name

    # Write tags and corresponding info to XML file
    xml_path = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0] + '.xml')
    tree = ET.ElementTree(annotation)
    tree.write(xml_path)


def create_xmls_for_dir(imgs_dir, label):
    # ensure that path is valid
    if not os.path.isdir(imgs_dir):
        raise ValueError("Provided directory is invalid")

    imgs_dir = os.path.abspath(imgs_dir)


    parent_dir = os.path.dirname(imgs_dir)
    annotations_dir = os.path.join(parent_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)

    # Go through each file in dir and create corresponding xml
    for filename in os.listdir(imgs_dir):
        if filename.lower().endswith('.png'):
            img_path = os.path.join(imgs_dir, filename)
            create_xml(img_path, annotations_dir, class_name=label)


def move_images(source_dir):
    cwd = os.getcwd()
    images_dir= os.path.join(cwd, 'images')
    os.makedirs(images_dir, exist_ok=True)

    if not os.path.isdir(source_dir):
        raise ValueError("Invalid source dir specified")

    for filename in os.listdir(source_dir):
        src_file = os.path.join(source_dir, filename)
        new_file = os.path.join(images_dir, filename)

        shutil.move(src_file, new_file)

# testing new script
img_dir = 'new_images/050505.png'
output_dir = 'new_annotations/'

#create_xml(img_dir,output_dir)

# Incorporating GRSTB dataset
create_xmls_for_dir('stop/', "stop")
move_images('stop/')
create_xmls_for_dir('speedlimit/', "speedlimit")
move_images('speedlimit/')
create_xmls_for_dir('crosswalk/', "crosswalk")
move_images('crosswalk/')
create_xmls_for_dir('trafficlight/', "trafficlight")
move_images('trafficlight/')
