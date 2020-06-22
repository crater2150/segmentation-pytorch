from tkinter import *
from tkinter import filedialog as fd
from PIL import ImageTk, Image
from segmentation.gui.xml_util import XMLGenerator
from typing import Tuple, List
import os
from enum import Enum
import threading
import numpy as np
from itertools import chain
from shapely.geometry import LineString, box
from segmentation.util import multiple_file_types

class App:
    '''
    Contains the GUI Elements and updates the GUI
    '''

    def __init__(self, controller, root):
        # ---create GUI Layout ---
        self.root = root  # create window
        self.windowwidth, self.windowheight = root.winfo_screenwidth(), root.winfo_screenheight()
        self.root.geometry("%dx%d+0+0" % (self.windowwidth, self.windowheight))
        # create imageframe
        self.imageframewidth = int(self.windowwidth * 0.4)
        self.imageframeheight = int(self.windowheight * 0.9)
        # imageframe width and height is hardcoded. This could be parameterized
        self.imageframe = Canvas(root, width=self.imageframewidth, height=self.imageframeheight)
        self.imageframe.grid(row=0, column=0, rowspan=30, pady=30)
        self.controller = controller  # get controller
        # create Buttons
        self.BUTTON_HEIGHT = 2
        self.BUTTON_WIDTH = 20
        self.load_img_Button = Button(root, text="load image", height=self.BUTTON_HEIGHT, width=self.BUTTON_WIDTH,
                                      command=controller.load_img_thread)
        self.set_save_dir_Button = Button(root, text="set save dir", height=self.BUTTON_HEIGHT, width=self.BUTTON_WIDTH,
                                          command=controller.set_save_dir)
        self.delete_bl_Button = Button(root, text="delete baselines", height=self.BUTTON_HEIGHT,
                                       width=self.BUTTON_WIDTH,
                                       command=controller.delete_bl)
        self.new_baseline_button = Button(root, text="create Baselines", height=self.BUTTON_HEIGHT,
                                          width=self.BUTTON_WIDTH,
                                          command=controller.create_baseline)
        self.new_con_baseline_button = Button(root, text="connects two Baselines", height=self.BUTTON_HEIGHT,
                                              width=self.BUTTON_WIDTH,
                                              command=controller.connect_baseline)

        self.new_split_baseline_button = Button(root, text="splits two Baselines", height=self.BUTTON_HEIGHT,
                                                width=self.BUTTON_WIDTH,
                                                command=controller.split_baseline)
        self.next_image = Button(root, text="Loads_next_image_in_directory", height=self.BUTTON_HEIGHT,
                                 width=self.BUTTON_WIDTH,
                                 command=controller.get_next_image)
        self.modify_Button = Button(root, text="modify", height=self.BUTTON_HEIGHT, width=self.BUTTON_WIDTH,
                                    command=controller.extend_baseline)
        self.save_Button = Button(root, text="save", height=self.BUTTON_HEIGHT, width=self.BUTTON_WIDTH,
                                  command=controller.save)
        # Button Position
        self.load_img_Button.grid(row=4, column=1, padx=min(self.windowwidth / 10, 5), pady=0)
        self.delete_bl_Button.grid(row=5, column=1, padx=min(self.windowwidth / 10, 5), pady=0)
        self.new_baseline_button.grid(row=6, column=1, padx=min(self.windowwidth / 5, 10), pady=0)
        self.modify_Button.grid(row=7, column=1, padx=min(self.windowwidth / 10, 5), pady=0)
        self.new_con_baseline_button.grid(row=8, column=1, padx=min(self.windowwidth / 5, 10), pady=0)
        self.new_split_baseline_button.grid(row=9, column=1, padx=min(self.windowwidth / 5, 10), pady=0)
        self.set_save_dir_Button.grid(row=10, column=1, padx=min(self.windowwidth / 5, 10), pady=0)
        self.save_Button.grid(row=11, column=1, padx=min(self.windowwidth / 10, 5), pady=0)
        self.next_image.grid(row=12, column=1, padx=min(self.windowwidth / 10, 5), pady=0)
        # create Labels
        self.state_label = Label(root, text='Start by loading an image', width=45, font=(None, 15))
        self.state_label.grid(row=2, rowspan=10, column=0, padx=10)
        self.warning_label = Label(root, fg="red", text='\n\n', width=45, font=(None, 15))
        self.warning_label.grid(row=11, rowspan=10, column=3, padx=10)
        # create Checkbutton
        self.run_inference = IntVar()
        self.run_inference.set(1)
        self.run_inference_checkbutton = Checkbutton(root, text="annotate image when loading image",
                                                     variable=self.run_inference)
        self.run_inference_checkbutton.grid(row=13, column=1, padx=min(self.windowwidth / 10, 5), pady=0)

    def show_img(self, img):
        '''
        Display an image
        :param img:
        :return:
        '''
        self.imageframe.create_image(0, 0, anchor=NW, image=img)

    def update_labels(self, state_label: str = None, warning_label: str = ""):
        '''
        Updates the labels displayed in the gui
        :param state_label: text the state_label should display
        :param warning_label: text the warning_label should display
        :return:
        '''
        if state_label is not None:
            self.state_label.config(text=state_label)
        self.warning_label.config(text=warning_label)


class Controller:
    '''
    Controls the flow of the program.
    '''

    def __init__(self, root, network):
        self.app = App(self, root)
        self.root = root
        self.imageframe = self.app.imageframe
        self.imfr_width = self.app.imageframewidth
        self.imfr_height = self.app.imageframeheight
        self.network = network

        self.img = None  # I need to store the image because of the python garbage collector
        self.im_name = None
        self.lock = States.LOAD_IMAGE  # sets the current lock for Buttons
        self.im_width = None  # original imagewidth
        self.im_height = None  # original imageheight
        self.save_directory = None  # output dirpath
        self.binder = Binder(self.root, self.imageframe, self)  # handles binds of events

        self.id_baselines = {}
        self.baselines = []

        self.directory = None

        # variables to delete baselines
        self.line_coords = []
        self.line = None

        # vairables to edit baselines
        self.edit_item = None
        self.edit_baseline = []
        self.edit_index = None

        # variables to connect baselines
        self.baseline_to_connect = []
        self.baseline_indexes = []

        self.scale = None

    # ----------BUTTON EVENTS----------

    def load_img_thread(self):
        '''
        creates second thread to run annotation when loading image
        :return:
        '''
        if self.lock != States.LOAD_IMAGE and self.lock != States.NO_STATE:
            return

        filename = fd.askopenfilename()  # starts dialogwindow
        if filename == "":
            return
        if self.directory is None:
            self.directory = os.path.dirname(filename)
        thread = threading.Thread(target=self.load_img,
                                  name="Thread1",
                                  args=[filename])
        thread.start()

    def load_img(self, filename):
        '''
        loads image from file to GUI
        :param filename: Name of image
        :return:
        '''
        self.imageframe.delete("all")
        self.app.state_label.grid_remove()

        # self.reset_imageframe()
        img = Image.open(filename)  # open image
        self.im_name, self.suffix = os.path.basename(filename).split('.')  # extract image name

        def get_scaled_dimensions(imagex, imagey, imagex2, imagey2):
            w_percent = imagex2 / float(imagex)
            h_percent = imagey2 / float(imagey)
            scale = min(1.0, w_percent, h_percent)
            w_size = int(float(imagex) * scale)
            h_size = int(float(imagey) * scale)
            return h_size, w_size, scale

        self.im_width, self.im_height = img.size  # save image size
        h_size, w_size, scale = get_scaled_dimensions(self.im_width, self.im_height, self.imfr_width, self.imfr_height)

        img = img.resize((w_size, h_size), Image.BICUBIC)  # resize image to imageframe size
        self.img = ImageTk.PhotoImage(img)  # tkinter needs PhotoImage
        self.app.show_img(self.img)
        if self.app.run_inference.get() == 0:
            self.lock = States.NO_STATE
            return
        # get the textregions already resized and resize to imageframe size
        self.lock = States.FULL_LOCK
        from segmentation.postprocessing.baseline_extraction import extraxct_baselines_from_probability_map
        probmap, scale_factor = self.network.predict_single_image_by_path(filename)
        baselines = extraxct_baselines_from_probability_map(probmap)
        if baselines and len(baselines) > 0:
            for ind, baseline in enumerate(baselines):
                from segmentation.postprocessing.simplify_line import VWSimplifier
                simplifier = VWSimplifier(np.asarray(baseline, dtype=np.float64))
                np_baseline = simplifier.from_number(5)
                baselines[ind] = list(np_baseline)
                pass
        else:
            baselines = []

        def scale_baselines(baselines, scale_factor=1.0):
            for b_idx, bline in enumerate(baselines):
                for c_idx, coord in enumerate(bline):
                    coord = (int(coord[0] * scale_factor), int(coord[1] * scale_factor))
                    baselines[b_idx][c_idx] = coord

        scale_baselines(baselines, 1 / scale_factor)
        scale_baselines(baselines, scale)
        self.scale = scale
        self.baselines = baselines
        self.lock = States.NO_STATE
        self.draw_baslines()
        self.lock = States.NO_STATE

    def draw_baslines(self):
        from itertools import chain
        self.imageframe.delete("baseline")
        for baseline in self.baselines:
            if len(baseline) <= 2:
                continue
            t = list(chain.from_iterable(baseline))
            lined_id = self.app.imageframe.create_line(t, fill='green', width=3, tag="baseline")
            self.id_baselines[lined_id] = baseline

    def baseline_delete_rec_on_button_press(self, event):
        # save mouse drag start position
        self.start_x = event.x
        self.start_y = event.y
        # create rectangle if not yet exist
        self.rect = self.app.imageframe.create_rectangle(0, 0, 1, 1, outline="black", tag="delrec")

    def baseline_delete_rec_on_move_press(self, event):
        curX, curY = (event.x, event.y)

        # expand rectangle as you drag the mouse
        self.app.imageframe.coords(self.rect, self.start_x, self.start_y, curX, curY)

    def baseline_delete_rec_on_button_release(self, event):
        if len(self.baselines) == 0:
            return

        def remove_items_from_list(list1, list2):
            list2 = [(int(x[0]), int(x[1])) for x in list2]
            set1 = set(list1)
            set2 = set(list2)
            set3 = set1 - set2
            ret = list(set3)
            if list2[-1][0] <= list1[-1][0] and list2[0][0] <= list1[0][0]:
                if list2[-1] not in ret:
                    ret.append(list2[-1])
            elif list2[-1][0] >= list1[-1][0] and list2[0][0] >= list1[0][0]:
                if list2[0] not in ret:
                    ret.append(list2[0])
            ret = sorted(ret, key=lambda x: x[0])
            return ret

        coords = self.app.imageframe.coords(self.rect)
        box_s = box(coords[0], coords[1], coords[2], coords[3])
        for ind, baseline in enumerate(self.baselines):
            base_l = baseline
            if len(baseline) < 2:
                continue
            linestring_s = LineString(baseline)
            if box_s.intersects(linestring_s):
                inters = box_s.intersection(linestring_s)

                if inters.type != "LineString":
                    continue
                coords = list(inters.coords)

                base_l = remove_items_from_list(base_l, coords)
                self.baselines[ind] = base_l

            pass

        self.imageframe.delete("delrec")
        self.rect = None
        self.draw_baslines()

    def baseline_create_on_button_press(self, event):
        # save mouse drag start position
        self.start_x = event.x
        self.start_y = event.y
        # create rectangle if not yet exist
        self.line_coords.append((self.start_x, self.start_y))
        if len(self.line_coords) >= 2:
            if not self.line:
                self.line = self.app.imageframe.create_line(list(chain.from_iterable(self.line_coords)),
                                                            fill='green', width=3, tag="baseline")
            else:
                self.app.imageframe.coords(self.line, list(chain.from_iterable(self.line_coords)))

    def baseline_create_on_button_press_right(self, event):
        if len(self.line_coords) >= 2:
            self.baselines.append(self.line_coords)
            self.draw_baslines()
        self.reset_bl_creation()

    def reset_bl_creation(self):
        self.line_coords = []
        self.line = None

    def delete_bl(self):
        '''
        Sets up the delete textregion mode.
        :return:
        '''
        if self.lock == States.NO_STATE:
            self.lock = States.DELETE
            self.binder.bind([Binds.de_rec])
            self.app.delete_bl_Button.config(text="stop deleting baselines")
        elif self.lock == States.DELETE:
            self.reset_bl_creation()
            self.lock = States.NO_STATE
            self.binder.bind([Binds.un_rec], tag='textregion_rec')
            self.app.delete_bl_Button.config(text="delete baselines")
        self.draw_baslines()

    def create_baseline(self):
        '''
        Sets up create textregion mode
        :return:
        '''
        if self.lock == States.NO_STATE:
            self.lock = States.NEW_BASELINE
            self.binder.bind([Binds.new_bline])
            self.app.new_baseline_button.config(text="stop creating Baselines")
        elif self.lock == States.NEW_BASELINE:
            self.lock = States.NO_STATE
            self.curr_tr = None
            self.binder.bind([Binds.un_new_bline])
            self.app.new_baseline_button.config(text="create Baselines")

    def baseline_extend_on_button_press(self, event):
        # save mouse drag start position

        if self.edit_item == None:
            item = self.app.imageframe.find_closest(event.x, event.y)
            tags = self.app.imageframe.itemcget(item, "tags")
            if "baseline" in tags:
                self.edit_item = item
                self.app.imageframe.itemconfig(self.edit_item, fill="red")
                self.edit_baseline = self.id_baselines[item[0]]
                self.edit_index = self.baselines.index(self.edit_baseline)
            return
        coord = (event.x, event.y)
        if abs(self.edit_baseline[0][0] - coord[0]) < abs(self.edit_baseline[-1][0] - coord[0]):
            self.edit_baseline.insert(0, coord)
        else:
            self.edit_baseline.append(coord)
        self.app.imageframe.coords(self.edit_item, list(chain.from_iterable(self.edit_baseline)))

    def baseline_extend_on_button_press_right(self, event):
        self.baselines[self.edit_index] = self.edit_baseline
        self.draw_baslines()
        self.reset_edit_creation()

    def reset_edit_creation(self):
        self.edit_item = None
        self.edit_baseline = []
        self.edit_index = None

    def extend_baseline(self):
        '''
        Sets up modify mode
        :return:
        '''
        if self.lock == States.NO_STATE:
            self.lock = States.MODIFY
            self.binder.bind([Binds.ext_baseline])
            self.app.modify_Button.config(text="stop extend baseline")
        elif self.lock == States.MODIFY:
            self.lock = States.NO_STATE
            self.tr_selected = False
            self.binder.bind([Binds.un_ext_baseline])
            self.app.modify_Button.config(text="Extend Basleine")

    def baseline_con_on_button_press(self, event):
        item = self.app.imageframe.find_closest(event.x, event.y)
        self.edit_item = item
        try:  # sometimes error
            self.app.imageframe.itemconfig(self.edit_item, fill="red")
        except:
            pass
        self.baseline_indexes.append(item[0])
        if len(self.baseline_indexes) == 2:
            index1 = self.baselines.index(self.id_baselines[self.baseline_indexes[0]])
            index2 = self.baselines.index(self.id_baselines[self.baseline_indexes[1]])
            min_index = min(index1, index2)  # just sort
            max_index = max(index1, index2)

            connected = self.id_baselines[self.baseline_indexes[0]] + self.id_baselines[self.baseline_indexes[1]]
            self.baselines[min_index] = sorted(connected, key=lambda x: x[0])
            del self.baselines[max_index]
            self.reset_con_creation()
            self.draw_baslines()

    def baseline_con_on_press_right(self, event):
        self.reset_con_creation()
        self.draw_baslines()

    def reset_con_creation(self):
        self.edit_item = None
        self.edit_baseline = []
        self.edit_index = None

    def connect_baseline(self):
        if self.lock == States.NO_STATE:
            self.lock = States.MODIFY
            self.binder.bind([Binds.con_baseline])
            self.app.new_con_baseline_button.config(text="stop com baseline")
        elif self.lock == States.MODIFY:
            self.lock = States.NO_STATE
            self.tr_selected = False
            self.binder.bind([Binds.un_con_baseline])
            self.app.new_con_baseline_button.config(text="combine Basleine")

    def split_baseline_by_rec_on_button_release(self, event):
        baselines_new = []
        if len(self.baselines) == 0:
            return

        def split_lists(list1, list2):
            min_x = list2[0][0]
            max_x = list2[-1][0]
            index_min = []
            index_max = []
            for ind, x in enumerate(list1):
                if min_x > x[0]:
                    index_min.append(ind)
                if x[0] > max_x:
                    index_max.append(ind)
            baseline1 = None
            baseline2 = None
            if len(index_min) >= 1:
                baseline1 = list1[:index_min[-1]] + [(int(list2[0][0]), int(list2[0][1]))]
            if len(index_max) >= 1:
                baseline2 = [(int(list2[-1][0]), int(list2[-1][1]))] + list1[index_max[0]:]

            return baseline1, baseline2

        coords = self.app.imageframe.coords(self.rect)
        box_s = box(coords[0], coords[1], coords[2], coords[3])
        for ind, baseline in enumerate(self.baselines):
            base_l = baseline
            if len(baseline) < 2:
                continue
            linestring_s = LineString(baseline)
            if box_s.intersects(linestring_s):
                inters = box_s.intersection(linestring_s)

                if inters.type != "LineString":
                    continue
                coords = list(inters.coords)

                b1, b2 = split_lists(base_l, coords)
                if b1 is None or b2 is None:
                    baselines_new.append(baseline)
                else:
                    baselines_new.append(b1)
                    baselines_new.append(b2)
            else:
                baselines_new.append(baseline)

            pass
        self.baselines = baselines_new
        self.imageframe.delete("delrec")
        self.rect = None
        self.draw_baslines()

    def split_baseline(self):
        '''
        Sets up modify mode
        :return:
        '''
        if self.lock == States.NO_STATE:
            self.lock = States.MODIFY
            self.binder.bind([Binds.split_baseline])
            self.app.new_split_baseline_button.config(text="stop split baseline")
        elif self.lock == States.MODIFY:
            self.lock = States.NO_STATE
            self.tr_selected = False
            self.binder.bind([Binds.un_split_baseline])
            self.app.new_split_baseline_button.config(text="Split Basleine")

    def get_next_image(self):
        '''
        displays next image
        :return:
        '''
        if self.lock == States.NO_STATE or self.lock == States.LOAD_IMAGE:
            files = list(multiple_file_types(os.path.join(self.directory,"*png"),os.path.join(self.directory, "*jpg")))
            index = files.index(os.path.join(self.directory,self.im_name)+ "." + self.suffix)
            index = index + 1 if len(files) - 1 > index else 0
            filename = files[index + 1]
            pass
            # dir_path = fd.askdirectory()
            # self.save_directory = dir_path
            thread = threading.Thread(target=self.load_img,
                                      name="Thread1",
                                      args=[filename])
            thread.start()
    def set_save_dir(self):
        '''
        sets up save directory mode
        :return:
        '''
        if self.lock == States.NO_STATE or self.lock == States.LOAD_IMAGE:
            dir_path = fd.askdirectory()
            self.save_directory = dir_path

    def save(self):
        '''
        sets up save mode
        :return:
        '''
        if self.lock == States.NO_STATE:
            if self.save_directory is None or self.save_directory == "":
                return
            self.extract_xml_info()

    def extract_xml_info(self):
        import copy
        cop_baselines = copy.deepcopy(self.baselines)
        cop_baselines = [x for x in cop_baselines if len(x) > 0]
        scale_baselines(cop_baselines, 1 / self.scale)

        xmlgen = XMLGenerator(self.im_width, self.im_height, self.im_name, cop_baselines)

        print(xmlgen.baselines_to_xml_string())
        xmlgen.save_textregions_as_xml(self.save_directory)
        pass


# Holds different Lock-States
class States(Enum):
    '''
    Saves the current GUI State
    '''
    NO_STATE = 0
    NEW_TEXTREGION = 1
    NEW_BASELINE = 2
    MODIFY = 3
    FULL_LOCK = 4
    LOAD_IMAGE = 5
    DELETE = 6


class Coord:
    '''
    Class which represents a coordinate. Can store the neighbouring coordinates.
    '''

    def __init__(self, coord):
        self.right = None
        self.left = None
        self.coord = coord

    def invert_left(self, recdict, linedict):
        temp = self.left
        self.left = self.right
        self.right = temp
        if self.right is not None:
            rec2 = recdict[self.right.coord]
            rec1 = recdict[self.coord]
            linedict[(rec1, rec2)] = linedict[(rec2, rec1)]
            linedict.pop((rec2, rec1))
            self.right.invert_left(recdict, linedict)

    def invert_right(self, recdict, linedict):
        temp = self.right
        self.right = self.left
        self.left = temp
        if self.left is not None:
            rec1 = recdict[self.left.coord]
            rec2 = recdict[self.coord]
            linedict[(rec1, rec2)] = linedict[(rec2, rec1)]
            linedict.pop((rec2, rec1))
            self.left.invert_right(recdict, linedict)


class Binds(Enum):
    '''
    Enum that connects binding names with int values
    '''
    # Bindings
    un_right = 24
    de_rec = 25
    un_rec = 26
    new_bline = 27
    un_new_bline = 28
    ext_baseline = 29
    un_ext_baseline = 30
    con_baseline = 31
    un_con_baseline = 32
    split_baseline = 33
    un_split_baseline = 34


class Binder:
    '''
        Class which lets easily remove or add new key and mousebindings
    '''

    def __init__(self, root, imageframe, controller):
        self.root = root
        self.imfr = imageframe
        self.cont = controller

    def bind(self, bind_list: List[Binds], tag: str = ""):
        print(list)
        if Binds.de_rec in bind_list:
            self.imfr.bind("<ButtonPress-1>", self.cont.baseline_delete_rec_on_button_press)
            self.imfr.bind("<B1-Motion>", self.cont.baseline_delete_rec_on_move_press)
            self.imfr.bind("<ButtonRelease-1>", self.cont.baseline_delete_rec_on_button_release)
        if Binds.un_rec in bind_list:
            self.imfr.unbind("<ButtonPress-1>")
            self.imfr.unbind("<B1-Motion>")
            self.imfr.unbind("<ButtonRelease-1>")
        if Binds.new_bline in bind_list:
            self.imfr.bind("<ButtonPress-1>", self.cont.baseline_create_on_button_press)
            self.imfr.bind("<ButtonPress-3>", self.cont.baseline_create_on_button_press_right)
        if Binds.un_new_bline in bind_list:
            self.imfr.unbind("<ButtonPress-1>")
            self.imfr.unbind("<ButtonPress-3>")
        if Binds.ext_baseline in bind_list:
            self.imfr.bind("<ButtonPress-1>", self.cont.baseline_extend_on_button_press)
            self.imfr.bind("<ButtonPress-3>", self.cont.baseline_extend_on_button_press_right)
        if Binds.un_ext_baseline in bind_list:
            self.imfr.unbind("<ButtonPress-1>")
            self.imfr.unbind("<ButtonPress-3>")
        if Binds.con_baseline in bind_list:
            self.imfr.tag_bind("baseline", "<ButtonPress-1>", self.cont.baseline_con_on_button_press)
            self.imfr.bind("<ButtonPress-3>", self.cont.baseline_con_on_press_right)

        if Binds.un_con_baseline in bind_list:
            self.imfr.tag_unbind("baseline", "<ButtonPress-1>")
            self.imfr.bind("<ButtonPress-3>")

        if Binds.split_baseline in bind_list:
            self.imfr.bind("<ButtonPress-1>", self.cont.baseline_delete_rec_on_button_press)
            self.imfr.bind("<B1-Motion>", self.cont.baseline_delete_rec_on_move_press)
            self.imfr.bind("<ButtonRelease-1>", self.cont.split_baseline_by_rec_on_button_release)
        if Binds.un_split_baseline in bind_list:
            self.imfr.unbind("<ButtonPress-1>")
            self.imfr.unbind("<B1-Motion>")
            self.imfr.unbind("<ButtonRelease-1>")


def start_GUI(network):
    '''
    Starts the GUI
    :return:
    '''
    root = Tk()
    Controller(root, network)
    root.mainloop()


def scale_baselines(baselines, scale_factor=1.0):
    for b_idx, bline in enumerate(baselines):
        for c_idx, coord in enumerate(bline):
            coord = (int(coord[0] * scale_factor), int(coord[1] * scale_factor))
            baselines[b_idx][c_idx] = coord


if __name__ == "__main__":
    from segmentation.dataset import dirs_to_pandaframe, load_image_map_from_file, XMLDataset, MaskDataset
    from segmentation.network import MaskSetting, PredictorSettings, Network, MaskType, PCGTSVersion, compose, \
        MaskGenerator

    f = dirs_to_pandaframe(
        ['/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/train/image/'],
        ['/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/train/page/'])

    e = dirs_to_pandaframe(
        ['/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/test/image/'],
        ['/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/test/page/'])

    c = dirs_to_pandaframe(
        ['/home/alexander/Dokumente/HBR2013/images/'],
        ['/home/alexander/Dokumente/HBR2013/masks/']
    )

    map = load_image_map_from_file(
        '/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/dataset-test/image_map.json')
    from segmentation.dataset import base_line_transform

    settings = MaskSetting(MASK_TYPE=MaskType.BASE_LINE, PCGTS_VERSION=PCGTSVersion.PCGTS2013, LINEWIDTH=5,
                           BASELINELENGTH=10)
    dt = XMLDataset(c, map, transform=compose([base_line_transform()]),
                    mask_generator=MaskGenerator(settings=settings))
    d_test = XMLDataset(c, map, transform=compose([base_line_transform()]),
                        mask_generator=MaskGenerator(settings=settings))
    import pandas as pd

    pd.set_option('display.max_colwidth', -1)  # or 199
    d_predict = MaskDataset(f, map)
    # transform=compose([base_line_transform()]))  # transform=compose([base_line_transform()]))
    from segmentation.settings import TrainSettings

    setting = TrainSettings(CLASSES=len(map), TRAIN_DATASET=dt, VAL_DATASET=d_test,
                            OUTPUT_PATH="/home/alexanderh/PycharmProjects/segmentation-pytorch/data/effnet2.torch",
                            MODEL_PATH='/home/alexanderh/PycharmProjects/segmentation-pytorch/data/effnet.torch.torch')
    p_setting = PredictorSettings(PREDICT_DATASET=d_predict,
                                  MODEL_PATH='/home/alexander/PycharmProjects/segmentation_pytorch/models/effnet2.torch.torch')
    trainer = Network(p_setting, color_map=map)
    start_GUI(network=trainer)
