import cv2
import numpy as np
import os

RAW_PATH = 'openimgv7/'
CROP_PATH = 'cropped/'
LABEL_PATH = 'labeled/'
PT_COLOR = (0, 0, 255)
PT_R = 2
LABEL_WIN_SIZE = 800
RECT_COLOR = (0, 0, 255)
RECT_THICKNESS = 2
MIN_IMG_DIM = 128

class LabelableImage:
    def __init__(self, fname, in_dir=RAW_PATH):
        # in_dir should end in /
        self.points = []
        self.redo_cache = []
        self.fname = fname
        self.base = cv2.imread(in_dir+fname)
        self.anno = self.base.copy()
    
    def _draw_circle(self, point, r=PT_R, color=PT_COLOR):
        # pt should be (x,y)
        self.anno = cv2.circle(self.anno, point, radius=PT_R, color=PT_COLOR, thickness=-1)

    def clicked(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt = (x,y)
            self.points.append(pt)
            self._draw_circle(pt)
            self.redo_cache = []
    
    def undo(self):
        if len(self.points) == 0:
            return
        undo_point = self.points.pop()
        self.redo_cache.append(undo_point)
        self.anno = self.base.copy()
        # redraw all but the last point
        for pt in self.points:
            self._draw_circle(pt)
    
    def redo(self):
        if len(self.redo_cache) == 0:
            return
        pt = self.redo_cache.pop()
        self.points.append(pt)
        self._draw_circle(pt)
    
    def save(self, out_dir=LABEL_PATH, verbose=False):
        # out_dir should end in /
        extension_idx = self.fname.rfind('.') # find index of extension
        file_prefix, ext = self.fname[:extension_idx], self.fname[extension_idx:]
        prefix = out_dir+file_prefix
        label_str = '\n'.join([f"{x} {y}" for (x,y) in self.points])
        with open(f"{prefix}_lbl.txt", 'w') as f:
            f.write(label_str)
        cv2.imwrite(f"{prefix}_anno{ext}", self.anno)
        if verbose:
            print(f'data saved for labeled version of {self.fname}')

class CropMarkableImage:
    # should resize to be 256x256
    def __init__(self, fname, in_dir=RAW_PATH):
        self.fname = fname
        self.base = cv2.imread(in_dir+fname)
        self.buffer = None
        self.anno = self.base.copy()
        self.ref_pts = []
        self.redo_cache = []
        self.is_cropping = False
    
    def _draw_rect(self, ref_pt_pair, color=RECT_COLOR, thickness=RECT_THICKNESS):
        cv2.rectangle(self.anno, ref_pt_pair[0], ref_pt_pair[1], color, thickness)

    def drag_and_crop(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ref_pts.append([(x,y)])
            self.is_cropping = True
            self.buffer = self.anno.copy()
        elif event == cv2.EVENT_MOUSEMOVE and self.is_cropping:
            self.anno = self.buffer.copy()
            self._draw_rect(self.ref_pts[-1]+[(x,y)])
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            self.ref_pts[-1].append((x,y))
            self.is_cropping = False
            # draw a rectangle around the region of interest
            self.redo_cache = []
            self._draw_rect(self.ref_pts[-1])

    def undo(self):
        if len(self.ref_pts) < 1:
            return
        undo_ref_pt_obj = self.ref_pts.pop()
        if len(undo_ref_pt_obj) == 2:
            self.redo_cache.append(undo_ref_pt_obj)
        self.anno = self.base.copy()
        for ref_pt_pair in self.ref_pts:
            self._draw_rect(ref_pt_pair)
    
    def redo(self):
        if len(self.redo_cache) < 1:
            return
        ref_pt_pair = self.redo_cache.pop()
        self.ref_pts.append(ref_pt_pair)
        self._draw_rect(ref_pt_pair)
    
    def save_crops(self, out_dir=CROP_PATH, verbose=False):
        if self.is_cropping:
            print('ERROR: cannot save while attempting to crop')
            return
        extension_idx = self.fname.rfind('.') # find index of extension
        file_prefix, ext = self.fname[:extension_idx], self.fname[extension_idx:]
        prefix = out_dir+file_prefix
        for i, [(x1, y1), (x2,y2)] in enumerate(self.ref_pts):
            xL, xR = sorted((x1,x2))
            yB, yT = sorted((y1,y2))
            roi = self.base[yB:yT, xL:xR]
            # resize if minimum dim is less than 128
            min_dim = min(roi.shape[0], roi.shape[1])
            scale = max(1, MIN_IMG_DIM/min_dim)
            scale_up = lambda l: int(np.ceil(scale*l))
            roi_out = cv2.resize(roi, (scale_up(roi.shape[1]), scale_up(roi.shape[0])))
            cv2.imwrite(f"{prefix}_crop{i}{ext}", roi_out)
        if verbose:
            print(f'saved {len(self.ref_pts)} crop(s) for {self.fname}')
        # reset
        self.ref_pts = []
        self.anno = self.base.copy()

def get_files(path):
    if not path.endswith('/'):
        path = path+'/'
    return [f for f in os.listdir(path) if os.path.isfile(path+f) and not f.startswith('.')]

def label_images(in_dir=CROP_PATH, out_dir=LABEL_PATH, win_name="LABEL_MODE"):
    files = get_files(in_dir)
    done_dir = out_dir
    for dir_name in [out_dir, done_dir]:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL) 
    for i,fn in enumerate(files):
        print(f'Annotating {fn} [{i+1}/{len(files)}]')
        anno_img = LabelableImage(fn, in_dir=in_dir)
        cv2.setMouseCallback(win_name, anno_img.clicked)
        key_was_released = True
        while True:
            cv2.imshow(win_name, anno_img.anno)
            cv2.resizeWindow(win_name, LABEL_WIN_SIZE, LABEL_WIN_SIZE)
            key = cv2.waitKey(1)# & 0xFF
            if key == ord('q'):
                print('quitting without saving')
                cv2.destroyAllWindows()
                return
            elif key == ord('t') and key_was_released:
                print('trashing')
                break
            elif key == ord('n') and key_was_released:
                anno_img.save(out_dir=out_dir)
                print('moving to next image')
                break
            elif key == ord('s') and key_was_released:
                anno_img.save(out_dir=out_dir)
            elif (key == ord('z') or key == ord('b')) and key_was_released:
                anno_img.undo()
            elif key == ord('r') and key_was_released:
                anno_img.redo()
            key_was_released = (key == -1)
        os.rename(in_dir+fn, done_dir+fn)
    cv2.destroyAllWindows()

def crop_images(in_dir=RAW_PATH, out_dir=CROP_PATH, win_name="CROP_MODE"):
    files = get_files(in_dir)
    done_dir = in_dir+"_done/"
    for dir_name in [out_dir, done_dir]:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
    cv2.namedWindow(win_name)
    for i,fn in enumerate(files):
        print(f'Annotating {fn} [{i+1}/{len(files)}]')
        anno_img = CropMarkableImage(fn, in_dir=in_dir)
        cv2.setMouseCallback(win_name, anno_img.drag_and_crop)
        key_was_released = True
        while True:
            cv2.imshow(win_name, anno_img.anno)
            key = cv2.waitKey(1)# & 0xFF
            if key == ord('q'):
                print('quitting without saving')
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                anno_img.save_crops(out_dir=out_dir)
                print('moving to next image')
                break
            elif key == ord('c') and key_was_released:
                anno_img.save_crops(out_dir=out_dir)
            elif key == ord('z') and key_was_released:
                anno_img.undo()
            elif key == ord('r') and key_was_released:
                anno_img.redo()
            key_was_released = (key == -1)
        os.rename(in_dir+fn, done_dir+fn)
    cv2.destroyAllWindows()


def main():
    # crop_images()
    label_images()
    # print(f"Cropping: {len(get_files(RAW_PATH+'_done'))} done, {len(get_files(RAW_PATH))} left")
    print(f"Labeling: {len(get_files(LABEL_PATH))//2} done, {len(get_files(CROP_PATH))} left")

if __name__ == '__main__':
    main()