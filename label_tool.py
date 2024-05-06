import cv2
import os

ROOT_PATH = 'to_label/'
SAVE_PATH = 'labeled/'
PT_COLOR = (0, 0, 255)
PT_R = 5
class AnnotatedImage:
    def __init__(self, fname, in_dir=ROOT_PATH):
        # in_dir should end in /
        self.points = []
        self.redo_cache = []
        self.fname = fname
        self.raw = cv2.imread(in_dir+fname)
        self.anno = self.raw.copy()
    
    def _draw_circle(self, point, r=PT_R, color=PT_COLOR):
        # pt should be (x,y)
        self.anno = cv2.circle(self.anno, point, radius=PT_R, color=PT_COLOR, thickness=-1)

    def clicked(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt = (x,y)
            self.points.append(pt)
            self._draw_circle(pt)
            self.redo_cache = []
            cv2.imshow("image", self.anno)
    
    def undo(self):
        if len(self.points) == 0:
            return
        undo_point = self.points.pop()
        self.redo_cache.append(undo_point)
        self.anno = self.raw.copy()
        # redraw all but the last point
        for pt in self.points:
            self._draw_circle(pt)
        cv2.imshow("image", self.anno)
    
    def redo(self):
        if len(self.redo_cache) == 0:
            return
        pt = self.redo_cache.pop()
        self.points.append(pt)
        self._draw_circle(pt)
        cv2.imshow("image", self.anno)
    
    def save(self, out_dir=SAVE_PATH, verbose=False):
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

def main(in_dir=ROOT_PATH, out_dir=SAVE_PATH):
    files = [f for f in os.listdir(in_dir) if os.path.isfile(ROOT_PATH+f)]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    cv2.namedWindow("image")
    for i,fn in enumerate(files):
        print(f'Annotating {fn} [{i+1}/{len(files)}]')
        anno_img = AnnotatedImage(fn, in_dir=in_dir)
        cv2.setMouseCallback("image", anno_img.clicked)
        key_was_released = True
        while True:
            cv2.imshow("image", anno_img.anno)
            key = cv2.waitKey(1)# & 0xFF
            if key == ord('q'):
                print('quitting without saving')
                return
            elif key == ord('n'):
                anno_img.save(out_dir=out_dir)
                print('moving to next image')
                break
            elif key == ord('s') and key_was_released:
                anno_img.save(out_dir=out_dir)
            elif key == ord('z') and key_was_released:
                anno_img.undo()
            elif key == ord('r') and key_was_released:
                anno_img.redo()
            key_was_released = (key == -1)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()