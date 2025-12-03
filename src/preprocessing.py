import cv2, numpy as np

class Preprocessing():
    """
    This class is used to preprocess the images from the camera.
    resize:         resize the image to the desired size according the soccer field dimensions.
    perspective:    apply a perspective transformation to the image. If inverted is True a mirror effect is applied.
    corners:        draw black rectangles in the corners of the image to hide undesired information.
    """
    def __init__(self, fx=150.0, fy=130.0, fgx=10.0, fgy=40.0, points=None, inverted=False):
        scale = 4.0
        self.fx = np.int32(fx*scale)
        self.fy = np.int32(fy*scale)
        self.fgx = np.int32(fgx*scale)
        self.fgy = np.int32(fgy*scale)
        self.fsize = (self.fx+2*self.fgx, self.fy)
        if points is not None: self.points = np.asarray(points)
        
        self.inverted = inverted

    def resize(self, frame):
        # cv2.INTER_NEAREST is used only to resize a bigger image to a smaller one
        # return cv2.resize(frame, self.fsize, interpolation=cv2.INTER_NEAREST)
        return cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    
    def perspective(self, frame):
        if not self.inverted: pts1 = np.float32([self.points[0], self.points[1], self.points[2], self.points[3]])
        else: pts1 = np.float32([self.points[2], self.points[3], self.points[0], self.points[1]])
        # pts2 = np.float32([[self.fgx, 0],[self.fgx, self.fy],[self.fgx+self.fx, self.fy], [self.fgx+self.fx, 0]])
        pts2 = np.float32([[0,0], [0, frame.shape[0]], [frame.shape[1], frame.shape[0]], [frame.shape[1], 0]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(frame, M, (frame.shape[1], frame.shape[0]))
    
    def corners(self, dst, offset=0):
        # offset modifies a little bit the size of the black rectangles
        cv2.rectangle(dst, (0,0), (self.fgx-offset, (self.fy-self.fgy)//2-offset), (0,0,0), -1)
        cv2.rectangle(dst, (self.fgx+self.fx+offset, 0), (2*self.fgx+self.fx, (self.fy-self.fgy)//2-offset), (0,0,0), -1)
        cv2.rectangle(dst, (0, (self.fy+self.fgy)//2+offset), (self.fgx-offset, self.fy), (0,0,0), -1)
        cv2.rectangle(dst, (self.fgx+self.fx+offset, (self.fy+self.fgy)//2+offset), (self.fgx*2+self.fx, self.fy), (0,0,0), -1)
        
class Points:
    """
    Points class is used to get the points of the soccer field corners to perform the perspective transformation.
    The params of the camera are set, the correction is applied to the image and then the points are selected by clicking on the image.
    The order of the points is: top-left, bottom-left, bottom-right, top-right.
    """
    def __init__(self, video_input, fix_distortion=None):
        self.pre = Preprocessing()
        self.points = []
        self.num_points = 0
        self.fix_distortion = fix_distortion
        self.camera = cv2.VideoCapture(video_input)
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.image_callback()
        
    def image_callback(self):
        for i in range(120):
            ret, self.frame = self.camera.read()
            if not ret: return
        if self.fix_distortion is not None:
            if self.fix_distortion == 'fisheye': self.frame = self.calibration.fish_eye_correction(self.frame)
            else: self.frame = self.calibration.normal_correction(self.frame)
        self.frame = self.pre.resize(self.frame)
        cv2.imshow('frame', self.frame)
        cv2.setMouseCallback('frame', self.mouse_click)
        cv2.waitKey()
        self.stop()
        
    def mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.num_points += 1
            self.points.append([x,y])
            cv2.circle(self.frame, (x,y), 2, (0, 0, 255), 2)
            cv2.imshow('frame', self.frame)
            if self.num_points == 4:
                self.pre.points = self.points
                dst = self.pre.perspective(self.frame)
                cv2.imshow('frame', dst)
                print(f"points = {self.points}")
                
    def stop(self):
        self.camera.release()
        cv2.destroyAllWindows()
        
def main():
    video_input = "real_data_regras/thechosenone.mp4"
    Points(video_input)

if __name__ == "__main__":
    main()