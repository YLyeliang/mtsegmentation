import cv2


class video_io(object):
    """
    A video readline and writeline process,which can be modified by
    any other functions.
    """

    def __init__(self, read_path, write=None, model=None, video_type='mp4'):
        self.read_path = read_path
        self.write = write
        self.video_type = video_type
        self.model = model

    def init_param(self):
        cap = cv2.VideoCapture(self.read_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read_video(self):
        if self.write is not None:
            self.write_video()
        cv2.namedWindow("video", cv2.WINDOW_NORMAL)
        cap = cv2.VideoCapture(self.read_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                print("start reading video.")

                cv2.imshow("video", frame)
                key = cv2.waitKey(20)
            else:
                break
            if key == ord('q'):
                break
        print("end reading video.")

    def write_video(self):
        self.VideoWriter = cv2.VideoWriter(self.write, cv2.CAP_PROP_FOURCC('M', 'J', 'P', 'G'), self.fps, self.size)
