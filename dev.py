from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys

from PyQt5.uic import loadUiType
import urllib.request
import pafy
import humanize

import os

ui,_=loadUiType('main.ui')


"""
Face detection
"""
import cv2
import os
from time import sleep
import numpy as np
import argparse
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file


class FaceCV(object):
    """
    Singleton class for face recongnition task
    """
    CASE_PATH = ".\\pretrained_models\\haarcascade_frontalface_alt.xml"
    WRN_WEIGHTS_PATH = "https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5"


    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        fpath = get_file('weights.18-4.06.hdf5',
                         self.WRN_WEIGHTS_PATH,
                         cache_subdir=model_dir)
        self.model.load_weights(fpath)

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def crop_face(self, imgarray, section, margin=40, size=64):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def detect_face(self):
        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)
        predicted_genders="0"
        predicted_ages=0

        # 0 means the default video capture device in OS
        video_capture = cv2.VideoCapture(0)
        # infinite loop, break by key ESC
        while True:
            if not video_capture.isOpened():
                sleep(5)
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(self.face_size, self.face_size)
            )
            # placeholder for cropped faces
            face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))

            for i, face in enumerate(faces):
                face_img, cropped = self.crop_face(frame, face, margin=40, size=self.face_size)
                (x, y, w, h) = cropped
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                face_imgs[i,:,:,:] = face_img
            
            if len(face_imgs) > 0:
                # predict ages and genders of the detected faces
                results = self.model.predict(face_imgs)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()
            
            # draw results
            for i, face in enumerate(faces):
                label = "{}, {}".format(int(predicted_ages[i]),
                                        "F" if predicted_genders[i][0] > 0.5 else "M")
                self.draw_label(frame, (face[0], face[1]), label)

            #cv2.imshow('Keras Faces', frame)
            if predicted_ages != 0:
                break

            if cv2.waitKey(5) == 27:  # ESC key press
                break
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()

        return predicted_ages[0],predicted_genders[0][1]


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    args = parser.parse_args()
    return args


class MainApp(QMainWindow, ui):
    def __init__(self,parent=None,age=20,gender=20):
        super(MainApp,self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.InitUI(age,gender)
        self.Handle_Buttons()
        self.age=age
        self.gender=gender

    def InitUI(self,age,gender):
        self.tabWidget.tabBar().setVisible(False)

        if(age<10):
            self.ApplyDark()
        elif(age<20):
            self.ApplyLight()
        elif(age<30):
            self.ApplyOrange()
        elif(age<40):
            self.ApplyQDark()
        else:
            self.ApplyNone()

    def Handle_Buttons(self):
        self.pushButton.clicked.connect(self.Download)
        self.pushButton_2.clicked.connect(self.Handle_Browse)

        self.pushButton_7.clicked.connect(self.Get_Video_Data)
        self.pushButton_6.clicked.connect(self.Save_Browse)
        self.pushButton_5.clicked.connect(self.Download_Video)

        self.pushButton_10.clicked.connect(self.Playlist_Download)
        self.pushButton_8.clicked.connect(self.Playlist_Browse)

        self.pushButton_3.clicked.connect(self.Open_Home)
        self.pushButton_4.clicked.connect(self.Open_Youtube)
        self.pushButton_9.clicked.connect(self.Open_Download)
        self.pushButton_11.clicked.connect(self.Open_Settings)

        self.pushButton_12.clicked.connect(self.ApplyLight)
        self.pushButton_13.clicked.connect(self.ApplyOrange)
        self.pushButton_14.clicked.connect(self.ApplyQDark)
        self.pushButton_15.clicked.connect(self.ApplyDark)
        self.pushButton_16.clicked.connect(self.ApplyNone)

    def Handle_Progress(self,blocknum,blocksize,totalsize):
        readed_data=blocknum*blocksize

        if totalsize>0:
            download_percentage=readed_data*100/totalsize
            self.progressBar.setValue(download_percentage)
            Mainlication.processEvents()

    def Handle_Browse(self):
        save_location=QFileDialog.getSaveFileName(self,caption="Save As",directory="D:",filter="All Files (*.*)")

        self.lineEdit_2.setText(str(save_location[0]))

    def Download(self):
        
        download_url=self.lineEdit.text()
        save_location=self.lineEdit_2.text()

        try:
            urllib.request.urlretrieve(download_url,save_location,self.Handle_Progress)
        except ValueError:
            QMessageBox.warning(self,"Data Error","The Provided URL or Save Location is invalid.")
            return
        except Exception:
            QMessageBox.warning(self,"Download Error",str(Exception))
            return

        QMessageBox.warning(self,"Download Complete","File was Downloaded Successfully.")
        

    def Get_Video_Data(self):
        
        video_url=self.lineEdit_6.text()
        video=pafy.new(video_url)
        try:
            video=pafy.new(video_url)

        except ValueError:
            QMessageBox.warning(self,"Incorrect Url","Url Provided is Incorrect.")
            return

        except Exception:
            QMessageBox.warning(self,"Unknown Error",str(Exception))
            return

        video_stream=video.videostreams

        for stream in video_stream:
            size=humanize.naturalsize(stream.get_filesize())
            data="{} {} {} {}".format(stream.mediatype,stream.extension,stream.quality,size)
            self.comboBox.addItem(data)
            


    def Download_Video(self):
        video_url=self.lineEdit_6.text()
        save_location=self.lineEdit_5.text()
        
        video=pafy.new(video_url)
        video_stream=video.videostreams
        video_quality=self.comboBox.currentIndex()

        try:            
            download=video_stream[video_quality].download(filepath=save_location,callback=self.Video_Progress)

        except ValueError:
            QMessageBox.warning(self,"Incorrect Url","Url Provided is Incorrect.")
            return

        except Exception:
            QMessageBox.warning(self,"Unknown Error",str(Exception))
            return

        QMessageBox.warning(self,"Download Complete","Video was downloaded Successfully.")

    def Video_Progress(self,total,received,ratio,rate,time):
        readed_data=received

        if total>0:
            download_percentage=readed_data*100/total
            self.progressBar_3.setValue(download_percentage)
            remaining_time=round(time/60,2)

            self.label_5.setText("{} minutes remaining".format(remaining_time))
            QApplication.processEvents()


    def Save_Browse(self):
        save_location=QFileDialog.getSaveFileName(self,caption="Save As",directory="D:",filter="All Files (*.*)")

        self.lineEdit_5.setText(str(save_location[0]))

    def Playlist_Download(self):
        playlist_url=self.lineEdit_7.text()
        save_location=self.lineEdit_8.text()

        try:
            playlist=pafy.get_playlist(playlist_url)

        except ValueError:
            QMessageBox.warning(self,"Incorrect Url","Url Provided is Incorrect.")
            return

        except Exception:
            QMessageBox.warning(self,"Unknown Error",str(Exception))
            return

        playlist_videos=playlist['items']
        self.lcdNumber_2.display(len(playlist_videos))

        os.chdir(save_location)

        if os.path.exists(str(playlist['title'])):
            os.chdir(str(playlist['title']))

        else:
            os.mkdir(str(playlist['title']))
            os.chdir(str(playlist['title']))

        current_video_in_download=1
        quality=self.comboBox_2.currentIndex()

        QApplication.processEvents()

        for video in playlist_videos:
            self.lcdNumber.display(current_video_in_download)

            current_video=video['pafy']
            current_video_stream=current_video.videostreams

            download=current_video_stream[quality].download(callback=self.Playlist_Progress)
            QApplication.processEvents()

            current_video_in_download+=1

        QMessageBox.warning(self,"Download Complete","Playlist was downloaded Successfully.")

    def Playlist_Browse(self):
        save_location=QFileDialog.getExistingDirectory(self,caption="Save As",directory="D:")

        self.lineEdit_8.setText(str(save_location))

    def Playlist_Progress(self,total,received,ratio,rate,time):
        readed_data=received

        if total>0:
            download_percentage=readed_data*100/total
            self.progressBar_4.setValue(download_percentage)
            remaining_time=round(time/60,2)

            self.label_6.setText("{} minutes remaining".format(remaining_time))
            QApplication.processEvents()

    def Open_Home(self):
        self.tabWidget.setCurrentIndex(0)

    def Open_Download(self):
        self.tabWidget.setCurrentIndex(1)

    def Open_Youtube(self):
        self.tabWidget.setCurrentIndex(2)

    def Open_Settings(self):
        self.tabWidget.setCurrentIndex(3)

    def ApplyOrange(self):
        self.setStyleSheet(None)
        style=open('themes/orange.css','r')
        style=style.read()
        self.setStyleSheet(style)

    def ApplyDark(self):
        self.setStyleSheet(None)
        style=open('themes/dark.css','r')
        style=style.read()
        self.setStyleSheet(style)
        
    def ApplyQDark(self):
        self.setStyleSheet(None)
        style=open('themes/qdark.css','r')
        style=style.read()
        self.setStyleSheet(style)

    def ApplyLight(self):
        self.setStyleSheet(None)
        style=open('themes/light.css','r')
        style=style.read()
        self.setStyleSheet(style)

    def ApplyNone(self):
        self.setStyleSheet(None)


def main():

    predicted_ages=int(input("[*] Enter Age:"))
    predicted_genders=input("[*] Enter Gender:")

    #Window Forming
    app=QApplication(sys.argv)
    window=MainApp(age=predicted_ages,gender=predicted_genders)
    window.show()
    app.exec_()

if __name__=='__main__':
    main()