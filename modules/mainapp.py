
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from PyQt5.uic import loadUiType
import urllib.request
import pafy
import humanize

import os

ui,_=loadUiType('main.ui')


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
            self.ApplyLight()
        elif(age<20):
            self.ApplyOrange()
        elif(age<30):
            self.ApplyQDark()
        elif(age<40):
            self.ApplyDark()
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