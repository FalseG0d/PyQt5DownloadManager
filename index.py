from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys

from PyQt5.uic import loadUiType
import urllib.request
from pafy import *
import humanize

ui,_=loadUiType('main.ui')

class MainApp(QMainWindow, ui):
    def __init__(self,parent=None):
        super(MainApp,self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.InitUI()
        self.Handle_Buttons()

    def InitUI(self):
        pass

    def Handle_Buttons(self):
        self.pushButton.clicked.connect(self.Download)
        self.pushButton_2.clicked.connect(self.Handle_Browse)
        self.pushButton_7.clicked.connect(self.Get_Video_Data)

    def Handle_Progress(self,blocknum,blocksize,totalsize):
        readed_data=blocknum*blocksize

        if totalsize>0:
            download_percentage=readed_data*100/totalsize
            self.progressBar.setValue(download_percentage)
            QApplication.processEvents()

    def Handle_Browse(self):
        save_location=QFileDialog.getSaveFileName(self,caption="Save As",directory="D:",filter="All Files (*.*)")

        self.lineEdit_2.setText(str(save_location[0]))

    def Download(self):
        print("Starting Download")
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

        self.lineEdit.setText('')
        self.lineEdit_2.setText('')
        self.progressBar.setValue(0)

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
        pass

    def Video_Progress(self):
        pass

    def Save_Browse(self):
        pass

def main():
    app=QApplication(sys.argv)
    window=MainApp()
    window.show()
    app.exec_()

if __name__=='__main__':
    main()