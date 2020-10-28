from modules.facecv import FaceCV,get_args
from modules.mainapp import MainApp,QApplication
import sys


def main():
    #Face Recognition
    args = get_args()
    depth = args.depth
    width = args.width

    face = FaceCV(depth=depth, width=width)

    predicted_ages,predicted_genders=face.detect_face()
    print(predicted_ages)
    print(predicted_genders)

    #Window Forming
    app=QApplication(sys.argv)

    window=MainApp(age=predicted_ages,gender=predicted_genders)
    
    window.show()
    app.exec_()

if __name__=='__main__':
    main()