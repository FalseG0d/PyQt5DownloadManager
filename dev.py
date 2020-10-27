from modules.mainapp import MainApp,QApplication
import sys


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