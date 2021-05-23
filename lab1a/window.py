from PyQt5 import QtWidgets
import factorization, interface.mainw
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap

def showError(text):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText("Something went wrong")
    msg.setInformativeText(text)
    msg.setWindowTitle("Error")
    msg.exec_()


class MyApp(QtWidgets.QMainWindow, interface.mainw.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.pushButton.clicked.connect(self.solve_btn)

    def solve_btn(self):
        try:
            n = int(self.lineEdit.text())
            if n % 2 == 0:
                showError("Number should be odd")
            else:
                result = factorization.cycle_fermat(n)
                if len(result) == 1:
                    self.textBrowser.clear()
                    self.textBrowser.append("Number is already prime")
                else:
                    text = "n = "
                    for i in result:
                        text += "{} \u00b7 ".format(i)
                    text = text[:-2]
                    self.textBrowser.clear()
                    self.textBrowser.append(text)
        except:
            showError("Check input data")
