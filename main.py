from GUI import MainWindow

from PyQt5.QtWidgets import QApplication, QStyleFactory
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    QApplication.setStyle(QStyleFactory.create('Plastique'))
    myGUI = MainWindow()
    sys.exit(app.exec_())
