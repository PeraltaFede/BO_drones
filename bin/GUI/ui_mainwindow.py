# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 5.15.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap, QRadialGradient)
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(922, 532)
        self.actionGuardar = QAction(MainWindow)
        self.actionGuardar.setObjectName(u"actionGuardar")
        self.actionReiniciar = QAction(MainWindow)
        self.actionReiniciar.setObjectName(u"actionReiniciar")
        self.actionSalir = QAction(MainWindow)
        self.actionSalir.setObjectName(u"actionSalir")
        self.actionMapa = QAction(MainWindow)
        self.actionMapa.setObjectName(u"actionMapa")
        self.actionMapa.setCheckable(True)
        self.actionMapa.setChecked(True)
        self.actionPredicci_n_GP = QAction(MainWindow)
        self.actionPredicci_n_GP.setObjectName(u"actionPredicci_n_GP")
        self.actionPredicci_n_GP.setCheckable(True)
        self.actionPredicci_n_GP.setChecked(True)
        self.actionIncertidumbre_GP = QAction(MainWindow)
        self.actionIncertidumbre_GP.setObjectName(u"actionIncertidumbre_GP")
        self.actionIncertidumbre_GP.setCheckable(True)
        self.actionIncertidumbre_GP.setChecked(True)
        self.actionFuncion_de_Adquisici_n = QAction(MainWindow)
        self.actionFuncion_de_Adquisici_n.setObjectName(u"actionFuncion_de_Adquisici_n")
        self.actionFuncion_de_Adquisici_n.setCheckable(True)
        self.action3D = QAction(MainWindow)
        self.action3D.setObjectName(u"action3D")
        self.action3D.setCheckable(False)
        self.action3D.setEnabled(False)
        self.actionAutomatic = QAction(MainWindow)
        self.actionAutomatic.setObjectName(u"actionAutomatic")
        self.actionAutomatic.setCheckable(True)
        self.actionAutomatic.setChecked(True)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 922, 26))
        self.menuArchivo = QMenu(self.menubar)
        self.menuArchivo.setObjectName(u"menuArchivo")
        self.menuEditar = QMenu(self.menubar)
        self.menuEditar.setObjectName(u"menuEditar")
        self.menuVista = QMenu(self.menubar)
        self.menuVista.setObjectName(u"menuVista")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuArchivo.menuAction())
        self.menubar.addAction(self.menuEditar.menuAction())
        self.menubar.addAction(self.menuVista.menuAction())
        self.menuArchivo.addAction(self.actionGuardar)
        self.menuArchivo.addSeparator()
        self.menuArchivo.addAction(self.actionSalir)
        self.menuEditar.addAction(self.actionReiniciar)
        self.menuVista.addAction(self.actionMapa)
        self.menuVista.addAction(self.actionPredicci_n_GP)
        self.menuVista.addAction(self.actionIncertidumbre_GP)
        self.menuVista.addAction(self.actionFuncion_de_Adquisici_n)
        self.menuVista.addSeparator()
        self.menuVista.addAction(self.actionAutomatic)
        self.menuVista.addAction(self.action3D)
        self.menuVista.addSeparator()

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Visualizador", None))
        self.actionGuardar.setText(QCoreApplication.translate("MainWindow", u"Guardar", None))
#if QT_CONFIG(shortcut)
        self.actionGuardar.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+S", None))
#endif // QT_CONFIG(shortcut)
        self.actionReiniciar.setText(QCoreApplication.translate("MainWindow", u"Reiniciar", None))
#if QT_CONFIG(shortcut)
        self.actionReiniciar.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+R", None))
#endif // QT_CONFIG(shortcut)
        self.actionSalir.setText(QCoreApplication.translate("MainWindow", u"Salir", None))
#if QT_CONFIG(shortcut)
        self.actionSalir.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+W", None))
#endif // QT_CONFIG(shortcut)
        self.actionMapa.setText(QCoreApplication.translate("MainWindow", u"Mapa", None))
        self.actionPredicci_n_GP.setText(QCoreApplication.translate("MainWindow", u"Predicci\u00f3n GP", None))
        self.actionIncertidumbre_GP.setText(QCoreApplication.translate("MainWindow", u"Incertidumbre GP", None))
        self.actionFuncion_de_Adquisici_n.setText(QCoreApplication.translate("MainWindow", u"Funcion de Adquisici\u00f3n", None))
        self.action3D.setText(QCoreApplication.translate("MainWindow", u"Send next", None))
#if QT_CONFIG(shortcut)
        self.action3D.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+N", None))
#endif // QT_CONFIG(shortcut)
        self.actionAutomatic.setText(QCoreApplication.translate("MainWindow", u"Automatic process", None))
#if QT_CONFIG(shortcut)
        self.actionAutomatic.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+S", None))
#endif // QT_CONFIG(shortcut)
        self.menuArchivo.setTitle(QCoreApplication.translate("MainWindow", u"Archivo", None))
        self.menuEditar.setTitle(QCoreApplication.translate("MainWindow", u"Editar", None))
        self.menuVista.setTitle(QCoreApplication.translate("MainWindow", u"Vista", None))
    # retranslateUi

