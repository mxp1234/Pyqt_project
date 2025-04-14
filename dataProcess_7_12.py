
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt,QThread
from PyQt5.QtGui import QIcon,QFont
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import time
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import reportlab
import tempfile
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from app_ui import *
from hypersonic import *

from nerual_speedup_bnn import BayesianRegressor,hypersonic_dobBased1
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')#必须加，否则会一直卡
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMessageBox, QWidget
from PyQt5.QtWidgets import QListView

import mplcursors
device = torch.device( "cpu")

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

class dataProcess(QMainWindow,Ui_MainWindow):
    #类的多重继承： dataProcess 类会继承 QMainWindow 类的功能（作为一个主窗口），同时也会使用 Ui_MainWindow 类中定义的 UI 布局和元素
    # my_signal = pyqtSignal(int, str, list, np.ndarray)可以定义任意类型的信号，但要保证对应
    # draw_table1_signal =  pyqtSignal(np.ndarray)
    draw_table2_fig1_signal =  pyqtSignal(np.ndarray)
    draw_table2_fig2_signal =  pyqtSignal(np.ndarray)
    neural_start_signal = pyqtSignal()
    
    def __init__(self) :
        super(dataProcess,self).__init__(parent=None)
        #我的理解是：父类初始化，即初始化QMainWindow与Ui_MainWindow，其中Q类是外部的，Ui_MainWindow是在app-ui里定义的，
        #Ui_MainWindow里面没有init，QMainWindow应该有，即初始化一个窗口。所以这里的self可以理解为由QMainWindow初始化的MainWindow
        self.setupUi(self)
        #第一个self是主函数中实例化的w，setupUi则是父类Ui_MainWindow（app_ui.py）里面定义的方法。
        #只要单纯的吧setupUi理解为一个方法就行，该方法需要一个参数（def setupUi(self, MainWindow):）参数为self而不是MainWindow
        #因为据gpt所说：setupUi 方法的第二个参数 MainWindow 实际上不需要手动传递。它会在 setupUi 方法内部根据 Ui_MainWindow 类的定义，作为一个默认的参数被传递使用。这通常是由 GUI 设计工具生成的代码，以确保正确地初始化和布局用户界面元素。
        #第二个self也是实例对象w，setupUi只需要传入一个self，表示对本体添加各式各样的控件与操作。
        
        #初始化training类，额外独立出training类的目的是能够多线程运行，training类继承qthread
        # self.train = training(200,0.001)
        self.comboBox_2.setView(QListView())
        self.comboBox_3.setView(QListView())
        self.fig_3D_path = os.path.join(tempfile.gettempdir(), 'temp_image.png')

        self.update_text_count = 0  #控制显示省略号的次数
        self.dot_count = 0  # 用于跟踪省略号的数量
        self.timer = QTimer(self)  # 创建一个定时器,用于显示“网络初始化”
        # self.timer.timeout.connect(self.update_text)  # 将定时器的timeout信号连接到更新函数
        self.timer_2 = QTimer(self)
    
        self.losses = []
        self.cd_rel = []
        self.cl_rel = []
        self.cy_rel = []
        self.cd_abs = []
        self.cl_abs = []
        self.cy_abs = []
        self.file1.clicked.connect(self.whereFile)
        self.file2.clicked.connect(self.whereFile2)
        self.button.clicked.connect(self.run)   
        self.button.setEnabled(True)
        
        self.stop_train.clicked.connect(self.stop_training)
        self.stop_train.setEnabled(False)
        
        self.erase.clicked.connect(self.erase_all)
        self.generate_file.clicked.connect(self.output_pdf_file)
        self.analysis.clicked.connect(self.output_data)
        
        self.comboBox.currentIndexChanged.connect(self.cd_bias_combo)
        self.comboBox_4.currentIndexChanged.connect(self.cl_bias_combo)
        self.comboBox_5.currentIndexChanged.connect(self.cy_bias_combo)
        self.comboBox_6.currentIndexChanged.connect(self.ro_bias_combo)
        
        self.comboBox_2.currentIndexChanged.connect(self.combo_position_param)  #combo_position_param包含了绘制图像的函数
        self.comboBox_3.currentIndexChanged.connect(self.combo_posture_param)
        self.comboBox_7.currentIndexChanged.connect(self.combo_tab_2_param)
        
        # self.generate_file.clicked.connect(self.save_to_file)
        #这下面几个run运行后即显示的信号，并不是类似button和combo的中断触发信号，run即是他们的触发指令
        #run是button click触发的，但run里面有emit，本身又相当于做为信号触发别的绘图函数，像是信号与槽的嵌套
        # self.draw_table1_signal.connect(self.draw)
        self.draw_table2_fig1_signal.connect(self.draw_table2_fig1) 
        self.draw_table2_fig2_signal.connect(self.draw_table2_fig2)
        self.neural_start_signal.connect(self.neural_start)
        # self.train.update_epoch_loss.connect(self.draw_loss)
        # # self.train.finished.connect(self.train_finished)
        # self.train.update_iteration_loss.connect(self.show_loss)
        # self.train.start_train.connect(self.print_start_train)

        self.file_path = []
        self.file_path2 = []
        
        #绘图区域初始化
        self.fig3d = plt.figure()
        self.cav3d = FigureCanvas(self.fig3d)  # FigureCanvas可以视为一个widget,图形fig的初始化放在前面，否则多次执行调用初始化会生成多个图

        self.verticalLayout_5.addWidget(self.cav3d)
        
        #第二个tab图，显示姿态角变化
        self.fig = plt.figure()
        self.cav1 = FigureCanvas(self.fig)
        self.verticalLayout_23.addWidget(self.cav1)
        
        #fig2_1为第三页tab里面的两个图
        self.fig2_1 = plt.figure()
        self.cav2_1 = FigureCanvas(self.fig2_1)
        self.verticalLayout_10.addWidget(self.cav2_1)
        
        self.fig2_2 = plt.figure()
        self.cav2_2 = FigureCanvas(self.fig2_2)
        self.verticalLayout_11.addWidget(self.cav2_2)
        
        self.fig3 = plt.figure()
        self.cav3 = FigureCanvas(self.fig3)
        # self.cav3.setStyleSheet("FigureCanvas { border: 10px solid rgb(190, 190, 190); }")
        # self.verticalLayout_20 = QtWidgets.QHBoxLayout(self.widget_10)
        self.verticalLayout_20.addWidget(self.cav3)
        
        self.fig4 = plt.figure()
        self.cav4 = FigureCanvas(self.fig4)
        # self.cav4.setStyleSheet("border: 10px solid rgb(190, 190, 190);")
        self.verticalLayout_79 = QtWidgets.QVBoxLayout(self.widget_7)
        self.verticalLayout_79.addWidget(self.cav4)
        
        self.fig5 = plt.figure()
        self.cav5 = FigureCanvas(self.fig5)
        # self.cav3.setStyleSheet("FigureCanvas { border: 10px solid rgb(190, 190, 190); }")
        # self.verticalLayout_710 = QtWidgets.QHBoxLayout(self.widget_11)
        self.verticalLayout_21.addWidget(self.cav5)
        self.ui_customize()
        
        '''
        input_data 范例： [6/57.3, 1/57.3, 2/57.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33589, 0, 5000, 0, 0, 0.03, 0, 0]
        goal_data  范例： [1/57.3,0,0]
        '''
     
    def whereFile(self):
        self.lineEdit.clear()
        initialDir = 'C:\\Users\\29961\\Desktop\\桌面文件\\data_file'
        '''记得最后此处路径删除'''
        self.file_path,_ = QFileDialog.getOpenFileNames(self,'打开轨迹文件',initialDir,'Excel files(*.xlsx)')   #,_是把后面跟着的Excel files(*.xlsx)给去了
        self.lineEdit.setText(str(self.file_path))

    def whereFile2(self): 
        initialDir = 'C:\\Users\\29961\\Desktop\\桌面文件\\data_file'
        self.file_path2,_ = QFileDialog.getOpenFileNames(self,'打开气动表文件',initialDir,'Excel files(*.xlsx)') 

    def run(self):
        epoch_max =  int(self.lineEdit_6.text())
        print("epoch",epoch_max)
        lr = float(self.lineEdit_5.text())
        print("lr",lr)
        self.train = training(epoch_max=epoch_max, lr=lr)
        self.train.update_epoch_loss.connect(self.draw_loss)
        # self.train.finished.connect(self.train_finished)
        self.train.update_iteration_loss.connect(self.show_loss)
        self.train.start_train.connect(self.print_start_train)
        #读取设定的参数
        #处理输入的初始飞行参数,第二个（绝对偏差由于不是按键触发，是即时读取的，也在这里处理）
        self.input_data = self.input.text()
        self.goal_data = self.goal.text()
        print(self.input_data)
        #cd_bias2是cd的绝对偏差
        self.cd_bias2 = float(self.lineEdit_2.text())
        self.cl_bias2 = float(self.lineEdit_3.text())
        self.cy_bias2 = float(self.lineEdit_4.text())
        
        # 如果输入框为空，则使用默认值,非空，处理成array。[2,2,2,3,4,5]形式
        if not self.input_data:
            self.input_data = x0  # 给定默认值
        else:
            self.input_data = [eval(item) for item in self.input_data.strip('[]').split(',')]
            self.input_data = np.array(self.input_data)
        if not self.goal_data:
            self.goal_data = x01  # 给定默认值
        else:
            self.goal_data = [eval(item) for item in self.goal_data.strip('[]').split(',')]
            self.goal_data = np.array(self.goal_data)
            
        self.input_data = self.input_data.astype(np.float64)
        self.goal_data = self.goal_data.astype(np.float64)
        self.cd_bias = float(self.comboBox.currentText().strip("%"))/100
        self.cl_bias = float(self.comboBox_4.currentText().strip("%"))/100
        self.cy_bias = float(self.comboBox_5.currentText().strip("%"))/100
        self.ro_bias = float(self.comboBox_6.currentText().strip("%"))/100
        print(self.ro_bias)
        #todo: 调用高超模型计算,fly_state是气动表结果，real_states是加了预设的“真实值”，但也是用hypersonic_dobBased_neural算的
        
        self.fly_states = odeint(hypersonic_dobBased,  self.input_data, tspan , args=(self.goal_data, Pid, z, mu, Vv),atol=1e-7,rtol=1e-6)
        self.real_states = odeint(hypersonic_dobBased1_numpy_forDrawing, self.input_data, tspan , args=( self.goal_data, Pid, z, mu, Vv ,[[self.cd_bias,self.cd_bias2],[self.cl_bias,self.cl_bias2],[self.cy_bias,self.cy_bias2]],self.ro_bias))
        
        self.calculate_3d_img = plot3D(self.real_states,self.fly_states,self.input_data,self.goal_data)
        self.calculate_table3 = plotTable3(self.real_states,self.fly_states,self.input_data,self.goal_data)
        
        self.train.threeDmap_signal.connect(self.calculate_3d_img.caculate_img)
        self.calculate_3d_img.plot_3d_data.connect(self.threeDmap)
        # self.calculate_3d_img.plot_3d_data.connect(self.draw)
        #plot_3d_data绑定两个槽函数，发射的data是三组数据：气动表值、真值、预测值
        self.train.draw_weight_and_bias.connect(self.calculate_table3.caculate_img)
        self.calculate_table3.plot_table3.connect(self.draw_table3)
        #检查文件（以下功能根据需求）
        # 1.当无输入文件时，弹窗提示  
        # 2.可以导入不确定数目的轨迹文件  
        # 3.输入trajectory的顺序没有区别，实际上气动表的顺序也没有区别
        # 4.检查气动表是否符合要求
        
        if not self.file_path:           
            QMessageBox.warning(self,'警告','未选择轨迹文件')
            return
        else:
            self.trajectory_file_num = len(self.file_path)
            self.df_traject = np.zeros(self.trajectory_file_num, dtype=object)
        
            for i in range(self.trajectory_file_num):
                self.df_traject[i] = np.array(pd.read_excel(self.file_path[i]))
                # self.df_traject2 = np.array(pd.read_excel(self.file_path[1]))
        
        if not self.file_path2:
            QMessageBox.warning(self,'警告','未选择气动表文件')
            return   
        else: 
            self.df_table1 = np.array(pd.read_excel(self.file_path2[0]))
            self.df_table2 = np.array(pd.read_excel(self.file_path2[1]))

        self.traject = self.df_traject.copy()
        self.table1 = self.df_table1.copy()
        self.table2 = self.df_table2.copy()
        
        # [6/57.3, 1/57.3, 2/57.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33589, 0, 5000, 0, 0, 0.03, 0, 0]
        # 滚转，俯仰，偏航，wx，wy，wz，
        
        #tab2默认显示的两张图像
        self.fly_state_one = self.fly_states[:,12]
        self.fly_state_two = self.fly_states[:,0]

        self.compare_data1 = CD_list
        self.compare_data2 = CL_list
  
        #train类，get到两个数据，一个是fly_states用于将数据训练，一个是预设偏差组成的list
        self.train.get_data(self.fly_states , [[self.cd_bias,self.cd_bias2],[self.cl_bias,self.cl_bias2],[self.cy_bias,self.cy_bias2]],self.goal_data,self.ro_bias)
        
        #输入形式[0.1,0.02],[1/57.3,0,0]
       
        
        #之所以把这两个信号的connect放这里是因为training实例化在这里。training实例化如果放在init下面，会跟下面冲突。因为training里面我还实例化了dataprocess
        # self.timer.start(500)  # 设置定时器每500毫秒触发一次
        self.neural_start_signal.emit()
        self.train.start()  #start用以执行Qthread里面的run函数
        self.stop_train.setEnabled(True)
        #假设处理数据后生成trajcet2，table2，分别用于绘制三维轨迹与二维图。两者均只能为list，可先封装为高维arrar转化为list
        #notice: 修改tab2里面的图，此为传入数据
        # self.draw_table1_signal.emit(self.compare_data1)
        self.draw_table2_fig1_signal.emit(self.fly_state_one)
        self.draw_table2_fig2_signal.emit(self.fly_state_two)
        # self.train.draw_weight_and_bias.emit(self.compare_data1,self.compare_data2)
        # 其他地方绑定好信号与槽，run里进行信号发射
       
    def erase_all(self):
        #清空lineedit，plainlinetext edit，
        self.losses = []
        self.lineEdit.clear()
        self.plainTextEdit.clear()
        self.fig3.clear()
        self.fig5.clear()
        self.cav3.draw()
        self.fig.clear()
        self.cav1.draw()
        self.fig2_1.clear()
        self.cav2_1.draw()
        self.fig2_2.clear()
        self.cav2_2.draw()
        self.fig3d.clear()
        self.cav3d.draw()
        self.fig4.clear()
        self.cav4.draw()
        
    def stop_training(self):
        self.train.stop()
        self.plainTextEdit.setPlainText("训练已停止")
    def output_data(self):
        
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Excel Files (*.xlsx);;All Files (*)", options=options)
        fileName = 'C:\\Users\\29961\\Desktop\\桌面文件\\data_file\\output_data.xlsx'
        if fileName:
            try:
                df = pd.DataFrame(self.modified_states)  # self.modified_states 是二维 numpy 数组
                df.to_excel(fileName, index=False, header=False)
            except Exception as e:
                QMessageBox.warning(self, "错误", f"保存文件时出错: {str(e)}")
        
    def output_pdf_file(self):
        text = self.plainTextEdit.toPlainText() 
        #notice: 后续记得更换路径
        pdfmetrics.registerFont(TTFont('SimSun', 'C:\WorkFile\Python_File\Formal_project\\visualize_learning\集成神经网络\SIMSUN.ttf'))
        if((self.train.on_running_flag==False)and(self.train.finished_flag==True)and(text!="")):
            options = QFileDialog.Options()
            fileName,_ = QFileDialog.getSaveFileName(self,"保存文件","","PDF Files (*.pdf);;All Files (*)", options=options)
            fileName = 'C:\\Users\\29961\\Desktop\\桌面文件\\data_file\\output_pdf.pdf'
            if fileName:
                # try:
                    c = canvas.Canvas(fileName, pagesize=letter)
                    c.setFont('SimSun', 12)
                    width, height = letter
                    y_start = height - 100
                    text_lines = text.split('\n')
                    total_text_height = len(text_lines) * 15
                    for i, line in enumerate(text_lines):
                        c.drawString(100, y_start - i * 15, line)
                    image_y_position = y_start - total_text_height - 50
                    c.drawImage(self.fig_3D_path, 100, image_y_position-400, width=400, height=400)
                    c.save()
                    
                    if os.path.exists(self.fig_3D_path):
                        os.remove(self.fig_3D_path)
                # except Exception as e:
                #     QMessageBox.warning(self, "错误", f"生成 PDF 时出错: {str(e)}")
        else:
            QMessageBox.information(self, "注意", "请检查训练是否结束 \n请检查输出框文本内容")
            return
              
    #绘制3d轨迹函数    
    def threeDmap(self,data) :
        
        self.fig3d.clear()
        data1 = data[0]
        data2 = data[1]
        data3 = data[2]
        
        ax = self.fig3d.add_subplot(projection = '3d')
       
        ax.plot(data1[:,12],data1[:,13],data1[:,14],label='风洞表轨迹')
        ax.plot(data2[:,12],data2[:,13],data2[:,14],label='真实轨迹')
        ax.plot(data3[:,12],data3[:,13],data3[:,14],label='修正后轨迹')
        
        ax.legend(loc='best')
        ax.set_xlabel('横坐标')
        ax.set_ylabel('纵坐标')
        ax.set_zlabel('相对高度')
        ax.set_title('轨迹可视化')
        self.fig3d.savefig(self.fig_3D_path)
        self.cav3d.draw()
    
    # "修正气动参数对比"这一个tab的三中姿态角对比图  
    def draw(self,data):
        self.fig.clear()
        data1 = data[0]
        data2 = data[1]
        data3 = data[2]
        combo_label = self.comboBox_7.currentText()
        ax1 = self.fig.add_subplot(111)
        if combo_label == "滚转角":
            ax1.plot(t, data1[:, 0], 'r', label='气动表')
            ax1.plot(t, data2[:, 0], 'b', label='真实值')
            ax1.plot(t, data3[:, 0], 'y', label='修正值')
        elif combo_label == "俯仰角":
            ax1.plot(t, data1[:, 1], 'r', label='气动表')
            ax1.plot(t, data2[:, 1], 'b', label='真实值')
            ax1.plot(t, data3[:, 1], 'y', label='修正值')
        elif combo_label == "偏航角":
            ax1.plot(t, data1[:, 2], 'r', label='气动表')
            ax1.plot(t, data2[:, 2], 'b', label='真实值')
            ax1.plot(t, data3[:, 2], 'y', label='修正值')
        ax1.set_xlabel('t (sec)')
        ax1.legend()
        ax1.grid()
        mplcursors.cursor(hover=True)
        # def on_scroll(event):
        #     axtemp = event.inaxes
        #     x_center = event.xdata  # 获取鼠标当前的 x 坐标
        #     y_center = event.ydata  
        #     x_min, x_max = axtemp.get_xlim()
        #     y_min, y_max = axtemp.get_ylim()
        #     x_range = x_max - x_min
        #     y_range = y_max - y_min

        #     # 缩放因子，可以根据需要调整
        #     scale_factor = 0.98 if event.button == 'up' else 1.02

        #     new_x_range = x_range * scale_factor
        #     new_y_range = y_range * scale_factor
            
        #     new_xmin = x_center - (x_center - x_min) * (new_x_range / x_range)
        #     new_xmax = x_center + (x_max - x_center) * (new_x_range / x_range)
        #     new_ymin = y_center - (y_center - y_min) * (new_y_range / y_range)
        #     new_ymax = y_center + (y_max - y_center) * (new_y_range / y_range)

        #     # 设置新的坐标范围
        #     axtemp.set_xlim(new_xmin, new_xmax)
        #     axtemp.set_ylim(new_ymin, new_ymax)

        #     self.cav1.draw()
            
        # self.cav1.mpl_connect('scroll_event', on_scroll)
        self.cav1.draw()
# axe是fig 的一个子类，表示在fig对象的子图，axe有很多属性。但是注意最终draw命令        
    
    def draw_table2_fig1(self,data):
        self.fig2_1.clear()
        combo_label = self.comboBox_2.currentText()
        ax1 = self.fig2_1.add_subplot(111)
        
        if combo_label in ["X坐标", "Y坐标", "Z坐标"]:
            ax1.plot(t, data, 'r', label='x20', linewidth=2)
        else:
            ax1.plot(t, 180 / np.pi * data, 'r', label='x20', linewidth=2)
        ax1.set_xlabel('t (sec)')
        ax1.set_ylabel(f'{combo_label}')
        ax1.grid()  
        self.cav2_1.draw()
      
    def draw_table2_fig2(self,data):
        combo_label = self.comboBox_3.currentText()
        self.fig2_2.clear()
        ax1 = self.fig2_2.add_subplot(111)        
        ax1.plot(t, 180 / np.pi * data, 'r', label='x20', linewidth=2)
        ax1.set_xlabel('t (sec)')
        ax1.set_ylabel(f'{combo_label}')
        ax1.grid()  
        self.cav2_2.draw()
        
    #绘制三条对比曲线，data3是用神经网络得到的权重和偏差，data1是原始气动表，data2是真值（预设偏差）,传入的list是神经网络训出来的系数
    def draw_table3(self, list):
        self.fig3.clear()
        self.fig5.clear()

        # 创建子图
        ax1 = self.fig3.add_subplot(111)
        ax2 = self.fig5.add_subplot(111)

        # 设置标签
        ax1.set_xlabel('训练进程')
        ax1.set_ylabel('相对气动偏差')
        ax2.set_xlabel('训练进程')
        ax2.set_ylabel('绝对气动偏差')

        # 更新数据列表
        self.cd_rel.append(list[0][0])
        self.cd_abs.append(list[0][1])
        self.cl_rel.append(list[1][0])
        self.cl_abs.append(list[1][1])
        self.cy_rel.append(list[2][0])
        self.cy_abs.append(list[2][1])

        # 定义颜色和线型
        colors = {'cd': 'red', 'cl': 'blue', 'cy': 'green'}  # 三种系数颜色
        actual_style = '--'  # 实际值用虚线
        target_style = '-'   # 目标值用实线

        # 绘制相对偏差（ax1）
        ax1.plot(self.cd_rel, color=colors['cd'], linestyle=actual_style, label='CD 相对偏差', linewidth=2, alpha=0.7)
        ax1.plot(self.cl_rel, color=colors['cl'], linestyle=actual_style, label='CL 相对偏差', linewidth=2, alpha=0.7)
        ax1.plot(self.cy_rel, color=colors['cy'], linestyle=actual_style, label='CY 相对偏差', linewidth=2, alpha=0.7)

        # 绘制绝对偏差（ax2）
        ax2.plot(self.cd_abs, color=colors['cd'], linestyle=actual_style, label='CD 绝对偏差', linewidth=2, alpha=0.7)
        ax2.plot(self.cl_abs, color=colors['cl'], linestyle=actual_style, label='CL 绝对偏差', linewidth=2, alpha=0.7)
        ax2.plot(self.cy_abs, color=colors['cy'], linestyle=actual_style, label='CY 绝对偏差', linewidth=2, alpha=0.7)

        ax1.axhline(y=self.cd_bias, color=colors['cd'], linestyle=target_style, label='CD 目标', linewidth=1.5)
        ax1.axhline(y=self.cl_bias, color=colors['cl'], linestyle=target_style, label='CL 目标', linewidth=1.5)
        ax1.axhline(y=self.cy_bias, color=colors['cy'], linestyle=target_style, label='CY 目标', linewidth=1.5)

        ax2.axhline(y=self.cd_bias2, color=colors['cd'], linestyle=target_style, label='CD 目标', linewidth=1.5)
        ax2.axhline(y=self.cl_bias2, color=colors['cl'], linestyle=target_style, label='CL 目标', linewidth=1.5)
        ax2.axhline(y=self.cy_bias2, color=colors['cy'], linestyle=target_style, label='CY 目标', linewidth=1.5)

        ax1.legend(loc='best')
        ax2.legend(loc='best')
        ax1.grid(True)
        ax2.grid(True)

        # 鼠标悬停显示数据点
        mplcursors.cursor([ax1, ax2], hover=True)

        def on_scroll(event):
            axtemp = event.inaxes
            if axtemp is None:
                return
            x_center = event.xdata
            y_center = event.ydata
            if x_center is None or y_center is None:
                return

            x_min, x_max = axtemp.get_xlim()
            y_min, y_max = axtemp.get_ylim()
            x_range = x_max - x_min
            y_range = y_max - y_min

            # 缩放因子
            scale_factor = 0.9 if event.button == 'up' else 1.1

            new_x_range = x_range * scale_factor
            new_y_range = y_range * scale_factor

            new_xmin = x_center - (x_center - x_min) * (new_x_range / x_range)
            new_xmax = x_center + (x_max - x_center) * (new_x_range / x_range)
            new_ymin = y_center - (y_center - y_min) * (new_y_range / y_range)
            new_ymax = y_center + (y_max - y_center) * (new_y_range / y_range)

            axtemp.set_xlim(new_xmin, new_xmax)
            axtemp.set_ylim(new_ymin, new_ymax)

            self.cav3.draw()
            self.cav5.draw()

        # 连接滚轮事件（为两个画布绑定）
        self.cav3.mpl_connect('scroll_event', on_scroll)
        self.cav5.mpl_connect('scroll_event', on_scroll)

        # 刷新画布
        self.cav3.draw()
        self.cav5.draw()

    def draw_loss(self,data):
        self.losses.append(data)
        self.fig4.clear()
        ax1 = self.fig4.add_subplot(111)
        ax1.plot(range(1, len(self.losses) + 1), self.losses)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel('Loss')
        ax1.grid()
        self.cav4.draw()

    #每一次迭代在文本框中显示loss，不绘图
    def show_loss(self,list):
        font = QFont()
        font.setPointSize(10)  # 设置字体大小
        font.setFamily("Arial")  # 设置字体样式，例如 "Arial"
        self.plainTextEdit.setFont(font)
        # self.plainTextEdit.setStyleSheet("color: blue;") 
        self.plainTextEdit.clear()
        cd_relative_deviation = list[0][0]
        cl_relative_deviation = list[1][0]
        cy_relative_deviation = list[2][0]
        cd_absolute_deviation = list[0][1]
        cl_absolute_deviation = list[1][1]
        cy_absolute_deviation = list[2][1]
        if self.train.finished_flag == False:
            self.plainTextEdit.setPlainText(f"""训练第{self.train.epoch+1}/{self.train.epoch_max}轮，总损失为 {self.train.total_loss} \n预设大气密度偏差 {self.ro_bias} 相对气动偏差:CD={self.cd_bias} CL={self.cl_bias} CY={self.cy_bias}，绝对气动偏差 CD={self.cd_bias2} CL={self.cl_bias2} CY={self.cy_bias2} \n
估算相对气动偏差均值：CD={cd_relative_deviation:.2f} CL={cl_relative_deviation:.2f} CY={cy_relative_deviation:.2f}, 估算绝对气动偏差均值: CD={cd_absolute_deviation:.2f}  CL={cl_absolute_deviation:.2f} CY={cy_absolute_deviation:.2f}""")
        else:
            self.plainTextEdit.setPlainText(f"""训练结束，总损失为 {self.train.total_loss} \n预设大气密度偏差 {self.ro_bias} 相对气动偏差:CD={self.cd_bias} CL={self.cl_bias} CY={self.cy_bias}，绝对气动偏差 CD={self.cd_bias2} CL={self.cl_bias2} CY={self.cy_bias2} \n 
估算相对气动偏差均值：CD={cd_relative_deviation:.2f} CL={cl_relative_deviation:.2f} CY={cy_relative_deviation:.2f}, 估算绝对气动偏差均值: CD={cd_absolute_deviation:.2f}  CL={cl_absolute_deviation:.2f} CY={cy_absolute_deviation:.2f}""")

    #开始训练提示
    def neural_start(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.setPlainText("网络初始化中")
        # time.sleep(2)
        # self.timer.start()  
    def print_start_train(self):
        
        self.plainTextEdit.clear()
        self.plainTextEdit.setPlainText("训练开始")    
        
    #偏差选择下拉菜单
    def cd_bias_combo(self):
        
        cd_bias = self.comboBox.currentText()
        if cd_bias =="10%":
            self.cd_bias = 0.1
        elif cd_bias =="5%" :
            self.cd_bias = 0.05
    def cl_bias_combo(self):
        cl_bias = self.comboBox_4.currentText()
        if cl_bias =="10%":
            self.cl_bias = 0.1
        elif cl_bias =="5%" :
            self.cl_bias = 0.05
    def cy_bias_combo(self):
        cy_bias = self.comboBox_5.currentText()
        if cy_bias =="10%":
            self.cy_bias = 0.1
        elif cy_bias =="5%" :
            self.cy_bias = 0.05 
    def ro_bias_combo(self):
        ro_bias = self.comboBox_6.currentText()
        if ro_bias =="10%":
            self.ro_bias = 0.1
        elif ro_bias =="0%" :
            self.ro_bias = 0 
    #修改下拉菜单的内容时，应该修改此处对应的气动模型里的东西。        
    def combo_position_param(self):
        index = self.comboBox_2.currentIndex()
        
        if index == 0:  #12是X，13是Y，14是Z，15是V，16弹道倾角，17弹道偏角
            self.fly_state_one = self.fly_states[:,12]
        elif index ==1:
            self.fly_state_one = self.fly_states[:,13]
        elif index ==2:
            self.fly_state_one = self.fly_states[:,14]
        elif index ==3:
            self.fly_state_one = self.fly_states[:,15]      
        elif index ==4:
            self.fly_state_one = self.fly_states[:,16] 
        elif index ==5:
            self.fly_state_one = self.fly_states[:,17]              
        
        self.draw_table2_fig1(self.fly_state_one)
        #todo: 这里传给最后一页下拉菜单的数据是fly-states
    #姿态下拉菜单    
    def combo_posture_param(self):
        index = self.comboBox_3.currentIndex()   
        
        if index ==0:
            self.fly_state_two = self.fly_states[:,0]   
        elif index ==1:
            self.fly_state_two = self.fly_states[:,1]
        elif index ==2:
            self.fly_state_two = self.fly_states[:,2]   
            
        self.draw_table2_fig2(self.fly_state_two)    
    def combo_tab_2_param(self) :
        self.draw([self.fly_states,self.real_states,self.calculate_3d_img.predicted_data])        
    def ui_customize(self):
        
        self.setStyleSheet("background-color: #FFFFE0; font-family: Arial;")
        self.resize(2000, 1200)
        # 设置标签阴影效果
        # self.label_2.setStyleSheet("color: #333; text-shadow: 1px 1px 2px grey; font-size: 40px;")
        self.widget_7.setStyleSheet("QWidget { border: 10px solid rgb(190, 190, 190); }")
        self.widget_10.setStyleSheet("QWidget { border: 10px solid rgb(190, 190, 190); }")
        self.widget_11.setStyleSheet("QWidget { border: 10px solid rgb(190, 190, 190); }")
        
        # 设置按钮样式
        self.file1.setStyleSheet("""
            QPushButton {
                background-color: #ADD8E6; 
                border: 1px solid #000; 
                border-radius: 5px;
                height: 35px; /* 添加这一行，设置按钮高度为40像素 */
            }
            QPushButton:hover {
                background-color: #E0FFFF;
            }
            QPushButton:pressed {
                background-color: #87CEEB; 
                border: 2px solid #000;
                border-radius: 5px;
                padding-left: 2px; 
                padding-top: 2px;
            }
        """)
        self.file2.setStyleSheet("""
            QPushButton {
                background-color: #ADD8E6; 
                border: 1px solid #000; 
                border-radius: 5px;
                height: 35px
            }
            QPushButton:hover {
                background-color: #E0FFFF;
            }
            QPushButton:pressed {
                background-color: #87CEEB; 
                border: 2px solid #000;
                border-radius: 5px;
                padding-left: 2px; 
                padding-top: 2px;
            }
        """)
        self.stop_train.setStyleSheet("""
            QPushButton {
                background-color: #ADD8E6; 
                border: 1px solid #000; 
                border-radius: 5px;
                height: 35px
            }
            QPushButton:hover {
                background-color: #E0FFFF;
            }
            QPushButton:pressed {
                background-color: #87CEEB; 
                border: 2px solid #000;
                border-radius: 5px;
                padding-left: 2px; 
                padding-top: 2px;
            }
        """)
        self.button.setStyleSheet("""
            QPushButton {
                background-color: #ADD8E6; 
                border: 1px solid #000; 
                border-radius: 5px;
                height: 35px;
            }
            QPushButton:hover {
                background-color: #E0FFFF;
            }
            QPushButton:pressed {
                background-color: #87CEEB; 
                border: 2px solid #000;
                border-radius: 5px;
                padding-left: 2px; 
                padding-top: 2px;
            }
        """)
        self.analysis.setStyleSheet("""
            QPushButton {
                background-color: #ADD8E6; 
                border: 1px solid #000; 
                border-radius: 5px;
                height: 35px;
            }
            QPushButton:hover {
                background-color: #E0FFFF;
            }
            QPushButton:pressed {
                background-color: #87CEEB; 
                border: 2px solid #000;
                border-radius: 5px;
                padding-left: 2px; 
                padding-top: 2px;
            }
        """)
        self.generate_file.setStyleSheet("""
            QPushButton {
                background-color: #ADD8E6; 
                border: 1px solid #000; 
                border-radius: 5px;
                height: 35px;
            }
            QPushButton:hover {
                background-color: #E0FFFF;
            }
            QPushButton:pressed {
                background-color: #87CEEB; 
                border: 2px solid #000;
                border-radius: 5px;
                padding-left: 2px; 
                padding-top: 2px;
            }
        """)
        self.erase.setStyleSheet("""
            QPushButton {
                background-color: #ADD8E6; 
                border: 1px solid #000; 
                border-radius: 5px;
                height: 35px
            }
            QPushButton:hover {
                background-color: #E0FFFF;
            }
            QPushButton:pressed {
                background-color: #87CEEB; 
                border: 2px solid #000;
                border-radius: 5px;
                padding-left: 2px; 
                padding-top: 2px;
            }
        """)

        # 设置TabWidget切换动画效果
        self.tabWidget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #87CEEB;
                border-radius: 20px;
            }
            QTabWidget::tab-bar {
                alignment: left;
            }
            QTabBar::tab {
                background-color: #87CEEB;
                color: #333;
                border: 1px solid #000;
                border-bottom: none;
                padding: 5px 10px;
                margin-right: 2px;  /* 添加间隔 */
                
            }
            QTabBar::tab:hover {
                background-color: #B0E0E6;  /* 鼠标悬停效果 */
            }
            QTabBar::tab:selected {
                background-color: #FFFFFF;  /* 更改选中标签的背景色 */
                color: #000000;  /* 更改选中标签的字体颜色 */
                font-weight: bold;  /* 确保字体加粗 */
                border: 2px solid #000000;  /* 增加选中标签的边框 */
            }
        """)
        #在 Qt 样式表中，注释应该使用 /* 和 */ ，而不是 Python 的 #
        # 设置ComboBox和LineEdit样式
        self.comboBox.setStyleSheet("""
            QComboBox {
                background-color: #F0F8FF;
                border: 1px solid #000;
                border-radius: 4px;
                padding: 2px 10px;
                color: #333;
                height: 30px;
            }
            QComboBox:hover {
                border: 2px solid #555; /* 鼠标悬停时边框颜色和深度变化 */
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left-width: 1px;
                border-left-color: #000;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            QComboBox::down-arrow {
                image: url(':\icon.png'); /* 确保提供正确的图片路径 */
                width : 5px;
                height: 5px;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #000;
                selection-background-color: #87CEEB;
                selection-color: #FFFFFF;
            }
        """)
        self.comboBox_2.setStyleSheet("""
            QComboBox {
                background-color: #F0F8FF;
                border: 2px solid #888;
                border-radius: 4px;
                padding: 2px 10px;
                color: #333;
                height: 30px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left-width: 1px;
                border-left-color: #000;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            QComboBox::down-arrow {
                image: url(path_to_arrow_image);
            }
            QComboBox QAbstractItemView {
                border: 1px solid #000;
                selection-background-color: #87CEEB;
                selection-color: #FFFFFF;
            }
        """)
        self.comboBox_7.setStyleSheet("""
            QComboBox {
                background-color: #F0F8FF;
                border: 2px solid #888;
                border-radius: 4px;
                padding: 2px 10px;
                color: #333;
                height: 30px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left-width: 1px;
                border-left-color: #000;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            QComboBox::down-arrow {
                image: url(path_to_arrow_image);
            }
            QComboBox QAbstractItemView {
                border: 1px solid #000;
                selection-background-color: #87CEEB;
                selection-color: #FFFFFF;
            }
        """)
        self.comboBox_3.setStyleSheet("""
            QComboBox {
                background-color: #F0F8FF;
                border: 2px solid #888;
                border-radius: 4px;
                padding: 2px 10px;
                color: #333;
                height: 30px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left-width: 1px;
                border-left-color: #000;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }

            QComboBox QAbstractItemView {
                border: 1px solid #000;
                selection-background-color: #87CEEB;
                selection-color: #FFFFFF;
            }
        """)
        self.comboBox_4.setStyleSheet("""
            QComboBox {
                background-color: #F0F8FF;
                border: 2px solid #888;
                border-radius: 4px;
                padding: 2px 10px;
                color: #333;
                height: 30px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left-width: 1px;
                border-left-color: #000;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }

            QComboBox QAbstractItemView {
                border: 1px solid #000;
                selection-background-color: #87CEEB;
                selection-color: #FFFFFF;
            }
        """)
        self.comboBox_5.setStyleSheet("""
            QComboBox {
                background-color: #F0F8FF;
                border: 2px solid #888;
                border-radius: 4px;
                padding: 2px 10px;
                color: #333;
                height: 30px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left-width: 1px;
                border-left-color: #000;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }

            QComboBox QAbstractItemView {
                border: 1px solid #000;
                selection-background-color: #87CEEB;
                selection-color: #FFFFFF;
            }
        """)        
        self.input.setStyleSheet("""
    QLineEdit {
        background-color: #F0F8FF;
        border: 1px solid #000;
        color: black;
        border-radius: 10px; /* 圆弧边框 */
        padding: 5px; /* 增加内边距以增加高度 */
        height: 30px; /* 可调整以适合您的界面 */
    }
    QLineEdit:hover {
        border: 2px solid #555; /* 鼠标悬停时边框颜色和深度变化 */
    }
""")
        self.goal.setStyleSheet("""
    QLineEdit {
        background-color: #F0F8FF;
        border: 1px solid #000;
        color: black;
        border-radius: 10px; /* 圆弧边框 */
        padding: 5px; /* 增加内边距以增加高度 */
        height: 30px; /* 可调整以适合您的界面 */
    }
    QLineEdit:hover {
        border: 2px solid #555; /* 鼠标悬停时边框颜色和深度变化 */
    }
""")
        self.comboBox_6.setStyleSheet("""
            QComboBox {
                background-color: #F0F8FF;
                border: 2px solid #888;
                border-radius: 4px;
                padding: 2px 10px;
                color: #333;
                height: 30px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left-width: 1px;
                border-left-color: #000;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }

            QComboBox QAbstractItemView {
                border: 1px solid #000;
                selection-background-color: #87CEEB;
                selection-color: #FFFFFF;
            }
        """)   
        self.lineEdit.setStyleSheet("""
    QLineEdit {
        background-color: #F0F8FF;
        border: 1px solid #000;
        color: black;
        border-radius: 10px; /* 圆弧边框 */
        padding: 5px; /* 增加内边距以增加高度 */
        height: 30px; /* 可调整以适合您的界面 */
    }
    QLineEdit:hover {
        border: 2px solid #555; /* 鼠标悬停时边框颜色和深度变化 */
    }
""")
        self.lineEdit_2.setStyleSheet("""
    QLineEdit {
        background-color: #F0F8FF;
        border: 1px solid #000;
        color: black;
        border-radius: 10px; /* 圆弧边框 */
        padding: 5px; /* 增加内边距以增加高度 */
        height: 30px; /* 可调整以适合您的界面 */
    }
    QLineEdit:hover {
        border: 2px solid #555; /* 鼠标悬停时边框颜色和深度变化 */
    }
""")
        self.lineEdit_3.setStyleSheet("""
    QLineEdit {
        background-color: #F0F8FF;
        border: 1px solid #000;
        color: black;
        border-radius: 10px; /* 圆弧边框 */
        padding: 5px; /* 增加内边距以增加高度 */
        height: 30px; /* 可调整以适合您的界面 */
    }
    QLineEdit:hover {
        border: 2px solid #555; /* 鼠标悬停时边框颜色和深度变化 */
    }
""")
        self.lineEdit_5.setStyleSheet("""
    QLineEdit {
        background-color: #F0F8FF;
        border: 1px solid #000;
        color: black;
        border-radius: 10px; /* 圆弧边框 */
        padding: 5px; /* 增加内边距以增加高度 */
        height: 30px; /* 可调整以适合您的界面 */
    }
    QLineEdit:hover {
        border: 2px solid #555; /* 鼠标悬停时边框颜色和深度变化 */
    }
""")
        self.lineEdit_6.setStyleSheet("""
    QLineEdit {
        background-color: #F0F8FF;
        border: 1px solid #000;
        color: black;
        border-radius: 10px; /* 圆弧边框 */
        padding: 5px; /* 增加内边距以增加高度 */
        height: 30px; /* 可调整以适合您的界面 */
    }
    QLineEdit:hover {
        border: 2px solid #555; /* 鼠标悬停时边框颜色和深度变化 */
    }
""")
        self.lineEdit_4.setStyleSheet("""
    QLineEdit {
        background-color: #F0F8FF;
        border: 1px solid #000;
        color: black;
        border-radius: 10px; /* 圆弧边框 */
        padding: 5px; /* 增加内边距以增加高度 */
        height: 30px; /* 可调整以适合您的界面 */
    }
    QLineEdit:hover {
        border: 2px solid #555; /* 鼠标悬停时边框颜色和深度变化 */
    }
""")

    def euler_method(hypersonic_dobBased, x, tspan, x01, Pid, z, mu, Vv):
        num_time_points = len(tspan)

        for i in range(1, num_time_points):
            dt = tspan[i] - tspan[i-1]
            x = x + dt * hypersonic_dobBased(x, tspan[i-1], x01, Pid, z, mu, Vv)

        return x           

#Training类被实例化时会创建一个finished信号，可以在其他地方连接这个信号。run方法会在调用start()方法时执行
class training(QThread):
    start_train = pyqtSignal()
    update_iteration_loss = pyqtSignal(list)
    update_epoch_loss = pyqtSignal(float)
    draw_weight_and_bias = pyqtSignal(list)
    finished = pyqtSignal()
    threeDmap_signal = pyqtSignal(list)

    def __init__(self,epoch_max,lr):
        super().__init__()
        self.total_loss = 0
        self.epoch_max = epoch_max
        self.finished_flag = False
        self.on_running_flag = False
        self.stop_training = False  # 添加一个停止训练的标志
        self.last_weight_bias = []
        self.lr = lr
    
    def get_data(self, data,biases , goal_data,ro_bias):
        self.data = data
        self.bias = biases
        self.goal = goal_data
        self.ro_bais = ro_bias
    def stop(self):
        self.stop_training = True  # 设置停止标志
    def run(self):

        
        # 模拟数据
        # dataset = torch.randn(10, 30)  
        dataset = self.data
        X_train = torch.Tensor(dataset[:, :])
        X_test = torch.Tensor(dataset[:, :])
        Force_narual,M,_ = hypersonic_dobBased1(X_train[:, :21], X_train[:, 20], [1 / 57.3, 0, 0], Pid, z, mu, Vv,
                                         [0,0,0, 0,0,0],0)
        print(dataset.shape)
        
        iteration = 0
        model = BayesianRegressor(self.goal)
        ds_train = torch.utils.data.TensorDataset(X_train, X_test[:])

        optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        Force, _, ddx = hypersonic_dobBased1(X_train[:, :21], X_train[:, 20], [1 / 57.3, 0, 0], Pid, z, mu, Vv,
                                         [self.bias[0][0],self.bias[0][1],self.bias[1][0],self.bias[1][1],self.bias[2][0],self.bias[2][1]],self.ro_bais)
        
        self.start_train.emit()
        a1 = time.time()
        weight_and_bias_list_print = [[0.0,0.0],[0.0,0.0],[0.0,0.0]]
        self.on_running_flag = True
        for self.epoch in range(self.epoch_max):
            model.train()                
            if self.stop_training:  # 检查停止标志
                    print("训练已被停止")
                    break  
            for i in range(10):
                optimizer.zero_grad()
                #notice: self.look是贝叶斯网络训出来的w1，w2，w3和b1，b2，b3的均值方差
                dx,self.look = model(X_train,Force_narual,M)  
                mse_loss_value = torch.nn.MSELoss()(dx,ddx.squeeze())
                self.total_loss = mse_loss_value
                self.total_loss.backward()
                optimizer.step()
                weight_and_bias_list = [[float(self.look[4 * i + 2 * j]) for j in range(2)] for i in range(3)]
                iteration += 1
                if iteration % 10 == 0:
                    print(f"Epoch [{self.epoch + 1}/{self.epoch_max}], Total Loss: {self.total_loss.item()}")
                    self.update_iteration_loss.emit(weight_and_bias_list_print)
                if iteration % 100 == 0:
                    print(self.look)
                    self.draw_weight_and_bias.emit(weight_and_bias_list)   #画图
                    self.update_iteration_loss.emit(weight_and_bias_list)   #文字显示loss
                    weight_and_bias_list_print = weight_and_bias_list  #用于第二行持续显示估计的相对偏差和绝对偏差
                    self.threeDmap_signal.emit(weight_and_bias_list)
            scheduler.step
            self.update_epoch_loss.emit(self.total_loss)

        # 模拟神经网络计算出的权重和偏置,self.look是返回的权重与偏置
        #list返回tensor([-0.9632,  1.1559,  0.2541, -0.2209, -0.8764,  0.9795, -0.7057, -1.1039,-0.3829,  0.4067, -1.1582, -0.6953])
        self.draw_weight_and_bias.emit(weight_and_bias_list)
        self.last_weight_bias = weight_and_bias_list
        a2 = time.time()
        print("total time",a2-a1)

        self.finished_flag = True
        self.on_running_flag = False
        '''
        try:

            dataset = self.data
            X_train = torch.Tensor(dataset[0:50, :])  # 输入就是一个轨迹的全部数据，尚不知为啥
            
            print(dataset.shape)

            iteration = 0
            model = BayesianRegressor()
            optimizer = optim.AdamW(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
            #
            self.epoch_max = 10
            for self.epoch in range(self.epoch_max):
                model.train()
                for i in range(X_train.shape[0]):

                    optimizer.zero_grad()

                    # 前向传播
                    err,self.look = model(X_train)

                    # 生成模拟数据，假设均值为1，方差为0.5
                    # 计算损失
                    mse_loss_value = torch.nn.MSELoss()(err[0], err[1])  # 对均值使用均方误差
                    # nll_loss_value = F.gaussian_nll_loss(mean, labels, variance)  # 对方差使用负对数似然损失
                    self.total_loss =  mse_loss_value
            # #
                    # 反向传播和参数更新
                    self.total_loss.backward()
                    optimizer.step()
                    iteration += 1
                    if iteration % 1 == 0:
                        print(f"Epoch [{self.epoch + 1}/{self.epoch_max}], Total Loss: {self.total_loss.item()}")
                    if iteration % 10 == 0:
                        print(self.look)

                scheduler.step
                self.updata_loss.emit(self.total_loss) 
            weight_and_bias_list = [[float(self.look[4*i+2*j]) for j in range(2)] for i in range(3)]   
            self.draw_weight_and_bias.emit(weight_and_bias_list)
        except:
            print("error")
        self.finished.emit()
        
        '''

class plot3D(QThread):
    plot_3d_data = pyqtSignal(list)
    def __init__(self,real_states,fly_states,input_data,goal_data) -> None:
        super().__init__()
        self.predicted_data = []
        self.real_states = real_states
        self.fly_states = fly_states
        self.input_data = input_data
        self.goal_data = goal_data
    
    def caculate_img(self,list):
        self.data3 = odeint(hypersonic_dobBased_neural,  self.input_data, tspan , args=( self.goal_data, Pid, z, mu, Vv, list),atol=1e-7,rtol=1e-6)
        self.data2 = self.real_states
        self.data1 = self.fly_states
        self.plot_3d_data.emit([self.data1,self.data2,self.data3])
        self.predicted_data = self.data3
class plotTable3(QThread):
    plot_table3 = pyqtSignal(list)
    def __init__(self,real_states,fly_states,input_data,goal_data) -> None:
        super().__init__()
    
        self.real_states = real_states
        self.fly_states = fly_states
        self.input_data = input_data
        self.goal_data = goal_data
    
    def caculate_img(self,list):
        # self.data3 = odeint(hypersonic_dobBased_neural,  self.input_data, tspan , args=( self.goal_data, Pid, z, mu, Vv, list),atol=1e-7,rtol=1e-6)
        # self.data2 = self.real_states
        # self.data1 = self.fly_states
        self.plot_table3.emit(list)
