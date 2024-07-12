<div align="center">

  <h1>PYQT_Project</h1>


  <p>
    <a href="https://github.com/misitebao/yakia/blob/main/LICENSE">
      <img alt="GitHub" src="https://img.shields.io/github/license/misitebao/yakia"/>
    </a>
    <a href="https://github.com/PyQt5/PyQt">
      <img alt="GitHub" src="https://img.shields.io/badge/blog-pyqt-green.svg"/>
    </a>
    <a href="https://www.codefactor.io/repository/github/misitebao/yakia">
      <img src="https://www.codefactor.io/repository/github/misitebao/yakia/badge" alt="CodeFactor" />
    </a>
    <a href="https://discord.gg/zRC5BfDhEu">
      <img alt="Discord" src="https://img.shields.io/discord/1044100035140390952">
    </a>
    <br/>
    
    
    
  </p>

  <div>
  <strong>
  <samp>

  </samp>
  </strong>
  </div>
</div>

## Table of Contents

<details>
  <summary>Click me to Open/Close the directory listing</summary>

- [Table of Contents](#table-of-contents)
- [Introductions](#introductions)
- [Structure](#structure)
- [Getting Started](#getting-started)
  - [Things to know](#things-to-know)
- [Maintainer](#maintainer)
- [License](#license)

</details>

## Introductions

This project implements a Qt visual interactive interface with a Bayesian deep learning module, which can export data files


## Structure

The project mainly includes the GUI part, the model training part, and the link between the two.

**Main structure:** app_ui.ui compiles and generates a callable python file app_ui.py , which contains the interface style set only through qt_creator. dataProcess_7_12.py is the main logic processing part of the program, including the main thread: datapreocess and 3 sub-threads training, plot3D, plotTable3. The main thread is set in the dataprocess class, including further refinement of the UI interface, and declares all globally bound signals and slots.

- **training** is responsible for training the model and returning signals according to the training process. plot3D and plotTable3 are responsible for asynchronously executing the data required for drawing the interface, which can avoid the interface being unresponsive due to the calculation required for drawing by the main thread.
- **plot3D and plotTable3** are relatively simple in content, which calculate the data required for drawing. Solve a differential equation.

## Getting Started

<mark>Before starting the project, a certain pyqt foundation is required

You need to pay attention to the libraries imported by each file in the project. For example, in order to export the software output content as a PDF file, you need to use the following command:


```markdown
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple reportlab
```

### Things to know

| File Name               | Style Preview                                                                                                  |
| ----------------------- | -------------------------------------------------------------------------------------------------------------- |
| hypersonic.py          | Contains a complete aerodynamic calculation model      |
| nerual_speedup_bnn.py   |  Contains a Bayesian network architecture|
| app_ui copy.ui | UI file built using Qt Creator |
| airparameter.py       | Important aerodynamic calculation parameter files |
| SIMSUN.ttf        |    Font Files  |

## Maintainer

Thanks to the maintainers of these projects:

<a href="https://github.com/mxp1234">
  <img src="https://github.com/mxp1234.png" width="40" height="40" alt="misitebao" title="misitebao"/>
</a>

<details>
  <summary>Click me to Open/Close the maintainer listing</summary>

- [pyqt_preoject](https://github.com/mxp1234) - Maintainer of pyqt_preoject.

</details>



## License

[License MIT](../LICENSE)

