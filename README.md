# NLU-2026 Laboratories

This repo contains all the laboratory lectures for the Natural Language Understanding course held at the University of Trento.
<br>

## Getting started

We suggest you install [Anaconda](https://www.anaconda.com/download) on your laptop and import the conda environment that we have prepared for you. The reason for this is to give you the same library versions that we used to test the labs. However, if you are not a conda lover, you can manually install on your favourite virtual env the libraries listed in the `requirements.txt` file.

```bash
conda env create -f nlu_env.yaml -n nlu26
conda activate nlu26
```
If you have a Mac or a Windows or you do not have a dedicated Nvidia gpu, you can install the environment in this way:

```bash
conda create -n nlu26 python=3.10.13
conda activate nlu26
pip install -r requirements_no_cuda.txt
```

To launch a lab run this line of code:
```bash
jupyter notebook
```


Then, you have to choose the lab that you want to open.

<br>

## Repo organization
In the repo `labs`, you can find the notebooks for each lab session and in `solutions` you can find the same notebooks with the solutions.
<br>

The solutions of each lab will be uploaded after the corresponding lab lecture.


## Exam

### Instructions

There will be one project assignment that will be graded (80% of the final exam grade) and Q&A during the oral exam on any of the topics covered in the class (lectures and labs) (20% of the final exam grade).
The mandatory project consists of two parts that are presented in lab 4 (LM) and 5 (NLU).

You will write a total of 2 reports: one for LM (1.A + 1.B) and one for NLU (2.A + 2.B). Each report must follow the LaTeX template in the zip folder `report_template.zip`. In particular, you have to write a mini-report of **max 1 page** (references, tables and images are excluded from the count) in which you explain all the parts of the project giving more weight to the part with higher points. **Reports longer than 1 page will not be evaluated**. The purpose of this is to give you a way to report **cleanly** the results and give you space to describe what you have done and/or the originality that you have added to the exercise. You can find more detail about the sections and relative content in the LaTeX template.

### Grading
The final grade is based on:
- Project (80%): Code and Report review;
- Q&A at the exam (20%):
    -  The questions will be related to the delivered project, and any of the topics covered in the class and during the labs.

### Use of AI tools
Use of AI tools is allowed but must be declared in the report.
You are responsible for all the code you deliver, and we may ask questions about your code.

### Submission format

The delivery must follow the directory schema that you can find in `exam/studentID_name_surname.zip`.

The `LM` and `NLU` folders contain two sub-folders one for part A and the other for part B.   Inside them, there are the following files and folders: `main.py`, `functions.py` ,  `utils.py`,  `model.py`,  `README.md`, `/dataset` and `/bin`.

- `utils.py`: you have to put all the functions needed to preprocess and load the dataset
- `model.py`: the class of the model defined in PyTorch.
- `functions.py`: you have to write all the other required functions (taken from the notebook and/or written on your own) needed to complete the exercise.
- `main.py`: you have to write the calls to the functions needed to output the results asked by the exercise.
- `README.md`: you may want to write a message for us related to your solution (optional).
- `/dataset`: the files of that dataset that you used.
- `/bin`: the binary files of the best models that you have trained.

 The **reports** have to be placed in the corresponding folders i.e. into the folders `LM` and `NLU`.

**Last but not least**, the code has to be **well-written** and **documented** with comments. Furthermore, the script has to **run** without bugs otherwise the exercise will not be evaluated. Jupyter notebooks are not accepted.

<br>

### How to submit
To deliver your project you have to:

1. register for the exam on esse3.
2. fill out this [Google form](https://forms.gle/6wvPsq49Xy85JLsA7).

The project must be delivered **7 days before** the date of the exam session that you want to attend. You can do multiple submissions as we will check only the last one.

<br>

## Acknowledgements
The notebooks that you can find here are an adaptation of the labs created by our colleagues [Evgeny A. Stepanov](https://github.com/esrel), and [Gabriel Roccabruna](https://github.com/BrownFortress). Also, we want to thank all the students who gave us feedback for improving these notebooks.
