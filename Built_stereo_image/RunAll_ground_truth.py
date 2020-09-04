import os

for i in range(1, 5):
    os.chdir('/Users/zauber/Desktop/Stereo_matching/Built_stereo_image')
    path = './Normal_case_' + str(i)
    pythonName = 'ground_truth' + str(i)
    os.chdir(path)
    f = os.listdir()
    print(f)
    for code in f:  # 文件夹下的文件
        if os.path.splitext(code)[0] == pythonName:  # 只运行py文件
            os.system('python {}'.format(code))  # 终端运行 python main.py
            print('End Python file:', i)
print('end All Python Files')
